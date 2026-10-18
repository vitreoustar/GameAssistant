"""A4 LightGBM LTR 排序层:训练 + 离线评测。

流程:
  1. 按**用户**留一(与 A3.1/A3.2 同一套协议),再把用户切成 train/valid/test
     —— 用户之间完全不重叠,所以不存在用户级泄漏;
  2. 对每个用户跑一次召回,得到 200 个候选;
  3. 标签 = 被藏起来那个物品(相关性等级由它的 playtime 分档给出),其余候选为 0;
  4. 训练 LightGBM `lambdarank`(query = 用户),用 valid 的 NDCG 早停;
  5. 在 test 用户上对比:随机 / 热门 / 协同单路 / 召回融合 / LTR 排序。

**为什么用"1 正 + N 负"的 LTR**:这是隐式反馈排序的标准做法(NCF/BPR 系列同款)。
随机留一的正样本噪声较大,但**对所有方法同等**地加噪,所以方法间的相对比较仍然成立
(绝对数字不能当线上预期,原因见 evaluate.py 的 MNAR 说明)。

⚠️ 已知泄漏边界(业界常见,如实记录):
  * 协同的 item-item 矩阵、内容 TopK 都是在**全量交互**上算的(含各用户的测试边),
    会让协同类特征略微偏乐观;
  * `train_popularity` 特征用的是**训练用户**的交互计数,这一项是干净的。

用法:
  cd backend
  python scripts/train_ranker.py
  python scripts/train_ranker.py --num-boost-round 300 --limit-users 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommend.evaluate import leave_one_out_split  # noqa: E402
from app.recommend.features import playtime_bucket  # noqa: E402
from app.recommend.index import RecommendIndex, cat_rows_of  # noqa: E402
from app.recommend.rank import FEATURE_NAMES, FeatureBuilder, Ranker  # noqa: E402
from app.recommend.recall import DEFAULT_QUOTAS, RecallConfig, RecallEngine  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERACTIONS_PATH = BACKEND_DIR / "data" / "interactions" / "interactions.parquet"
ITEMS_PATH = BACKEND_DIR / "data" / "interactions" / "items.parquet"
OUT_DIR = BACKEND_DIR / "data" / "recommend"
KS = (10, 20, 50)


# ------------------------------------------------------------------ 数据准备


def load_inputs(args):
    index = RecommendIndex.load()
    items = pd.read_parquet(ITEMS_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    interactions["game_idx"] = cat_rows_of(
        index.catalog["appid"].to_numpy(), items["appid"].to_numpy()
    )[interactions["game_idx"].to_numpy()]
    split = leave_one_out_split(
        interactions, min_items=args.min_items, seed=args.seed, n_items=index.n_items
    )
    return index, split


def split_users(users: np.ndarray, rng: np.random.Generator, ratios=(0.7, 0.15, 0.15)):
    """按用户切分(而不是按交互对)—— 同一用户不能同时出现在训练和测试里。"""
    perm = rng.permutation(len(users))
    n_train = int(len(users) * ratios[0])
    n_valid = int(len(users) * ratios[1])
    return (
        users[perm[:n_train]],
        users[perm[n_train : n_train + n_valid]],
        users[perm[n_train + n_valid :]],
    )


def quality_from_counts(index, counts):
    catalog = index.catalog
    pop = np.log1p(counts)
    pop = pop / pop.max() if pop.max() > 0 else pop
    owners = np.log1p(catalog["estimated_owners_lower"].to_numpy(dtype=np.float64))
    owners = owners / owners.max() if owners.max() > 0 else owners
    return (pop + catalog["bayes_rating"].to_numpy(dtype=np.float64) + 0.5 * owners).astype(
        np.float32
    )


def build_dataset(
    users, split, engine, builder, *, popularity_index: dict[int, int], verbose_every=2000
):
    """对一批用户各跑一次召回 + 特征,拼成 LTR 训练集。

    `popularity_index` 把 user_idx 映射到它在 split.users 里的位置(取测试物品用)。
    """
    indptr, indices, data = (
        split.train_user_item.indptr,
        split.train_user_item.indices,
        split.train_user_item.data,
    )
    feats, labels, groups = [], [], []
    start = time.time()
    for done, user in enumerate(users, 1):
        u = int(user)
        library = indices[indptr[u] : indptr[u + 1]]
        weights = data[indptr[u] : indptr[u + 1]]
        candidates = engine.recall_rows(library, weights, owned_rows=library)
        if not candidates:
            continue
        rows = np.array([c.cat_idx for c in candidates], dtype=np.int64)
        context = builder.build(library, weights, rows)
        target = int(split.test_game[popularity_index[u]])
        label = np.zeros(len(rows), dtype=np.float32)
        hit = np.flatnonzero(rows == target)
        if len(hit):
            # 相关性等级用被藏物品的游玩时长分档(而不是 0/1),给 LTR 更多信息
            label[hit[0]] = float(playtime_bucket(float(split.test_weight[popularity_index[u]])) + 1)
        feats.append(context.features)
        labels.append(label)
        groups.append(len(rows))
        if verbose_every and done % verbose_every == 0:
            print(f"      {done:,}/{len(users):,} 用户  已用 {time.time()-start:.0f}s")
    if not feats:
        raise SystemExit("没有构造出任何训练样本")
    return np.vstack(feats), np.concatenate(labels), np.array(groups, dtype=np.int32)


# ------------------------------------------------------------------ 评测


def evaluate_rankings(rank_fn, users, split, popularity_index, ks=KS) -> dict:
    """用给定的打分函数对召回候选重排,统计 candidate→ranking 的口径。"""
    indptr, indices, data = (
        split.train_user_item.indptr,
        split.train_user_item.indices,
        split.train_user_item.data,
    )
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = 0.0
    covered = {k: set() for k in ks}
    evaluated = 0
    for user in users:
        u = int(user)
        library = indices[indptr[u] : indptr[u + 1]]
        weights = data[indptr[u] : indptr[u + 1]]
        order = rank_fn(library, weights)
        if len(order) == 0:
            continue
        evaluated += 1
        target = int(split.test_game[popularity_index[u]])
        where = np.flatnonzero(order == target)
        position = int(where[0]) if len(where) else 10**9
        for k in ks:
            if position < k:
                hits[k] += 1
                ndcg[k] += 1.0 / np.log2(position + 2)
            covered[k].update(order[:k].tolist())
        if position < 10**9:
            mrr += 1.0 / (position + 1)
    n_items = split.n_items
    out = {"n_users": evaluated}
    for k in ks:
        out[f"recall@{k}"] = hits[k] / max(evaluated, 1)
        out[f"ndcg@{k}"] = ndcg[k] / max(evaluated, 1)
        out[f"coverage@{k}"] = len(covered[k]) / n_items
    out["mrr"] = mrr / max(evaluated, 1)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="A4 LTR 排序层训练与评测")
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-users", type=int, default=0, help="只用前 N 个合格用户")
    parser.add_argument("--total-candidates", type=int, default=200)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "eval_rank.json")
    args = parser.parse_args()

    print("[1/6] 加载索引与留一切分 ...")
    index, split = load_inputs(args)
    rng = np.random.default_rng(args.seed)
    users = split.users
    if args.limit_users > 0:
        users = users[: args.limit_users]
    train_u, valid_u, test_u = split_users(users, rng)
    print(f"      用户 {len(users):,} = 训练 {len(train_u):,} / 验证 {len(valid_u):,} / 测试 {len(test_u):,}")

    # user_idx -> 它在 split.users 里的位置
    popularity_index = {int(u): i for i, u in enumerate(split.users)}

    print("[2/6] 构建召回引擎与特征构造器 ...")
    # 热度只用**训练用户**的交互统计,避免把验证/测试用户的行为当特征(泄漏红线)
    train_users_mask = np.zeros(split.n_users, dtype=bool)
    train_users_mask[train_u] = True
    train_counts = np.bincount(
        split.train_pairs[train_users_mask[split.train_pairs[:, 0]], 1],
        minlength=index.n_items,
    ).astype(np.float64)
    engine = RecallEngine(
        index,
        quality_scores=quality_from_counts(index, train_counts),
        config=RecallConfig(quotas=dict(DEFAULT_QUOTAS), total=args.total_candidates),
    )
    builder = FeatureBuilder(index, engine, train_popularity=train_counts)
    print(f"      特征维度 {len(FEATURE_NAMES)} | 候选上限 {args.total_candidates}")

    print("[3/6] 构造训练集(逐用户召回 + 特征) ...")
    start = time.time()
    x_train, y_train, g_train = build_dataset(
        train_u, split, engine, builder, popularity_index=popularity_index
    )
    x_valid, y_valid, g_valid = build_dataset(
        valid_u, split, engine, builder, popularity_index=popularity_index, verbose_every=0
    )
    print(f"      训练 {x_train.shape[0]:,} 行 x {x_train.shape[1]} 列 / "
          f"验证 {x_valid.shape[0]:,} 行  (耗时 {time.time()-start:.0f}s)")
    print(f"      正样本:训练 {int((y_train>0).sum()):,} / 验证 {int((y_valid>0).sum()):,}")

    print("[4/6] 训练 LightGBM lambdarank ...")
    train_set = lgb.Dataset(x_train, label=y_train, group=g_train,
                            feature_name=list(FEATURE_NAMES), free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, group=g_valid, reference=train_set,
                           free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": args.learning_rate,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "lambdarank_truncation_level": 30,
        "verbosity": -1,
        "seed": args.seed,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=args.num_boost_round,
        valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                   lgb.log_evaluation(50)],
    )
    print(f"      best_iteration = {booster.best_iteration}")

    print("[5/6] 在测试用户上对比各方案 ...")

    def make_recall_order(config=None):
        """把配置**绑进闭包**,而不是改 engine 的实例状态。

        曾经的写法是临时改 `engine.config` 再赋值给 baselines,结果恢复得太早,
        `collab_only` / `popular_only` / 基线跑出了完全相同的数字 —— 共享可变状态
        导致的静默串台。
        """
        def order_fn(library, weights):
            candidates = engine.recall_rows(
                library, weights, owned_rows=library, config=config
            )
            return np.array([c.cat_idx for c in candidates], dtype=np.int64)
        return order_fn

    def make_ltr_order(model):
        def order_fn(library, weights):
            cands = engine.recall_rows(library, weights, owned_rows=library)
            if not cands:
                return np.empty(0, dtype=np.int64)
            rows = np.array([c.cat_idx for c in cands], dtype=np.int64)
            context = builder.build(library, weights, rows)
            scores = model.predict(context.features)
            return rows[np.argsort(-scores)]
        return order_fn

    def single_channel(channel):
        return RecallConfig(
            quotas={c: (args.total_candidates if c == channel else 0)
                    for c in DEFAULT_QUOTAS},
            total=args.total_candidates,
        )

    baselines = {
        "recall_merge(基线)": make_recall_order(),
        "collab_only": make_recall_order(single_channel("collab_itemknn")),
        "popular_only": make_recall_order(single_channel("quality")),
        "ltr_lambdarank": make_ltr_order(booster),
    }

    results = {"protocol": {"n_train_users": int(len(train_u)),
                            "n_valid_users": int(len(valid_u)),
                            "n_test_users": int(len(test_u)),
                            "n_features": len(FEATURE_NAMES),
                            "best_iteration": int(booster.best_iteration)},
               "baselines": {}}
    header = "  " + f"{'method':<24}" + "".join(
        f"{'r@'+str(k):>9}" for k in KS
    ) + "".join(f"{'ndcg@'+str(k):>10}" for k in KS) + f"{'mrr':>8}"
    print()
    print(header)
    for name, fn in baselines.items():
        metrics = evaluate_rankings(fn, test_u, split, popularity_index)
        results["baselines"][name] = metrics
        row = f"  {name:<24}"
        row += "".join(f"{metrics[f'recall@{k}']:>8.2%} " for k in KS)
        row += "".join(f"{metrics[f'ndcg@{k}']:>9.4f} " for k in KS)
        row += f"{metrics['mrr']:>7.3f}"
        print(row)

    print("[6/6] 保存模型与特征重要性 ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_DIR / "ranker.txt"), num_iteration=booster.best_iteration)
    importance = sorted(
        zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain").tolist()),
        key=lambda kv: -kv[1],
    )
    results["feature_importance_gain"] = [
        {"feature": name, "gain": round(gain, 2)} for name, gain in importance
    ]
    print("      特征重要性 (gain) top 12:")
    for name, gain in importance[:12]:
        print(f"        {name:<26} {gain:>12.1f}")
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"      模型 -> {OUT_DIR / 'ranker.txt'}")
    print(f"      结果 -> {args.out}")


if __name__ == "__main__":
    main()
