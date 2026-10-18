"""A6 端到端嵌入配置裁决:把每套 embedding 装进**完整管线**,重训排序器,比端到端指标。

这是 A3.1 挂起的那个决策的收口。单通道口径下 cur128 与 desc256 各有胜负
(recall@10 选 cur128、recall@50 选 desc256),所以当时决定"等端到端指标再定"。
现在管线齐了,直接看最终排序结果。

公平性保证(必须严格做到,否则比的是别的东西):
  * **同一套用户切分**(同 seed)、同一批测试用户;
  * **同一套 LightGBM 参数**与轮数上限;
  * 换 embedding 时**必须连 content TopK 一起重算** —— 只换向量不换 TopK
    等于比"半新半旧"的管线;
  * 除 embedding 外,其余 A2 矩阵(协同 TopK、目录、质量先验)完全不动。

用法:
  cd backend
  python scripts/eval_embedding_config.py
  python scripts/eval_embedding_config.py --configs cur128 desc256 --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommend.evaluate import leave_one_out_split  # noqa: E402
from app.recommend.index import (  # noqa: E402
    RecommendIndex,
    cat_rows_of,
    content_topk_from_embeddings,
)
from app.recommend.rank import FEATURE_NAMES, FeatureBuilder  # noqa: E402
from app.recommend.recall import DEFAULT_QUOTAS, RecallConfig, RecallEngine  # noqa: E402
from scripts.train_ranker import (  # noqa: E402  # noqa: E402
    KS,
    build_dataset,
    evaluate_rankings,
    quality_from_counts,
    split_users,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERACTIONS_PATH = BACKEND_DIR / "data" / "interactions" / "interactions.parquet"
ITEMS_PATH = BACKEND_DIR / "data" / "interactions" / "items.parquet"
OUT_DIR = BACKEND_DIR / "data" / "recommend"
CONFIG_EMB_DIR = OUT_DIR / "config_emb"
TOPK_CACHE_DIR = OUT_DIR / "config_topk"

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "lambdarank_truncation_level": 30,
    "verbosity": -1,
}


def load_config_embeddings(index: RecommendIndex, config: str) -> np.ndarray:
    """取该配置的全库向量;cur128 直接用 A2 索引里的(省一次编码)。"""
    if config == "cur128":
        return np.asarray(index.content_emb)
    path = CONFIG_EMB_DIR / f"{config}.npy"
    if not path.exists():
        raise SystemExit(
            f"找不到 {path};请先运行 scripts/eval_recall.py --configs {config} 生成缓存"
        )
    return np.load(path)


def patch_index_for_config(
    index: RecommendIndex, config: str, k: int, use_cache: bool = True
) -> RecommendIndex:
    """返回一个换了内容向量与内容 TopK 的索引副本(其余矩阵不动)。"""
    cache = TOPK_CACHE_DIR / f"{config}_k{k}.npz"
    embeddings = load_config_embeddings(index, config)
    if use_cache and cache.exists():
        cached = np.load(cache)
        return replace(
            index,
            content_emb=embeddings,
            content_topk_idx=cached["indices"],
            content_topk_score=cached["scores"],
        )
    has_content = index.catalog["has_content"].to_numpy()
    start = time.time()
    idx, scores = content_topk_from_embeddings(embeddings, has_content, k)
    print(f"      重算 content TopK 用时 {time.time()-start:.0f}s")
    TOPK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, indices=idx, scores=scores)
    return replace(
        index,
        content_emb=embeddings,
        content_topk_idx=idx,
        content_topk_score=scores,
    )


def run_pipeline(
    index: RecommendIndex,
    interactions: pd.DataFrame,
    *,
    seed: int,
    min_items: int,
    total_candidates: int,
    num_boost_round: int,
    early_stopping: int,
    limit_users: int,
    config: str,
) -> dict:
    """完整跑一遍:留一切分 -> 召回 -> 特征 -> 训练 -> 端到端评测。"""
    split = leave_one_out_split(
        interactions, min_items=min_items, seed=seed, n_items=index.n_items
    )
    users = split.users if limit_users <= 0 else split.users[:limit_users]
    rng = np.random.default_rng(seed)
    train_u, valid_u, test_u = split_users(users, rng)
    popularity_index = {int(u): i for i, u in enumerate(split.users)}

    mask = np.zeros(split.n_users, dtype=bool)
    mask[train_u] = True
    train_counts = np.bincount(
        split.train_pairs[mask[split.train_pairs[:, 0]], 1], minlength=index.n_items
    ).astype(np.float64)

    engine = RecallEngine(
        index,
        quality_scores=quality_from_counts(index, train_counts),
        config=RecallConfig(quotas=dict(DEFAULT_QUOTAS), total=total_candidates),
    )
    builder = FeatureBuilder(index, engine, train_popularity=train_counts)

    x_train, y_train, g_train = build_dataset(
        train_u, split, engine, builder, popularity_index=popularity_index, verbose_every=0
    )
    x_valid, y_valid, g_valid = build_dataset(
        valid_u, split, engine, builder, popularity_index=popularity_index, verbose_every=0
    )

    params = dict(LGB_PARAMS, seed=seed)
    train_set = lgb.Dataset(
        x_train, label=y_train, group=g_train,
        feature_name=list(FEATURE_NAMES), free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, group=g_valid,
        reference=train_set, free_raw_data=False,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    def make_ltr_order(model):
        def order_fn(library, weights):
            cands = engine.recall_rows(library, weights, owned_rows=library)
            if not cands:
                return np.empty(0, dtype=np.int64)
            rows = np.array([c.cat_idx for c in cands], dtype=np.int64)
            scores = model.predict(builder.build(library, weights, rows).features)
            return rows[np.argsort(-scores)]
        return order_fn

    def recall_order(library, weights):
        return np.array(
            [c.cat_idx for c in engine.recall_rows(library, weights, owned_rows=library)],
            dtype=np.int64,
        )

    return {
        "config": config,
        "seed": seed,
        "n_train_users": int(len(train_u)),
        "n_test_users": int(len(test_u)),
        "n_train_rows": int(x_train.shape[0]),
        "best_iteration": int(booster.best_iteration),
        "recall_merge": evaluate_rankings(recall_order, test_u, split, popularity_index),
        "ltr": evaluate_rankings(make_ltr_order(booster), test_u, split, popularity_index),
        "feature_importance_gain": [
            {"feature": name, "gain": round(gain, 1)}
            for name, gain in sorted(
                zip(FEATURE_NAMES,
                    booster.feature_importance(importance_type="gain").tolist()),
                key=lambda kv: -kv[1],
            )
        ],
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="A6 端到端嵌入配置裁决")
    parser.add_argument("--configs", nargs="*", default=["cur128", "cur256", "desc256"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--total-candidates", type=int, default=200)
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--early-stopping", type=int, default=60)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "eval_embedding_config.json")
    args = parser.parse_args()

    print("[1/3] 加载索引与交互 ...")
    base_index = RecommendIndex.load()
    items = pd.read_parquet(ITEMS_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    interactions["game_idx"] = cat_rows_of(
        base_index.catalog["appid"].to_numpy(), items["appid"].to_numpy()
    )[interactions["game_idx"].to_numpy()]
    print(f"      目录 {base_index.n_items:,} | 交互 {len(interactions):,}")

    results: dict = {"configs": {}, "protocol": {
        "configs": args.configs, "seeds": args.seeds,
        "total_candidates": args.total_candidates,
        "num_boost_round": args.num_boost_round,
        "note": "同一套用户切分与 LightGBM 参数;仅 embedding(+其派生的 content TopK)不同",
    }}

    print("[2/3] 逐配置重建索引副本 ...")
    patched: dict[str, RecommendIndex] = {}
    for config in args.configs:
        print(f"  -> {config}")
        patched[config] = patch_index_for_config(base_index, config, k=base_index.manifest["topk"])

    print("[3/3] 逐配置跑完整管线 ...")
    for config in args.configs:
        for seed in args.seeds:
            print(f"\n  === {config} (seed={seed}) ===")
            start = time.time()
            outcome = run_pipeline(
                patched[config], interactions,
                seed=seed, min_items=args.min_items,
                total_candidates=args.total_candidates,
                num_boost_round=args.num_boost_round,
                early_stopping=args.early_stopping,
                limit_users=args.limit_users, config=config,
            )
            outcome["seconds"] = round(time.time() - start, 1)
            results["configs"].setdefault(config, {})[str(seed)] = outcome
            merge, ltr = outcome["recall_merge"], outcome["ltr"]
            print(f"      best_iter={outcome['best_iteration']}  "
                  f"训练 {outcome['n_train_rows']:,} 行  测试 {outcome['n_test_users']:,} 用户"
                  f"  ({outcome['seconds']}s)")
            print("      " + f"{'':<10}" + "".join(f"{'r@'+str(k):>9}" for k in KS)
                  + "".join(f"{'ndcg@'+str(k):>10}" for k in KS) + f"{'mrr':>8}")
            for label, m in (("融合基线", merge), ("LTR", ltr)):
                row = f"      {label:<10}"
                row += "".join(f"{m[f'recall@{k}']:>8.2%} " for k in KS)
                row += "".join(f"{m[f'ndcg@{k}']:>9.4f} " for k in KS)
                row += f"{m['mrr']:>7.3f}"
                print(row)

    print("\n  三配置汇总(LTR,跨 seed 均值)")
    print("  " + f"{'config':<12}" + "".join(f"{'r@'+str(k):>9}" for k in KS)
          + "".join(f"{'ndcg@'+str(k):>10}" for k in KS) + f"{'mrr':>8}")
    summary = {}
    for config, per_seed in results["configs"].items():
        runs = [v["ltr"] for v in per_seed.values()]
        agg = {key: float(np.mean([r[key] for r in runs]))
               for key in runs[0] if isinstance(runs[0][key], float)}
        summary[config] = agg
        row = f"  {config:<12}"
        row += "".join(f"{agg[f'recall@{k}']:>8.2%} " for k in KS)
        row += "".join(f"{agg[f'ndcg@{k}']:>9.4f} " for k in KS)
        row += f"{agg['mrr']:>7.3f}"
        print(row)
    results["summary_ltr"] = summary

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")


if __name__ == "__main__":
    main()
