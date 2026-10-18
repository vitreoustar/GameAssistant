"""A3.2 多路召回层的离线评测:单通道消融 + 融合对比。

对每个测试用户用训练边构造游戏库,跑一遍召回,看被藏起的那个物品有没有进候选集。
指标是**候选生成**口径(`recall@K` 与覆盖率),不是排序口径 —— 排序交给 A4。

⚠️ 与 A3.1 同样的警告:全库口径下热度主导,`quality` 通道会天然很强。
所以除了 recall 还要一起看**覆盖率**(`quality` 的覆盖率极低,因为它给所有人推同一批),
以及下面 `--config merge=score` 的对照。

用法:
  cd backend
  python scripts/eval_recall_channels.py                 # 单通道消融 + 两种融合
  python scripts/eval_recall_channels.py --limit 2000     # 快速试跑
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommend.evaluate import leave_one_out_split  # noqa: E402
from app.recommend.index import RecommendIndex  # noqa: E402
from app.recommend.recall import (  # noqa: E402
    CHANNEL_ORDER,
    RecallConfig,
    RecallEngine,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERACTIONS_PATH = BACKEND_DIR / "data" / "interactions" / "interactions.parquet"
ITEMS_PATH = BACKEND_DIR / "data" / "interactions" / "items.parquet"
OUT_DIR = BACKEND_DIR / "data" / "recommend"
KS = (20, 50, 100)


def quality_prior(index: RecommendIndex, popularity_counts: np.ndarray) -> np.ndarray:
    """质量/热度先验 = 热度 + 好评率 + 销量,各自归一化后相加。

    `popularity_counts` 由调用方给:评测时传**训练集**计数,线上传全量。
    `bayes_rating` / `estimated_owners_lower` 是物品侧统计(来自 Steam 评价与销量估计),
    与我们的留一切分无关,所以不存在泄漏。
    """
    catalog = index.catalog
    pop = np.log1p(popularity_counts)
    pop = pop / pop.max() if pop.max() > 0 else pop
    owners = np.log1p(catalog["estimated_owners_lower"].to_numpy(dtype=np.float64))
    owners = owners / owners.max() if owners.max() > 0 else owners
    return (pop + catalog["bayes_rating"].to_numpy(dtype=np.float64) + 0.5 * owners).astype(
        np.float32
    )


def evaluate_engine(engine: RecallEngine, split, users, ks=KS) -> dict:
    """逐用户跑召回,统计候选生成口径的 recall@K 与覆盖率@K。"""
    indptr = split.train_user_item.indptr
    indices = split.train_user_item.indices
    data = split.train_user_item.data
    n_items = split.n_items
    max_k = max(ks)
    hits = {k: np.zeros(len(users), dtype=bool) for k in ks}
    top_items = np.full((len(users), max_k), -1, dtype=np.int32)
    sizes = np.zeros(len(users), dtype=np.int32)

    for i, u in enumerate(users):
        rows = indices[indptr[u] : indptr[u + 1]]
        weights = data[indptr[u] : indptr[u + 1]]
        candidates = engine.recall_rows(rows, weights, owned_rows=rows)
        sizes[i] = len(candidates)
        order = np.array([c.cat_idx for c in candidates], dtype=np.int32)
        top_items[i, : min(len(order), max_k)] = order[:max_k]
        target = split.test_game[i]
        where = np.flatnonzero(order == target)
        position = int(where[0]) if len(where) else 10**9
        for k in ks:
            hits[k][i] = position < k

    metrics = {
        "n_users": int(len(users)),
        "mean_candidates": float(sizes.mean()),
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(hits[k].mean())
        seen = top_items[:, :k]
        metrics[f"coverage@{k}"] = float(
            len(np.unique(seen[seen >= 0])) / n_items
        )
    return metrics


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="A3.2 多路召回评测")
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 个用户(0=全部)")
    parser.add_argument("--total", type=int, default=200, help="融合候选总数")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "eval_recall_channels.json")
    args = parser.parse_args()

    print("[1/3] 加载索引与交互 ...")
    index = RecommendIndex.load()
    items = pd.read_parquet(ITEMS_PATH)
    from app.recommend.index import cat_rows_of

    interactions = pd.read_parquet(INTERACTIONS_PATH)
    interactions["game_idx"] = cat_rows_of(
        index.catalog["appid"].to_numpy(), items["appid"].to_numpy()
    )[interactions["game_idx"].to_numpy()]

    split = leave_one_out_split(
        interactions, min_items=args.min_items, seed=args.seed, n_items=index.n_items
    )
    users = split.users if args.limit <= 0 else split.users[: args.limit]
    print(f"      评测用户 {len(users):,} | 目录 {index.n_items:,} | 融合总数 {args.total}")

    print("[2/3] 构建召回引擎(先验用训练集计数,避免泄漏) ...")
    train_counts = np.bincount(
        split.train_pairs[:, 1], minlength=index.n_items
    ).astype(np.float64)
    engine = RecallEngine(
        index,
        quality_scores=quality_prior(index, train_counts),
        config=RecallConfig(total=args.total),
    )

    print("[3/3] 单通道消融 + 融合 ...")
    runs: dict[str, RecallConfig] = {}
    for channel in CHANNEL_ORDER:
        runs[f"only:{channel}"] = RecallConfig(
            quotas={c: (args.total if c == channel else 0) for c in CHANNEL_ORDER},
            merge="quota",
            total=args.total,
        )
    runs["merge:quota"] = RecallConfig(merge="quota", total=args.total)
    runs["merge:score"] = RecallConfig(merge="score", total=args.total)
    # 边际贡献消融:去掉某一路后融合结果掉多少,直接回答"这一路值不值它的配额"
    from app.recommend.recall import DEFAULT_QUOTAS

    runs["quota_no_content"] = RecallConfig(
        quotas={**DEFAULT_QUOTAS, "content_user": 0, "content_itemknn": 0},
        merge="quota", total=args.total,
    )
    runs["quota_no_collab"] = RecallConfig(
        quotas={**DEFAULT_QUOTAS, "collab_itemknn": 0},
        merge="quota", total=args.total,
    )
    runs["quota_no_publisher"] = RecallConfig(
        quotas={**DEFAULT_QUOTAS, "publisher": 0},
        merge="quota", total=args.total,
    )

    results: dict = {
        "protocol": {
            "min_items": args.min_items, "seed": args.seed,
            "n_users": int(len(users)), "total_candidates": args.total,
        },
        "runs": {},
    }
    header = "  " + f"{'run':<20}" + "".join(
        f"{'recall@'+str(k):>11}" for k in KS
    ) + "".join(f"{'cov@'+str(k):>10}" for k in KS)
    print()
    print(header)
    for name, config in runs.items():
        engine.config = config
        start = time.time()
        metrics = evaluate_engine(engine, split, users)
        metrics["seconds"] = round(time.time() - start, 1)
        results["runs"][name] = metrics
        row = f"  {name:<20}"
        row += "".join(f"{metrics[f'recall@{k}']:>10.2%} " for k in KS)
        row += "".join(f"{metrics[f'coverage@{k}']:>9.1%} " for k in KS)
        print(row + f"  ({metrics['seconds']}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  结果已写入 {args.out}")


if __name__ == "__main__":
    main()
