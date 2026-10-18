"""A6 补充:用同一套留一协议评测**现有 RAG 方案**,作为"改造前"的对照。

忠实复刻 `app/agent/graph.py` 的召回逻辑:
  * 画像:取游玩时长 top-30,统计类型频率取 top-6
  * 多查询:1 条类型查询 + top-3 游戏的 "Games similar to X"
  * 每条查询走 chroma 语义检索,合并去重后按 distance 升序

**唯一为了公平而做的调整**:单查询 `k` 从应用里的 8 提到 50,让候选预算(≈200)
与新方案的召回层一致 —— 否则比的是"候选池大小",而不是"检索质量"。
这一调整对 RAG 是**有利**的(给它更多候选),所以如果新方案仍然赢,结论更稳。

用法:
  cd backend
  python scripts/eval_rag_baseline.py
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
from app.recommend.index import RecommendIndex, cat_rows_of  # noqa: E402
from scripts.train_ranker import KS, evaluate_rankings, split_users  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERACTIONS_PATH = BACKEND_DIR / "data" / "interactions" / "interactions.parquet"
ITEMS_PATH = BACKEND_DIR / "data" / "interactions" / "items.parquet"
OUT_DIR = BACKEND_DIR / "data" / "recommend"

PROFILE_SIZE = 30        # 画像取样数(与 graph.py 的 PROFILE_SAMPLE_SIZE 一致)
TOP_GENRES = 6
QUERY_K = 50             # 单查询候选数(应用里是 8,这里放大以对齐候选预算)


def build_profile_queries(library_rows, weights, catalog, names) -> list[str]:
    """复刻 graph.py 的 _build_profile + _build_queries(用目录里的类型,不调 Steam)。"""
    order = np.argsort(-np.asarray(weights))[:PROFILE_SIZE]
    sample = np.asarray(library_rows)[order]

    genre_counts: dict[str, int] = {}
    for row in sample:
        raw = catalog["genres"].iloc[int(row)]
        for g in str(raw).split(","):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:TOP_GENRES]

    queries: list[str] = []
    if top_genres:
        queries.append(f"Games in genres: {', '.join(top_genres[:3])}")
    for row in np.asarray(library_rows)[np.argsort(-np.asarray(weights))[:3]]:
        queries.append(f"Games similar to {names[int(row)]}")
    return queries[:4]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="A6 RAG 基线评测")
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--query-k", type=int, default=QUERY_K)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "eval_rag_baseline.json")
    args = parser.parse_args()

    print("[1/4] 加载索引 / 交互 / chroma ...")
    index = RecommendIndex.load()
    catalog = index.catalog
    names = catalog["name"].to_numpy()
    items = pd.read_parquet(ITEMS_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    interactions["game_idx"] = cat_rows_of(
        catalog["appid"].to_numpy(), items["appid"].to_numpy()
    )[interactions["game_idx"].to_numpy()]

    from app.rag import store as rag_store

    collection = rag_store.get_client().get_collection(rag_store.COLLECTION)
    print(f"      目录 {index.n_items:,} | chroma {collection.count():,}")

    split = leave_one_out_split(
        interactions, min_items=args.min_items, seed=args.seed, n_items=index.n_items
    )
    users = split.users if args.limit_users <= 0 else split.users[: args.limit_users]
    rng = np.random.default_rng(args.seed)
    _, _, test_u = split_users(users, rng)
    popularity_index = {int(u): i for i, u in enumerate(split.users)}
    print(f"      测试用户 {len(test_u):,}")

    print("[2/4] 为测试用户构造多查询(与 graph.py 同逻辑) ...")
    indptr, indices, data = (
        split.train_user_item.indptr,
        split.train_user_item.indices,
        split.train_user_item.data,
    )
    per_user_queries: dict[int, list[str]] = {}
    all_queries: list[str] = []
    start = time.time()
    for user in test_u:
        u = int(user)
        library = indices[indptr[u] : indptr[u + 1]]
        weights = data[indptr[u] : indptr[u + 1]]
        queries = build_profile_queries(library, weights, catalog, names)
        per_user_queries[u] = queries
        all_queries.extend(queries)
    print(f"      共 {len(all_queries):,} 条查询 ({time.time()-start:.0f}s)")

    print("[3/4] 批量编码 + 批量检索 chroma ...")
    from app.rag.embedder import get_model

    model = get_model()
    start = time.time()
    vectors = model.encode(
        all_queries, normalize_embeddings=True, batch_size=128,
        show_progress_bar=False, convert_to_numpy=True,
    )
    print(f"      编码用时 {time.time()-start:.0f}s")

    start = time.time()
    hits: dict[str, list[tuple[int, float]]] = {}
    offset = 0
    with_queries = [(u, per_user_queries[u]) for u in test_u]
    BATCH = 128
    for i in range(0, len(with_queries), BATCH):
        chunk = with_queries[i : i + BATCH]
        chunk_vectors, owners = [], []
        for u, qs in chunk:
            chunk_vectors.extend(vectors[offset : offset + len(qs)])
            owners.extend([u] * len(qs))
            offset += len(qs)
        result = collection.query(
            query_embeddings=np.asarray(chunk_vectors, dtype=np.float32),
            n_results=args.query_k,
            include=["distances"],
        )
        for owner, ids, dists in zip(owners, result["ids"], result["distances"]):
            bucket = hits.setdefault(owner, [])
            for doc_id, dist in zip(ids, dists):
                bucket.append((int(doc_id), float(dist)))
    print(f"      检索用时 {time.time()-start:.0f}s")

    print("[4/4] 评测 ...")
    # evaluate_rankings 的回调只拿到 (library, weights),所以先把"库指纹 -> 用户"建好。
    # 用带权重的指纹降低碰撞概率,并断言无碰撞(碰撞会让两个用户共用一份排序表)。
    owner_by_library: dict[tuple, int] = {}
    collisions = 0
    for u in test_u:
        u = int(u)
        rows = indices[indptr[u] : indptr[u + 1]]
        fingerprint = tuple(sorted(zip(rows.tolist(), np.round(data[indptr[u] : indptr[u + 1]], 4).tolist())))
        if fingerprint in owner_by_library:
            collisions += 1
        owner_by_library[fingerprint] = u
    if collisions:
        raise SystemExit(f"有 {collisions} 个测试用户的库指纹碰撞,评测不可信")

    appid_to_row = {int(a): i for i, a in enumerate(catalog["appid"].to_numpy())}

    def rank_fn(library, weights):
        fingerprint = tuple(sorted(zip(
            np.asarray(library).tolist(), np.round(np.asarray(weights), 4).tolist()
        )))
        u = owner_by_library.get(fingerprint)
        if u is None:
            return np.empty(0, dtype=np.int64)
        best: dict[int, float] = {}
        for appid, dist in hits.get(u, []):
            row = appid_to_row.get(appid)
            if row is None:
                continue
            if row not in best or dist < best[row]:
                best[row] = dist
        ordered = [row for row, _ in sorted(best.items(), key=lambda kv: kv[1])]
        return np.array(ordered, dtype=np.int64)

    metrics = evaluate_rankings(rank_fn, test_u, split, popularity_index)
    print()
    print("  " + f"{'method':<22}" + "".join(f"{'r@'+str(k):>9}" for k in KS)
          + "".join(f"{'ndcg@'+str(k):>10}" for k in KS) + f"{'mrr':>8}")
    row = f"  {'rag_multiquery(旧)':<22}"
    row += "".join(f"{metrics[f'recall@{k}']:>8.2%} " for k in KS)
    row += "".join(f"{metrics[f'ndcg@{k}']:>9.4f} " for k in KS)
    row += f"{metrics['mrr']:>7.3f}"
    print(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "protocol": {
                    "min_items": args.min_items, "seed": args.seed,
                    "n_test_users": int(len(test_u)), "query_k": args.query_k,
                    "note": "单查询 k 由应用的 8 提到 50,以对齐候选预算(对新方案不利、对旧方案有利)",
                },
                "rag_multiquery": metrics,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  结果 -> {args.out}")


if __name__ == "__main__":
    main()
