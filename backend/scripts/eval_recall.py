"""A3 第一步:离线召回评测 + 嵌入配置裁决。

先用真实 leave-one-out 交互算 Recall@K,回答两个问题:
  1. 各召回通道(随机/热门/内容/协同)的相对强弱与覆盖率;
  2. **嵌入配置到底选哪个** —— A2 发现文本被 128 token 静默截断,但"类型重合率"
     这个代理指标反而在放宽截断后下降(A2 §待裁决),必须用真实 Recall 裁决。

三个候选配置(描述 800 字 ≈ 194 token):
  cur128  现模板 + 128      = 当前线上(chroma 也是这个),作为基线
  cur256  现模板 + 256      放宽截断,但描述仍排在 Genres/Categories 之后
  desc256 描述前置 + 256    描述提到最前,让它在截断里活下来

`--check-config cur128` 会把重算出的向量与 A2 索引里的向量做一致性比对,
用来验证评测链路本身没有搭错。

用法:
  cd backend
  python scripts/eval_recall.py                       # 只跑通道基线(快)
  python scripts/eval_recall.py --configs cur128 cur256 desc256
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

from app.recommend.evaluate import (  # noqa: E402
    evaluate_bucketed,
    evaluate_scores,
    leave_one_out_split,
    make_collab_score_fn,
    make_content_score_fn,
    make_popularity_score_fn,
    make_random_score_fn,
    topk_to_sparse,
)
from app.recommend.index import RecommendIndex, cat_rows_of  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BACKEND_DIR / "data" / "corpus.jsonl"
ITEMS_PATH = BACKEND_DIR / "data" / "interactions" / "items.parquet"
INTERACTIONS_PATH = BACKEND_DIR / "data" / "interactions" / "interactions.parquet"
OUT_DIR = BACKEND_DIR / "data" / "recommend"

DEFAULT_KS = (10, 20, 50)
CONFIGS = ("cur128", "cur256", "desc256")


def interactions_in_catalog_space(interactions: pd.DataFrame, index: RecommendIndex):
    """把交互表的 `game_idx` 从『交互物品空间』重映射到『目录空间』。

    项目里存在两个物品下标空间(见 ROADMAP §3-A2):
      * 交互物品空间 `0..9,934`  —— `items.parquet` 的 `game_idx`
      * 目录空间     `0..11,443` —— `catalog.parquet` 的 `cat_idx`,A2 所有矩阵的下标

    混用会静默错位(轻则 shape 不匹配,重则把分数算在错的物品上)。评测全程走目录
    空间,和 `content_emb` / `collab_topk` 对齐。
    """
    items = pd.read_parquet(ITEMS_PATH)
    cat_rows = cat_rows_of(
        index.catalog["appid"].to_numpy(), items["appid"].to_numpy()
    )
    unmapped = int((cat_rows < 0).sum())
    if unmapped:
        raise SystemExit(f"有 {unmapped} 个交互物品找不到目录行,索引与交互表不匹配")
    out = interactions.copy()
    out["game_idx"] = cat_rows[interactions["game_idx"].to_numpy()]
    return out


def parse_corpus_parts(path: Path) -> pd.DataFrame:
    """把 corpus.jsonl 的 text 拆回 (name, Genres 行, Categories 行, 描述)。

    语料里只存了拼好的 text,而对照实验要重新排列模板,所以必须拆开。
    """
    docs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    names, genres, categories, descs = [], [], [], []
    for doc in docs:
        lines = doc["text"].split("\n")
        name = lines[0]
        g = next((x for x in lines[1:] if x.startswith("Genres:")), "")
        c = next((x for x in lines[1:] if x.startswith("Categories:")), "")
        consumed = 1 + (1 if g else 0) + (1 if c else 0)
        names.append(name)
        genres.append(g)
        categories.append(c)
        descs.append("\n".join(lines[consumed:]) if len(lines) > consumed else "")
    return pd.DataFrame(
        {
            "appid": [d["appid"] for d in docs],
            "name": names,
            "genres_line": genres,
            "categories_line": categories,
            "description": descs,
        }
    )


def build_text(parts: pd.DataFrame, *, description_first: bool) -> list[str]:
    texts = []
    for name, g, c, desc in zip(
        parts["name"], parts["genres_line"], parts["categories_line"], parts["description"]
    ):
        head = [name]
        tail = [x for x in (g, c) if x]
        body = [desc[:800]] if desc else []
        texts.append("\n".join(head + (body + tail if description_first else tail + body)))
    return texts


CONFIG_EMB_DIR = OUT_DIR / "config_emb"


def embed_config(config: str, index: RecommendIndex, *, use_cache: bool = True) -> np.ndarray:
    """按配置重算**全库**向量:corpus 物品重编码,薄文本物品复用 A2 的向量。

    薄文本物品的文本是 name+Genres+Categories(约 50~80 token),三个配置下都不会
    碰到截断上限,所以它们的向量不随配置变化 —— 复用即可,省掉 6,082 次编码。
    结果缓存到磁盘:一次编码要 2~3 分钟,而换 seed 重跑评测不该再付这个代价。
    """
    from app.rag.embedder import get_model

    cache = CONFIG_EMB_DIR / f"{config}.npy"
    if use_cache and cache.exists():
        cached = np.load(cache)
        if cached.shape == (index.n_items, index.content_emb.shape[1]):
            print(f"      复用缓存 {cache.name}")
            return cached

    parts = parse_corpus_parts(CORPUS_PATH)
    appids = index.catalog["appid"].to_numpy()
    row_of = {int(a): i for i, a in enumerate(appids)}

    embeddings = np.asarray(index.content_emb).copy()
    rows = np.array([row_of[int(a)] for a in parts["appid"] if int(a) in row_of])

    if config == "cur128":
        max_len, desc_first = 128, False
    elif config == "cur256":
        max_len, desc_first = 256, False
    elif config == "desc256":
        max_len, desc_first = 256, True
    else:
        raise SystemExit(f"未知配置 {config}")

    model = get_model()
    model.max_seq_length = max_len
    texts = build_text(parts[parts["appid"].isin(row_of)], description_first=desc_first)
    print(f"      编码 {len(texts):,} 条 corpus 文本 (max_seq_length={max_len}, "
          f"描述前置={desc_first}) ...")
    start = time.time()
    vectors = model.encode(
        texts, normalize_embeddings=True, batch_size=64,
        show_progress_bar=False, convert_to_numpy=True,
    )
    print(f"      用时 {time.time() - start:.1f}s")
    embeddings[rows] = np.asarray(vectors, dtype=np.float32)
    CONFIG_EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    return embeddings


def format_row(name: str, m: dict, ks) -> str:
    cells = [f"{name:<10}"]
    for k in ks:
        cells.append(f"{m[f'recall@{k}']:>8.2%}")
    cells.append(f"{m['ndcg@20']:>8.3f}")
    cells.append(f"{m['mrr']:>7.3f}")
    cells.append(f"{m['median_rank']:>7.0f}")
    cells.append(f"{m['coverage@20']:>8.1%}")
    return "  ".join(cells)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="方向 A 离线召回评测")
    parser.add_argument("--configs", nargs="*", default=[], choices=CONFIGS)
    parser.add_argument("--min-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neg", type=int, default=100,
                        help="协议 B 每个测试物品配的同热度负样本数")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0],
                        help="配置对比用的随机种子(检稳定性)")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略 config_emb/ 缓存,强制重新编码")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "eval_recall.json")
    args = parser.parse_args()

    print("[1/3] 加载索引与交互 ...")
    index = RecommendIndex.load()
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    interactions = interactions_in_catalog_space(interactions, index)
    print(f"      目录 {index.n_items:,} 物品 | 交互 {len(interactions):,} 行"
          f" (已重映射到目录空间)")

    print(f"[2/3] 按用户留一切分 (min_items={args.min_items}, seed={args.seed}) ...")
    split = leave_one_out_split(
        interactions, min_items=args.min_items, seed=args.seed, n_items=index.n_items
    )
    print(f"      评测用户 {len(split.users):,} / {split.n_users:,}"
          f" | 训练边 {len(split.train_pairs):,} | 测试边 {len(split.test_pairs):,}")

    print("[3/3] 评测 ...")
    ks = DEFAULT_KS
    n_items = split.n_items
    results: dict = {
        "protocol": {
            "min_items": args.min_items,
            "seed": args.seed,
            "n_users": int(len(split.users)),
            "n_items": int(n_items),
            "n_train_edges": int(len(split.train_pairs)),
            "note": "MNAR 数据集,绝对 Recall 偏乐观,只用于相对比较",
        },
        "channels": {},
        "embedding_configs": {},
    }

    header = ["channel   "] + [f"recall@{k:<2}" for k in ks] + \
             ["ndcg@20 ", "mrr    ", "med_rank", "cov@20  "]
    print()
    print("  【协议 A】全库排序(受热度主导,只看相对强弱与覆盖率)")
    print("  " + "  ".join(header))

    channels = {
        "random": make_random_score_fn(n_items, seed=args.seed),
        "popular": make_popularity_score_fn(split),
        "content": make_content_score_fn(np.asarray(index.content_emb), split),
        "collab": make_collab_score_fn(
            topk_to_sparse(index.collab_topk_idx, index.collab_topk_score, n_items),
            split,
        ),
    }
    for name, fn in channels.items():
        t = time.time()
        m = evaluate_scores(fn, split, ks=ks)
        m["seconds"] = round(time.time() - t, 1)
        results["channels"][name] = m
        print("  " + format_row(name, m, ks))

    print()
    print(f"  【协议 B】热度匹配候选集(1 正 + {args.n_neg} 同热度桶负样本,"
          f"随机命中率 ≈ {10 / (args.n_neg + 1):.1%})")
    print("  channel     recall@10  recall@20   lift@10")
    for name, fn in channels.items():
        m = evaluate_bucketed(
            fn, split, n_neg=args.n_neg, ks=(10, 20), seed=args.seed
        )
        results["channels"][name]["bucketed"] = m
        print(f"  {name:<10}  {m['recall@10']:>8.2%}  {m['recall@20']:>8.2%}"
              f"  {m['lift_over_chance@10']:>8.2f}x")

    if args.configs:
        print()
        print("  嵌入配置对比(内容通道,多 seed 检验稳定性):")
        print("  ⚠️ 内容通道在 hybrid 里是**候选生成器**,最终排序交给 A4 的 LTR,")
        print("     所以选型该看 recall@50(通道配额),而不是 recall@10(那是排序指标)")
        print("  config      seed   recall@10  recall@20  recall@50   lift@10   (协议 B)")
        config_embeddings = {}
        for config in args.configs:
            embeddings = embed_config(config, index, use_cache=not args.no_cache)
            config_embeddings[config] = embeddings
            if config == "cur128" and "cur128_selfcheck_cosine_mean" not in results["embedding_configs"]:
                cos = np.sum(embeddings * np.asarray(index.content_emb), axis=1)
                has = index.catalog["has_content"].to_numpy()
                print(f"      [自检] cur128 与 A2 索引向量余弦: 均值 {cos[has].mean():.4f} "
                      f"最小 {cos[has].min():.4f}")
                results["embedding_configs"]["cur128_selfcheck_cosine_mean"] = float(
                    cos[has].mean()
                )

        for config, embeddings in config_embeddings.items():
            per_seed = []
            for seed in args.seeds:
                sp = leave_one_out_split(
                    interactions, min_items=args.min_items, seed=seed,
                    n_items=index.n_items,
                )
                fn = make_content_score_fn(embeddings, sp)
                m = evaluate_bucketed(
                    fn, sp, n_neg=args.n_neg, ks=(10, 20, 50), seed=seed
                )
                full = evaluate_scores(fn, sp, ks=(20, 50))
                m["full_recall@20"] = full["recall@20"]
                m["full_recall@50"] = full["recall@50"]
                per_seed.append(m)
                print(f"  {config:<10}  {seed:<4}  {m['recall@10']:>8.2%}  "
                      f"{m['recall@20']:>8.2%}  {m['recall@50']:>8.2%}  "
                      f"{m['lift_over_chance@10']:>8.2f}x    (全库 {full['recall@50']:.2%})")
            results["embedding_configs"][config] = {
                "per_seed": per_seed,
                **{
                    f"mean_{key}": float(np.mean([m[key] for m in per_seed]))
                    for key in (
                        "recall@10", "recall@20", "recall@50",
                        "full_recall@20", "full_recall@50",
                    )
                },
            }
        print()
        print("  三 seed 均值:")
        print("  config      recall@10  recall@20  recall@50  全库recall@50")
        for config, agg in results["embedding_configs"].items():
            if not isinstance(agg, dict) or "mean_recall@10" not in agg:
                continue
            print(f"  {config:<10}  {agg['mean_recall@10']:>8.2%}  "
                  f"{agg['mean_recall@20']:>8.2%}  {agg['mean_recall@50']:>8.2%}  "
                  f"{agg['mean_full_recall@50']:>11.2%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"  结果已写入 {args.out}")


if __name__ == "__main__":
    main()
