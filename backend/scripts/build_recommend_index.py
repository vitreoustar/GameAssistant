"""A2 离线索引构建:内容向量 + item-item 相似度矩阵 + 用户画像向量。

输入:
  backend/data/interactions/{interactions,users,items}.parquet   A1.2 的产物
  backend/data/corpus.jsonl                                     RAG 语料(含描述文本)

输出 backend/data/recommend/:
  catalog.parquet        cat_idx -> appid + 全部结构化特征与标志位
  content_emb.npy        [N, 384] float32,L2 归一化(无文本的物品为零向量)
  content_topk.npz       [N, K] 内容相似邻居
  collab_topk.npz        [N, K] 协同共现邻居
  user_emb.npy           [U, 384] float32,训练集用户的画像向量
  user_item_csr.npz      [N, U] 交互稀疏矩阵(playtime_log 加权)
  manifest.json          形状 / 构建参数 / 校验结果

关键设计(详见 docs/ROADMAP.md §3-A2):
  1. **物品空间取并集**:corpus 物品 ∪ 交互物品 = 11,444。
     交互空间(9,935)是 A4 训练/评测所必需的;corpus 独有的物品虽无交互数据,
     但线上仍可推荐(冷物品),内容通道能覆盖它们。
  2. **文本模板与 build_corpus 共用同一个实现**(`build_item_text`),且 corpus 物品
     直接复用 corpus.jsonl 里已拼好的 text —— 保证与 ChromaDB 嵌入的文本逐字节一致,
     否则 A6 和现有 RAG 基线对比时,差异里会混进"文本模板不同"这个混淆因素。
  3. **矩阵分块计算**:11k×11k 的相似度矩阵不整体物化,按 chunk 做
     `[chunk, d] @ [d, N]` 再逐块取 TopK,峰值内存只有几十 MB。
  4. **协同相似度用 playtime_log 加权**(不是二值),"真的玩了很久"贡献更大;
     行 L2 归一化后点积即余弦。

⚠️ 数据泄漏警告:`n_interactions`、`bayes_rating` 等**物品侧全局统计**都是在
全量交互上算的。用它们当 A4 排序特征、又在同一份交互上做 leave-one-out 评测,
会把测试集信息漏进特征。A4 必须按训练切分重算这些特征(本脚本的 catalog 只作为
"描述性统计"与冷启动先验使用)。

用法:
  cd backend
  python scripts/build_recommend_index.py
  python scripts/build_recommend_index.py --k 200 --batch-size 64
  python scripts/build_recommend_index.py --skip-embed    # 只重算矩阵(复用已有向量)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommend.features import (  # noqa: E402
    bayesian_rating,
    build_item_text,
    owners_lower_bound,
    positive_ratio,
    rating_text,
    release_year,
)
from app.recommend.index import (  # noqa: E402
    CATALOG_FILE,
    COLLAB_TOPK_FILE,
    CONTENT_DIM,
    CONTENT_EMB_FILE,
    CONTENT_TOPK_FILE,
    MANIFEST_FILE,
    USER_EMB_FILE,
    USER_ITEM_FILE,
    cat_rows_of,
    content_topk_from_embeddings,
    topk_from_dense,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
INTERACTIONS_DIR = BACKEND_DIR / "data" / "interactions"
CORPUS_PATH = BACKEND_DIR / "data" / "corpus.jsonl"
OUT_DIR = BACKEND_DIR / "data" / "recommend"

PRIOR_STRENGTH = 50.0  # 贝叶斯好评率的先验票数
EMBED_CHUNK = 512      # 相似度矩阵分块行数


# ---------------------------------------------------------------- 输入


def load_corpus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"找不到语料 {path},请先运行 scripts/build_corpus.py")
    docs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    df = pd.DataFrame(docs)
    df["appid"] = df["appid"].astype("int64")
    return df


def load_interactions_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "interactions": INTERACTIONS_DIR / "interactions.parquet",
        "users": INTERACTIONS_DIR / "users.parquet",
        "items": INTERACTIONS_DIR / "items.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            raise SystemExit(f"找不到 {p},请先运行 scripts/build_interactions.py")
    return (
        pd.read_parquet(paths["interactions"]),
        pd.read_parquet(paths["users"]),
        pd.read_parquet(paths["items"]),
    )


# ---------------------------------------------------------------- 物品目录


def _str_array(series: pd.Series, fill: str = "") -> np.ndarray:
    """转成 object 数组并把 NaN / 非字符串都变成 fill。"""
    return np.array(
        [v if isinstance(v, str) else fill for v in series.to_numpy()], dtype=object
    )


def _num_array(series: pd.Series, fill: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(fill).to_numpy()


def build_catalog(
    items: pd.DataFrame, corpus: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """构建并集物品目录 + 每个物品的嵌入文本。返回 (catalog, texts)。"""
    all_appids = np.union1d(items["appid"].to_numpy(), corpus["appid"].to_numpy())

    corpus_side = pd.DataFrame(
        {
            "appid": corpus["appid"],
            "c_name": _str_array(corpus["name"]),
            "c_text": _str_array(corpus["text"]),
            "c_genres": [
                ", ".join(v or []) for v in corpus["genres"].to_numpy()
            ],
            "c_categories": [
                ", ".join(v or []) for v in corpus["categories"].to_numpy()
            ],
            "c_owners_lower": _num_array(corpus["estimated_owners"].map(owners_lower_bound)),
            "c_price_cents": _num_array(corpus["price_cents"]),
            "c_positive": _num_array(corpus["positive"]),
            "c_negative": _num_array(corpus["negative"]),
            "c_metacritic": _num_array(corpus["metacritic_score"]),
            "c_release_date": _str_array(corpus["release_date"]),
            "c_dlc_count": _num_array(corpus["dlc_count"]),
            "c_header_image": _str_array(corpus["header_image"]),
        }
    ).set_index("appid")

    I = items.set_index("appid").reindex(all_appids)
    C = corpus_side.reindex(all_appids)

    def pick(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a 优先;其中为空串/NaN 的位置用 b 补。"""
        out = np.array(a, dtype=object)
        blank = np.array([not (isinstance(v, str) and v.strip()) for v in out])
        out[blank] = b[blank]
        return out

    def pick_num(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype="float64")
        alt = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype="float64")
        out = np.where(np.isnan(out), alt, out)
        return np.nan_to_num(out, nan=fill)

    name = pick(_str_array(I["name"]), _str_array(C["c_name"]))
    genres = pick(_str_array(I["genres"]), _str_array(C["c_genres"]))
    categories = pick(_str_array(I["categories"]), _str_array(C["c_categories"]))
    publishers = _str_array(I["publishers"])
    header_image = pick(_str_array(I["header_image"]), _str_array(C["c_header_image"]))

    positive = pick_num(_num_array(I["positive"]), C["c_positive"].to_numpy())
    negative = pick_num(_num_array(I["negative"]), C["c_negative"].to_numpy())
    price_cents = pick_num(_num_array(I["price_cents"]), C["c_price_cents"].to_numpy())
    metacritic = pick_num(
        _num_array(I["metacritic_score"]), C["c_metacritic"].to_numpy()
    )
    dlc_count = pick_num(_num_array(I["dlc_count"]), C["c_dlc_count"].to_numpy())
    owners_lower = pick_num(
        _num_array(I["estimated_owners_lower"]), C["c_owners_lower"].to_numpy()
    )
    release_year_arr = pick_num(
        _num_array(I["release_year"]), C["c_release_date"].map(release_year).to_numpy()
    )
    n_interactions = np.nan_to_num(
        pd.to_numeric(I["n_interactions"], errors="coerce").to_numpy(dtype="float64")
    )

    # 嵌入文本:corpus 物品直接用 corpus.jsonl 里已拼好的 text(与 ChromaDB 一致),
    # 仅数据集侧的物品用同一模板现场拼(无描述)
    texts = np.array(C["c_text"].to_numpy(), dtype=object)
    need_build = np.array([not (isinstance(v, str) and v.strip()) for v in texts])
    texts[need_build] = [
        build_item_text(n, g, c)
        for n, g, c in zip(
            name[need_build], genres[need_build], categories[need_build], strict=True
        )
    ]
    has_content = np.array([bool(str(t).strip()) for t in texts])

    # 全局好评率先验均值(只统计有票数的物品),供贝叶斯平滑用
    votes = positive + negative
    with_votes = votes > 0
    prior_mean = (
        float(positive[with_votes].sum() / votes[with_votes].sum())
        if with_votes.any()
        else 0.5
    )

    catalog = pd.DataFrame(
        {
            "cat_idx": np.arange(len(all_appids), dtype=np.int32),
            "appid": all_appids.astype("int64"),
            "name": name,
            "genres": genres,
            "categories": categories,
            "publishers": publishers,
            "price_cents": price_cents.astype("int64"),
            "is_free": I["is_free"].fillna(False).to_numpy(dtype=bool),
            "positive": positive.astype("int64"),
            "negative": negative.astype("int64"),
            "positive_ratio": [
                positive_ratio(int(p), int(n)) for p, n in zip(positive, negative)
            ],
            "bayes_rating": [
                bayesian_rating(int(p), int(n), prior_mean, PRIOR_STRENGTH)
                for p, n in zip(positive, negative)
            ],
            "rating": [
                rating_text(int(p), int(n)) for p, n in zip(positive, negative)
            ],
            "metacritic_score": metacritic.astype("int64"),
            "estimated_owners_lower": owners_lower.astype("int64"),
            "release_year": release_year_arr.astype("int64"),
            "dlc_count": dlc_count.astype("int64"),
            "header_image": header_image,
            "n_interactions": n_interactions.astype("int64"),
            "in_corpus": C["c_name"].notna().to_numpy(),
            "in_dataset": I["in_dataset"].fillna(False).to_numpy(dtype=bool),
            "has_content": has_content,
            "has_interactions": n_interactions > 0,
        }
    )
    catalog.attrs["prior_mean"] = prior_mean
    return catalog, texts


# ---------------------------------------------------------------- 相似度矩阵


def embed_texts(texts: np.ndarray, batch_size: int) -> np.ndarray:
    """用项目既有的本地 embedding 模型编码。模型只在这里加载,在线路径不会碰它。"""
    from app.rag.embedder import get_model

    model = get_model()
    start = time.time()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"      编码 {len(texts):,} 条文本用时 {time.time() - start:.1f}s")
    return np.asarray(vectors, dtype=np.float32)


def content_topk(
    embeddings: np.ndarray, has_content: np.ndarray, k: int, chunk: int = EMBED_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """内容相似度 TopK —— 实现已下沉到 `app.recommend.index`,这里只固定 chunk 大小。"""
    return content_topk_from_embeddings(embeddings, has_content, k, chunk=chunk)


def collab_topk(
    interactions: pd.DataFrame,
    appid_of_row: np.ndarray,
    cat_appids: np.ndarray,
    n_cat: int,
    n_users: int,
    k: int,
    chunk: int = EMBED_CHUNK,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """用户-物品共现的 item-item 余弦 TopK(playtime_log 加权)。

    行 L2 归一化之后 `X @ X.T` 就是余弦相似度。整块乘积不物化:每 chunk 取一次
    TopK,峰值内存可控。结果映射回 cat_idx 空间,和内容通道保持同一套下标。
    """
    n_items = len(appid_of_row)
    weights = interactions["playtime_log"].to_numpy(dtype=np.float32)
    X = sp.csr_matrix(
        (weights, (interactions["game_idx"].to_numpy(), interactions["user_idx"].to_numpy())),
        shape=(n_items, n_users),
        dtype=np.float32,
    )
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    Xn = (sp.diags(1.0 / norms) @ X).tocsr()
    XnT = Xn.T.tocsr()

    idx_out = np.full((n_cat, k), -1, dtype=np.int32)
    score_out = np.zeros((n_cat, k), dtype=np.float32)

    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        scores = np.asarray((Xn[start:stop] @ XnT).todense(), dtype=np.float32)
        local_idx, local_val = topk_from_dense(
            scores, k, exclude_self_offset=start
        )
        rows = cat_rows_of(cat_appids, appid_of_row[start:stop])
        idx_out[rows] = cat_rows_of(cat_appids, appid_of_row[local_idx])
        score_out[rows] = local_val
    return idx_out, score_out, X


def precompute_user_vectors(
    interactions: pd.DataFrame,
    embeddings: np.ndarray,
    appid_of_row: np.ndarray,
    cat_appids: np.ndarray,
    n_users: int,
) -> np.ndarray:
    """训练集用户的画像向量 = playtime_log 加权的内容向量之和再归一化。

    用稀疏矩阵乘一次算完,比 Python 循环快几个数量级:
        W [U, N] (稀疏权重) @ E [N, 384] -> [U, 384]
    无内容向量的物品是零向量,天然不贡献方向。
    """
    game_idx = interactions["game_idx"].to_numpy()
    rows = cat_rows_of(cat_appids, appid_of_row[game_idx])
    W = sp.csr_matrix(
        (
            interactions["playtime_log"].to_numpy(dtype=np.float32),
            (interactions["user_idx"].to_numpy(), rows),
        ),
        shape=(n_users, embeddings.shape[0]),
        dtype=np.float32,
    )
    vectors = np.asarray(W @ embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------- 主流程


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="A2 离线索引构建")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--k", type=int, default=200, help="每个物品保留的邻居数")
    parser.add_argument("--batch-size", type=int, default=64, help="embedding 批大小")
    parser.add_argument(
        "--skip-embed", action="store_true", help="复用已有的 content_emb.npy,只重算矩阵"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] 读取 A1.2 产物与语料 ...")
    interactions, users, items = load_interactions_inputs()
    corpus = load_corpus(CORPUS_PATH)
    print(f"      交互 {len(interactions):,} 行 | 交互物品 {len(items):,} | "
          f"语料 {len(corpus):,} | 用户 {len(users):,}")

    print("[2/6] 构建并集物品目录 + 文本 ...")
    catalog, texts = build_catalog(items, corpus)
    prior_mean = float(catalog.attrs["prior_mean"])
    n_cat = len(catalog)
    n_users = len(users)
    print(f"      目录 {n_cat:,} 个物品(有内容文本 "
          f"{int(catalog['has_content'].sum()):,} / 有交互 "
          f"{int(catalog['has_interactions'].sum()):,})")

    emb_path = args.out_dir / CONTENT_EMB_FILE
    if args.skip_embed and emb_path.exists():
        print("[3/6] 复用已有 content_emb.npy(跳过编码)")
        embeddings = np.load(emb_path)
        if embeddings.shape != (n_cat, CONTENT_DIM):
            raise SystemExit(
                f"已有向量形状 {embeddings.shape} 与目录 {n_cat} 不匹配,请去掉 --skip-embed"
            )
    else:
        print(f"[3/6] 编码内容向量(模型 {CONTENT_DIM} 维,批 {args.batch_size}) ...")
        embeddings = np.zeros((n_cat, CONTENT_DIM), dtype=np.float32)
        has = catalog["has_content"].to_numpy()
        embeddings[has] = embed_texts(texts[has], args.batch_size)
        np.save(emb_path, embeddings)

    print(f"[4/6] 内容相似度 TopK (K={args.k}) ...")
    c_idx, c_score = content_topk(embeddings, catalog["has_content"].to_numpy(), args.k)
    np.savez(args.out_dir / CONTENT_TOPK_FILE, indices=c_idx, scores=c_score)

    print(f"[5/6] 协同共现 TopK (K={args.k}) ...")
    appid_of_row = items["appid"].to_numpy()
    cat_appids = catalog["appid"].to_numpy()
    k_idx, k_score, user_item = collab_topk(
        interactions, appid_of_row, cat_appids, n_cat, n_users, args.k
    )
    np.savez(args.out_dir / COLLAB_TOPK_FILE, indices=k_idx, scores=k_score)
    sp.save_npz(args.out_dir / USER_ITEM_FILE, user_item.tocsr())

    print("[6/6] 用户画像向量 + 写出 ...")
    user_emb = precompute_user_vectors(
        interactions, embeddings, appid_of_row, cat_appids, n_users
    )
    np.save(args.out_dir / USER_EMB_FILE, user_emb)
    catalog.to_parquet(args.out_dir / CATALOG_FILE, index=False)

    # 校验:这些不变式一旦破坏,下游召回会静默给出错误结果
    has = catalog["has_content"].to_numpy()
    # collab 自环:把 cat_idx 映射回交互行号再比
    row_of_cat = np.full(n_cat, -1, dtype=np.int64)
    row_of_cat[cat_rows_of(cat_appids, appid_of_row)] = np.arange(len(appid_of_row))
    collab_valid = k_idx >= 0
    collab_neighbor_row = row_of_cat[k_idx.clip(0)]
    collab_own_row = np.broadcast_to(row_of_cat[:, None], k_idx.shape)

    user_norms = np.linalg.norm(user_emb, axis=1)
    checks = {
        "content_topk 行数 == 目录大小": c_idx.shape[0] == n_cat,
        "content_topk 无自环": bool((c_idx[has] != np.arange(n_cat)[has, None]).all()),
        "content_topk 的 idx 都指向有内容的物品": bool(
            (c_idx[has] >= 0).all() and has[c_idx[has]].all()
        ),
        "content_topk 的分数降序": bool(
            (np.diff(c_score[has], axis=1) <= 1e-6).all()
        ),
        "collab_topk 无自环": bool(
            (collab_neighbor_row[collab_valid] != collab_own_row[collab_valid]).all()
        ),
        "collab_topk 的 idx 都指向有交互的物品": bool(
            (k_idx[catalog["has_interactions"].to_numpy()] >= 0).all()
        ),
        "content_emb 已 L2 归一化": bool(
            np.allclose(np.linalg.norm(embeddings[has], axis=1), 1.0, atol=1e-4)
        ),
        "user_emb 非零行已 L2 归一化": bool(
            np.allclose(user_norms[user_norms > 0], 1.0, atol=1e-4)
        ),
        "user_item 非零数 == 交互行数": int(user_item.nnz) == len(interactions),
        "无内容的物品向量为零": bool(
            np.allclose(embeddings[~has], 0.0)
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"校验失败: {failed}")

    manifest = {
        "n_items": int(n_cat),
        "n_items_with_content": int(has.sum()),
        "n_items_with_interactions": int(catalog["has_interactions"].sum()),
        "n_users": int(n_users),
        "content_dim": CONTENT_DIM,
        "topk": int(args.k),
        "params": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "text_template_shared_with": "scripts/build_corpus.py",
            "prior_strength": PRIOR_STRENGTH,
            "prior_mean": round(prior_mean, 6),
            "embed_chunk": EMBED_CHUNK,
            "collab_weight": "playtime_log",
        },
        "checks": checks,
        "leakage_warning": (
            "n_interactions / bayes_rating 等物品侧统计在全量交互上计算,"
            "用于 A4 排序特征时必须按训练切分重算,否则评测会泄漏"
        ),
    }
    (args.out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("A2 离线索引构建完成")
    print("=" * 68)
    print(f"  物品目录              {n_cat:>10,}   (有内容 {int(has.sum()):,} / 有交互 {int(catalog['has_interactions'].sum()):,})")
    print(f"  用户                  {n_users:>10,}")
    print(f"  TopK 邻居数           {args.k:>10,}")
    print(f"  先验好评率均值        {prior_mean:>10.4f}")
    print("-" * 68)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 68)
    for f in sorted(args.out_dir.iterdir()):
        print(f"    {f.name:<24} {f.stat().st_size / 1e6:>8.2f} MB")
    print("=" * 68)


if __name__ == "__main__":
    main()
