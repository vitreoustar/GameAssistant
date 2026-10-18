"""A1.2 数据落盘:把公开交互数据集与现有语料标准化成推荐算法可直接加载的表。

输入:
  <dataset-dir>/zenodo_interactions_dataset.csv   原始交互(1,009,745 行 / 116 MB)
  <dataset-dir>/zenodo_games_dataset.csv          数据集物品侧元数据(14,736 款)
  backend/data/corpus.jsonl                       现有 RAG 语料物品侧特征(5,362 款)

输出 backend/data/interactions/:
  interactions.parquet   user_idx, game_idx, user_id, appid, playtime_minutes, playtime_log, playtime_bucket
  users.parquet          user_idx, user_id, n_items, playtime_minutes, group_ids
  items.parquet          game_idx, appid, 合并后的物品侧特征, in_corpus, in_dataset, n_interactions
  stats.json             全部统计指标(供 README / 面试材料引用)

关键处理(依据 docs/ROADMAP.md §3-A1.1 的实测结论):
  1. 数据集的真实粒度是 (user_id, group_name, appid);同一用户属于多个群组时同款游戏会重复出现,
     必须按 (user_id, appid) 去重(playtime 取 max),否则会污染训练集(实测重复 169,471 行)。
  2. playtime 极长尾(p99.9 ≈ 3,669 小时),因此额外落 log1p 与 4 档分级两列。
  3. items 表按 game_idx 升序,下游可直接 np.asarray 建特征矩阵。
  4. 只读需要的列,不把 116 MB 交互表整个读进内存。

用法:
  cd backend
  python scripts/build_interactions.py
  python scripts/build_interactions.py --dataset-dir ../Steam_November2025_Dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.recommend.features import (  # noqa: E402
    owners_lower_bound,
    playtime_bucket,
    positive_ratio,
    rating_text,
    release_year,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = BACKEND_DIR.parent / "Steam_November2025_Dataset"
CORPUS_PATH = BACKEND_DIR / "data" / "corpus.jsonl"
OUT_DIR = BACKEND_DIR / "data" / "interactions"

INTERACTION_COLUMNS = ["user_id", "group_name", "appid", "playtime_forever", "is_played"]
CORPUS_NUMERIC = ["price_cents", "positive", "negative", "metacritic_score", "dlc_count"]


# ---------------------------------------------------------------- 读取与清洗


def load_interactions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """读原始交互,按 (user_id, appid) 去重。返回 (去重后的表, 用户->群组, 统计)。"""
    raw = pd.read_csv(
        path,
        usecols=INTERACTION_COLUMNS,
        dtype={
            "user_id": str,
            "group_name": str,
            "appid": "int64",
            "playtime_forever": "float64",
            "is_played": str,
        },
    )
    if raw.empty:
        raise SystemExit(f"{path} 没读到任何行")

    stats = {
        "raw_rows": int(len(raw)),
        "raw_unique_users": int(raw["user_id"].nunique()),
        "raw_unique_items": int(raw["appid"].nunique()),
        # 这一列在数据集里恒为 1,实测确认:数据集记录的是"玩过 >=10 分钟"而非完整拥有列表
        "is_played_values": sorted(raw["is_played"].unique().tolist()),
        "playtime_min_minutes": float(raw["playtime_forever"].min()),
        "n_groups": int(raw["group_name"].nunique()),
    }

    groups = (
        raw.groupby("user_id")["group_name"]
        .agg(lambda s: ",".join(sorted(set(s))))
        .rename("group_ids")
        .reset_index()
    )
    stats["users_in_multiple_groups"] = int(groups["group_ids"].str.contains(",").sum())

    dedup = (
        raw.groupby(["user_id", "appid"], as_index=False)["playtime_forever"]
        .max()
        .rename(columns={"playtime_forever": "playtime_minutes"})
    )
    stats["duplicate_rows_removed"] = stats["raw_rows"] - int(len(dedup))
    return dedup, groups, stats


def load_corpus_items(path: Path) -> pd.DataFrame:
    """读 corpus.jsonl 的物品侧特征。字段带 _corpus 后缀,便于与数据集元数据合并。"""
    if not path.exists():
        raise SystemExit(f"找不到语料 {path},请先运行 scripts/build_corpus.py")
    # 与 rag/store.py 保持一致:用 split("\n") 而不是 splitlines(),避免 U+2028 切坏 JSON
    docs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    df = pd.DataFrame(docs)
    df["appid"] = df["appid"].astype("int64")
    df["genres"] = df["genres"].apply(lambda v: ", ".join(v or []))
    df["categories"] = df["categories"].apply(lambda v: ", ".join(v or []))
    cols = [
        "appid", "name", "genres", "categories", "price_cents", "positive", "negative",
        "metacritic_score", "release_date", "dlc_count", "estimated_owners", "header_image",
    ]
    # 只给两路都有的列加 _corpus 后缀;数值列(price/positive/...)数据集侧没有,保持原名,
    # 免得下游到处写 _corpus
    conflicting = {"name", "genres", "categories", "release_date"}
    return df[cols].rename(columns={c: f"{c}_corpus" for c in cols if c in conflicting})


def load_dataset_items(path: Path) -> pd.DataFrame:
    """读数据集自带的物品侧元数据(提供 corpus 缺失的 publishers / recommendations 等)。"""
    df = pd.read_csv(path, dtype=str).fillna("")
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df = df.dropna(subset=["appid"])
    df["appid"] = df["appid"].astype("int64")
    df["is_free"] = df["is_free"].str.strip().str.lower().eq("true")
    df["recommendations"] = (
        pd.to_numeric(df["recommendations"], errors="coerce").fillna(0).astype("int64")
    )
    cols = [
        "appid", "name", "genres", "categories", "publishers", "is_free",
        "release_date", "recommendations",
    ]
    return df[cols].rename(columns={c: f"{c}_dataset" for c in cols if c != "appid"})


# ---------------------------------------------------------------- 组装


def build_tables(
    dedup: pd.DataFrame,
    groups: pd.DataFrame,
    corpus_items: pd.DataFrame,
    dataset_items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把去重交互 + 两路物品侧特征组装成三张最终表。"""
    users = pd.DataFrame({"user_id": np.sort(dedup["user_id"].unique())})
    users.insert(0, "user_idx", np.arange(len(users), dtype=np.int32))
    users = users.merge(groups, on="user_id", how="left")
    users["group_ids"] = users["group_ids"].fillna("")

    game_ids = pd.DataFrame({"appid": np.sort(dedup["appid"].unique())})
    game_ids.insert(0, "game_idx", np.arange(len(game_ids), dtype=np.int32))

    interactions = dedup.merge(users[["user_id", "user_idx"]], on="user_id", how="left")
    interactions = interactions.merge(game_ids, on="appid", how="left")
    interactions["playtime_log"] = np.log1p(interactions["playtime_minutes"])
    interactions["playtime_bucket"] = (
        interactions["playtime_minutes"].map(playtime_bucket).astype("int8")
    )
    # Steam 的 playtime_forever 是整数分钟(已实测)。转 int32 前先断言,避免将来
    # 数据源变了之后静默截断小数
    if not (interactions["playtime_minutes"] % 1 == 0).all():
        raise SystemExit("playtime_minutes 出现非整数值,请先确认数据源再决定是否转 int32")
    interactions["playtime_minutes"] = interactions["playtime_minutes"].astype("int32")
    interactions = (
        interactions[
            [
                "user_idx", "game_idx", "user_id", "appid",
                "playtime_minutes", "playtime_log", "playtime_bucket",
            ]
        ]
        .sort_values(["user_idx", "game_idx"], ignore_index=True)
    )

    users = users.merge(
        interactions.groupby("user_idx")
        .agg(n_items=("game_idx", "size"), playtime_minutes=("playtime_minutes", "sum"))
        .reset_index(),
        on="user_idx",
        how="left",
    )
    users = users[["user_idx", "user_id", "n_items", "playtime_minutes", "group_ids"]]

    items = (
        game_ids.merge(corpus_items, on="appid", how="left")
        .merge(dataset_items, on="appid", how="left")
    )
    items["name"] = items["name_corpus"].fillna(items["name_dataset"]).fillna("")
    items["genres"] = items["genres_corpus"].fillna(items["genres_dataset"]).fillna("")
    items["categories"] = (
        items["categories_corpus"].fillna(items["categories_dataset"]).fillna("")
    )
    items["in_corpus"] = items["name_corpus"].notna()
    items["in_dataset"] = items["name_dataset"].notna()

    # 缺一侧元数据的物品在 left-join 后是 NaN。文本列填空串,数值/布尔列显式定型 ——
    # 否则 parquet 里会出现 float64 的"整数"列和 object 的"布尔"列(实测踩到过)
    items["publishers"] = items["publishers_dataset"].fillna("")
    items["is_free"] = items["is_free_dataset"].fillna(False).astype(bool)
    items["recommendations"] = (
        pd.to_numeric(items["recommendations_dataset"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    for col in CORPUS_NUMERIC:
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0).astype("int64")
    items["estimated_owners_lower"] = (
        items["estimated_owners"].fillna("").map(owners_lower_bound).astype("int64")
    )
    # 好评率 / 评价标签都是按 positive+negative 现算,而不是沿用任何一方的旧字段
    items["positive_ratio"] = [
        positive_ratio(int(p), int(n))
        for p, n in zip(items["positive"], items["negative"])
    ]
    items["rating"] = [
        rating_text(int(p), int(n)) for p, n in zip(items["positive"], items["negative"])
    ]
    items["release_year"] = (
        items["release_date_corpus"]
        .fillna(items["release_date_dataset"])
        .fillna("")
        .map(release_year)
        .astype("int64")
    )

    items = items.merge(
        interactions.groupby("game_idx").size().rename("n_interactions").reset_index(),
        on="game_idx",
        how="left",
    )
    items["n_interactions"] = items["n_interactions"].fillna(0).astype("int64")
    items = items[
        [
            "game_idx", "appid", "name", "genres", "categories", "publishers",
            "is_free", "recommendations", "price_cents", "positive",
            "negative", "positive_ratio", "rating", "metacritic_score",
            "estimated_owners_lower", "release_year", "dlc_count", "header_image",
            "in_corpus", "in_dataset", "n_interactions",
        ]
    ]
    return interactions, users, items


def build_stats(
    interactions: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, base: dict
) -> dict:
    """汇总统计指标。这些数字同时写进 stats.json,供文档与面试材料直接引用。"""
    per_user = interactions.groupby("user_idx").size()
    per_item = interactions.groupby("game_idx").size()
    playtime = interactions["playtime_minutes"].to_numpy()

    n_users, n_items = len(users), len(items)
    in_corpus = items["in_corpus"].to_numpy()
    game_idx = interactions["game_idx"].to_numpy()
    pairs_in_corpus = int(in_corpus[game_idx].sum())

    def buckets(series: pd.Series, thresholds: list[int]) -> dict[str, int]:
        return {f">={t}": int((series >= t).sum()) for t in thresholds}

    in_corpus_items = items.loc[items["in_corpus"], "n_interactions"]

    stats = {
        **base,
        "interactions": int(len(interactions)),
        "users": n_users,
        "items": n_items,
        "density": round(len(interactions) / (n_users * n_items), 6),
        "interactions_per_user": {
            "min": int(per_user.min()),
            "median": float(per_user.median()),
            "max": int(per_user.max()),
        },
        "interactions_per_item": {
            "min": int(per_item.min()),
            "median": float(per_item.median()),
            "max": int(per_item.max()),
        },
        "playtime_minutes": {
            "min": float(playtime.min()),
            "p50": float(np.percentile(playtime, 50)),
            "p90": float(np.percentile(playtime, 90)),
            "p99": float(np.percentile(playtime, 99)),
            "p999": float(np.percentile(playtime, 99.9)),
            "max": float(playtime.max()),
        },
        "playtime_bucket_counts": {
            str(k): int(v)
            for k, v in interactions["playtime_bucket"].value_counts().sort_index().items()
        },
        "users_by_interaction_count": buckets(per_user, [5, 10, 20, 30, 50, 100]),
        "items_by_interaction_count": buckets(per_item, [20, 50, 100, 500]),
        "coverage": {
            "items_in_corpus": int(items["in_corpus"].sum()),
            "items_in_dataset": int(items["in_dataset"].sum()),
            "items_in_both": int((items["in_corpus"] & items["in_dataset"]).sum()),
            "items_only_in_dataset": int((~items["in_corpus"] & items["in_dataset"]).sum()),
            "items_with_no_features": int((~items["in_corpus"] & ~items["in_dataset"]).sum()),
            "interaction_pairs_with_corpus_item": pairs_in_corpus,
            "interaction_pairs_covered": round(pairs_in_corpus / len(interactions), 4),
        },
        "ltr_pool": {
            "items_ge20": int((per_item >= 20).sum()),
            "items_ge50": int((per_item >= 50).sum()),
            "in_corpus_items_ge20": int((in_corpus_items >= 20).sum()),
            "in_corpus_items_ge50": int((in_corpus_items >= 50).sum()),
        },
        "top_items": [
            {
                "appid": int(items.loc[i, "appid"]),
                "name": str(items.loc[i, "name"]),
                "n_interactions": int(items.loc[i, "n_interactions"]),
            }
            for i in items["n_interactions"].nlargest(10).index
        ],
        "distinct_publishers": int(
            items.loc[items["publishers"] != "", "publishers"].str.split(";").explode().nunique()
        ),
    }
    return stats


def print_summary(stats: dict, out_dir: Path) -> None:
    cov = stats["coverage"]
    print("\n" + "=" * 68)
    print("A1.2 数据落盘完成")
    print("=" * 68)
    print(f"  原始交互行           {stats['raw_rows']:>12,}")
    print(f"  去重后交互           {stats['interactions']:>12,}"
          f"   (移除重复 {stats['duplicate_rows_removed']:,})")
    print(f"  用户 / 物品 / 群组    {stats['users']:>12,} / {stats['items']:,}"
          f" / {stats['n_groups']}")
    print(f"  密度                 {stats['density']:>12.4%}")
    print(f"  交互/用户 (中位)      {stats['interactions_per_user']['median']:>12.0f}")
    print(f"  交互/物品 (中位)      {stats['interactions_per_item']['median']:>12.0f}")
    pt = stats["playtime_minutes"]
    print(f"  时长中位 / p99        {pt['p50']:>12.0f} 分钟 / {pt['p99']:.0f} 分钟")
    print(f"  有语料特征的物品      {cov['items_in_corpus']:>12,}"
          f"   ({cov['items_in_corpus'] / stats['items']:.1%})")
    print(f"  交互对被语料覆盖      {cov['interaction_pairs_covered']:>12.1%}")
    print(f"  仅数据集有的物品      {cov['items_only_in_dataset']:>12,}")
    print(f"  两路都无特征的物品    {cov['items_with_no_features']:>12,}")
    print(f"  LTR 池 (物品>=20/>=50)  {stats['ltr_pool']['items_ge20']:,}"
          f" / {stats['ltr_pool']['items_ge50']:,}")
    print(f"  可评测用户 (>=10 交互) {stats['users_by_interaction_count']['>=10']:>12,}")
    print(f"  不同发行商           {stats['distinct_publishers']:>12,}")
    print("-" * 68)
    print(f"  输出目录: {out_dir}")
    for name in ("interactions.parquet", "users.parquet", "items.parquet", "stats.json"):
        p = out_dir / name
        size = p.stat().st_size / 1e6 if p.exists() else 0.0
        print(f"    {name:<24} {size:>8.2f} MB")
    print("=" * 68)


def main() -> None:
    # 控制台默认可能是 GBK,显式转 UTF-8,避免中文摘要变成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="A1.2 数据落盘:交互数据集 + 语料 -> 推荐算法可加载的表"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    inter_path = args.dataset_dir / "zenodo_interactions_dataset.csv"
    games_path = args.dataset_dir / "zenodo_games_dataset.csv"
    for path in (inter_path, games_path):
        if not path.exists():
            raise SystemExit(f"找不到 {path}\n可用 --dataset-dir 指定数据集目录")

    print(f"[1/5] 读取交互并按 (user, appid) 去重 ... {inter_path.name}")
    dedup, groups, base_stats = load_interactions(inter_path)

    print("[2/5] 读取物品侧特征 (corpus.jsonl + games csv) ...")
    corpus_items = load_corpus_items(CORPUS_PATH)
    dataset_items = load_dataset_items(games_path)

    print("[3/5] 组装 interactions / users / items ...")
    interactions, users, items = build_tables(dedup, groups, corpus_items, dataset_items)

    # 内部一致性断言:这些不变式一旦被破坏,下游的矩阵构建会静默出错
    assert not interactions.duplicated(["user_idx", "game_idx"]).any(), "去重失败:仍有重复 (user,item)"
    assert interactions["user_idx"].notna().all(), "有交互没能映射到 user_idx"
    assert interactions["game_idx"].notna().all(), "有交互没能映射到 game_idx"
    assert len(interactions) == base_stats["raw_rows"] - base_stats["duplicate_rows_removed"]
    assert interactions["playtime_minutes"].min() >= 0

    print("[4/5] 汇总统计 ...")
    stats = build_stats(interactions, users, items, base_stats)

    print("[5/5] 写出 ...")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(args.out_dir / "interactions.parquet", index=False)
    users.to_parquet(args.out_dir / "users.parquet", index=False)
    items.to_parquet(args.out_dir / "items.parquet", index=False)
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(stats, args.out_dir)


if __name__ == "__main__":
    main()
