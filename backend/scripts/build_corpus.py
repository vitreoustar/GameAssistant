"""RAG 语料构建脚本(步骤5)。

从 FronkonGames/steam-games-dataset(parquet)构建语料:
  1. (可选)自动下载 parquet(缺文件时,支持 HF_ENDPOINT 镜像)
  2. 过滤:有描述 + 销量估计下界 >= MIN_OWNERS
  3. 字段拼接 name + Genres + Categories + 描述 -> 文本
  4. 输出 JSONL(data/corpus.jsonl)

用法:
    cd backend && .venv/Scripts/python scripts/build_corpus.py

国内下载慢可先设: set HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# 允许 `python scripts/build_corpus.py` 直接从 backend 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "steam_games.parquet"
OUTPUT_PATH = DATA_DIR / "corpus.jsonl"

MIN_OWNERS = 200_000  # estimated_owners 下界
DESC_LIMIT = 800      # 描述截断长度

DATASET_PATH = "datasets/FronkonGames/steam-games-dataset/resolve/main/data/train-00000-of-00001.parquet"


def ensure_parquet() -> None:
    if PARQUET_PATH.exists():
        return
    base = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{base}/{DATASET_PATH}"
    print(f"正在下载数据集: {url}")
    import urllib.request

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, PARQUET_PATH)
    print("下载完成")


def parse_owners(value: str | None) -> int:
    """'1000000 - 2000000' -> 1000000"""
    m = re.match(r"([\d,]+)", (value or "").strip())
    return int(m.group(1).replace(",", "")) if m else 0


def build_text(name: str, genres: list[str], categories: list[str], desc: str) -> str:
    parts = [name]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if categories:
        parts.append(f"Categories: {', '.join(categories)}")
    if desc:
        parts.append(desc[:DESC_LIMIT])
    return "\n".join(parts)


def main() -> None:
    ensure_parquet()
    table = pq.read_table(PARQUET_PATH)

    appids = table.column("appID").to_pylist()
    names = table.column("name").to_pylist()
    owners = table.column("estimated_owners").to_pylist()
    detailed = table.column("detailed_description").to_pylist()
    short = table.column("short_description").to_pylist()
    price = table.column("price").to_pylist()
    positive = table.column("positive").to_pylist()
    negative = table.column("negative").to_pylist()
    metacritic = table.column("metacritic_score").to_pylist()
    release = table.column("release_date").to_pylist()
    dlc_count = table.column("dlc_count").to_pylist()
    # parquet 中列表列名统一为 element:35=Categories, 36=Genres
    categories = table.column(35).to_pylist()
    genres = table.column(36).to_pylist()

    docs: list[dict] = []
    for i in range(len(appids)):
        desc = (detailed[i] or "").strip() or (short[i] or "").strip()
        if not desc:
            continue
        if parse_owners(owners[i]) < MIN_OWNERS:
            continue
        g = genres[i] or []
        c = categories[i] or []
        docs.append(
            {
                "appid": int(appids[i]),
                "name": names[i],
                "text": build_text(names[i], g, c, desc),
                "genres": g,
                "categories": c,
                "price_cents": int(round((price[i] or 0.0) * 100)),
                "positive": positive[i] or 0,
                "negative": negative[i] or 0,
                "metacritic_score": metacritic[i] or 0,
                "release_date": release[i],
                "dlc_count": dlc_count[i] or 0,
                "estimated_owners": owners[i],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"构建完成: {len(docs)} 条 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
