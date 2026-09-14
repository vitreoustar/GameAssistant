"""构建向量索引:读取 corpus.jsonl -> 嵌入 -> 写入 ChromaDB。

用法:
    cd backend && .venv/Scripts/python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python scripts/build_index.py` 直接从 backend 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.store import build_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    corpus = DATA_DIR / "corpus.jsonl"
    if not corpus.exists():
        print(f"未找到 {corpus},请先运行 scripts/build_corpus.py")
        return
    n = build_index(corpus)
    print(f"索引完成: {n} 条 -> {DATA_DIR / 'chroma'}")


if __name__ == "__main__":
    main()
