"""向量库:ChromaDB 持久化 + 语义检索。"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.embedder import embed_documents, embed_query

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
COLLECTION = "steam_games"

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(DATA_DIR / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def rating_text(positive: int, negative: int) -> str:
    """Steam 好评率 -> 中文评价标签。"""
    total = (positive or 0) + (negative or 0)
    if total < 50:
        return "评价较少"
    pct = (positive or 0) / total * 100
    if pct >= 95:
        return "好评如潮"
    if pct >= 80:
        return "特别好评"
    if pct >= 70:
        return "多半好评"
    if pct >= 40:
        return "褒贬不一"
    if pct >= 20:
        return "多半差评"
    return "差评如潮"


def _scalar_metadata(doc: dict) -> dict:
    """Chroma metadata 只接受标量,列表字段转成逗号字符串。"""
    return {
        "name": doc.get("name", ""),
        "genres": ", ".join(doc.get("genres") or []),
        "categories": ", ".join(doc.get("categories") or []),
        "price_cents": doc.get("price_cents", 0),
        "positive": doc.get("positive", 0),
        "negative": doc.get("negative", 0),
        "metacritic_score": doc.get("metacritic_score", 0),
        "release_date": doc.get("release_date", "") or "",
        "dlc_count": doc.get("dlc_count", 0),
        "estimated_owners": doc.get("estimated_owners", "") or "",
        "header_image": doc.get("header_image", "") or "",
    }


def build_index(corpus_path: Path | str) -> int:
    """从 corpus.jsonl 构建(重建)向量库,返回文档数。"""
    corpus_path = Path(corpus_path)
    # 注意:用 \n 分割而非 splitlines()——描述里可能含 U+2028 等 Unicode 行分隔符
    docs = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]

    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids = [str(d["appid"]) for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [_scalar_metadata(d) for d in docs]
    embeddings = embed_documents(texts)

    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(docs)


def _to_result(doc_id: str, meta: dict, distance: float | None = None) -> dict:
    pos = meta.get("positive", 0)
    neg = meta.get("negative", 0)
    return {
        "appid": int(doc_id),
        "name": meta.get("name", ""),
        "header_image": meta.get("header_image", ""),
        "genres": meta.get("genres", ""),
        "categories": meta.get("categories", ""),
        "price_cents": meta.get("price_cents", 0),
        "positive": pos,
        "negative": neg,
        "rating": rating_text(pos, neg),
        "metacritic_score": meta.get("metacritic_score", 0),
        "distance": distance,
    }


def search(query: str, k: int = 10, exclude: set[int] | None = None) -> list[dict]:
    """语义检索 top-k。"""
    client = get_client()
    try:
        col = client.get_collection(COLLECTION)
    except Exception as exc:
        raise RuntimeError("向量库尚未构建,请先运行 scripts/build_index.py") from exc

    qe = embed_query(query)
    res = col.query(query_embeddings=[qe], n_results=k)
    exclude = exclude or set()

    results: list[dict] = []
    for i, doc_id in enumerate(res["ids"][0]):
        appid = int(doc_id)
        if appid in exclude:
            continue
        meta = res["metadatas"][0][i] or {}
        dist = res["distances"][0][i]
        results.append(_to_result(doc_id, meta, dist))
    return results


def get_game(appid: int) -> dict | None:
    """按 appid 取单款游戏的元数据(用于「某游戏怎么样」)。"""
    client = get_client()
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        return None
    res = col.get(ids=[str(appid)])
    if res and res["metadatas"] and res["metadatas"][0]:
        return _to_result(str(appid), res["metadatas"][0])
    return None
