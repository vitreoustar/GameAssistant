"""本地 sentence-transformers 嵌入(免费、离线、无需额外 key)。

注意:必须先导入 app.config 再导入 sentence_transformers,
因为 huggingface_hub 会在导入时把 HF_ENDPOINT 冻结为常量,
若顺序颠倒,镜像地址(hf-mirror.com)将不生效。
"""
from __future__ import annotations

from app.config import settings  # 会在模块级设置 os.environ["HF_ENDPOINT"]

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()
