"""游戏推荐相关端点。

TODO(步骤6-7): RAG 检索 + LLM 生成推荐。
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/recommend")
def get_recommendations(steamid: str | None = None) -> dict:
    return {"status": "not_implemented", "steamid": steamid}
