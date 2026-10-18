"""推荐端点(A5)。

POST /api/recommend  body: {"library": [Game], "top_k": 5}

返回结构化候选,每条带:
  * `score` / `channels` —— 由哪几路召回、精排得分(可解释性)
  * `because_of` —— "因为你在玩《X》",直接来自 TopK 邻居
  * `match_genres` —— 与用户画像重叠的类型

与 `/api/chat` 里走 LLM 的推荐不同,这个端点**不调用 LLM**,纯查表 + 精排
(实测 P50 7.4ms / P99 8.4ms,常规库大小),适合前端做即时预览或压测。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.recommend import service as recommend_service
from app.schemas import Game

router = APIRouter()


class RecommendRequest(BaseModel):
    library: list[Game]
    top_k: int = 5


@router.post("/recommend")
async def get_recommendations(req: RecommendRequest) -> dict:
    recommender = recommend_service.get_recommender()
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "推荐索引不可用:"
                f"{recommend_service.unavailable_reason() or '未知原因'}。"
                "请先运行 scripts/build_recommend_index.py 与 scripts/train_ranker.py"
            ),
        )
    if not req.library:
        raise HTTPException(status_code=400, detail="请先提供游戏库")

    games = [g.model_dump() for g in req.library]
    result = recommender.recommend(games, top_k=max(1, min(req.top_k, 20)))
    return {
        "profile": result.profile,
        "cold_start": result.cold_start,
        "unmatched_in_library": result.unmatched_in_library,
        "library_matched": result.library_matched,
        "recommendations": [r.as_dict() for r in result.recommendations],
    }
