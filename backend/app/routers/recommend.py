"""推荐端点(A5 / A7)。

POST /api/recommend  body:
    {
      "library": [Game],
      "top_k": 5,
      "exclude_appids": [123, 456],      # 已经推荐过的,不重复
      "prefer": ["Strategy"],            # 用户明说想要的类型/玩法
      "avoid": ["Gore"]                  # 用户明说不要的
    }

返回结构化候选,每条带:
  * `score` / `channels` —— 由哪几路召回、精排得分(可解释性)
  * `because_of` —— "因为你在玩《X》",直接来自 TopK 邻居
  * `match_genres` —— 与用户**长期画像**重叠的类型
  * `pref_match`   —— 与用户**本轮明说偏好**重叠的类型(空 = 这条没对上)

与 `/api/chat` 里走 LLM 的推荐不同,这个端点**不调用 LLM**,纯查表 + 精排
(实测 P50 7.4ms / P99 8.4ms,常规库大小),适合前端做即时预览或压测。
`prefer`/`avoid` 只接受 `preferences.FILTER_VOCAB` 里的词,词表外的词会被忽略。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.recommend import service as recommend_service
from app.schemas import Game

router = APIRouter()


class RecommendRequest(BaseModel):
    library: list[Game]
    top_k: int = 5
    exclude_appids: list[int] = Field(default_factory=list, max_length=1000)
    prefer: list[str] = Field(default_factory=list, max_length=16)
    avoid: list[str] = Field(default_factory=list, max_length=16)


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
    result = recommender.recommend(
        games,
        top_k=max(1, min(req.top_k, 20)),
        exclude_appids=req.exclude_appids,
        prefer=req.prefer,
        avoid=req.avoid,
    )
    return {
        "profile": result.profile,
        "cold_start": result.cold_start,
        "unmatched_in_library": result.unmatched_in_library,
        "library_matched": result.library_matched,
        "recommendations": [r.as_dict() for r in result.recommendations],
        # 诊断字段:让调用方(bot / 前端 / 压测)能判断"是不是真的命中了"
        "prefer": list(result.prefer),
        "avoid": list(result.avoid),
        "pool_size": result.pool_size,
        "matched_count": result.matched_count,
        "relaxed": result.relaxed,
        "no_match": result.no_match,
        "exhausted": result.exhausted,
        "skipped_shown": result.skipped_shown,
    }
