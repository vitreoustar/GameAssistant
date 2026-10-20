"""Agent 聊天端点(SSE 流式)。

POST /api/chat  body: {"message": str, "library": [Game] | null, "context": {...}}
响应: text/event-stream,每行 `data: {json}\n\n`

`context` 是**前端持有的对话状态**,服务端不存会话:

| 字段 | 用途 |
|---|---|
| `candidates` | 上次推荐的候选,用于解析「第 x 个」 |
| `recommend`  | 累积的偏好 + 已展示 appid,用于多轮去重与偏好过滤 |
| `history`    | 最近几轮对话,供路由 LLM 消解省略与指代 |

三者都由 `done` 事件回传、前端下一轮原样带上。
"""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.stream import stream_agent
from app.schemas import Game

router = APIRouter()

MAX_HISTORY_TURNS = 8
MAX_TURN_CHARS = 400


class ChatTurn(BaseModel):
    role: str = "user"
    text: str = Field(default="", max_length=MAX_TURN_CHARS)


class RecommendStateIn(BaseModel):
    """多轮推荐状态(前端回传)。字段含义见 `app/recommend/preferences.py`。"""

    genres: list[str] = Field(default_factory=list, max_length=16)
    avoid: list[str] = Field(default_factory=list, max_length=16)
    unmapped: list[str] = Field(default_factory=list, max_length=16)
    shown: list[int] = Field(default_factory=list, max_length=1000)
    turn: int = 0


class ChatContext(BaseModel):
    candidates: list[Game] | None = None  # 上次推荐候选,用于解析「第x个」
    recommend: RecommendStateIn | None = None
    history: list[ChatTurn] | None = None


class ChatRequest(BaseModel):
    message: str
    library: list[Game] | None = None
    context: ChatContext | None = None


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    library = [g.model_dump() for g in (req.library or [])]
    context = req.context
    context_candidates = (
        [g.model_dump() for g in (context.candidates or [])] if context else []
    )
    recommend_state = (
        context.recommend.model_dump() if context and context.recommend else None
    )
    history = (
        [t.model_dump() for t in (context.history or [])][-MAX_HISTORY_TURNS:]
        if context
        else []
    )

    async def event_stream():
        try:
            async for ev in stream_agent(
                req.message, library, context_candidates, recommend_state, history
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 兜底:把错误也作为事件返回,前端可显示
            err = {"type": "error", "data": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
