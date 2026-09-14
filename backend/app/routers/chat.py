"""Agent 聊天端点(SSE 流式)。

POST /api/chat  body: {"message": str, "library": [Game] | null}
响应: text/event-stream,每行 `data: {json}\n\n`
"""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.stream import stream_agent
from app.schemas import Game

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    library: list[Game] | None = None


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    library = [g.model_dump() for g in (req.library or [])]

    async def event_stream():
        try:
            async for ev in stream_agent(req.message, library):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 兜底:把错误也作为事件返回,前端可显示
            err = {"type": "error", "data": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
