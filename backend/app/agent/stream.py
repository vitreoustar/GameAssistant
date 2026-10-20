"""Agent 流式执行:通过 astream_events 产出 SSE 事件。

事件类型:
  {"type": "node",  "data": {"node": str, "status": "start"|"end"}}  节点状态
  {"type": "token", "node": str, "data": str}                        LLM token 流
  {"type": "done",  "data": {...}}                                   最终结果

`done` 里多一个 `recommend_state`:本轮之后的偏好 + 已展示 appid。
前端必须把它**原样回传**(`ChatContext.recommend`),否则下一轮去重记录归零,
用户会重新看到刚推荐过的游戏 —— 这是多轮推荐最容易踩的坑。
"""
from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from app.agent import agent

NODE_NAMES = {"route", "dlc", "dlc_detail", "recommend", "game_ask", "chat"}
# 只有这些节点的 token 才向前端流式输出(route 的内部 JSON 不外泄)
STREAM_NODES = {"dlc", "recommend", "game_ask", "chat"}


async def stream_agent(
    message: str,
    library: list[dict],
    context_candidates: list[dict] | None = None,
    recommend_state: dict | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    input_payload = {
        "messages": [HumanMessage(content=message)],
        "library": library or [],
        "context_candidates": context_candidates or [],
        "recommend_state": recommend_state or None,
        "history": history or [],
    }
    final_state: dict | None = None

    async for event in agent.astream_events(input_payload, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        meta = event.get("metadata", {})

        if kind == "on_chat_model_stream":
            node = meta.get("langgraph_node")
            if node in STREAM_NODES:
                chunk = event["data"].get("chunk")
                if chunk is not None and chunk.content:
                    yield {"type": "token", "node": node, "data": str(chunk.content)}
        elif kind == "on_chain_start" and name in NODE_NAMES:
            yield {"type": "node", "data": {"node": name, "status": "start"}}
        elif kind == "on_chain_end":
            if name in NODE_NAMES:
                yield {"type": "node", "data": {"node": name, "status": "end"}}
            elif name == "LangGraph" and not event.get("parent_ids"):
                final_state = event["data"].get("output")

    if final_state is None:
        final_state = {}

    messages = final_state.get("messages") or []
    text = str(messages[-1].content) if messages else ""
    yield {
        "type": "done",
        "data": {
            "text": text,
            "intent": final_state.get("intent"),
            "dlc_report": final_state.get("dlc_report"),
            "profile": final_state.get("profile"),
            "candidates": final_state.get("candidates"),
            "game_analysis": final_state.get("game_analysis"),
            "game_candidates": final_state.get("game_candidates"),
            "recommend_state": final_state.get("recommend_state"),
        },
    }
