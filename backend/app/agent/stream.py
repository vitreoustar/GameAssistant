"""Agent 流式执行:通过 astream_events 产出 SSE 事件。

事件类型:
  {"type": "node",  "data": {"node": str, "status": "start"|"end"}}  节点状态
  {"type": "token", "node": str, "data": str}                        LLM token 流
  {"type": "done",  "data": {text, intent, dlc_report, candidates}}  最终结果
"""
from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from app.agent import agent

NODE_NAMES = {"route", "dlc", "recommend", "chat"}
# 只有这些节点的 token 才向前端流式输出(route 的内部 JSON 不外泄)
STREAM_NODES = {"dlc", "recommend", "chat"}


async def stream_agent(message: str, library: list[dict]) -> AsyncIterator[dict]:
    input_payload = {
        "messages": [HumanMessage(content=message)],
        "library": library or [],
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
            "candidates": final_state.get("candidates"),
        },
    }
