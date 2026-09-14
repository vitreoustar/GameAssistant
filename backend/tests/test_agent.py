"""Agent 图流转测试:用假 LLM 验证三种意图的节点流转(不依赖真实 key)。

运行: cd backend && STEAM_PROXY=... .venv/Scripts/python tests/test_agent.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

import app.agent.graph as graph_mod


class FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def bind(self, **kwargs) -> "FakeLLM":
        return self

    async def ainvoke(self, messages, **kwargs) -> FakeResp:
        sys_msg = str(messages[0].content) if messages else ""
        human = str(messages[-1].content) if messages else ""
        if "判断用户意图" in sys_msg:
            h = human.lower()
            if "dlc" in h:
                return FakeResp('{"intent":"dlc"}')
            if "推荐" in h or "recommend" in h:
                return FakeResp('{"intent":"recommend"}')
            return FakeResp('{"intent":"chat"}')
        if "DLC 报告" in sys_msg:
            return FakeResp("(测试)这是 DLC 摘要。")
        if "推荐助手" in sys_msg:
            return FakeResp("(测试)这是推荐结果。")
        return FakeResp("(测试)这是闲聊回复。")


LIBRARY = [
    {"appid": 570, "name": "Dota 2", "playtime_forever": 500},
    {"appid": 292030, "name": "The Witcher 3: Wild Hunt", "playtime_forever": 120},
    {"appid": 1086940, "name": "Baldur's Gate 3", "playtime_forever": 80},
]


async def main() -> None:
    graph_mod.get_llm = lambda: FakeLLM()  # type: ignore[method-assign]

    for message in ["整理一下我的 DLC", "推荐几款我可能喜欢的游戏", "你好呀"]:
        result = await graph_mod.agent.ainvoke(
            {"messages": [HumanMessage(content=message)], "library": LIBRARY}
        )
        intent = result.get("intent")
        reply = result["messages"][-1].content
        has_dlc = result.get("dlc_report") is not None
        n_cand = len(result.get("candidates") or [])
        print(f"[{message}] -> intent={intent} | dlc_report={has_dlc} | candidates={n_cand}")
        print(f"    reply: {str(reply)[:80]}")
        assert intent in {"dlc", "recommend", "chat"}

    print("OK: agent flow test passed")


if __name__ == "__main__":
    asyncio.run(main())
