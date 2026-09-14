"""流式事件测试:用假 LLM 验证 node/token/done 事件产出。

运行: cd backend && .venv/Scripts/python tests/test_stream.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.agent.graph as graph_mod
from app.agent.stream import stream_agent


class FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def bind(self, **kwargs) -> "FakeLLM":
        return self

    async def ainvoke(self, messages, **kwargs) -> FakeResp:
        return FakeResp('{"intent":"dlc"}')

    async def astream(self, messages, **kwargs):
        sys_msg = str(messages[0].content) if messages else ""
        if "DLC 信息" in sys_msg:
            yield FakeChunk("这是")
            yield FakeChunk("DLC 摘要")
        else:
            yield FakeChunk("这是回复")


LIBRARY = [
    {"appid": 570, "name": "Dota 2", "playtime_forever": 500},
    {"appid": 292030, "name": "The Witcher 3: Wild Hunt", "playtime_forever": 120},
]


async def main() -> None:
    graph_mod.get_llm = lambda: FakeLLM()  # type: ignore[method-assign]

    events = []
    async for ev in stream_agent("整理 DLC", LIBRARY):
        events.append(ev)

    types = [e["type"] for e in events]
    print("event types:", types)
    assert "node" in types and "done" in types, "缺少 node/done 事件"

    done = events[-1]
    assert done["type"] == "done"
    assert done["data"]["intent"] == "dlc"
    assert done["data"]["dlc_report"] is not None
    assert "DLC 摘要" in done["data"]["text"]
    print("done data keys:", list(done["data"].keys()))
    print("OK: stream test passed")


if __name__ == "__main__":
    asyncio.run(main())
