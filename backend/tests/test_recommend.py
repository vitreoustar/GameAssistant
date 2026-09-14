"""推荐 & 游戏评价流程测试(使用真实 LLM/Steam/RAG)。

运行: cd backend && .venv/Scripts/python tests/test_recommend.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from app.agent import agent  # noqa: E402
from app.rag import store as rag_store  # noqa: E402
from app.agent.graph import _build_profile  # noqa: E402

LIB = [
    {"appid": 570, "name": "Dota 2", "playtime_forever": 1200},
    {"appid": 292030, "name": "The Witcher 3: Wild Hunt", "playtime_forever": 150},
    {"appid": 1086940, "name": "Baldur's Gate 3", "playtime_forever": 90},
]


async def main() -> None:
    print("nodes:", list(agent.get_graph().nodes.keys()))

    # 1) RAG 富元数据
    rs = rag_store.search("cozy farming simulation", k=2)
    for r in rs:
        print("RAG:", r["name"], "| rating=", r["rating"], "| header=", r["header_image"][:40])

    # 2) 画像
    profile = await _build_profile(LIB)
    print("profile:", profile["summary"])

    # 3) 推荐全流程
    res = await agent.ainvoke(
        {"messages": [HumanMessage(content="推荐几款我可能喜欢的游戏")], "library": LIB}
    )
    print("recommend intent =", res.get("intent"))
    print("recommend text =", str(res["messages"][-1].content)[:120])
    cands = res.get("candidates") or []
    print(f"candidates = {len(cands)}")
    for c in cands[:3]:
        print(f"  - {c['name']} | {c['rating']} | reason={c.get('reason','')[:40]}")

    # 4) 游戏评价全流程
    res2 = await agent.ainvoke(
        {"messages": [HumanMessage(content="赛博朋克2077怎么样")], "library": LIB}
    )
    print("game_ask intent =", res2.get("intent"))
    ga = res2.get("game_analysis")
    if ga:
        print("game_analysis:", ga["name"], "| rating=", ga["rating"], "| match=", ga["match_genres"], "| owned=", ga["owned"])
    print("game_ask text =", str(res2["messages"][-1].content)[:150])

    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
