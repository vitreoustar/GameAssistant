"""前端方案后端验证:推荐限5、第x个解析、多结果卡片、真实评价。

运行: cd backend && .venv/Scripts/python tests/test_frontend_plan.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from app.agent import agent  # noqa: E402
from app.agent.graph import _extract_ordinal  # noqa: E402

LIB = [
    {"appid": 570, "name": "Dota 2", "playtime_forever": 1200},
    {"appid": 292030, "name": "The Witcher 3: Wild Hunt", "playtime_forever": 150},
    {"appid": 1086940, "name": "Baldur's Gate 3", "playtime_forever": 90},
]


async def main() -> None:
    # 1) 序数解析
    for s in ["第2个游戏", "第 3 款", "第一款"]:
        print("ordinal", s, "->", _extract_ordinal(s))

    # 2) 推荐限 5
    res = await agent.ainvoke(
        {"messages": [HumanMessage(content="推荐几款我可能喜欢的游戏")], "library": LIB}
    )
    cands = res.get("candidates") or []
    print("推荐候选数:", len(cands))
    assert len(cands) <= 5

    # 3) 第2个 -> 详细介绍(带上下文)
    ctx = [{"appid": c["appid"], "name": c["name"]} for c in cands]
    res2 = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="第2个游戏好像不错,详细说说")],
            "library": LIB,
            "context_candidates": ctx,
        }
    )
    ga = res2.get("game_analysis")
    print("第2个 -> intent =", res2.get("intent"), "| game =", ga["name"] if ga else None)
    print("  分析文本:", str(res2["messages"][-1].content)[:120])

    # 4) 多结果 -> 卡片
    res3 = await agent.ainvoke(
        {"messages": [HumanMessage(content="Call of Duty怎么样")], "library": LIB}
    )
    gc = res3.get("game_candidates") or []
    print("Call of Duty 多结果卡片数:", len(gc))
    for c in gc:
        print("  -", c["name"], "|", c["rating"], "| overlap=", c["match_genres"])

    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
