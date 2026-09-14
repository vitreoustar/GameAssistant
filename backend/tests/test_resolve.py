"""游戏名解析测试(中文/英文名 → appid)。

运行: cd backend && .venv/Scripts/python tests/test_resolve.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.agent.graph as g  # noqa: E402

LIB = [
    {"appid": 1718570, "name": "ASTLIBRA Revision", "playtime_forever": 50},
    {"appid": 570, "name": "Dota 2", "playtime_forever": 500},
]


async def main() -> None:
    # 1) 英文名子串匹配
    en = g._find_games_mentioned("展开 Dota 2 的 DLC", LIB)
    print("英文子串匹配 Dota 2 ->", en)
    assert en == [570]

    # 2) 中文名经商店搜索解析(模拟 LLM 提取)
    async def fake_extract(msg: str) -> str:
        return "神之天平"

    g._extract_game_name = fake_extract  # type: ignore[method-assign]
    ids = await g._resolve_game_appids("展开神之天平的 DLC", LIB)
    print("中文名解析 神之天平 ->", ids)
    assert ids == [1718570], ids

    print("OK: resolve test passed")


if __name__ == "__main__":
    asyncio.run(main())
