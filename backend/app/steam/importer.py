"""游戏库导入:把 SteamID 解析成规范化的游戏列表。"""
from __future__ import annotations

from app.steam import client

SUPPORTED_MODES = ("steamid",)


async def resolve_library(mode: str, value: str) -> list[dict]:
    """按模式解析用户游戏库。

    mode:
      - steamid : value 为 steamid64,走 GetOwnedGames(需 STEAM_API_KEY +
                  用户「游戏详情」隐私设为公开)

    返回 [{"appid", "name", "playtime_forever"}]。

    注:早期版本还支持主页 HTML 抓取(url)与粘贴游戏名匹配(names),
    现已移除 —— 前者依赖 Steam 社区页面结构、脆弱且易被限流,
    后者匹配准确率不足以支撑推荐质量。
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("请提供 steamid64")

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"不支持的导入模式:{mode}(仅支持 {'/'.join(SUPPORTED_MODES)})"
        )

    return await client.get_owned_games(value)
