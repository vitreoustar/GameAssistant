"""游戏库导入:统一三种模式,返回规范化的游戏列表。"""
from __future__ import annotations

from app.steam import client


async def resolve_library(mode: str, value: str) -> list[dict]:
    """按模式解析用户游戏库。

    mode:
      - steamid : value 为 steamid64,走 GetOwnedGames(需 key)
      - url     : value 为个人主页 URL 或 /id/xxx,抓取 games 页
      - names   : value 为游戏名列表(每行一个),名称匹配 appid

    返回 [{"appid", "name", "playtime_forever"}]。
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("请提供 steamid / 主页地址 / 游戏名列表")

    if mode == "steamid":
        return await client.get_owned_games(value)
    if mode == "url":
        return await client.scrape_profile_games(value)
    if mode == "names":
        names = [n.strip() for n in value.splitlines() if n.strip()]
        return await client.match_games_by_name(names)
    raise ValueError(f"未知导入模式: {mode}")
