"""DLC 整理:基于已拥有游戏库,计算每款游戏的已拥有/未拥有 DLC 与价格。

性能优化:
  - top_games:只分析游玩时长最靠前的 N 款游戏(默认 20),避免全库遍历
  - max_dlc_per_game:每款游戏最多列出 N 个未拥有/已拥有 DLC(默认 8)
  - only_appids:只分析指定游戏(用于用户追问时展开完整列表)
"""
from __future__ import annotations

from app.steam import client


def _dlc_item(appid: int, details: dict | None, owned: bool) -> dict:
    name = (details or {}).get("name") or f"App {appid}"
    price_cents: int | None = None
    currency: str | None = None
    if details:
        po = details.get("price_overview")
        if po:
            price_cents = po.get("final")
            currency = po.get("currency")
    return {
        "appid": appid,
        "name": name,
        "owned": owned,
        "price_cents": price_cents,
        "currency": currency,
    }


def _empty_report(total_games: int = 0) -> dict:
    return {
        "games_with_dlc": 0,
        "total_missing_dlc": 0,
        "total_missing_price_cents": 0,
        "processed_games": 0,
        "total_games": total_games,
        "capped": False,
        "rows": [],
    }


async def build_dlc_report(
    games: list[dict],
    top_games: int | None = 20,
    max_dlc_per_game: int | None = 8,
    only_appids: list[int] | None = None,
) -> dict:
    """构建 DLC 报告。

    - top_games=None 表示不限制游戏数
    - max_dlc_per_game=None 表示不限 DLC 数量(追问展开时用)
    - only_appids 只分析指定的游戏 appid
    """
    if not games:
        return _empty_report(0)

    total_games = len(games)
    owned_ids = {g["appid"] for g in games}

    if only_appids:
        want = set(only_appids)
        target = [g for g in games if g["appid"] in want]
    else:
        target = sorted(games, key=lambda g: g.get("playtime_forever") or 0, reverse=True)
        if top_games:
            target = target[:top_games]

    target_ids = [g["appid"] for g in target]
    details = await client.get_many_app_details(target_ids)
    base = [(aid, d) for aid, d in details.items() if d and d.get("type") == "game"]

    # 1) 纯集合运算分类 DLC(不额外请求)
    per_game: dict[int, tuple[list[int], list[int]]] = {}
    to_fetch: set[int] = set()
    for aid, d in base:
        dlc_ids = [int(x) for x in (d.get("dlc") or [])]
        owned = [x for x in dlc_ids if x in owned_ids]
        missing = [x for x in dlc_ids if x not in owned_ids]
        per_game[aid] = (owned, missing)
        limit = max_dlc_per_game if max_dlc_per_game is not None else len(dlc_ids)
        to_fetch.update(missing[:limit])
        to_fetch.update(owned[:limit])

    # 2) 只拉取需要展示的 DLC 详情(名称/价格)
    need = [x for x in to_fetch if x not in details]
    if need:
        details.update(await client.get_many_app_details(need))

    playtime = {g["appid"]: g.get("playtime_forever") for g in target}

    rows: list[dict] = []
    total_missing = 0
    total_price = 0
    for aid, d in base:
        owned_list, missing_list = per_game[aid]
        limit = max_dlc_per_game if max_dlc_per_game is not None else len(missing_list)
        missing_shown = [_dlc_item(x, details.get(x), owned=False) for x in missing_list[:limit]]
        owned_shown = [_dlc_item(x, details.get(x), owned=True) for x in owned_list[:limit]]
        total_missing += len(missing_list)
        total_price += sum(m["price_cents"] or 0 for m in missing_shown)

        rows.append(
            {
                "appid": aid,
                "name": d.get("name", ""),
                "header_image": d.get("header_image"),
                "playtime_forever": playtime.get(aid),
                "owned_dlc": owned_shown,
                "missing_dlc": missing_shown,
                "missing_count": len(missing_list),
                "owned_count": len(owned_list),
                "truncated": len(missing_list) > len(missing_shown),
            }
        )

    rows.sort(key=lambda r: r["missing_count"], reverse=True)
    return {
        "games_with_dlc": len(rows),
        "total_missing_dlc": total_missing,
        "total_missing_price_cents": total_price,
        "processed_games": len(target),
        "total_games": total_games,
        "capped": total_games > len(target),
        "rows": rows,
    }
