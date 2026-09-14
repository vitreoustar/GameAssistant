"""DLC 整理:基于已拥有游戏库,计算每款游戏的已拥有/未拥有 DLC 与价格。"""
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


async def build_dlc_report(games: list[dict]) -> dict:
    """返回 DLC 报告。

    games: [{"appid", "name", "playtime_forever"}]（已拥有的游戏，含 DLC 条目时更准确）。
    """
    if not games:
        return {
            "games_with_dlc": 0,
            "total_missing_dlc": 0,
            "total_missing_price_cents": 0,
            "rows": [],
        }

    owned_ids = {g["appid"] for g in games}
    details = await client.get_many_app_details(list(owned_ids))

    # 只保留本体游戏(type == "game")
    base = [(aid, d) for aid, d in details.items() if d and d.get("type") == "game"]

    # 收集所有 DLC appid
    base_dlc: dict[int, list[int]] = {}
    all_dlc_ids: set[int] = set()
    for aid, d in base:
        dlc_ids = [int(x) for x in (d.get("dlc") or [])]
        base_dlc[aid] = dlc_ids
        all_dlc_ids.update(dlc_ids)

    # 补充拉取尚未有详情的 DLC
    need = [x for x in all_dlc_ids if x not in details]
    if need:
        details.update(await client.get_many_app_details(need))

    rows: list[dict] = []
    total_missing = 0
    total_price = 0
    for aid, d in base:
        owned_dlc: list[dict] = []
        missing_dlc: list[dict] = []
        for did in base_dlc[aid]:
            dd = details.get(did)
            if did in owned_ids:
                owned_dlc.append(_dlc_item(did, dd, owned=True))
            else:
                item = _dlc_item(did, dd, owned=False)
                missing_dlc.append(item)
                total_missing += 1
                if item["price_cents"]:
                    total_price += item["price_cents"]
        if owned_dlc or missing_dlc:
            rows.append(
                {
                    "appid": aid,
                    "name": d.get("name", ""),
                    "header_image": d.get("header_image"),
                    "owned_dlc": owned_dlc,
                    "missing_dlc": missing_dlc,
                    "missing_count": len(missing_dlc),
                }
            )

    rows.sort(key=lambda r: r["missing_count"], reverse=True)
    return {
        "games_with_dlc": len(rows),
        "total_missing_dlc": total_missing,
        "total_missing_price_cents": total_price,
        "rows": rows,
    }
