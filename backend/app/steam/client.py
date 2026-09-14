"""Steam 数据访问客户端(异步)。

覆盖三种数据来源:
  * Steam Web API      —— GetOwnedGames / GetAppList(需 Steamworks key 的仅前者)
  * Steam Store API    —— appdetails(无需 key)
  * Steam Community    —— 个人主页游戏列表抓取(无需 key,需资料公开)

内置 TTL 缓存 + 简单限速,避免重复请求。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

WEB_API = "https://api.steampowered.com"
STORE_API = "https://store.steampowered.com/api"
COMMUNITY = "https://steamcommunity.com"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class TTLCache:
    """进程内 TTL 缓存。"""

    def __init__(self, ttl: float = 3600.0) -> None:
        self._ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.time() < expires_at:
            return value
        self._data.pop(key, None)
        return None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.time() + self._ttl, value)


class RateLimiter:
    """保证相邻两次请求至少间隔 min_interval 秒。"""

    def __init__(self, min_interval: float = 0.3) -> None:
        self._min_interval = min_interval
        self._next_allowed = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_allowed:
            await asyncio.sleep(self._next_allowed - now)
        self._next_allowed = time.monotonic() + self._min_interval


class SteamClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._cache = TTLCache(ttl=3600)
        self._limiter = RateLimiter(min_interval=0.3)

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            proxy = httpx.Proxy(settings.steam_proxy) if settings.steam_proxy else None
            self._http = httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                verify=settings.steam_verify_ssl,
                proxy=proxy,
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ---------- 基础请求 ----------

    async def _get_json(
        self, url: str, params: dict[str, Any], *, use_limiter: bool = True
    ) -> Any:
        """带缓存 + 限速的 GET，返回解析后的 JSON。"""
        cache_key = f"{url}?{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if use_limiter:
            await self._limiter.wait()
        client = await self._client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        self._cache.set(cache_key, data)
        return data

    # ---------- Steam Web API ----------

    async def get_owned_games(self, steamid: str) -> list[dict]:
        """GetOwnedGames：读取某玩家拥有/玩过的游戏(需 key + 资料公开)。"""
        if not settings.steam_api_key:
            raise ValueError("未配置 STEAM_API_KEY,无法调用 GetOwnedGames")
        data = await self._get_json(
            f"{WEB_API}/IPlayerService/GetOwnedGames/v1/",
            {
                "key": settings.steam_api_key,
                "steamid": steamid,
                "include_appinfo": 1,
                "include_played_free_games": 1,
                "format": "json",
            },
        )
        games = (data.get("response") or {}).get("games") or []
        return [
            {
                "appid": g["appid"],
                "name": g.get("name", ""),
                "playtime_forever": g.get("playtime_forever"),
            }
            for g in games
        ]

    async def search_store(self, term: str, lang: str = "english") -> list[dict]:
        """Store 搜索接口(替代已下线的 GetAppList),按名称搜索。

        lang 支持 english / schinese,中文名搜索需用 schinese。
        """
        cc = "cn" if lang == "schinese" else "us"
        data = await self._get_json(
            f"{STORE_API}/storesearch/",
            {"term": term, "l": lang, "cc": cc},
        )
        return (data or {}).get("items") or []

    # ---------- Steam Store API ----------

    async def get_app_details(self, appid: int, *, use_limiter: bool = True) -> dict | None:
        """appdetails：返回单款游戏详情(名称/类型/dlc 列表/价格/标签等)。"""
        data = await self._get_json(
            f"{STORE_API}/appdetails",
            {"appids": appid, "l": "english", "cc": "us"},
            use_limiter=use_limiter,
        )
        entry = (data or {}).get(str(appid)) or {}
        if not entry.get("success"):
            return None
        return entry.get("data")

    async def get_many_app_details(
        self, appids: list[int], *, concurrency: int = 5
    ) -> dict[int, dict | None]:
        """并发批量拉取 appdetails(信号量限流,用于 DLC/库分析)。"""
        sem = asyncio.Semaphore(concurrency)

        async def one(appid: int) -> tuple[int, dict | None]:
            async with sem:
                try:
                    return appid, await self.get_app_details(appid, use_limiter=False)
                except Exception:  # 单个失败不拖垮整体
                    return appid, None

        pairs = await asyncio.gather(*(one(a) for a in appids))
        return dict(pairs)

    # ---------- Steam Community 抓取 ----------

    async def scrape_profile_games(self, profile: str) -> list[dict]:
        """抓取个人主页 games 页(无需 key,需「游戏详情」公开)。

        profile 可为:完整 URL、/id/自定义、/profiles/steamid64、或裸 id。
        """
        url = self._profile_url(profile)
        client = await self._client()
        await self._limiter.wait()
        resp = await client.get(f"{url}/games/", params={"tab": "all", "l": "english"})
        resp.raise_for_status()
        games = _parse_games_html(resp.text)
        if not games:
            raise ValueError(
                "未能从主页解析到游戏:请确认「游戏详情」隐私设为公开,"
                "或改用 steamid / 粘贴导入模式"
            )
        return games

    @staticmethod
    def _profile_url(profile: str) -> str:
        p = profile.strip().rstrip("/")
        if p.startswith("http"):
            return p
        # 17 位纯数字 -> steamid64
        if re.fullmatch(r"\d{17}", p):
            return f"{COMMUNITY}/profiles/{p}"
        return f"{COMMUNITY}/id/{p}"

    # ---------- 名称匹配导入 ----------

    async def match_games_by_name(self, names: list[str]) -> list[dict]:
        """把粘贴的游戏名列表匹配成 appid(用于无 key/无公开资料场景)。

        用 Store 搜索接口逐个匹配:先取归一化名称完全一致的 app;
        否则用 appdetails 的 type 字段找真正的 game(排除 DLC/原声带等)。
        """
        result: list[dict] = []
        for raw in names:
            key = _norm(raw)
            items = await self.search_store(raw)
            apps = [it for it in items if it.get("type") == "app"]

            matched: dict | None = None
            # pass 1: 名称归一化后完全一致
            for it in apps:
                if _norm(it.get("name", "")) == key:
                    matched = it
                    break
            # pass 2: 用 appdetails 的 type 字段找真正的 game
            if matched is None:
                for it in apps:
                    details = await self.get_app_details(it["id"])
                    if details and details.get("type") == "game":
                        matched = it
                        break
            if matched is None and apps:
                matched = apps[0]
            if matched is None:
                continue  # 无法匹配的忽略

            result.append(
                {
                    "appid": matched["id"],
                    "name": matched.get("name", raw),
                    "playtime_forever": None,
                }
            )
        return result


def _norm(name: str) -> str:
    """归一化游戏名用于匹配:去空白、转小写、去版权符号与撇号/连字符等。"""
    s = name.strip().lower()
    s = re.sub(r"[™®©'’\"\-—:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_games_html(html: str) -> list[dict]:
    """解析 games 页 HTML 的 gameListRow 块。"""
    soup = BeautifulSoup(html, "html.parser")
    games: list[dict] = []
    seen: set[int] = set()

    for row in soup.find_all("div", class_="gameListRow"):
        appid: int | None = None
        m = re.search(r"game_(\d+)", row.get("id", ""))
        if m:
            appid = int(m.group(1))

        name = ""
        name_el = row.find("div", class_="gameListRowItemName")
        if name_el is not None:
            name = name_el.get_text(strip=True)

        # 兜底:从商店链接提取
        link = row.find("a", href=re.compile(r"store\.steampowered\.com/app/\d+"))
        if link is not None:
            if appid is None:
                lm = re.search(r"/app/(\d+)", link["href"])
                appid = int(lm.group(1)) if lm else None
            if not name:
                name = link.get_text(strip=True)

        playtime: float | None = None
        hours_el = row.find("div", class_="gameListRowItem")
        if hours_el is not None:
            hm = re.search(r"([\d.,]+)\s*hrs", hours_el.get_text(" ", strip=True))
            if hm:
                try:
                    playtime = float(hm.group(1).replace(",", ""))
                except ValueError:
                    playtime = None

        if appid is not None and appid not in seen:
            seen.add(appid)
            games.append(
                {"appid": appid, "name": name, "playtime_forever": playtime}
            )

    return games
