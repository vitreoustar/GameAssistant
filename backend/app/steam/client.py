"""Steam 数据访问客户端(异步)。

覆盖两种数据来源:
  * Steam Web API   —— GetOwnedGames(需 Steamworks key)
  * Steam Store API —— appdetails / storesearch / appreviews(无需 key)

内置:
  * `TTLCache`     —— 进程内响应缓存,避免重复请求
  * `RateLimiter`  —— 全局令牌桶限速,所有请求(含并发批处理)都经过它
  * in-flight 去重 —— 同一 URL 的并发请求只打一次
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

WEB_API = "https://api.steampowered.com"
STORE_BASE = "https://store.steampowered.com"
STORE_API = "https://store.steampowered.com/api"

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
    """全局令牌桶限速器。

    与"相邻两次请求至少间隔 N 秒"相比,令牌桶有两个好处:
      1. 允许突发(burst):并发批处理可以一次性拿到多个令牌,不必逐条排队等 0.3s;
      2. 有总量上限:无论并发多高,长期平均速率都不超过 rate/秒。

    之前 `get_many_app_details` 通过 `use_limiter=False` 完全绕过限速,
    导致 5 路并发无节流,是真实的 429 风险点;现在所有请求都走这里。
    """

    def __init__(self, rate: float = 10.0, burst: int = 5) -> None:
        if rate <= 0:
            raise ValueError("rate 必须为正数")
        self._rate = float(rate)
        self._burst = max(1, int(burst))
        self._tokens = float(self._burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self._burst, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self._rate)


class SteamClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._http_lock = asyncio.Lock()
        self._cache = TTLCache(ttl=3600)
        self._limiter = RateLimiter(
            rate=settings.steam_rate_limit, burst=settings.steam_rate_burst
        )
        # cache_key -> 正在进行的请求,避免同一 appid 被并发重复拉取
        self._inflight: dict[str, asyncio.Task] = {}

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            async with self._http_lock:
                if self._http is None:
                    proxy = (
                        httpx.Proxy(settings.steam_proxy) if settings.steam_proxy else None
                    )
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

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        """带缓存 + 限速 + in-flight 去重的 GET,返回解析后的 JSON。"""
        cache_key = f"{url}?{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        task = self._inflight.get(cache_key)
        if task is None:
            task = asyncio.ensure_future(self._fetch_json(url, params))
            self._inflight[cache_key] = task
            task.add_done_callback(lambda _t, k=cache_key: self._inflight.pop(k, None))
        # shield:某个等待者被取消时不牵连共享请求
        return await asyncio.shield(task)

    async def _fetch_json(self, url: str, params: dict[str, Any]) -> Any:
        await self._limiter.wait()
        client = await self._client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        self._cache.set(f"{url}?{sorted(params.items())}", data)
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

    async def get_app_details(self, appid: int) -> dict | None:
        """appdetails:返回单款游戏详情(名称/类型/dlc 列表/价格/标签等)。"""
        data = await self._get_json(
            f"{STORE_API}/appdetails",
            {"appids": appid, "l": "english", "cc": "us"},
        )
        entry = (data or {}).get(str(appid)) or {}
        if not entry.get("success"):
            return None
        return entry.get("data")

    async def get_many_app_details(
        self, appids: list[int], *, concurrency: int = 5
    ) -> dict[int, dict | None]:
        """并发批量拉取 appdetails。

        并发只用来"同时发出",总量仍由全局令牌桶(STEAM_RATE_LIMIT)约束;
        同一 appid 的重复请求会被 in-flight 去重合并成一次网络请求。
        """
        sem = asyncio.Semaphore(concurrency)

        async def one(appid: int) -> tuple[int, dict | None]:
            async with sem:
                try:
                    return appid, await self.get_app_details(appid)
                except Exception:  # 单个失败不拖垮整体
                    return appid, None

        pairs = await asyncio.gather(*(one(a) for a in appids))
        return dict(pairs)

    async def get_app_reviews(
        self, appid: int, num: int = 3, lang: str = "schinese"
    ) -> dict:
        """Steam 用户评价(免费,无需 key)。"""
        data = await self._get_json(
            f"{STORE_BASE}/appreviews/{appid}",
            {
                "json": 1,
                "language": lang,
                "purchase_type": "all",
                "num_per_page": num,
                "filter": "summary",
            },
        )
        return {
            "summary": (data or {}).get("query_summary") or {},
            "reviews": (data or {}).get("reviews") or [],
        }

