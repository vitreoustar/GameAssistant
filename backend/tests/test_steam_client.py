"""Steam 数据层单元测试(全离线,不访问网络)。

覆盖:
  * TTLCache     —— 命中 / 未命中 / 过期
  * RateLimiter  —— 令牌桶的突发与总量约束(「并发绕过限速」修复的回归测试)
  * steam/dlc.py —— 空库报告、DLC 条目构造

运行: cd backend && .venv/Scripts/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import time
import unittest

from app.steam.client import RateLimiter, TTLCache
from app.steam.dlc import _dlc_item, _empty_report, build_dlc_report


class TestTTLCache(unittest.TestCase):
    def test_set_then_get(self):
        cache = TTLCache(ttl=60)
        cache.set("k", {"value": 1})
        self.assertEqual(cache.get("k"), {"value": 1})

    def test_missing_key_returns_none(self):
        self.assertIsNone(TTLCache(ttl=60).get("missing"))

    def test_entry_expires(self):
        cache = TTLCache(ttl=0)
        cache.set("k", 1)
        time.sleep(0.01)
        self.assertIsNone(cache.get("k"))


class TestRateLimiter(unittest.TestCase):
    """令牌桶:burst 内立即放行,超出部分按 rate 排队。"""

    def test_rejects_non_positive_rate(self):
        with self.assertRaises(ValueError):
            RateLimiter(rate=0)

    def test_burst_is_immediate(self):
        async def run() -> float:
            limiter = RateLimiter(rate=1.0, burst=5)
            start = time.monotonic()
            for _ in range(5):
                await limiter.wait()
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        self.assertLess(elapsed, 0.2, f"burst 内的请求不应排队,却耗时 {elapsed:.3f}s")

    def test_excess_requests_are_throttled(self):
        """burst=2、rate=20/s,共 6 个请求 -> 至少 (6-2)/20 = 0.2s。"""

        async def run() -> float:
            limiter = RateLimiter(rate=20.0, burst=2)
            start = time.monotonic()
            for _ in range(6):
                await limiter.wait()
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        self.assertGreaterEqual(elapsed, 0.15, f"限速失效,仅耗时 {elapsed:.3f}s")

    def test_concurrent_waiters_share_one_bucket(self):
        """并发 10 个请求走同一个桶:burst=2、rate=20/s -> 至少 (10-2)/20 = 0.4s。

        这是修复前的漏洞:并发路径通过 use_limiter=False 完全绕过限速。
        """

        async def run() -> float:
            limiter = RateLimiter(rate=20.0, burst=2)
            start = time.monotonic()
            await asyncio.gather(*(limiter.wait() for _ in range(10)))
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        self.assertGreaterEqual(elapsed, 0.3, f"并发请求未被限速,仅耗时 {elapsed:.3f}s")


class TestDlcHelpers(unittest.TestCase):
    def test_empty_report_shape(self):
        report = _empty_report(total_games=7)
        self.assertEqual(report["total_games"], 7)
        self.assertEqual(report["rows"], [])
        self.assertEqual(report["total_missing_dlc"], 0)
        self.assertFalse(report["capped"])

    def test_dlc_item_without_details(self):
        item = _dlc_item(123, None, owned=False)
        self.assertEqual(item["appid"], 123)
        self.assertEqual(item["name"], "App 123")
        self.assertIsNone(item["price_cents"])
        self.assertFalse(item["owned"])

    def test_dlc_item_with_price(self):
        details = {
            "name": "Some DLC",
            "price_overview": {"final": 499, "currency": "USD"},
        }
        item = _dlc_item(9, details, owned=True)
        self.assertEqual(item["name"], "Some DLC")
        self.assertEqual(item["price_cents"], 499)
        self.assertEqual(item["currency"], "USD")
        self.assertTrue(item["owned"])


class TestBuildDlcReport(unittest.IsolatedAsyncioTestCase):
    async def test_empty_library_makes_empty_report(self):
        report = await build_dlc_report([])
        self.assertEqual(report["total_games"], 0)
        self.assertEqual(report["processed_games"], 0)
        self.assertEqual(report["rows"], [])


if __name__ == "__main__":
    unittest.main()
