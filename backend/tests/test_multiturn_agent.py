"""多轮推荐的**整图**行为测试(A7)。

验证的是 `AgentState` 这一层的接线,不是推荐算法本身:

    第 1 轮推荐 5 款 -> 状态里记下这 5 个 appid
    第 2 轮"再推荐点别的" -> 必须换一批(不能原样返回)
    第 3 轮"我更喜欢策略类的游戏" -> 偏好进入状态,排序被约束

整条路径**不调 LLM、不联网**:没有 `DEEPSEEK_API_KEY` 时路由走关键词 +
`parse_message` 抽取偏好,推荐器用 `tests/test_recommend_preferences.py` 里的假索引替换掉。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_multiturn_agent -v
"""
from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage

import app.agent.graph as graph_mod
from tests.test_recommend_preferences import LIBRARY as SINGLE, make_recommender

LIBRARY = [
    {"appid": 1000, "name": "Game0", "playtime_forever": 200.0},
    {"appid": 1003, "name": "Game3", "playtime_forever": 80.0},
]


def run(message: str, state: dict | None = None) -> dict:
    return _invoke(message, state)


async def _ainvoke(message: str, state: dict | None) -> dict:
    payload = {
        "messages": [HumanMessage(content=message)],
        "library": LIBRARY,
        "recommend_state": state,
        "history": [],
    }
    return await graph_mod.agent.ainvoke(payload)


def _invoke(message: str, state: dict | None) -> dict:
    import asyncio

    return asyncio.run(_ainvoke(message, state))


class TestMultiTurnGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._rec = make_recommender()
        cls._orig_local = graph_mod._local_recommender
        graph_mod._local_recommender = staticmethod(lambda: cls._rec)  # type: ignore[assignment]

    @classmethod
    def tearDownClass(cls):
        graph_mod._local_recommender = cls._orig_local  # type: ignore[assignment]

    def setUp(self):
        self.assertTrue(SINGLE)  # 复用同一份假索引,确保两处口径一致

    def test_first_turn_records_shown_appids(self):
        result = run("推荐几款游戏")
        self.assertEqual(result.get("intent"), "recommend")
        candidates = result.get("candidates") or []
        self.assertEqual(len(candidates), 5)
        state = result.get("recommend_state") or {}
        self.assertEqual(len(state.get("shown") or []), 5)
        self.assertEqual(state.get("turn"), 1)

    def test_second_turn_does_not_repeat(self):
        """回归:用户说「再推荐点别的」时原样返回上一批。"""
        first = run("推荐几款游戏")
        first_ids = [c["appid"] for c in first["candidates"]]

        second = run("再推荐点别的", first.get("recommend_state"))
        second_ids = [c["appid"] for c in second["candidates"]]

        self.assertEqual(len(second_ids), 5)
        self.assertEqual(
            set(first_ids) & set(second_ids), set(),
            "第二轮出现了第一轮已经推荐过的游戏",
        )
        self.assertEqual(len(second["recommend_state"]["shown"]), 10)

    def test_stated_preference_reaches_the_ranker(self):
        # 用干净状态:上一轮已经展示过的会被去重掉,不能用来验证"排前面"
        result = run("我更喜欢策略类的游戏")

        state = result["recommend_state"]
        self.assertIn("Strategy", state["genres"])
        candidates = result["candidates"]
        self.assertTrue(candidates)
        self.assertEqual(result["intent"], "recommend")
        # 假索引里 4 款带 Strategy,应当全部排在前面并带上 pref_match 证据
        self.assertTrue(
            all("Strategy" in c["genres"] for c in candidates[:4]),
            f"策略类应当排在前面:{[(c['name'], c['genres']) for c in candidates]}",
        )
        self.assertTrue(all(c["pref_match"] for c in candidates[:4]))
        self.assertFalse(any(c["pref_match"] for c in candidates[4:]))

    def test_avoid_reaches_the_ranker(self):
        result = run("不要策略")
        state = result["recommend_state"]
        self.assertEqual(state["genres"], [])
        self.assertIn("Strategy", state["avoid"])
        for c in result["candidates"]:
            self.assertNotIn("Strategy", c["genres"])

    def test_state_survives_a_turn_with_no_candidates(self):
        """即使这一轮没东西可推,`shown` 也不能丢 —— 否则去重记录会整体回退。"""
        shown = [1000 + i for i in range(1, 12)]
        state = {"genres": [], "avoid": [], "unmapped": [], "shown": shown, "turn": 5}
        result = run("再推荐点别的", state)
        self.assertEqual(result.get("candidates") or [], [])
        self.assertEqual(result["recommend_state"]["shown"], shown)

    def test_keyword_route_covers_follow_ups(self):
        """没有 LLM 时「换一批」这类追问不能被判成闲聊。"""
        for message in ("换一批", "再推荐点别的", "还有别的吗", "看看其他类型"):
            self.assertEqual(graph_mod._keyword_route(message), "recommend", message)


if __name__ == "__main__":
    unittest.main()
