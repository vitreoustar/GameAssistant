"""A5 在线服务层单测。

用一个最小的假索引 + 假模型跑通 `Recommender` 的全部路径(含冷启动),
不读磁盘、不加载 LightGBM。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_service -v
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from app.recommend.rank import FeatureBuilder
from app.recommend.recall import RecallConfig, RecallEngine
from app.recommend.service import Recommender

EMB = np.array(
    [
        [1.00, 0.00, 0.00, 0.00],
        [0.90, 0.10, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
        [0.70, 0.70, 0.00, 0.00],
        [0.20, 0.90, 0.00, 0.00],
    ],
    dtype=np.float32,
)


class FakeModel:
    """把第 0 维(merge_rank)当分数,便于断言排序来自模型而不是别的。"""

    def predict(self, features):
        from app.recommend.rank import FEATURE_NAMES

        return -features[:, FEATURE_NAMES.index("merge_rank")]


class FakeIndex:
    def __init__(self) -> None:
        from app.recommend.index import RecommendIndex
        from pathlib import Path

        n = len(EMB)
        norm = np.linalg.norm(EMB, axis=1, keepdims=True)
        catalog = pd.DataFrame(
            {
                "appid": np.arange(1000, 1000 + n, dtype=np.int64),
                "name": [f"Game{i}" for i in range(n)],
                "genres": ["A, B", "A", "B", "C", "D", "D", "D", "D"],
                "categories": ["Single-player"] * n,
                "publishers": ["Pub0", "Pub0", "Pub1"] + ["Pub2"] * 5,
                "bayes_rating": np.linspace(0.5, 0.95, n),
                "positive_ratio": np.full(n, 0.8),
                "estimated_owners_lower": np.arange(1, n + 1) * 1000,
                "metacritic_score": np.full(n, 80),
                "price_cents": np.full(n, 1000),
                "release_year": np.full(n, 2015),
                "dlc_count": np.zeros(n),
                "n_interactions": np.arange(n, dtype=np.int64),
                "header_image": [f"http://img/{i}.jpg" for i in range(n)],
                "rating": ["特别好评"] * n,
                "positive": np.full(n, 1000),
                "negative": np.full(n, 100),
                "has_content": np.ones(n, dtype=bool),
            }
        )
        self.catalog = catalog
        self.content_emb = (EMB / norm).astype(np.float32)
        self.content_topk_idx = np.stack(
            [(np.arange(n) + 1 + i) % n for i in range(3)], 1
        ).astype(np.int32)
        self.content_topk_score = np.full((n, 3), 0.5, dtype=np.float32)
        self.collab_topk_idx = np.stack(
            [(np.arange(n) + 4 + i) % n for i in range(3)], 1
        ).astype(np.int32)
        self.collab_topk_score = np.full((n, 3), 0.8, dtype=np.float32)
        self.n_items = n
        self.manifest = {"topk": 3}
        self.path = Path(".")
        self._row = {int(a): i for i, a in enumerate(catalog["appid"].to_numpy())}

    def rows_of(self, appids):
        return np.array([self._row.get(int(a), -1) for a in appids], dtype=np.int64)


def make_recommender():
    index = FakeIndex()
    engine = RecallEngine(
        index,
        quality_scores=index.catalog["n_interactions"].to_numpy(dtype=np.float32),
        config=RecallConfig(total=6),
    )
    builder = FeatureBuilder(
        index, engine, train_popularity=index.catalog["n_interactions"].to_numpy(dtype=float)
    )
    return Recommender(index, _RankerLike(FakeModel(), builder, engine), builder, engine)


class _RankerLike:
    def __init__(self, model, builder, engine):
        self.model = model
        self.builder = builder
        self.engine = engine


class TestRecommend(unittest.TestCase):
    def setUp(self):
        self.rec = make_recommender()

    def test_returns_top_k_and_no_owned_items(self):
        library = [{"appid": 1000, "playtime_forever": 100.0}]
        result = self.rec.recommend(library, top_k=3)
        self.assertLessEqual(len(result.recommendations), 3)
        self.assertNotIn(1000, [r.appid for r in result.recommendations])
        self.assertFalse(result.cold_start)

    def test_profile_comes_from_catalog(self):
        library = [
            {"appid": 1000, "playtime_forever": 100.0},   # genres A, B
            {"appid": 1001, "playtime_forever": 50.0},    # genres A
        ]
        result = self.rec.recommend(library, top_k=2)
        self.assertIn("A", result.profile["top_genres"])
        self.assertIn("B", result.profile["top_genres"])
        self.assertIn("偏爱类型", result.profile["summary"])

    def test_match_genres_overlap_reported(self):
        library = [{"appid": 1002, "playtime_forever": 100.0}]  # genres B
        result = self.rec.recommend(library, top_k=5)
        # Game0 与 Game2 都含 B,应出现在某些候选的 match_genres 里
        all_matches = {g for r in result.recommendations for g in r.match_genres}
        self.assertIn("B", all_matches)

    def test_library_games_outside_catalog_are_counted(self):
        library = [
            {"appid": 1000, "playtime_forever": 100.0},
            {"appid": 999999, "playtime_forever": 5.0},   # 不在目录里
        ]
        result = self.rec.recommend(library, top_k=2)
        self.assertEqual(result.unmatched_in_library, 1)
        self.assertEqual(result.library_matched, 1)
        self.assertFalse(result.cold_start)

    def test_as_dict_keeps_frontend_contract(self):
        library = [{"appid": 1000, "playtime_forever": 100.0}]
        payload = self.rec.recommend(library, top_k=1).recommendations[0].as_dict()
        for key in (
            "appid", "name", "header_image", "genres", "categories", "price_cents",
            "rating", "positive", "negative", "metacritic_score", "distance", "reason",
        ):
            self.assertIn(key, payload)
        # distance 与 score 是互补的,保持旧字段语义一致
        self.assertAlmostEqual(payload["distance"], 1.0 - payload["score"], places=6)


class TestColdStart(unittest.TestCase):
    def setUp(self):
        self.rec = make_recommender()

    def test_empty_library_uses_quality_channel(self):
        result = self.rec.recommend([], top_k=3)
        self.assertTrue(result.cold_start)
        self.assertTrue(result.recommendations, "空库也必须有兜底候选")
        self.assertTrue(
            all("quality" in r.channels for r in result.recommendations)
        )

    def test_library_entirely_outside_catalog_is_cold_start(self):
        result = self.rec.recommend(
            [{"appid": 42, "playtime_forever": 10.0}], top_k=2
        )
        self.assertTrue(result.cold_start)
        self.assertEqual(result.unmatched_in_library, 1)

    def test_cold_start_profile_is_empty_but_valid(self):
        result = self.rec.recommend([], top_k=1)
        self.assertEqual(result.profile["top_genres"], [])
        self.assertIn("暂无", result.profile["summary"])


class TestExplain(unittest.TestCase):
    def setUp(self):
        self.rec = make_recommender()

    def test_because_of_points_at_a_library_game(self):
        library = [
            {"appid": 1000, "playtime_forever": 100.0},
            {"appid": 1004, "playtime_forever": 50.0},
        ]
        result = self.rec.recommend(library, top_k=6)
        explained = [r for r in result.recommendations if r.because_of]
        self.assertTrue(explained, "至少应有候选能给出『因为你在玩 X』")
        library_names = {"Game0", "Game4"}
        for r in explained:
            self.assertIn(r.because_of, library_names)

    def test_library_covering_whole_catalog_returns_empty_gracefully(self):
        """库把目录占满时没有可推的了 —— 应当返回空,而不是崩或推荐已拥有的。"""
        library = [{"appid": 1000 + i, "playtime_forever": 10.0} for i in range(8)]
        result = self.rec.recommend(library, top_k=5)
        self.assertEqual(result.recommendations, [])
        self.assertFalse(result.cold_start)   # 库是能被识别的,只是没剩什么可推

    def test_explain_only_cites_library_games(self):
        """回归:早期实现对 2,000 款的库要 230ms(双层 Python 循环),现已向量化。

        这里断言**结果正确性**而不测时间(避免 CI 抖动):解释来源必须落在库内。
        """
        library = [{"appid": 1000 + i, "playtime_forever": float(100 - i)} for i in range(4)]
        owned = [g["appid"] for g in library]
        result = self.rec.recommend(library, top_k=4)
        self.assertTrue(result.recommendations)
        library_names = {f"Game{i}" for i in range(4)}
        for r in result.recommendations:
            self.assertNotIn(r.appid, owned)
            if r.because_of:
                self.assertIn(r.because_of, library_names)


if __name__ == "__main__":
    unittest.main()
