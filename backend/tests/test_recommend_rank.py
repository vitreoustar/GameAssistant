"""A4 特征构造器单测(小号假索引,不读磁盘、不加载 LightGBM 模型)。

重点覆盖:
  * 特征维度与顺序稳定(训练脚本按 FEATURE_NAMES 取名次,错位会静默毁掉模型)
  * 缺失通道的哨兵值(名次 -1 / 分数 0)与"命中几路"计数
  * 类型重合 / Jaccard / 时长亲和 / 发行商亲和的精确取值
  * **泄漏守卫**:热度特征必须来自注入的训练集计数,而不是 catalog 的全量统计

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_rank -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.recommend.rank import ABSENT_RANK, FEATURE_NAMES, FeatureBuilder, Ranker
from app.recommend.recall import RecallConfig, RecallEngine

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
GENRES = ["A, B", "A", "B", "C", "D", "D", "D", "D"]
PUBLISHERS = ["Pub0", "Pub0", "Pub1", "Pub2", "Pub2", "Pub2", "Pub2", "Pub2"]


class FakeIndex:
    def __init__(self) -> None:
        n = len(EMB)
        norm = np.linalg.norm(EMB, axis=1, keepdims=True)
        self.content_emb = (EMB / norm).astype(np.float32)
        self.content_topk_idx = np.stack([(np.arange(n) + 1 + i) % n for i in range(3)], 1).astype(np.int32)
        self.content_topk_score = np.full((n, 3), 0.5, dtype=np.float32)
        self.collab_topk_idx = np.stack([(np.arange(n) + 4 + i) % n for i in range(3)], 1).astype(np.int32)
        self.collab_topk_score = np.full((n, 3), 0.8, dtype=np.float32)
        self.n_items = n
        self.catalog = pd.DataFrame(
            {
                "appid": np.arange(1000, 1000 + n, dtype=np.int64),
                "genres": GENRES,
                "publishers": PUBLISHERS,
                "bayes_rating": np.full(n, 0.8),
                "positive_ratio": np.full(n, 0.8),
                "estimated_owners_lower": np.arange(1, n + 1) * 1000,
                "metacritic_score": np.full(n, 80),
                "price_cents": np.full(n, 1000),
                "release_year": np.full(n, 2015),
                "dlc_count": np.zeros(n),
                # 全量统计,故意设成与训练计数不同的值,用来验证泄漏守卫
                "n_interactions": np.full(n, 9999),
            }
        )
        self._row = {int(a): i for i, a in enumerate(self.catalog["appid"].to_numpy())}

    def rows_of(self, appids):
        return np.array([self._row.get(int(a), -1) for a in appids], dtype=np.int64)


def build_builder(train_popularity=None):
    index = FakeIndex()
    n = index.n_items
    engine = RecallEngine(
        index,
        quality_scores=np.zeros(n, dtype=np.float32),
        config=RecallConfig(total=10),
    )
    if train_popularity is None:
        train_popularity = np.arange(n, dtype=np.float64)
    return index, engine, FeatureBuilder(index, engine, train_popularity=train_popularity)


class TestFeatureMatrix(unittest.TestCase):
    def setUp(self):
        self.index, self.engine, self.builder = build_builder()
        self.library = np.array([0])
        self.weights = np.array([1.0], dtype=np.float32)
        self.candidates = np.array([1, 2, 3], dtype=np.int64)

    def test_shape_matches_feature_names(self):
        context = self.builder.build(self.library, self.weights, self.candidates)
        self.assertEqual(context.features.shape, (3, len(FEATURE_NAMES)))
        self.assertEqual(context.candidate_rows.tolist(), [1, 2, 3])

    def test_merge_rank_is_positional(self):
        context = self.builder.build(self.library, self.weights, self.candidates)
        col = FEATURE_NAMES.index("merge_rank")
        np.testing.assert_array_equal(context.features[:, col], [0.0, 1.0, 2.0])

    def test_absent_channel_uses_sentinel(self):
        """item3 的 genres 是 C,不在 item0 的 TopK 邻居里时应记为"未召回"。"""
        context = self.builder.build(self.library, self.weights, self.candidates)
        ranks = context.channel_ranks["content_itemknn"]
        # content_topk 里 item0 -> [1,2,3],所以三个候选都被召回
        self.assertTrue(all(c in ranks for c in (1, 2, 3)))
        # publisher 通道只覆盖 Pub0 的两个物品(item0、item1)
        self.assertIn(1, context.channel_ranks["publisher"])
        self.assertNotIn(3, context.channel_ranks["publisher"])
        col = FEATURE_NAMES.index("publisher_rank")
        self.assertEqual(context.features[2, col], ABSENT_RANK)
        self.assertEqual(
            context.features[2, FEATURE_NAMES.index("publisher_score")], 0.0
        )

    def test_n_channels_counts_hits(self):
        context = self.builder.build(self.library, self.weights, self.candidates)
        col = FEATURE_NAMES.index("n_channels")
        self.assertTrue((context.features[:, col] >= 1).all())
        self.assertTrue((context.features[:, col] <= len(FEATURE_NAMES) ).all())


class TestGenreAndPublisherFeatures(unittest.TestCase):
    def setUp(self):
        self.index, self.engine, self.builder = build_builder()
        self.library = np.array([0])          # 用户只玩 item0:genres=A,B / Pub0
        self.weights = np.array([1.0], dtype=np.float32)

    def test_overlap_jaccard_and_affinity(self):
        # item1 = "A"(与用户共享 A);item3 = "C"(无交集)
        context = self.builder.build(self.library, self.weights, np.array([1, 3]))
        overlap = context.features[:, FEATURE_NAMES.index("genre_overlap")]
        jaccard = context.features[:, FEATURE_NAMES.index("genre_jaccard")]
        affinity = context.features[:, FEATURE_NAMES.index("user_genre_affinity")]
        self.assertAlmostEqual(float(overlap[0]), 1.0)      # {A} ∩ {A,B} = 1
        self.assertAlmostEqual(float(jaccard[0]), 1 / 2)    # 1 / (1 + 2 - 1)
        self.assertAlmostEqual(float(affinity[0]), 1.0)     # 用户的全部时长都在 A/B 上
        self.assertAlmostEqual(float(overlap[1]), 0.0)
        self.assertAlmostEqual(float(jaccard[1]), 0.0)
        self.assertAlmostEqual(float(affinity[1]), 0.0)

    def test_publisher_affinity(self):
        context = self.builder.build(self.library, self.weights, np.array([1, 2, 3]))
        affinity = context.features[:, FEATURE_NAMES.index("user_publisher_affinity")]
        self.assertAlmostEqual(float(affinity[0]), 1.0, places=5)  # item1 同属 Pub0
        self.assertAlmostEqual(float(affinity[1]), 0.0, places=5)  # item2 是 Pub1
        self.assertAlmostEqual(float(affinity[2]), 0.0, places=5)

    def test_content_cosine_is_one_for_own_library_item(self):
        context = self.builder.build(self.library, self.weights, np.array([0, 3]))
        cosine = context.features[:, FEATURE_NAMES.index("content_cosine")]
        self.assertAlmostEqual(float(cosine[0]), 1.0, places=5)


class TestLeakageGuard(unittest.TestCase):
    def test_popularity_feature_comes_from_injected_counts(self):
        """热度特征必须等于注入的训练集计数,不能用 catalog.n_interactions(全量统计)。"""
        counts = np.arange(8, dtype=np.float64)
        index, engine, builder = build_builder(train_popularity=counts)
        context = builder.build(
            np.array([0]), np.array([1.0], dtype=np.float32), np.arange(8, dtype=np.int64)
        )
        col = FEATURE_NAMES.index("train_popularity")
        expected = np.log1p(counts) / np.log1p(counts.max())
        np.testing.assert_allclose(context.features[:, col], expected, rtol=1e-5)
        # catalog 里那列是 9999,归一化后是 1.0,绝不能出现在这里
        self.assertFalse(np.allclose(context.features[1:, col], 1.0))


class TestRanker(unittest.TestCase):
    def test_rank_respects_top_k_and_sorts_by_score(self):
        class FakeModel:
            def predict(self, features):
                return features[:, FEATURE_NAMES.index("merge_rank")] * -1.0

        index, engine, builder = build_builder()
        ranker = Ranker(FakeModel(), builder, engine)
        out = ranker.rank(np.array([0]), np.array([1.0], dtype=np.float32), top_k=3)
        self.assertLessEqual(len(out), 3)
        scores = [c.score for c in out]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_library_falls_back_to_quality_channel(self):
        """空库不是返回空,而是走冷启动兜底(quality 通道)。

        这是有意设计:新用户/库为空时仍要能给出推荐,否则前端只会看到一片空白。
        """
        class FakeModel:
            def predict(self, features):
                return np.zeros(len(features))

        index, engine, builder = build_builder()
        ranker = Ranker(FakeModel(), builder, engine)
        out = ranker.rank(np.array([], dtype=np.int64), np.array([]), top_k=3)
        self.assertTrue(out, "空库应当有冷启动兜底候选")
        self.assertTrue(all("quality" in c.channels for c in out))


if __name__ == "__main__":
    unittest.main()
