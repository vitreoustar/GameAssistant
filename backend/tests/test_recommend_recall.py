"""多路召回层单测(用小号假索引,不读磁盘、不加载模型)。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_recall -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.recommend.recall import CHANNEL_ORDER, RecallConfig, RecallEngine


class FakeIndex:
    """只提供 RecallEngine 需要的最小接口。"""

    def __init__(self, n_items: int = 10, dim: int = 4, k: int = 3) -> None:
        rng = np.random.default_rng(0)
        emb = rng.random((n_items, dim), dtype=np.float32)
        self.content_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.content_topk_idx = np.stack(
            [(np.arange(n_items) + 1 + i) % n_items for i in range(k)], axis=1
        ).astype(np.int32)
        self.content_topk_score = np.full((n_items, k), 0.5, dtype=np.float32)
        self.collab_topk_idx = np.stack(
            [(np.arange(n_items) + 5 + i) % n_items for i in range(k)], axis=1
        ).astype(np.int32)
        self.collab_topk_score = np.full((n_items, k), 0.9, dtype=np.float32)
        self.n_items = n_items
        self.catalog = pd.DataFrame(
            {
                "appid": np.arange(1000, 1000 + n_items, dtype=np.int64),
                "publishers": [f"Pub{i % 3}" for i in range(n_items)],
                "bayes_rating": np.linspace(0.5, 0.95, n_items),
                "estimated_owners_lower": np.arange(n_items) * 1000,
            }
        )
        self._row = {int(a): i for i, a in enumerate(self.catalog["appid"].to_numpy())}

    def rows_of(self, appids):
        return np.array([self._row.get(int(a), -1) for a in appids], dtype=np.int64)


def zero_quotas(**kwargs) -> dict:
    quotas = {c: 0 for c in CHANNEL_ORDER}
    quotas.update(kwargs)
    return quotas


class TestQuotaMergeIsolation(unittest.TestCase):
    """回归测试:`only:X` 的消融必须真的只用 X。

    曾经的 bug:补齐剩余名额时遍历了**所有**通道,导致 `only:content_user`
    悄悄混进了协同/质量/发行商的候选,单通道指标虚高约 20 倍。
    """

    def setUp(self):
        self.engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(content_user=2), total=5),
        )
        self.channels = {
            "content_user": (np.array([1, 2, 3]), np.array([0.9, 0.8, 0.7])),
            # 其余通道给出与 content_user 完全不相交的候选
            "content_itemknn": (np.array([7, 8]), np.array([0.9, 0.8])),
            "collab_itemknn": (np.array([7, 8]), np.array([0.95, 0.85])),
            "quality": (np.array([9]), np.array([0.99])),
            "publisher": (np.array([6]), np.array([0.5])),
        }

    def test_only_enabled_channel_leaks_nothing(self):
        """配额是**下限**不是上限:补齐阶段可以在启用通道内继续取,但绝不能跨通道。"""
        result = self.engine._merge_by_quota(self.channels, np.array([], dtype=int), self.engine.config)
        produced = {c.cat_idx for c in result}
        self.assertTrue(produced, "不该返回空候选")
        # content_user 自己的候选是 {1,2,3};其余通道的 {6,7,8,9} 一个都不许出现
        self.assertTrue(
            produced <= {1, 2, 3},
            f"出现了 content_user 之外的候选: {sorted(produced - {1, 2, 3})}",
        )
        self.assertFalse(
            produced & {6, 7, 8, 9},
            f"配额为 0 的通道泄漏进了结果: {sorted(produced & {6, 7, 8, 9})}",
        )

    def test_quota_is_a_floor_and_total_is_the_cap(self):
        result = self.engine._merge_by_quota(self.channels, np.array([], dtype=int), self.engine.config)
        produced = {c.cat_idx for c in result}
        self.assertIn(1, produced, "启用通道的头部候选必须被取到")
        self.assertIn(2, produced)
        self.assertLessEqual(len(result), self.engine.config.total, "不得超过 total")

    def test_fill_phase_uses_enabled_channels(self):
        """配额没占满时,补齐也只能从启用的通道里补,且不超过 total。"""
        engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(content_itemknn=1), total=2),
        )
        result = engine._merge_by_quota(self.channels, np.array([], dtype=int), engine.config)
        self.assertEqual({c.cat_idx for c in result}, {7, 8})
        self.assertLessEqual(len(result), 2)


class TestOwnedExclusion(unittest.TestCase):
    def test_owned_items_never_returned(self):
        engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(content_user=5), total=5),
        )
        owned = np.array([1, 2], dtype=int)
        result = engine.recall_rows(np.array([0]), np.array([1.0]), owned_rows=owned)
        self.assertFalse({c.cat_idx for c in result} & set(owned.tolist()))


class TestRecallConfigValidation(unittest.TestCase):
    def test_weights_length_mismatch_rejected(self):
        engine = RecallEngine(FakeIndex())
        with self.assertRaises(ValueError):
            engine.recall_rows(np.array([0, 1]), np.array([1.0]))

    def test_appid_entry_point_matches_rows(self):
        engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(collab_itemknn=3), total=3),
        )
        by_appid = engine.recall([1000, 1001])
        by_row = engine.recall_rows(np.array([0, 1]), None)
        self.assertEqual(
            [c.cat_idx for c in by_appid], [c.cat_idx for c in by_row]
        )

    def test_minute_weights_are_log_compressed(self):
        """传入的权重像分钟数(>30)时应自动取 log1p,避免极端值主导。"""
        engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(collab_itemknn=3), total=3),
        )
        minutes = engine.recall_rows(np.array([0, 1]), np.array([100.0, 500000.0]))
        logged = engine.recall_rows(
            np.array([0, 1]), np.log1p(np.array([100.0, 500000.0]))
        )
        self.assertEqual(
            [c.cat_idx for c in minutes], [c.cat_idx for c in logged]
        )


class TestScoreMerge(unittest.TestCase):
    def test_higher_quota_channel_dominates(self):
        engine = RecallEngine(
            FakeIndex(),
            quality_scores=np.ones(10, dtype=np.float32),
            config=RecallConfig(quotas=zero_quotas(content_user=9, collab_itemknn=1),
                                merge="score", total=4),
        )
        result = engine.recall_rows(np.array([0]), np.array([1.0]))
        self.assertTrue(all(c.channels for c in result))
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
