"""A2 数组助手单测(topk / 用户向量)。只依赖 numpy,不加载 torch。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_index -v
"""
from __future__ import annotations

import unittest

import numpy as np

from app.recommend.index import topk_from_dense, user_vector


class TestTopkFromDense(unittest.TestCase):
    def test_orders_by_score_descending(self):
        scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.6]], dtype=np.float32)
        idx, val = topk_from_dense(scores, 2)
        np.testing.assert_array_equal(idx, [[1, 2], [0, 2]])
        np.testing.assert_allclose(val, [[0.9, 0.5], [0.7, 0.6]], atol=1e-6)

    def test_dtypes_are_compact(self):
        idx, val = topk_from_dense(np.eye(4, dtype=np.float32), 2)
        self.assertEqual(idx.dtype, np.int32)
        self.assertEqual(val.dtype, np.float32)

    def test_exclude_self_drops_the_diagonal(self):
        """自身是最相似的时候,必须把自己排掉,否则会推荐"你已经拥有的那款"。"""
        scores = np.array([[1.0, 0.9, 0.5], [0.7, 1.0, 0.6]], dtype=np.float32)
        with_self, _ = topk_from_dense(scores, 2)
        without_self, _ = topk_from_dense(scores, 2, exclude_self_offset=0)
        np.testing.assert_array_equal(with_self[0], [0, 1])
        np.testing.assert_array_equal(without_self[0], [1, 2])

    def test_exclude_self_offset_handles_chunk_offset(self):
        """分块计算时,第 r 行的自身列是 r + offset,不是 r。"""
        scores = np.zeros((1, 8), dtype=np.float32)
        scores[0, 5] = 0.5  # 分块起点为 5 时的"自身"列
        scores[0, 2] = 1.0
        idx, _ = topk_from_dense(scores, 2, exclude_self_offset=5)
        self.assertNotIn(5, idx[0].tolist())
        self.assertEqual(int(idx[0][0]), 2)

    def test_k_is_clamped_to_available_columns(self):
        idx, _ = topk_from_dense(np.eye(3, dtype=np.float32), 10)
        self.assertEqual(idx.shape, (3, 3))

    def test_nan_scores_sort_last(self):
        scores = np.array([[np.nan, 0.2, 0.9]], dtype=np.float32)
        idx, val = topk_from_dense(scores, 2)
        self.assertEqual(int(idx[0][0]), 2)
        self.assertFalse(np.isnan(val).any())

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            topk_from_dense(np.zeros(5, dtype=np.float32), 2)


class TestUserVector(unittest.TestCase):
    def setUp(self):
        # 未归一化的物品向量;user_vector 自己会归一化结果
        self.emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    def test_weighted_sum_is_normalised(self):
        vec = user_vector(self.emb, np.array([0, 1]), np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(vec, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)

    def test_weights_change_direction(self):
        vec = user_vector(self.emb, np.array([0, 1]), np.array([3.0, 1.0], dtype=np.float32))
        self.assertGreater(float(vec[0]), float(vec[1]))

    def test_empty_rows_returns_zero_vector(self):
        vec = user_vector(self.emb, np.array([], dtype=np.int64), np.array([], dtype=np.float32))
        np.testing.assert_array_equal(vec, np.zeros(2, dtype=np.float32))

    def test_cancelling_weights_returns_zero_not_nan(self):
        """正负抵消时模长为 0,必须返回零向量而不是 NaN(否则下游归一化会炸)。"""
        vec = user_vector(self.emb, np.array([0, 0]), np.array([1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(vec, np.zeros(2, dtype=np.float32))
        self.assertFalse(np.isnan(vec).any())

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            user_vector(self.emb, np.array([0, 1]), np.array([1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
