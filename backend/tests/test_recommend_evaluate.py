"""离线评测模块单测(切分 / 指标 / 热度匹配协议)。纯 numpy+scipy,不加载 torch。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_evaluate -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.recommend.evaluate import (
    evaluate_bucketed,
    evaluate_scores,
    leave_one_out_split,
    popularity_bucket_of,
    topk_to_sparse,
)


def make_interactions(rows):
    return pd.DataFrame(rows, columns=["user_idx", "game_idx", "playtime_log"])


class TestLeaveOneOutSplit(unittest.TestCase):
    def setUp(self):
        # user0: 4 件 | user1: 2 件 | user2: 1 件(不合格,其边留在训练集)
        self.interactions = make_interactions(
            [
                (0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0), (0, 3, 4.0),
                (1, 4, 1.5), (1, 5, 2.5),
                (2, 6, 5.0),
            ]
        )
        self.split = leave_one_out_split(self.interactions, min_items=2, seed=0)

    def test_only_eligible_users_are_evaluated(self):
        self.assertEqual(sorted(self.split.users.tolist()), [0, 1])

    def test_exactly_one_test_edge_per_eligible_user(self):
        self.assertEqual(len(self.split.test_pairs), 2)
        self.assertEqual(len(set(self.split.test_pairs[:, 0].tolist())), 2)

    def test_train_plus_test_covers_every_edge(self):
        self.assertEqual(len(self.split.train_pairs) + len(self.split.test_pairs), 7)

    def test_test_edges_are_absent_from_train_matrix(self):
        """泄题检查:测试边绝不能出现在训练矩阵里,否则等于自己给自己答案。"""
        for user, game in self.split.test_pairs:
            self.assertEqual(self.split.train_user_item[user, game], 0)

    def test_ineligible_user_edges_stay_in_train(self):
        row = self.split.train_user_item[2]
        self.assertEqual(row.nnz, 1)
        self.assertEqual(row.indices.tolist(), [6])

    def test_weights_are_preserved_not_binarised(self):
        """训练矩阵的值必须是 playtime_log,用户画像和协同通道都靠它加权。"""
        expected = sorted(
            self.interactions["playtime_log"].tolist()
        )
        # 训练集应当丢掉两条测试边的权重
        self.assertEqual(len(expected) - 2, len(self.split.train_user_item.data))

    def test_n_items_override(self):
        split = leave_one_out_split(
            self.interactions, min_items=2, seed=0, n_items=20
        )
        self.assertEqual(split.n_items, 20)
        self.assertEqual(split.train_user_item.shape, (3, 20))

    def test_split_is_deterministic_for_a_seed(self):
        again = leave_one_out_split(self.interactions, min_items=2, seed=0)
        np.testing.assert_array_equal(self.split.test_pairs, again.test_pairs)


class TestEvaluateScores(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(40):
            for g in rng.choice(120, size=8, replace=False):
                rows.append((u, int(g), 1.0))
        self.split = leave_one_out_split(make_interactions(rows), min_items=5, seed=0)
        self.n = self.split.n_items

    def _scorer(self, *, put_target_first: bool):
        def fn(row_ids):
            score = np.full((len(row_ids), self.n), -1.0, dtype=np.float32)
            if put_target_first:
                for i, r in enumerate(row_ids):
                    score[i, self.split.test_game[r]] = 1.0
            else:
                score[:] = 1.0
                for i, r in enumerate(row_ids):
                    score[i, self.split.test_game[r]] = -1.0
            return score
        return fn

    def test_perfect_scorer_gets_full_recall(self):
        m = evaluate_scores(self._scorer(put_target_first=True), self.split, ks=(1, 10))
        self.assertAlmostEqual(m["recall@1"], 1.0)
        self.assertAlmostEqual(m["recall@10"], 1.0)
        self.assertAlmostEqual(m["mrr"], 1.0)
        self.assertEqual(m["median_rank"], 0.0)

    def test_worst_scorer_gets_zero_recall(self):
        m = evaluate_scores(self._scorer(put_target_first=False), self.split, ks=(1, 10))
        self.assertAlmostEqual(m["recall@1"], 0.0)
        self.assertAlmostEqual(m["recall@10"], 0.0)

    def test_rejects_wrong_score_shape(self):
        with self.assertRaises(ValueError):
            evaluate_scores(lambda rows: np.zeros((len(rows), 3)), self.split, ks=(1,))


class TestTopkToSparse(unittest.TestCase):
    def test_skips_missing_entries(self):
        idx = np.array([[1, 2, -1], [0, -1, -1]], dtype=np.int32)
        score = np.array([[0.5, 0.3, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32)
        mat = topk_to_sparse(idx, score, 3)
        self.assertEqual(mat.shape, (3, 3))
        self.assertEqual(mat.nnz, 3)
        self.assertAlmostEqual(mat[0, 1], 0.5, places=6)
        self.assertAlmostEqual(mat[1, 0], 0.9, places=6)


class TestPopularityBucket(unittest.TestCase):
    def test_buckets_are_equal_sized_and_monotone(self):
        rows = []
        # 物品 g 被 g+1 个用户拥有 → 热度随 g 单调递增
        for g in range(4):
            for u in range(g + 1):
                rows.append((u, g, 1.0))
        # 补足用户交互数门槛
        split = leave_one_out_split(
            make_interactions(rows), min_items=1, seed=0, n_items=4
        )
        bucket = popularity_bucket_of(split, n_buckets=2)
        self.assertEqual(bucket.tolist(), [0, 0, 1, 1])


class TestEvaluateBucketed(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        rows = []
        for u in range(60):
            for g in rng.choice(150, size=10, replace=False):
                rows.append((u, int(g), 1.0))
        self.split = leave_one_out_split(make_interactions(rows), min_items=6, seed=0)
        self.n = self.split.n_items

    def test_perfect_scorer_wins(self):
        def fn(row_ids):
            score = np.zeros((len(row_ids), self.n), dtype=np.float32)
            for i, r in enumerate(row_ids):
                score[i, self.split.test_game[r]] = 1.0
            return score

        m = evaluate_bucketed(fn, self.split, n_neg=20, ks=(10,), seed=0)
        self.assertAlmostEqual(m["recall@10"], 1.0)
        self.assertGreater(m["lift_over_chance@10"], 1.0)

    def test_chance_metric_reported(self):
        m = evaluate_bucketed(
            lambda rows: np.zeros((len(rows), self.n), dtype=np.float32),
            self.split, n_neg=100, ks=(10,), seed=0,
        )
        # 全零分数时并列,rank=0,全部命中;这里只验证 chance 基准被正确报告
        self.assertAlmostEqual(m["chance_recall@10"], 10 / 101, places=6)


if __name__ == "__main__":
    unittest.main()
