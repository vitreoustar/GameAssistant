"""推荐算法纯函数单测(离线,不加载 torch / pandas)。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_features -v
"""
from __future__ import annotations

import unittest

from app.recommend.features import (
    DESC_LIMIT,
    MIN_RATING_VOTES,
    bayesian_rating,
    build_item_text,
    owners_lower_bound,
    playtime_bucket,
    positive_ratio,
    rating_text,
    release_year,
)


class TestRatingText(unittest.TestCase):
    def test_too_few_votes(self):
        self.assertEqual(rating_text(10, 1), "评价较少")
        self.assertEqual(rating_text(0, 0), "评价较少")
        self.assertEqual(rating_text(MIN_RATING_VOTES - 1, 0), "评价较少")

    def test_thresholds(self):
        # (好评数, 差评数, 期望标签) —— 总数固定 100,百分比即好评数
        cases = [
            (100, 0, "好评如潮"),
            (95, 5, "好评如潮"),
            (94, 6, "特别好评"),
            (80, 20, "特别好评"),
            (79, 21, "多半好评"),
            (70, 30, "多半好评"),
            (69, 31, "褒贬不一"),
            (40, 60, "褒贬不一"),
            (39, 61, "多半差评"),
            (20, 80, "多半差评"),
            (19, 81, "差评如潮"),
            (0, 100, "差评如潮"),
        ]
        for positive, negative, expected in cases:
            with self.subTest(positive=positive, negative=negative):
                self.assertEqual(rating_text(positive, negative), expected)


class TestPositiveRatio(unittest.TestCase):
    def test_none_when_no_votes(self):
        """没有票数时返回 None,而不是伪造成 0 —— 下游才能用均值填充。"""
        self.assertIsNone(positive_ratio(0, 0))
        self.assertIsNone(positive_ratio(0, 0))

    def test_ratio(self):
        self.assertAlmostEqual(positive_ratio(75, 25), 0.75)
        self.assertAlmostEqual(positive_ratio(1, 0), 1.0)


class TestOwnersLowerBound(unittest.TestCase):
    def test_parses_range_lower_bound(self):
        self.assertEqual(owners_lower_bound("500000 - 1000000"), 500_000)
        self.assertEqual(owners_lower_bound("200000 - 500000"), 200_000)
        self.assertEqual(owners_lower_bound("1,000,000 - 2,000,000"), 1_000_000)

    def test_unparseable_returns_zero(self):
        self.assertEqual(owners_lower_bound(""), 0)
        self.assertEqual(owners_lower_bound(None), 0)
        self.assertEqual(owners_lower_bound("unknown"), 0)

    def test_non_string_returns_zero(self):
        """pandas left-join 后的缺失值是 float NaN,而 NaN 是真值 —— 必须显式挡住。"""
        self.assertEqual(owners_lower_bound(float("nan")), 0)
        self.assertEqual(owners_lower_bound(0.0), 0)
        self.assertEqual(owners_lower_bound(500_000), 0)


class TestReleaseYear(unittest.TestCase):
    def test_both_source_formats(self):
        """corpus 用 'Nov 16, 2018',数据集用 '1 Nov, 2000'。"""
        self.assertEqual(release_year("Nov 16, 2018"), 2018)
        self.assertEqual(release_year("1 Nov, 2000"), 2000)

    def test_unparseable_returns_zero(self):
        self.assertEqual(release_year(""), 0)
        self.assertEqual(release_year(None), 0)
        self.assertEqual(release_year("Coming soon"), 0)

    def test_non_string_returns_zero(self):
        self.assertEqual(release_year(float("nan")), 0)


class TestPlaytimeBucket(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(playtime_bucket(0), 0)
        self.assertEqual(playtime_bucket(59), 0)
        self.assertEqual(playtime_bucket(60), 1)
        self.assertEqual(playtime_bucket(599), 1)
        self.assertEqual(playtime_bucket(600), 2)
        self.assertEqual(playtime_bucket(2999), 2)
        self.assertEqual(playtime_bucket(3000), 3)
        self.assertEqual(playtime_bucket(597_266), 3)

    def test_missing_treated_as_zero(self):
        self.assertEqual(playtime_bucket(None), 0)


class TestBayesianRating(unittest.TestCase):
    """直接用好评率会让"1 好评 0 差评 = 100%"的冷门物品霸榜,所以要做平滑。"""

    def test_no_votes_falls_back_to_prior(self):
        self.assertAlmostEqual(bayesian_rating(0, 0, 0.85, 50.0), 0.85)

    def test_small_sample_pulled_toward_prior(self):
        # 1 好评 0 差评:原始好评率 1.0,但只有 1 票,应被拉向 0.85
        value = bayesian_rating(1, 0, 0.85, 50.0)
        self.assertLess(value, 0.90)
        self.assertGreater(value, 0.85)

    def test_large_sample_approaches_raw_ratio(self):
        # 9000/1000 的好评率正好是 0.9;C=50 时先验仍有 50/10050 的拉力,
        # 结果约 0.89975 —— 所以用 places=3 而不是更严的阈值
        value = bayesian_rating(9000, 1000, 0.85, 50.0)
        self.assertAlmostEqual(value, 0.9, places=3)
        self.assertGreater(value, 0.899)

    def test_worthless_item_stays_low(self):
        self.assertLess(bayesian_rating(2, 98, 0.85, 50.0), 0.4)


class TestBuildItemText(unittest.TestCase):
    """模板必须与 build_corpus 一致 —— A2 的内容向量要和 ChromaDB 的语料可比。"""

    def test_name_only(self):
        self.assertEqual(build_item_text("Dota 2", "", "", ""), "Dota 2")

    def test_full_template(self):
        text = build_item_text("Dota 2", "Action, Strategy", "Multi-player", "A MOBA.")
        self.assertEqual(
            text, "Dota 2\nGenres: Action, Strategy\nCategories: Multi-player\nA MOBA."
        )

    def test_description_truncated(self):
        text = build_item_text("X", "", "", "d" * (DESC_LIMIT + 500))
        self.assertEqual(len(text), 1 + 1 + DESC_LIMIT)  # name + \n + desc

    def test_publishers_deliberately_excluded(self):
        """Publishers 走 A4 的结构化特征,不混进文本,便于归因。"""
        self.assertNotIn("Publishers", build_item_text("X", "Action", "Co-op", "desc"))


if __name__ == "__main__":
    unittest.main()
