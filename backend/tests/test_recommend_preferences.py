"""多轮对话约束(A7)单测:偏好状态 + 去重 + 偏好过滤 + 诚实回退。

覆盖两层:
  * `app/recommend/preferences.py` —— 纯函数与状态机
  * `app/recommend/service.py`     —— 排序之后施加约束的分桶选品

运行: cd backend && .venv/Scripts/python -m unittest tests.test_recommend_preferences -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.recommend.preferences import (
    FILTER_VOCAB,
    PreferenceState,
    compile_term_matrix,
    match_mask,
    normalise_terms,
    parse_message,
    term_columns,
)
from app.recommend.rank import FeatureBuilder
from app.recommend.recall import RecallConfig, RecallEngine
from app.recommend.service import Recommender


# ---------------------------------------------------------------- 状态机

class TestNormaliseTerms(unittest.TestCase):
    def test_case_insensitive_canonicalisation(self):
        ok, bad = normalise_terms(["strategy", "RPG", "co-op"])
        self.assertEqual(ok, ["Strategy", "RPG", "Co-op"])
        self.assertEqual(bad, [])

    def test_unknown_terms_are_rejected_not_guessed(self):
        """词表外的词必须进 bad —— 绝不能悄悄当成"筛过了"。"""
        ok, bad = normalise_terms(["Strategy", "潜行", "恐怖"])
        self.assertEqual(ok, ["Strategy"])
        self.assertEqual(bad, ["潜行", "恐怖"])

    def test_string_input_and_dedup(self):
        ok, _ = normalise_terms("Strategy")
        self.assertEqual(ok, ["Strategy"])
        ok, _ = normalise_terms(["RPG", "rpg", "RPG"])
        self.assertEqual(ok, ["RPG"])

    def test_garbage_input_is_ignored(self):
        for payload in (None, 42, {"a": 1}, [None, 3, []], ""):
            ok, bad = normalise_terms(payload)
            self.assertEqual(ok, [])
            self.assertEqual(bad, [])


class TestParseMessage(unittest.TestCase):
    def test_chinese_genre_keyword(self):
        self.assertEqual(parse_message("我更喜欢策略类的游戏")["genres"], ["Strategy"])

    def test_ascii_keyword_adjacent_to_cjk(self):
        """回归:`\\brpg\\b` 在「推荐点rpg」里匹配不到,因为 CJK 也算 \\w。"""
        self.assertEqual(parse_message("推荐点rpg")["genres"], ["RPG"])

    def test_negation_goes_to_avoid(self):
        result = parse_message("不要策略")
        self.assertEqual(result["genres"], [])
        self.assertEqual(result["avoid"], ["Strategy"])

    def test_unsupported_topic_is_reported_as_unmapped(self):
        result = parse_message("我最近不想玩恐怖游戏")
        self.assertEqual(result["genres"], [])
        self.assertIn("恐怖", result["unmapped"])

    def test_plain_chitchat_yields_nothing(self):
        for message in ("换一批", "你好呀", "随便推荐点别的"):
            result = parse_message(message)
            self.assertEqual(result["genres"], [], message)
            self.assertEqual(result["avoid"], [], message)


class TestPreferenceState(unittest.TestCase):
    def test_from_dict_tolerates_dirty_input(self):
        state = PreferenceState.from_dict(
            {"genres": ["strategy", "潜行"], "avoid": None, "shown": ["x", 1000, 1000, -3],
             "turn": "2", "unmapped": "not-a-list"}
        )
        self.assertEqual(state.genres, ["Strategy"])
        self.assertEqual(state.shown, [1000])
        self.assertEqual(state.turn, 2)
        self.assertIn("潜行", state.unmapped)

    def test_from_dict_handles_none(self):
        self.assertTrue(PreferenceState.from_dict(None).is_empty())

    def test_merge_accumulates(self):
        state = PreferenceState(genres=["Strategy"])
        merged = state.merge({"genres": ["RPG"], "unmapped": ["潜行"]})
        self.assertEqual(merged.genres, ["Strategy", "RPG"])
        self.assertEqual(merged.unmapped, ["潜行"])
        self.assertEqual(state.genres, ["Strategy"], "merge 不应修改原状态")

    def test_merge_dedupes(self):
        state = PreferenceState(genres=["Strategy"])
        self.assertEqual(state.merge({"genres": ["strategy"]}).genres, ["Strategy"])

    def test_negative_wins_over_positive(self):
        """同一轮里既说"要策略"又说"不要策略"时以负向为准:宁可不推,不要推错。"""
        merged = PreferenceState().merge({"genres": ["Strategy"], "avoid": ["Strategy"]})
        self.assertEqual(merged.genres, [])
        self.assertEqual(merged.avoid, ["Strategy"])

    def test_restart_clears_preferences_but_keeps_dedup(self):
        state = PreferenceState(genres=["Strategy"], shown=[1000, 1001])
        restarted = state.merge({"genres": ["RPG"]}, restart=True)
        self.assertEqual(restarted.genres, ["RPG"])
        self.assertEqual(
            restarted.shown, [1000, 1001],
            "restart 不能清掉 shown,否则会立刻看到刚推荐过的游戏",
        )

    def test_with_shown_dedupes_and_bumps_turn(self):
        state = PreferenceState(shown=[1000])
        next_state = state.with_shown([1001, 1000, 1002])
        self.assertEqual(next_state.shown, [1000, 1001, 1002])
        self.assertEqual(next_state.turn, 1)
        self.assertEqual(state.shown, [1000], "with_shown 不应修改原状态")

    def test_roundtrip(self):
        state = PreferenceState(genres=["RPG"], avoid=["Gore"], unmapped=["潜行"],
                                shown=[1, 2, 3], turn=4)
        self.assertEqual(PreferenceState.from_dict(state.to_dict()), state)


class TestTermMatrix(unittest.TestCase):
    def test_matches_genres_and_categories(self):
        matrix = compile_term_matrix(
            np.array(["Action, Indie", "Strategy", "Indie"]),
            np.array(["Single-player", "Co-op", "Co-op"]),
        )
        rows = np.arange(3)
        self.assertEqual(match_mask(matrix, rows, term_columns(["Strategy"])).tolist(),
                         [False, True, False])
        self.assertEqual(match_mask(matrix, rows, term_columns(["Co-op"])).tolist(),
                         [False, True, True])
        self.assertEqual(match_mask(matrix, rows, term_columns(["Indie"])).tolist(),
                         [True, False, True])

    def test_empty_terms_match_nothing(self):
        matrix = compile_term_matrix(np.array(["Action"]), np.array([""]))
        self.assertEqual(match_mask(matrix, np.arange(1), np.empty(0, np.int64)).tolist(),
                         [False])

    def test_unknown_term_yields_no_columns(self):
        self.assertEqual(len(term_columns(["这不是一个类型"])), 0)

    def test_vocab_has_no_duplicates(self):
        self.assertEqual(len(FILTER_VOCAB), len(set(FILTER_VOCAB)))


# ---------------------------------------------------------------- 服务层

# 12 个物品,类型用真实 Steam 词汇(否则过滤矩阵全是 False,测不出东西)
GENRES = [
    "Action, Indie",        # 0  <- 用户库里的游戏
    "Strategy, Indie",      # 1
    "Strategy",             # 2
    "RPG",                  # 3
    "Action",               # 4
    "Indie",                # 5
    "Strategy, Simulation", # 6
    "RPG, Adventure",       # 7
    "Action, Gore",         # 8  <- avoid 测试用
    "Casual",               # 9
    "Strategy, Action",     # 10
    "Sports",               # 11
]
N = len(GENRES)


def make_index():
    rng = np.random.default_rng(7)
    emb = rng.random((N, 4)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    catalog = pd.DataFrame(
        {
            "appid": np.arange(1000, 1000 + N, dtype=np.int64),
            "name": [f"Game{i}" for i in range(N)],
            "genres": GENRES,
            "categories": ["Single-player" if i % 2 == 0 else "Co-op" for i in range(N)],
            "publishers": [f"Pub{i % 3}" for i in range(N)],
            "bayes_rating": np.linspace(0.5, 0.95, N),
            "positive_ratio": np.full(N, 0.8),
            "estimated_owners_lower": np.arange(1, N + 1) * 1000,
            "metacritic_score": np.full(N, 80),
            "price_cents": np.full(N, 1000),
            "release_year": np.full(N, 2015),
            "dlc_count": np.zeros(N),
            "n_interactions": np.arange(N, dtype=np.int64),
            "header_image": [f"http://img/{i}.jpg" for i in range(N)],
            "rating": ["特别好评"] * N,
            "positive": np.full(N, 1000),
            "negative": np.full(N, 100),
            "has_content": np.ones(N, dtype=bool),
        }
    )
    index = SimpleNamespace()
    index.catalog = catalog
    index.content_emb = emb
    index.content_topk_idx = np.stack(
        [(np.arange(N) + 1 + i) % N for i in range(3)], 1
    ).astype(np.int32)
    index.content_topk_score = np.full((N, 3), 0.5, dtype=np.float32)
    index.collab_topk_idx = np.stack(
        [(np.arange(N) + 4 + i) % N for i in range(3)], 1
    ).astype(np.int32)
    index.collab_topk_score = np.full((N, 3), 0.8, dtype=np.float32)
    index.n_items = N
    index.manifest = {"topk": 3}
    index.path = Path(".")
    index._row = {int(a): i for i, a in enumerate(catalog["appid"].to_numpy())}
    index.rows_of = lambda appids: np.array(
        [index._row.get(int(a), -1) for a in appids], dtype=np.int64
    )
    return index


class _Model:
    """把 merge_rank 取负当分数 —— 于是"排序"就是融合顺序,分桶效果一眼可见。"""

    def predict(self, features):
        from app.recommend.rank import FEATURE_NAMES

        return -features[:, FEATURE_NAMES.index("merge_rank")]


class _RankerLike:
    def __init__(self, model, builder, engine):
        self.model = model
        self.builder = builder
        self.engine = engine


def make_recommender():
    index = make_index()
    engine = RecallEngine(
        index,
        quality_scores=index.catalog["n_interactions"].to_numpy(dtype=np.float32),
        config=RecallConfig(total=N + 1),
    )
    builder = FeatureBuilder(
        index, engine,
        train_popularity=index.catalog["n_interactions"].to_numpy(dtype=float),
    )
    return Recommender(index, _RankerLike(_Model(), builder, engine), builder, engine)


LIBRARY = [{"appid": 1000, "playtime_forever": 100.0}]  # 只有 Game0,类型 Action/Indie


def make_same_name_recommender():
    """把 Game1 / Game2 改成同名 —— 复现目录里"同名不同 appid"的情况。

    真实数据里有 319 组(Counter-Strike: Condition Zero 的 appid 80 与 100 就是一对),
    只按 appid 去重会让用户连着看到两张一模一样的卡。
    """
    index = make_index()
    names = index.catalog["name"].to_numpy().copy()
    names[1] = "SameGame"
    names[2] = "SameGame"
    index.catalog["name"] = names
    engine = RecallEngine(
        index,
        quality_scores=index.catalog["n_interactions"].to_numpy(dtype=np.float32),
        config=RecallConfig(total=N + 1),
    )
    builder = FeatureBuilder(
        index, engine,
        train_popularity=index.catalog["n_interactions"].to_numpy(dtype=float),
    )
    return Recommender(index, _RankerLike(_Model(), builder, engine), builder, engine)


class TestSameNameDedup(unittest.TestCase):
    def setUp(self):
        self.rec = make_same_name_recommender()

    def test_same_name_appears_once_per_turn(self):
        result = self.rec.recommend(LIBRARY, top_k=N)
        names = [r.name for r in result.recommendations]
        self.assertEqual(len(names), len(set(names)), f"出现同名卡片:{names}")
        self.assertIn("SameGame", names)

    def test_excluding_one_appid_also_excludes_its_namesake(self):
        """回归:`_name_keys` 按 cat_idx 索引,却被用池内位置取用 —— 同名去重会静默失效。"""
        result = self.rec.recommend(LIBRARY, top_k=N, exclude_appids=[1001])
        names = [r.name for r in result.recommendations]
        self.assertNotIn("SameGame", names, "另一个 appid 的同名物品也必须被去掉")
        self.assertNotIn(1002, [r.appid for r in result.recommendations])

    def test_namesake_does_not_swallow_other_slots(self):
        """去掉同名之后,后面的候选要能顶上来,不能少给。"""
        result = self.rec.recommend(LIBRARY, top_k=5)
        self.assertEqual(len(result.recommendations), 5)


class TestMultiTurnRecommend(unittest.TestCase):
    def setUp(self):
        self.rec = make_recommender()

    # ---- 去重 ----

    def test_no_constraints_keeps_original_behaviour(self):
        result = self.rec.recommend(LIBRARY, top_k=5)
        self.assertEqual(len(result.recommendations), 5)
        self.assertFalse(result.relaxed)
        self.assertFalse(result.no_match)
        self.assertEqual(result.prefer, ())
        self.assertEqual(result.prefer, ())

    def test_excluded_appids_never_come_back(self):
        """回归:「再推荐点别的」原样返回上一批。"""
        first = self.rec.recommend(LIBRARY, top_k=5)
        shown = [r.appid for r in first.recommendations]
        self.assertEqual(len(shown), 5)

        second = self.rec.recommend(LIBRARY, top_k=5, exclude_appids=shown)
        again = [r.appid for r in second.recommendations]
        self.assertEqual(
            set(shown) & set(again), set(), "第二轮不能出现第一轮推荐过的游戏"
        )
        self.assertEqual(len(again), 5, "池子里还有候选,应该照常给满")
        self.assertEqual(second.skipped_shown, 5)

    def test_exhausted_when_everything_shown(self):
        everything = [1000 + i for i in range(1, N)]  # 除库内那款以外的全部
        result = self.rec.recommend(LIBRARY, top_k=5, exclude_appids=everything)
        self.assertEqual(result.recommendations, [])
        self.assertTrue(result.exhausted)

    # ---- 偏好过滤 ----

    def test_preferred_genre_comes_first(self):
        result = self.rec.recommend(LIBRARY, top_k=5, prefer=["Strategy"])
        genres = [r.genres for r in result.recommendations]
        self.assertEqual(len(genres), 5)
        # 4 款 Strategy 应全部排在前面,第 5 款才是补齐的
        for g in genres[:4]:
            self.assertIn("Strategy", g)
        self.assertNotIn("Strategy", genres[4])
        self.assertEqual(result.matched_count, 4)
        self.assertTrue(result.relaxed)
        self.assertFalse(result.no_match)

    def test_preference_hits_are_reported_per_item(self):
        result = self.rec.recommend(LIBRARY, top_k=5, prefer=["Strategy"])
        for r in result.recommendations:
            if "Strategy" in r.genres:
                self.assertIn("Strategy", r.pref_match)
            else:
                self.assertEqual(r.pref_match, [])

    def test_avoid_genre_is_never_returned(self):
        result = self.rec.recommend(LIBRARY, top_k=N, avoid=["Gore"])
        self.assertNotIn(1008, [r.appid for r in result.recommendations])

    def test_no_match_is_reported_honestly(self):
        """词表里没有 Racing 之外匹配不上时,必须报 no_match 而不是假装命中。"""
        result = self.rec.recommend(LIBRARY, top_k=5, prefer=["Racing"])
        self.assertEqual(result.matched_count, 0)
        self.assertTrue(result.no_match)
        self.assertFalse(result.relaxed)
        # 仍然给出"最接近的"作为备选,而不是空手而归
        self.assertEqual(len(result.recommendations), 5)

    def test_all_matched_means_not_relaxed(self):
        result = self.rec.recommend(LIBRARY, top_k=2, prefer=["Strategy"])
        self.assertEqual(result.matched_count, 2)
        self.assertFalse(result.relaxed)
        self.assertFalse(result.no_match)

    def test_unknown_prefer_term_is_ignored(self):
        """词表外的词不该悄悄改变结果,也不该报 no_match。"""
        plain = self.rec.recommend(LIBRARY, top_k=5)
        weird = self.rec.recommend(LIBRARY, top_k=5, prefer=["潜行"])
        self.assertEqual([r.appid for r in plain.recommendations],
                         [r.appid for r in weird.recommendations])
        self.assertFalse(weird.no_match)

    def test_prefer_and_exclude_compose(self):
        result = self.rec.recommend(
            LIBRARY, top_k=5, prefer=["Strategy"], exclude_appids=[1002, 1006]
        )
        appids = [r.appid for r in result.recommendations]
        self.assertNotIn(1002, appids)
        self.assertNotIn(1006, appids)
        self.assertEqual(result.matched_count, 2)   # 只剩 1001 与 1010 是 Strategy

    # ---- 冷启动路径也要受约束 ----

    def test_cold_start_respects_preferences(self):
        """空库只有质量通道可走,偏好仍然要生效(命中不够就补齐,并如实标记)。"""
        result = self.rec.recommend([], top_k=3, prefer=["Gore"])
        self.assertTrue(result.cold_start)
        self.assertEqual(len(result.recommendations), 3)
        # 目录里只有 Game8 带 Gore,所以必然是"命中 1 款 + 补 2 款"
        self.assertEqual(result.matched_count, 1)
        self.assertTrue(result.relaxed)
        self.assertIn("Gore", result.recommendations[0].genres)


# ---------------------------------------------------------------- 文案

class TestConstraintNote(unittest.TestCase):
    """`constraint_note` 是"实话"的唯一来源,它必须准确。

    这里直接测 `app.recommend.notes` 而不是 `app.agent.graph`:`app/agent/__init__.py`
    会连带导入整张图与 `sentence-transformers`,一个纯字符串函数的测试要等 14 秒。
    """

    def _result(self, **kwargs):
        base = dict(prefer=(), pool_size=200, matched_count=0,
                    relaxed=False, no_match=False, exhausted=False,
                    recommendations=[1, 2, 3, 4, 5])
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_silent_when_nothing_to_say(self):
        from app.recommend.notes import constraint_note

        self.assertEqual(constraint_note(self._result()), "")

    def test_relaxed_mentions_real_counts(self):
        from app.recommend.notes import constraint_note

        note = constraint_note(
            self._result(prefer=("Strategy",), matched_count=2, relaxed=True)
        )
        self.assertIn("Strategy", note)
        self.assertIn("2", note)
        self.assertIn("3", note)   # 5 - 2 款是补齐的

    def test_no_match_admits_it(self):
        from app.recommend.notes import constraint_note

        note = constraint_note(self._result(prefer=("Strategy",), no_match=True))
        self.assertIn("一个都没对上", note)

    def test_no_match_reports_unmapped_terms(self):
        from app.recommend.notes import constraint_note

        note = constraint_note(
            self._result(prefer=("Strategy",), no_match=True), unmapped=["潜行"]
        )
        self.assertIn("潜行", note)

    def test_full_hit_is_silent(self):
        from app.recommend.notes import constraint_note

        self.assertEqual(
            constraint_note(
                self._result(prefer=("Strategy",), matched_count=5, relaxed=False)
            ),
            "",
        )

    def test_unmapped_only_is_reported_not_swallowed(self):
        """用户说了词表里没有的词(恐怖/潜行)时,不能装作没听见。

        `prefer` 会是空的、`no_match` 也是 False,如果不单独处理,用户就得不到任何反馈,
        会误以为自己的要求被采纳了。
        """
        from app.recommend.notes import constraint_note

        note = constraint_note(self._result(), unmapped=["恐怖"])
        self.assertIn("恐怖", note)
        self.assertIn("没有", note)

    def test_works_on_the_real_result_dataclass(self):
        """回归:曾把 `unmapped` 当成 `RecommendationResult` 的字段,线上直接 AttributeError。

        `SimpleNamespace` 测不出这种漂移,所以这里用**真的** dataclass 跑一遍。
        """
        from app.recommend.notes import constraint_note
        from app.recommend.service import RecommendationResult

        result = RecommendationResult(
            recommendations=[], profile={}, cold_start=False, quality_fallback=False,
            unmatched_in_library=0, library_matched=3, prefer=("Strategy",),
            pool_size=200, matched_count=0, no_match=True,
        )
        self.assertIn("一个都没对上", constraint_note(result))
        self.assertIn("潜行", constraint_note(result, unmapped=["潜行"]))


if __name__ == "__main__":
    unittest.main()
