"""A5 在线服务:把 A2~A4 的离线资产串成一个可用的推荐入口。

一次加载(进程内单例),之后每次推荐只做**查表 + 点积 + LightGBM 打分**:
  * 索引 107 ms / 常驻 73 MB / 纯 CPU(A6 实测)
  * 端到端 P50 **9.1 ms**、P99 **36.6 ms**
  * **不加载 embedding 模型、不调 Steam API** —— 画像直接读目录里的类型

冷启动:
  * 库为空             -> 只走质量/热度通道(实测仍能给出候选)
  * 库里的游戏不在目录里 -> 能映射多少算多少;一个都映射不上时同样退回 quality

## 多轮对话约束(A7)

`recommend()` 额外接受三个参数:

* `exclude_appids` —— 已经展示过的游戏(去重)
* `prefer`         —— 用户**明说**想要的类型/玩法
* `avoid`          —— 用户明说不要的

⚠️ **约束在排序之后施加,不是之前**。排序特征里的 `merge_rank` 是候选在 200 条
召回池里的**位置**;如果先按类型过滤再排序,送进模型的池子变成另一个分布,
`merge_rank` 就和训练时不一致(训练用的是完整池子的位置)。所以流程是:
先用 LightGBM 给**完整池子**打分,再按"是否命中用户明说的偏好"分桶取用 ——
桶内仍按精排分数排序,于是"明说的偏好"决定**要不要**,游戏历史决定**要哪个**。

⚠️ 与评测的差别:评测里质量先验必须用**训练切分**计数(否则泄漏),线上没有这个问题,
直接用目录里的全量 `n_interactions` 即可。这个差异由调用方通过 `popularity` 选择。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.recommend.index import DEFAULT_INDEX_DIR, RecommendIndex
from app.recommend.preferences import (
    FILTER_VOCAB,
    compile_term_matrix,
    match_mask,
    term_columns,
)
from app.recommend.rank import FeatureBuilder, Ranker
from app.recommend.recall import DEFAULT_QUOTAS, RecallConfig, RecallEngine

MODEL_FILE = "ranker.txt"
CHANNEL_LABEL = {
    "content_user": "整体口味",
    "content_itemknn": "玩法相似",
    "collab_itemknn": "同好也在玩",
    "quality": "口碑与热度",
    "publisher": "同厂商",
}


@dataclass
class Recommendation:
    appid: int
    name: str
    header_image: str
    genres: str
    categories: str
    price_cents: int | None
    rating: str
    positive: int
    negative: int
    metacritic_score: int
    score: float
    channels: tuple[str, ...] = ()
    because_of: str | None = None
    match_genres: list[str] = field(default_factory=list)
    pref_match: list[str] = field(default_factory=list)
    owned: bool = False

    def as_dict(self) -> dict:
        """转成前端/Agent 用的字典,字段与既有 `Candidate` 类型兼容。

        `distance` 只是为兼容旧字段而保留(越小越相似 = `1 - score`),
        新代码请使用 `score`。
        """
        return {
            "appid": self.appid,
            "name": self.name,
            "header_image": self.header_image,
            "genres": self.genres,
            "categories": self.categories,
            "price_cents": self.price_cents,
            "rating": self.rating,
            "positive": self.positive,
            "negative": self.negative,
            "metacritic_score": self.metacritic_score,
            "score": self.score,
            "distance": 1.0 - self.score,
            "reason": None,
            "channels": list(self.channels),
            "because_of": self.because_of,
            "match_genres": self.match_genres,
            "pref_match": self.pref_match,
            "owned": self.owned,
        }


@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    profile: dict
    cold_start: bool
    quality_fallback: bool
    unmatched_in_library: int
    library_matched: int
    # ---- 多轮对话约束的诊断信息(供上层写"实话")----
    prefer: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    pool_size: int = 0        # 召回池大小(过滤前)
    matched_count: int = 0    # 最终列表里命中"明说偏好"的条数
    relaxed: bool = False     # 命中的不够,补了不完全符合的
    no_match: bool = False    # 一个都没命中
    exhausted: bool = False   # 池子里的候选被去重/排除规则清空了
    skipped_shown: int = 0    # 因"已经展示过"被跳过的条数


class Recommender:
    """召回 + 精排的对外入口。线程安全(只读)。"""

    def __init__(self, index: RecommendIndex, ranker: Ranker, builder: FeatureBuilder,
                 engine: RecallEngine) -> None:
        self.index = index
        self.ranker = ranker
        self.builder = builder
        self.engine = engine
        self._catalog = index.catalog
        self._appids = self._catalog["appid"].to_numpy()
        self._names = self._catalog["name"].to_numpy()
        self._genres = self._catalog["genres"].to_numpy()
        self._categories = self._catalog["categories"].to_numpy()
        self._header = self._catalog["header_image"].to_numpy()
        self._rating = self._catalog["rating"].to_numpy()
        self._price = self._catalog["price_cents"].to_numpy()
        self._positive = self._catalog["positive"].to_numpy()
        self._negative = self._catalog["negative"].to_numpy()
        self._metacritic = self._catalog["metacritic_score"].to_numpy()
        # 类型/玩法 -> 布尔矩阵,一次构建(11,444 × 25 ≈ 286 KB),之后每轮只切列
        self._term_matrix = compile_term_matrix(self._genres, self._categories)
        # 同名去重用:归一化名字 + appid -> 行号(见 `name_key`)
        self._name_keys = np.array([name_key(n) for n in self._names], dtype=object)
        self._row_by_appid = {int(a): i for i, a in enumerate(self._appids)}

    # ------------------------------------------------------------ 加载

    @classmethod
    def load(cls, index_dir: Path | str = DEFAULT_INDEX_DIR) -> "Recommender":
        import lightgbm as lgb

        index_dir = Path(index_dir)
        model_path = index_dir / MODEL_FILE
        if not model_path.exists():
            raise RuntimeError(
                f"找不到排序器 {model_path};请先运行 scripts/train_ranker.py"
            )
        index = RecommendIndex.load(index_dir)
        model = lgb.Booster(model_file=str(model_path))
        # 线上用目录里的全量热度(评测才需要按训练切分重算,见 evaluate.py 的泄漏说明)
        popularity = index.catalog["n_interactions"].to_numpy(dtype=np.float64)
        engine = RecallEngine(
            index,
            quality_scores=_quality_scores(index, popularity),
            config=RecallConfig(quotas=dict(DEFAULT_QUOTAS), total=200),
        )
        builder = FeatureBuilder(index, engine, train_popularity=popularity)
        return cls(index, Ranker(model, builder, engine), builder, engine)

    # ------------------------------------------------------------ 画像

    def build_profile(self, rows: np.ndarray, weights: np.ndarray,
                      *, sample: int = 30) -> dict:
        """从**目录**统计类型/玩法画像 —— 不调 Steam appdetails。

        与 `graph.py` 原来的 `_build_profile` 口径一致(按时长取 top-30、各取 top-6),
        但省掉了每次推荐 30 次 Steam 请求。
        """
        if len(rows) == 0:
            return {"top_genres": [], "top_categories": [], "summary": "暂无游戏库"}
        order = np.argsort(-np.asarray(weights))[:sample]
        sample_rows = np.asarray(rows)[order]

        def top_of(column, limit=6) -> list[str]:
            counter: dict[str, int] = {}
            for row in sample_rows:
                for token in str(column[int(row)]).split(","):
                    token = token.strip()
                    if token:
                        counter[token] = counter.get(token, 0) + 1
            return sorted(counter, key=counter.get, reverse=True)[:limit]

        top_genres = top_of(self._genres)
        top_cats = top_of(self._categories)
        summary = (
            f"偏爱类型:{'、'.join(top_genres) or '暂无'};"
            f"常见玩法:{'、'.join(top_cats) or '暂无'}"
        )
        return {"top_genres": top_genres, "top_categories": top_cats, "summary": summary}

    # ------------------------------------------------------------ 推荐

    def recommend(
        self,
        games: list[dict],
        *,
        top_k: int = 5,
        pool: int = 200,
        exclude_appids: Iterable[int] | None = None,
        prefer: Sequence[str] = (),
        avoid: Sequence[str] = (),
    ) -> RecommendationResult:
        """`games` 为 `[{"appid": int, "playtime_forever": float|None}, ...]`。

        `prefer` / `avoid` 里的词必须已经在 `preferences.FILTER_VOCAB` 里
        (调用方用 `PreferenceState` 保证这一点);词表外的词在这里会被静默丢弃,
        因为"筛不了的词"不该悄悄变成"筛过了"。
        """
        appids = [int(g["appid"]) for g in games]
        weights = np.array(
            [float(g.get("playtime_forever") or 0.0) for g in games], dtype=np.float32
        )
        rows = self.index.rows_of(appids) if appids else np.empty(0, dtype=np.int64)
        keep = rows >= 0
        matched_rows, matched_weights = rows[keep], weights[keep]
        unmatched = int((~keep).sum())

        profile = self.build_profile(matched_rows, matched_weights)
        prefer_t, avoid_t = tuple(prefer), tuple(avoid)
        exclude_set = appids_of(exclude_appids)

        # 库里完全没有目录内的游戏(或库为空)-> 只能走质量/热度兜底
        quality_fallback = len(matched_rows) == 0
        if quality_fallback:
            candidates = self._quality_candidates(top_k, pool)
            scored = [(c, c.score) for c in candidates]
        else:
            recalled = self.engine.recall_rows(
                matched_rows, matched_weights, owned_rows=matched_rows
            )
            if not recalled:
                # 库能识别、但已把候选池占满(例如库覆盖了整个目录):
                # 这是"没剩下可推的",**不是**冷启动 —— 标成冷启动会让上层给出错误文案
                return RecommendationResult(
                    [], profile, False, False, unmatched, int(keep.sum()),
                    prefer=prefer_t, avoid=avoid_t,
                )
            candidate_rows = np.array([c.cat_idx for c in recalled], dtype=np.int64)
            # 只构造**一次**特征:Ranker.rank 内部也会构造,这里直接复用模型打分,
            # 避免把 26 维特征算两遍(实测特征占端到端延迟的四成)
            features = self.builder.build(
                matched_rows, matched_weights, candidate_rows
            ).features
            scores = self.ranker.model.predict(features)
            # 完整池子按分数降序,后面的分桶从这里切 —— 不要先过滤再排序(见模块注释)
            order = np.argsort(-scores)
            scored = [(recalled[int(i)], float(scores[int(i)])) for i in order]

        picked, diag = self._select(
            scored, top_k=top_k, prefer=prefer_t, avoid=avoid_t, exclude=exclude_set
        )
        selected = [scored[i] for i in picked]
        selected_rows = np.array([c.cat_idx for c, _ in selected], dtype=np.int64)

        because = self._explain(selected_rows, matched_rows, matched_weights)
        user_genres = set(profile.get("top_genres") or [])
        pref_cols = term_columns(prefer_t)
        recommendations = [
            self._to_recommendation(
                candidate, score, because.get(candidate.cat_idx), user_genres, pref_cols
            )
            for candidate, score in selected
        ]
        return RecommendationResult(
            recommendations=recommendations,
            profile=profile,
            cold_start=quality_fallback,
            quality_fallback=quality_fallback,
            unmatched_in_library=unmatched,
            library_matched=int(keep.sum()),
            prefer=prefer_t,
            avoid=avoid_t,
            pool_size=len(scored),
            **diag,
        )

    # ------------------------------------------------------------ 选择

    def _select(
        self,
        scored: list[tuple],
        *,
        top_k: int,
        prefer: Sequence[str] = (),
        avoid: Sequence[str] = (),
        exclude: frozenset[int] = frozenset(),
    ) -> tuple[list[int], dict]:
        """在**已排好序**的候选上施加对话约束,返回被选中的下标与诊断信息。

        规则(按优先级):

        1. 剔除 `exclude`(已经推荐过的)和命中 `avoid` 的;
        2. 剔除**同名**的 —— 目录里 319 组同名不同 appid 的物品,不去重会看到两张一样的卡;
        3. 剩下的人里,命中 `prefer` 的排前面,桶内保持精排顺序;
        4. 命中不够 `top_k` 时用未命中的补齐,并标记 `relaxed` / `no_match`。
        """
        n = len(scored)
        empty_diag = {
            "matched_count": 0, "relaxed": False, "no_match": False,
            "exhausted": n == 0, "skipped_shown": 0,
        }
        if n == 0:
            return [], empty_diag

        rows = np.array([c.cat_idx for c, _ in scored], dtype=np.int64)
        pref_cols = term_columns(prefer)
        avoid_cols = term_columns(avoid)
        hit_pref = match_mask(self._term_matrix, rows, pref_cols)
        hit_avoid = match_mask(self._term_matrix, rows, avoid_cols)

        if exclude:
            excluded = np.isin(
                self._appids[rows],
                np.fromiter(exclude, dtype=np.int64, count=len(exclude)),
            )
        else:
            excluded = np.zeros(n, dtype=bool)
        skipped_shown = int((excluded & ~hit_avoid).sum())

        # 已经展示过的**名字**也要记住:换了个 appid 的同名物品同样是"重复推荐"
        taken_names: set[str] = set()
        for appid in exclude:
            row = self._row_by_appid.get(appid)
            if row is not None:
                taken_names.add(str(self._name_keys[row]))

        eligible = np.flatnonzero(~excluded & ~hit_avoid)
        matched = eligible[hit_pref[eligible]]
        rest = eligible[~hit_pref[eligible]]

        # ⚠️ 索引空间:下面 `picked` 里存的是**池内位置**,而 `_name_keys` 是按
        # `cat_idx` 索引的。这里先把名字按池内位置重排,避免拿位置当 cat_idx 用 ——
        # 踩过这个错,表现是"同名去重偶尔失效",极难定位。
        pool_name_keys = self._name_keys[rows]

        picked: list[int] = []

        def take(order: np.ndarray) -> int:
            added = 0
            for idx in order.tolist():
                if len(picked) >= top_k:
                    break
                key = str(pool_name_keys[idx])
                if key in taken_names:
                    continue
                taken_names.add(key)
                picked.append(int(idx))
                added += 1
            return added

        take(matched)
        matched_count = len(picked)
        relaxed = False
        no_match = False
        if len(picked) < top_k and len(rest):
            filled = take(rest)
            if pref_cols.size:
                # 有正向偏好却没填满 -> 必须区分"部分命中"和"完全没命中"
                no_match = matched_count == 0
                relaxed = not no_match and filled > 0

        return picked, {
            "matched_count": min(matched_count, top_k),
            "relaxed": relaxed,
            "no_match": no_match,
            "exhausted": len(picked) == 0,
            "skipped_shown": skipped_shown,
        }

    def _quality_candidates(self, top_k: int, pool: int):
        """冷启动兜底:只走质量/热度通道。"""
        from app.recommend.recall import Candidate

        idx, scores = self.engine._quality_channel(
            np.empty(0, dtype=np.int64), self.engine.config
        )
        return [Candidate(int(i), float(s), ("quality",)) for i, s in zip(idx, scores)][
            : max(top_k, pool)
        ]

    # ------------------------------------------------------------ 可解释

    def _explain(
        self, candidate_rows: np.ndarray, library_rows: np.ndarray,
        library_weights: np.ndarray,
    ) -> dict[int, str]:
        """为每个候选找"最像它"的库内游戏,给 LLM 当理由素材。

        做法:把库内每个物品的 TopK 邻居摊平,**只保留命中候选集**的那些条目
        (库可能有两千多款,不筛的话是 20 万级条目),再按 (邻居, 分数) 排序取最大值。
        全程 numpy,没有 Python 双层循环 —— 早期版本对 2,143 款的库要 230ms。
        """
        if len(candidate_rows) == 0 or len(library_rows) == 0:
            return {}
        candidate_set = set(int(r) for r in candidate_rows)
        best: dict[int, tuple[float, int]] = {}
        for topk_idx, topk_score in (
            (self.engine.content_idx, self.engine.content_score),
            (self.engine.collab_idx, self.engine.collab_score),
        ):
            depth = min(self.engine.config.knn_depth, topk_idx.shape[1])
            neighbours = topk_idx[library_rows, :depth]          # [L, depth]
            scores = topk_score[library_rows, :depth]
            hit = (neighbours >= 0) & np.isin(neighbours, candidate_rows)
            if not hit.any():
                continue
            flat = np.flatnonzero(hit.ravel())
            flat_neighbour = neighbours.ravel()[flat]
            flat_score = scores.ravel()[flat]
            source_row = library_rows[flat // depth]
            # 同一个邻居可能被库里多个物品指向,按分数取最大
            order = np.lexsort((flat_score, flat_neighbour))
            ordered_neighbour = flat_neighbour[order]
            last = np.flatnonzero(
                np.r_[ordered_neighbour[1:] != ordered_neighbour[:-1], True]
            )
            for neighbour, score, source in zip(
                ordered_neighbour[last].tolist(),
                flat_score[order][last].tolist(),
                source_row[order][last].tolist(),
            ):
                if neighbour not in candidate_set:
                    continue
                current = best.get(neighbour)
                if current is None or score > current[0]:
                    best[neighbour] = (score, source)
        return {
            row: str(self._names[best[row][1]]) for row in (int(r) for r in candidate_rows)
            if row in best
        }

    def _to_recommendation(self, candidate, score: float, because_of, user_genres,
                           pref_cols: np.ndarray | None = None) -> Recommendation:
        row = int(candidate.cat_idx)
        genres = str(self._genres[row])
        overlap = sorted(
            {g.strip() for g in genres.split(",") if g.strip()} & user_genres
        )
        pref_match: list[str] = []
        if pref_cols is not None and len(pref_cols):
            hit = self._term_matrix[row, pref_cols]
            pref_match = [FILTER_VOCAB[int(c)] for c in pref_cols[hit]]
        price = int(self._price[row])
        return Recommendation(
            appid=int(self._appids[row]),
            name=str(self._names[row]),
            header_image=str(self._header[row]),
            genres=genres,
            categories=str(self._categories[row]),
            price_cents=price if price > 0 else None,
            rating=str(self._rating[row]),
            positive=int(self._positive[row]),
            negative=int(self._negative[row]),
            metacritic_score=int(self._metacritic[row]),
            score=round(float(score), 6),
            channels=tuple(candidate.channels),
            because_of=because_of,
            match_genres=overlap,
            pref_match=pref_match,
        )


def appids_of(values: Iterable[int] | None) -> frozenset[int]:
    """把外部传入的"已展示 appid"转成集合,对脏数据容错。"""
    result: set[int] = set()
    if not values:
        return frozenset()
    for value in values:
        try:
            appid = int(value)
        except (TypeError, ValueError):
            continue
        if appid > 0:
            result.add(appid)
    return frozenset(result)


def name_key(name: object) -> str:
    """游戏名的归一化键,用于"同名即重复"的判断。

    目录里存在 **319 组同名不同 appid** 的物品(例如 appid 80 与 100 都叫
    `Counter-Strike: Condition Zero`)。只按 appid 去重的话,用户会连着看到两张
    名字完全一样的卡片 —— 看起来就是"没有去重"。
    """
    text = str(name).lower()
    for token in ("™", "®", "©"):
        text = text.replace(token, "")
    return " ".join(text.split())


def _quality_scores(index: RecommendIndex, popularity: np.ndarray) -> np.ndarray:
    """质量先验 = 热度 + 贝叶斯好评率 + 销量(与 scripts/train_ranker.py 同一套公式)。"""
    catalog = index.catalog
    pop = np.log1p(np.asarray(popularity, dtype=np.float64))
    pop = pop / pop.max() if pop.max() > 0 else pop
    owners = np.log1p(catalog["estimated_owners_lower"].to_numpy(dtype=np.float64))
    owners = owners / owners.max() if owners.max() > 0 else owners
    return (
        pop + catalog["bayes_rating"].to_numpy(dtype=np.float64) + 0.5 * owners
    ).astype(np.float32)


# ---------------------------------------------------------------- 单例

_instance: Recommender | None = None
_lock = threading.Lock()
_failed: str | None = None


def get_recommender() -> Recommender | None:
    """懒加载单例。索引/模型不存在时返回 None(调用方需能降级)。

    加载失败只记录一次原因,不做重试 —— 缺文件时每次请求都去尝试加载会拖慢服务。
    """
    global _instance, _failed
    if _instance is not None:
        return _instance
    if _failed is not None:
        return None
    with _lock:
        if _instance is not None:
            return _instance
        try:
            _instance = Recommender.load()
        except Exception as exc:  # 缺索引/缺模型/依赖缺失都走这里
            _failed = str(exc)
            return None
    return _instance


def unavailable_reason() -> str | None:
    return _failed
