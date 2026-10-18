"""A4 特征工程 + LTR 排序器。

**核心约束:训练与线上必须共用同一个特征构造器**。否则会出现训练-服务偏斜
(train/serve skew):离线指标很好,线上却不是那个分布。所以 `FeatureBuilder`
放在 `app/` 而不是训练脚本里,`scripts/train_ranker.py` 与未来的 `service.py`
都调它。

特征分四组:
  * **召回证据**(每路的分/名次、命中几路、融合名次)—— 来自 `RecallEngine.channel_outputs`
  * **用户×物品交互**(内容余弦、类型重合/Jaccard/时长亲和、发行商亲和)
  * **物品静态**(好评率 / 销量 / Metacritic / 价格 / 年份 / DLC / 热度)
  * **用户规模**(库大小)

⚠️ **泄漏红线**:物品热度必须用**训练切分**的计数,不能用 `catalog.n_interactions`
(那是全量统计,含测试边)。调用方通过 `train_popularity` 注入。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from app.recommend.recall import CHANNEL_ORDER, Candidate, RecallEngine

FEATURE_NAMES: tuple[str, ...] = (
    "content_user_score", "content_user_rank",
    "content_itemknn_score", "content_itemknn_rank",
    "collab_score", "collab_rank",
    "quality_score", "quality_rank",
    "publisher_score", "publisher_rank",
    "n_channels", "merge_rank",
    "content_cosine",
    "bayes_rating", "positive_ratio", "log_owners", "metacritic_norm",
    "price_norm", "release_year_norm", "dlc_count_norm",
    "train_popularity",
    "genre_overlap", "genre_jaccard", "user_genre_affinity",
    "user_publisher_affinity", "user_size_log",
)
ABSENT_RANK = -1.0  # 该路没有召回这个物品
REFERENCE_YEAR = 1997.0


@dataclass
class FeatureContext:
    """一次查询的全部中间量,便于调试与解释("为什么推荐这个")。"""

    candidate_rows: np.ndarray
    features: np.ndarray
    channel_ranks: dict[str, dict[int, int]]


class FeatureBuilder:
    def __init__(
        self,
        index,
        engine: RecallEngine,
        *,
        train_popularity: np.ndarray,
        prior_positive_ratio: float = 0.87,
    ) -> None:
        self.index = index
        self.engine = engine
        self.n_items = index.n_items
        self.emb = np.asarray(index.content_emb)
        catalog = index.catalog

        # ---- 类型矩阵(稀疏的一热表示,用于向量化算重合/Jaccard/亲和)----
        genre_lists = [
            [g.strip() for g in str(raw).split(",") if g.strip()]
            for raw in catalog["genres"].to_numpy()
        ]
        self.genre_names = sorted({g for gs in genre_lists for g in gs})
        genre_col = {g: i for i, g in enumerate(self.genre_names)}
        rows, cols = [], []
        for i, gs in enumerate(genre_lists):
            for g in gs:
                rows.append(i)
                cols.append(genre_col[g])

        self.item_genre_count = np.zeros(self.n_items, dtype=np.float32)
        if rows:
            self.item_genre = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(self.n_items, len(self.genre_names)),
            )
            self.item_genre_count = np.asarray(
                self.item_genre.sum(axis=1)
            ).ravel().astype(np.float32)
        else:
            self.item_genre = sp.csr_matrix((self.n_items, 1), dtype=np.float32)

        # ---- 物品静态特征(预先归一化,查表即可)----
        owners = np.log1p(catalog["estimated_owners_lower"].to_numpy(dtype=np.float64))
        owners = owners / owners.max() if owners.max() > 0 else owners
        price = np.clip(catalog["price_cents"].to_numpy(dtype=np.float64), 0, 6000) / 6000
        year = catalog["release_year"].to_numpy(dtype=np.float64)
        year = np.clip((year - REFERENCE_YEAR) / 25.0, 0.0, 1.0)
        dlc = np.clip(catalog["dlc_count"].to_numpy(dtype=np.float64), 0, 50) / 50
        self.item_static = np.column_stack(
            [
                catalog["bayes_rating"].to_numpy(dtype=np.float64),
                catalog["positive_ratio"].fillna(prior_positive_ratio).to_numpy(dtype=np.float64),
                owners,
                catalog["metacritic_score"].to_numpy(dtype=np.float64) / 100.0,
                price,
                year,
                dlc,
            ]
        ).astype(np.float32)

        popularity = np.log1p(np.asarray(train_popularity, dtype=np.float64))
        self.train_popularity = (
            popularity / popularity.max() if popularity.max() > 0 else popularity
        ).astype(np.float32)
        self.item_pub = engine._item_pubs
        self.n_pubs = len(engine._pub_items)

    # ------------------------------------------------------------------

    def build(
        self, user_rows: np.ndarray, user_weights: np.ndarray, candidate_rows: np.ndarray
    ) -> FeatureContext:
        """构造 `[n_candidates, n_features]` 特征矩阵。

        `candidate_rows` 必须是**已按融合分数排序**的候选(融合名次也是一维特征)。
        """
        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        n_cand = len(candidate_rows)
        features = np.zeros((n_cand, len(FEATURE_NAMES)), dtype=np.float32)
        if n_cand == 0:
            return FeatureContext(candidate_rows, features, {})

        user_rows = np.asarray(user_rows, dtype=np.int64)
        user_weights = np.asarray(user_weights, dtype=np.float32)

        channels = self.engine.channel_outputs(user_rows, user_weights)
        rank_maps: dict[str, dict[int, int]] = {}
        score_maps: dict[str, dict[int, float]] = {}
        for name, (idx, scores) in channels.items():
            rank_maps[name] = {int(v): r for r, v in enumerate(idx)}
            score_maps[name] = {int(v): float(s) for v, s in zip(idx, scores)}

        col = {name: i for i, name in enumerate(FEATURE_NAMES)}
        hit_counts = np.zeros(n_cand, dtype=np.int32)
        for name in CHANNEL_ORDER:
            key = "collab" if name == "collab_itemknn" else name
            score_col, rank_col = col[f"{key}_score"], col[f"{key}_rank"]
            ranks, scores = rank_maps[name], score_maps[name]
            for i, item in enumerate(candidate_rows):
                item = int(item)
                if item in ranks:
                    features[i, score_col] = scores[item]
                    features[i, rank_col] = float(ranks[item])
                    hit_counts[i] += 1
                else:
                    features[i, rank_col] = ABSENT_RANK
        features[:, col["n_channels"]] = hit_counts
        features[:, col["merge_rank"]] = np.arange(n_cand, dtype=np.float32)

        # ---- 用户×物品 ----
        if len(user_rows):
            vector = (self.emb[user_rows] * user_weights[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                features[:, col["content_cosine"]] = (
                    self.emb[candidate_rows] @ (vector / norm)
                )
            user_genre_weight = np.asarray(
                self.item_genre[user_rows].T @ user_weights
            ).ravel().astype(np.float32)                       # [n_genres]
            user_genre_binary = (user_genre_weight > 0).astype(np.float32)
            user_genre_count = float(user_genre_binary.sum())
            overlap = np.asarray(
                self.item_genre[candidate_rows] @ user_genre_binary
            ).ravel().astype(np.float32)
            union = self.item_genre_count[candidate_rows] + user_genre_count - overlap
            features[:, col["genre_overlap"]] = overlap
            features[:, col["genre_jaccard"]] = np.divide(
                overlap, union, out=np.zeros_like(overlap), where=union > 0
            )
            total_weight = float(user_weights.sum())
            affinity = np.asarray(
                self.item_genre[candidate_rows] @ user_genre_weight
            ).ravel().astype(np.float32)
            features[:, col["user_genre_affinity"]] = (
                affinity / total_weight if total_weight > 0 else 0.0
            )
            # 发行商亲和:该物品的发行商在你的库里占多少权重
            pub_weight = np.zeros(self.n_pubs, dtype=np.float32)
            user_pubs = self.item_pub[user_rows]
            np.add.at(pub_weight, user_pubs[user_pubs >= 0],
                      user_weights[user_pubs >= 0])
            cand_pubs = self.item_pub[candidate_rows]
            valid = cand_pubs >= 0
            features[valid, col["user_publisher_affinity"]] = (
                pub_weight[cand_pubs[valid]] / total_weight if total_weight > 0 else 0.0
            )
            features[:, col["user_size_log"]] = np.log1p(len(user_rows))

        # ---- 物品静态 + 训练集热度 ----
        static_cols = [
            col["bayes_rating"], col["positive_ratio"], col["log_owners"],
            col["metacritic_norm"], col["price_norm"], col["release_year_norm"],
            col["dlc_count_norm"],
        ]
        features[:, static_cols] = self.item_static[candidate_rows]
        features[:, col["train_popularity"]] = self.train_popularity[candidate_rows]
        return FeatureContext(candidate_rows, features, rank_maps)


class Ranker:
    """线上排序入口:召回 -> 特征 -> LightGBM 打分 -> top-k。

    模型由 `scripts/train_ranker.py` 产出(`data/recommend/ranker.txt`)。
    """

    def __init__(self, model, builder: FeatureBuilder, engine: RecallEngine) -> None:
        self.model = model
        self.builder = builder
        self.engine = engine

    def rank(
        self, user_rows: np.ndarray, user_weights: np.ndarray, *, top_k: int = 5
    ) -> list[Candidate]:
        candidates = self.engine.recall_rows(user_rows, user_weights, owned_rows=user_rows)
        if not candidates:
            return []
        rows = np.array([c.cat_idx for c in candidates], dtype=np.int64)
        context = self.builder.build(user_rows, user_weights, rows)
        scores = self.model.predict(context.features)
        order = np.argsort(-scores)[:top_k]
        return [
            Candidate(
                cat_idx=int(rows[i]),
                score=float(scores[i]),
                channels=candidates[i].channels,
            )
            for i in order
        ]
