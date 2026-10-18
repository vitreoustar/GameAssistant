"""A3.2 多路召回层:把若干**离线预计算**的信号合成一个候选集。

在线只做查表 + 点积 + bincount,不加载任何模型(A2 已实测:load 后 torch 未被 import)。

五路通道:

| 通道 | 信号 | 来源 |
|---|---|---|
| `content_user` | 用户画像向量 @ 内容向量 —— 整体口味 | `content_emb.npy` |
| `content_itemknn` | 库内物品的内容邻居 —— "因为你玩了 X" | `content_topk.npz` |
| `collab_itemknn` | 库内物品的协同邻居 —— "喜欢 X 的人也喜欢 Y" | `collab_topk.npz` |
| `quality` | 质量/热度先验 —— 冷启动兜底 | 调用方注入 |
| `publisher` | 同发行商 —— 系列/厂商偏好 | `catalog.parquet` |

**为什么两路 item-kNN 单独设通道**:用户向量是对整库的加权平均,库一大就趋近全局质心
(A2 实测:2,143 款的用户召回 Top-10 全是冷门物品)。item-kNN 走的是"逐个物品找邻居再累加",
不会质心塌缩,两者互补。

⚠️ **先验必须由调用方注入**,不要在这里现算:`quality` 通道在评测时必须传**训练集统计**,
否则会把测试边算进热度里(见 `evaluate.py` 的泄漏说明)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_QUOTAS: dict[str, int] = {
    "content_user": 20,
    "content_itemknn": 20,
    "collab_itemknn": 50,
    "quality": 20,
    "publisher": 20,
}
# 内容配额为什么是 20 而不是 50:实测扫描(7,495 用户,total=200)显示
#   content 50/50 -> recall@20 26.10% / cov@50 19.7% / cov@100 42.2%
#   content 20/20 -> recall@20 26.35% / cov@50 22.2% / cov@100 39.3%
#   content  0/0  -> recall@20 27.53% / cov@50 18.0% / cov@100 30.2%
# 20/20 在 recall@50(37.69% vs 37.16%)和 cov@50 上同时更好;继续降到 0 虽然 recall 更高,
# 但覆盖率掉得快 —— 而内容路是**唯一能覆盖冷物品**(无交互记录)的通道,不能砍掉。
CHANNEL_ORDER = ("content_user", "content_itemknn", "collab_itemknn", "quality", "publisher")


@dataclass(frozen=True)
class RecallConfig:
    """召回配置。

    `merge="quota"` 按路轮转取(每路都有代表,多样性好,推荐线上用);
    `merge="score"` 各路归一化后加权求和(排序更优,但容易被强通道垄断)。
    """

    quotas: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    merge: str = "quota"
    total: int = 200
    knn_depth: int = 100  # 每路从 TopK 矩阵里取前多少个邻居


@dataclass
class Candidate:
    cat_idx: int
    score: float
    channels: tuple[str, ...]


class RecallEngine:
    """一次构建(预建倒排/缓存),多次查询。

    评测要对 7,495 个用户各查一次,所以派生结构必须预建,不能每次重算。
    """

    def __init__(self, index, *, quality_scores: np.ndarray | None = None,
                 config: RecallConfig | None = None) -> None:
        self.index = index
        self.config = config or RecallConfig()
        self.n_items = index.n_items
        self.emb = np.asarray(index.content_emb)
        self.content_idx = index.content_topk_idx
        self.content_score = index.content_topk_score
        self.collab_idx = index.collab_topk_idx
        self.collab_score = index.collab_topk_score
        self.quality = (
            np.asarray(quality_scores, dtype=np.float32)
            if quality_scores is not None
            else np.zeros(self.n_items, dtype=np.float32)
        )
        # quality 通道的候选顺序只取决于先验,预先排好即可
        self._quality_rank = np.argsort(-self.quality)[: max(200, self.config.quotas.get("quality", 0) * 4)]
        self._pub_items, self._item_pubs = self._build_publisher_index()

    # ------------------------------------------------------------ 预建结构

    def _build_publisher_index(self):
        """发行商倒排:一个物品只取**第一个**发行商。

        简化是有意的 —— 多发行商串("Ubisoft;Reflections")的精确匹配价值有限,
        而取首个能把"同厂商"这个粗信号稳定地用起来。
        """
        pubs = self.index.catalog["publishers"].to_numpy()
        pub_id: dict[str, int] = {}
        item_pubs = np.full(self.n_items, -1, dtype=np.int32)
        for i, raw in enumerate(pubs):
            if not isinstance(raw, str) or not raw.strip():
                continue
            key = raw.split(";")[0].strip().lower()
            if not key:
                continue
            if key not in pub_id:
                pub_id[key] = len(pub_id)
            item_pubs[i] = pub_id[key]
        pub_items = [
            np.flatnonzero(item_pubs == pid) for pid in range(len(pub_id))
        ]
        return pub_items, item_pubs

    # ------------------------------------------------------------ 各通道

    def _content_user(self, rows, weights, cfg: RecallConfig) -> tuple[np.ndarray, np.ndarray]:
        if len(rows) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        vector = (self.emb[rows] * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        scores = self.emb @ (vector / norm)
        depth = min(cfg.knn_depth, self.n_items)
        idx = np.argpartition(-scores, depth - 1)[:depth]
        order = np.argsort(-scores[idx])
        idx = idx[order]
        return idx, scores[idx]

    def _itemknn(self, rows, weights, topk_idx, topk_score, cfg: RecallConfig) -> tuple[np.ndarray, np.ndarray]:
        """库内物品的 TopK 邻居按权重累加(`np.bincount` 加权,避免 Python 循环)。"""
        if len(rows) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        depth = min(cfg.knn_depth, topk_idx.shape[1])
        neighbours = topk_idx[rows, :depth]                 # [L, depth]
        scores = topk_score[rows, :depth]
        valid = neighbours >= 0
        if not valid.any():
            return np.empty(0, np.int64), np.empty(0, np.float32)
        contribution = np.where(valid, weights[:, None] * scores, 0.0)
        accumulated = np.bincount(
            neighbours[valid], weights=contribution[valid], minlength=self.n_items
        ).astype(np.float32)
        depth = min(depth, int(valid.sum(axis=1).sum()))
        idx = np.argpartition(-accumulated, depth - 1)[:depth]
        order = np.argsort(-accumulated[idx])
        idx = idx[order]
        return idx, accumulated[idx]

    def _quality_channel(self, owned: np.ndarray, cfg: RecallConfig) -> tuple[np.ndarray, np.ndarray]:
        idx = self._quality_rank
        if len(owned):
            idx = idx[~np.isin(idx, owned, assume_unique=False)]
        depth = min(cfg.quotas.get("quality", 20) * 4, len(idx))
        idx = idx[:depth]
        return idx, self.quality[idx]

    def _publisher_channel(self, rows: np.ndarray, cfg: RecallConfig) -> tuple[np.ndarray, np.ndarray]:
        if len(rows) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        pub_ids = self._item_pubs[rows]
        pub_ids = pub_ids[pub_ids >= 0]
        if len(pub_ids) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        counts = np.bincount(pub_ids)
        hits = np.zeros(self.n_items, dtype=np.float32)
        for pid in np.flatnonzero(counts):
            hits[self._pub_items[pid]] += counts[pid]
        depth = min(cfg.knn_depth, self.n_items)
        idx = np.argpartition(-hits, depth - 1)[:depth]
        order = np.argsort(-hits[idx])
        idx = idx[order]
        keep = hits[idx] > 0
        return idx[keep], hits[idx][keep]

    # ------------------------------------------------------------ 主入口

    def recall(
        self,
        library_appids,
        library_weights=None,
        *,
        owned_rows: np.ndarray | None = None,
    ) -> list[Candidate]:
        """对一份游戏库(按 appid)做多路召回。"""
        rows = self.index.rows_of(library_appids)
        if library_weights is not None and len(library_weights) != len(rows):
            raise ValueError("library_weights 必须与 library_appids 等长")
        keep = rows >= 0
        weights = None if library_weights is None else np.asarray(
            library_weights, dtype=np.float32
        )[keep]
        return self.recall_rows(rows[keep], weights, owned_rows=owned_rows)

    def channel_outputs(
        self, rows, weights=None, *, config: RecallConfig | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """返回每一路的 `(候选下标, 分数)`,不做融合。

        A4 的排序器要拿各路的原始分数与名次当特征,所以在这里暴露出来 ——
        **训练与线上调用的是同一个函数**,避免训练/服务特征口径不一致(训练-服务偏斜)。

        `config` 可**逐次覆盖**:它不走实例状态,否则"临时改配额跑消融"会因为
        共享可变状态而串台 —— 实测踩过,`collab_only` 与 `popular_only` 跑出了
        一模一样的数字。显式传参同时也让引擎可以被并发调用。
        """
        cfg = config or self.config
        rows, weights = self._prepare(rows, weights)
        owned_set = np.unique(rows)
        return {
            "content_user": self._content_user(rows, weights, cfg),
            "content_itemknn": self._itemknn(
                rows, weights, self.content_idx, self.content_score, cfg
            ),
            "collab_itemknn": self._itemknn(
                rows, weights, self.collab_idx, self.collab_score, cfg
            ),
            "quality": self._quality_channel(owned_set, cfg),
            "publisher": self._publisher_channel(rows, cfg),
        }

    @staticmethod
    def _prepare(rows, weights):
        rows = np.asarray(rows, dtype=np.int64)
        if weights is None:
            return rows, np.ones(len(rows), dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if len(weights) != len(rows):
            raise ValueError("weights 必须与 rows 等长")
        if weights.max(initial=0) > 30:  # 看起来是分钟数,取 log1p
            weights = np.log1p(weights)
        return rows, weights

    def recall_rows(
        self,
        rows: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        owned_rows: np.ndarray | None = None,
        config: RecallConfig | None = None,
    ) -> list[Candidate]:
        """核心入口:`rows` 直接是目录下标。

        评测里训练边本来就是 `cat_idx`(见 `evaluate.py`),走这里可以省掉
        "cat_idx -> appid -> cat_idx" 的来回转换。
        `weights` 默认等权;若传入的像是分钟数(最大值 > 30)会自动取 `log1p`。
        `config` 逐次覆盖(见 `channel_outputs` 的说明)。
        """
        cfg = config or self.config
        rows, weights = self._prepare(rows, weights)
        owned_set = np.unique(owned_rows if owned_rows is not None else rows)
        channels = self.channel_outputs(rows, weights, config=cfg)

        if cfg.merge == "score":
            return self._merge_by_score(channels, owned_set, cfg)
        return self._merge_by_quota(channels, owned_set, cfg)

    # ------------------------------------------------------------ 融合

    def _normalise(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        top = float(scores.max())
        return scores / top if top > 0 else scores

    def _merge_by_score(self, channels, owned_set, cfg: RecallConfig) -> list[Candidate]:
        """各路归一化到 [0,1] 后加权求和(权重 = 配额占比)。"""
        total_quota = sum(cfg.quotas.values()) or 1
        fused: dict[int, float] = {}
        provenance: dict[int, list[str]] = {}
        for name in CHANNEL_ORDER:
            idx, scores = channels[name]
            if len(idx) == 0:
                continue
            weight = cfg.quotas.get(name, 0) / total_quota
            for row, value in zip(idx, self._normalise(scores)):
                if row in owned_set:
                    continue
                fused[int(row)] = fused.get(int(row), 0.0) + weight * float(value)
                provenance.setdefault(int(row), []).append(name)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[: cfg.total]
        return [
            Candidate(cat_idx=row, score=score, channels=tuple(provenance[row]))
            for row, score in ranked
        ]

    def _merge_by_quota(self, channels, owned_set, cfg: RecallConfig) -> list[Candidate]:
        """按路轮转取(每路先取自己的配额,再依次补齐),保证每路都有代表。

        单纯按分数合并会让最强的那一路吃掉全部名额(实测 `content_user` 在
        大库上会挤掉 item-kNN);配额轮转是保证多样性的常规做法。
        """
        taken: dict[int, list[str]] = {}
        scores: dict[int, float] = {}
        for name in CHANNEL_ORDER:
            idx, vals = channels[name]
            quota = cfg.quotas.get(name, 0)
            added = 0
            for row, value in zip(idx, vals):
                if added >= quota:
                    break
                row = int(row)
                if row in owned_set:
                    continue
                if row in taken:
                    taken[row].append(name)
                    continue
                taken[row] = [name]
                scores[row] = float(value)
                added += 1

        # 还没满就按各路剩余分数补齐,避免配额小的路浪费名额。
        # ⚠️ 只从**配额非零**的通道补齐:否则 `only:X` 的消融实验会串台 ——
        # 实测过这个 bug,它让单通道评测看起来强了 20 倍。
        if len(taken) < cfg.total:
            pool: dict[int, float] = {}
            for name in CHANNEL_ORDER:
                if cfg.quotas.get(name, 0) <= 0:
                    continue
                idx, vals = channels[name]
                normalised = self._normalise(vals)
                for row, value in zip(idx, normalised):
                    row = int(row)
                    if row in owned_set or row in taken:
                        continue
                    pool[row] = max(pool.get(row, 0.0), float(value))
            for row, value in sorted(pool.items(), key=lambda kv: -kv[1]):
                if len(taken) >= cfg.total:
                    break
                taken[row] = ["fill"]
                scores[row] = value

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: cfg.total]
        return [
            Candidate(cat_idx=row, score=score, channels=tuple(taken[row]))
            for row, score in ranked
        ]
