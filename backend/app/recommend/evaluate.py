"""方向 A 的离线评测:leave-one-out 切分 + 排序指标。

**为什么必须按用户切分**:同一用户的物品之间高度相关。若随机切分交互对,
"同用户的其他物品"会同时出现在训练和评测两侧,指标会虚高到没有参考价值。
所以这里按用户留一:每个用户藏起 1 个物品,用剩下的当画像去把它找回来。

**为什么绝对数字会偏乐观(必须写进文档)**:这份数据集是"玩过 ≥10 分钟"的
列表(见 ROADMAP §3-A1.1 的坑 2),未观测不等于负样本(MNAR)。因此绝对 Recall
不能当作线上预期,只能用于**算法/配置之间的相对比较**。

**泄漏边界**:内容通道只用物品侧向量,不碰交互 → 嵌入配置的对比是干净的。
协同通道的 item-item 矩阵是在全量交互上算的(含被藏起来的那条边),属于业界
常见做法,但会略微抬高它的分数 —— 跨通道比较时要记得这一点。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LooSplit:
    """按用户留一的切分结果。

    `train_user_item` 的**数据部分是 playtime_log 权重**(不是 1):
    用户画像向量要用它加权,协同通道也要用它;需要二值计数的地方(热门基线)
    用 `train_pairs` 现算,避免两种语义混在一个矩阵里。
    """

    users: np.ndarray            # 参与评测的用户 id
    test_game: np.ndarray        # 与 users 一一对应的被藏起的物品
    test_weight: np.ndarray      # 该物品的 playtime_log(A4 用它定相关性等级)
    test_pairs: np.ndarray       # [n_test, 2] (user_idx, game_idx)
    train_pairs: np.ndarray      # [n_train, 2] 其余交互
    train_user_item: sp.csr_matrix  # [n_users, n_items],值为 playtime_log
    n_users: int
    n_items: int


def leave_one_out_split(
    interactions, *, min_items: int = 10, seed: int = 0, n_items: int | None = None
) -> LooSplit:
    """按用户留一切分(每个合格用户藏起 1 个物品)。

    `train_user_item` **只含训练边**:评测时要把用户"已经玩过的"排除掉,
    如果混进测试边,目标物品会先被判为负样本再被找回来,等于自己泄题给自己。

    `n_items` 显式传入时以它为准。这是必要的:交互里的 `game_idx` 最大取值
    未必等于目录大小(有物品没有交互),而 A2 的矩阵都在目录空间,靠
    `max(game_idx)+1` 推断会静默算错维度。
    """
    user = interactions["user_idx"].to_numpy()
    game = interactions["game_idx"].to_numpy()
    weight = interactions["playtime_log"].to_numpy(dtype=np.float32)
    n_users = int(user.max()) + 1
    if n_items is None:
        n_items = int(game.max()) + 1

    counts = np.bincount(user, minlength=n_users)
    eligible = counts >= min_items

    rng = np.random.default_rng(seed)
    # 每条边一个随机键,同一用户内键最小的成为测试边(等价于无放回随机抽一条)
    key = rng.random(len(user))
    order = np.lexsort((key, user))
    user_s, game_s, weight_s = user[order], game[order], weight[order]

    first = np.ones(len(user_s), dtype=bool)
    first[1:] = user_s[1:] != user_s[:-1]
    is_test = first & eligible[user_s]

    train_u, train_g, train_w = user_s[~is_test], game_s[~is_test], weight_s[~is_test]
    test_u, test_g = user_s[is_test], game_s[is_test]

    train_user_item = sp.csr_matrix(
        (train_w, (train_u, train_g)), shape=(n_users, n_items)
    )
    return LooSplit(
        users=test_u,
        test_game=test_g,
        test_weight=weight_s[is_test],
        test_pairs=np.stack([test_u, test_g], axis=1),
        train_pairs=np.stack([train_u, train_g], axis=1),
        train_user_item=train_user_item,
        n_users=n_users,
        n_items=n_items,
    )


def evaluate_scores(
    score_fn,
    split: LooSplit,
    *,
    ks: tuple[int, ...] = (10, 20, 50),
    chunk: int = 1024,
) -> dict:
    """对打分函数做分块评测,返回 Recall@K / NDCG@K / MRR / Coverage@K。

    `score_fn(rows) -> [len(rows), n_items]` 只负责产出原始分数;函数内部统一把
    每个用户自己的训练物品置为 `-inf`,保证任何通道都不会"推荐已经玩过的"。

    分块的意义:7,495 × 11,444 的打分矩阵一次性物化要 343 MB;按 1024 行分块后
    峰值只有 ~47 MB,而且 `(scores > target)` 得到的就是精确名次,不必真的排序。
    """
    user_rows, test_game = split.users, split.test_game
    n_users, n_items = len(user_rows), split.n_items
    max_k = max(ks)
    ranks = np.empty(n_users, dtype=np.int64)
    topk_items = np.zeros((n_users, max_k), dtype=np.int32)
    indptr, indices = split.train_user_item.indptr, split.train_user_item.indices

    for start in range(0, n_users, chunk):
        stop = min(start + chunk, n_users)
        rows = np.arange(start, stop)
        scores = np.asarray(score_fn(rows), dtype=np.float32)
        if scores.shape != (stop - start, n_items):
            raise ValueError(
                f"score_fn 返回 {scores.shape},期望 {(stop - start, n_items)}"
            )
        for local, u in enumerate(user_rows[rows]):
            scores[local, indices[indptr[u] : indptr[u + 1]]] = -np.inf

        target = scores[np.arange(stop - start), test_game[rows]]
        ranks[rows] = (scores > target[:, None]).sum(axis=1)
        topk_items[rows] = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]

    metrics: dict = {"n_users": int(n_users)}
    for k in ks:
        hit = ranks < k
        metrics[f"recall@{k}"] = float(hit.mean())
        metrics[f"ndcg@{k}"] = float(
            np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean()
        )
        metrics[f"coverage@{k}"] = float(len(np.unique(topk_items[:, :k])) / n_items)
    metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def popularity_bucket_of(split: LooSplit, *, n_buckets: int = 10) -> np.ndarray:
    """按训练集热度把物品等频分桶(桶内物品热度接近)。

    用途见 `evaluate_bucketed`:在**同一热度桶内**挑负样本,把"热度"这个最容易
    拿分的信号控制掉,剩下的差异才反映真正的个性化能力。
    """
    counts = np.bincount(split.train_pairs[:, 1], minlength=split.n_items)
    order = np.argsort(counts, kind="stable")
    bucket = np.empty(split.n_items, dtype=np.int8)
    bucket[order] = (np.arange(split.n_items) * n_buckets) // split.n_items
    return bucket


def evaluate_bucketed(
    score_fn,
    split: LooSplit,
    *,
    n_neg: int = 100,
    ks: tuple[int, ...] = (10, 20),
    n_buckets: int = 10,
    seed: int = 0,
    chunk: int = 1024,
) -> dict:
    """**热度匹配**的候选集评测:每个测试物品配 `n_neg` 个同热度桶负样本。

    为什么要多这一套协议:全库排序下"热门"基线 recall@20 能到 25%,而内容通道
    只有 0.45%(约等于随机)—— 因为随机留一的目标物品本身就大概率是热门游戏,
    这个协议主要在考"能不能猜中热门",而不是"懂不懂这个用户"。

    把负样本限制在同一热度桶内,热门先验拿不到任何优势(桶内热度一样),
    于是指标衡量的是**个性化**:随机命中率约 `k / (n_neg + 1)`。
    """
    max_k = max(ks)
    bucket = popularity_bucket_of(split, n_buckets=n_buckets)
    bucket_items = [np.flatnonzero(bucket == b) for b in range(n_buckets)]

    rng = np.random.default_rng(seed)
    hits = {k: np.zeros(len(split.users), dtype=bool) for k in ks}
    indptr, indices = split.train_user_item.indptr, split.train_user_item.indices
    owned_mask = np.zeros(split.n_items, dtype=bool)

    for start in range(0, len(split.users), chunk):
        stop = min(start + chunk, len(split.users))
        rows = np.arange(start, stop)
        scores = np.asarray(score_fn(rows), dtype=np.float32)
        for local, u in enumerate(split.users[rows]):
            owned_mask.fill(False)
            owned_mask[indices[indptr[u] : indptr[u + 1]]] = True
            target = split.test_game[start + local]
            pool = bucket_items[bucket[target]]
            pool = pool[~owned_mask[pool]]
            size = min(n_neg, len(pool))
            negatives = rng.choice(pool, size=size, replace=False)
            candidates = np.concatenate(([target], negatives))
            cand_scores = scores[local, candidates]
            rank = int((cand_scores[1:] > cand_scores[0]).sum())
            for k in ks:
                hits[k][start + local] = rank < k

    metrics: dict = {"n_users": int(len(split.users)), "n_neg": int(n_neg),
                     "chance_recall@10": 10 / (n_neg + 1)}
    for k in ks:
        metrics[f"recall@{k}"] = float(hits[k].mean())
    metrics["lift_over_chance@10"] = float(
        metrics["recall@10"] / (10 / (n_neg + 1))
    ) if 10 in ks else float("nan")
    return metrics


def make_content_score_fn(embeddings: np.ndarray, split: LooSplit):
    """内容通道:用户画像向量(训练物品的 playtime_log 加权平均)与全库点积。

    **必须用训练边重建用户向量**。A2 预计算的 `user_emb.npy` 是用全部交互算的
    (含被藏起来的那条),直接拿来评测就是泄题 —— 那份向量只服务线上。
    """
    def score_fn(rows: np.ndarray) -> np.ndarray:
        block = split.train_user_item[rows]                  # [c, n_items] 含权重
        vectors = np.asarray(block @ embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms) @ embeddings.T

    return score_fn


def make_collab_score_fn(item_sim: sp.csr_matrix, split: LooSplit):
    """协同通道:用户训练物品那几行的 item-item 相似度加权求和。

    `item_sim` 是 `[n_items, n_items]` 的稀疏相似度 TopK 展开,于是
    `train_user_item @ item_sim` 一次矩阵乘就给所有用户打了分。
    """
    def score_fn(rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            (split.train_user_item[rows] @ item_sim).todense(), dtype=np.float32
        )

    return score_fn


def make_popularity_score_fn(split: LooSplit):
    """热门基线:只用**训练边**做二值计数。

    用全量统计会把测试边算进去,目标物品的热度被自己抬高、基线虚高;
    用加权计数又会让基线变成"时长榜"而不是"热度榜"。
    """
    counts = np.bincount(
        split.train_pairs[:, 1], minlength=split.n_items
    ).astype(np.float32)

    def score_fn(rows: np.ndarray) -> np.ndarray:
        return np.tile(counts, (len(rows), 1))

    return score_fn


def make_random_score_fn(n_items: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def score_fn(rows: np.ndarray) -> np.ndarray:
        return rng.random((len(rows), n_items), dtype=np.float32)

    return score_fn


def topk_to_sparse(
    topk_idx: np.ndarray, topk_score: np.ndarray, n_items: int
) -> sp.csr_matrix:
    """把 `[N, K]` 的 TopK 邻居展开成 `[N, N]` 稀疏相似度矩阵(-1 视为缺失)。"""
    valid = (topk_idx >= 0).ravel()
    rows = np.repeat(np.arange(topk_idx.shape[0]), topk_idx.shape[1])[valid]
    cols = topk_idx.ravel()[valid]
    data = topk_score.ravel()[valid]
    return sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
