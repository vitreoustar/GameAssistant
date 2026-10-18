"""推荐索引的加载与在线数组计算(A2 产物)。

产物由 `scripts/build_recommend_index.py` 生成,放在 `backend/data/recommend/`:

| 文件 | 形状 | 说明 |
|---|---|---|
| `catalog.parquet` | N 行 | `cat_idx -> appid` + 全部结构化特征与标志位 |
| `content_emb.npy` | `[N, 384]` float32 | 内容向量,L2 已归一化;无文本的物品为零向量 |
| `content_topk.npz` | `[N, K]` int32 + float32 | 内容相似邻居(`indices` / `scores`) |
| `collab_topk.npz` | `[N, K]` int32 + float32 | 协同共现邻居 |
| `user_emb.npy` | `[U, 384]` float32 | 训练集用户的画像向量(离线预计算) |
| `user_item_csr.npz` | `[N, U]` sparse | 用户-物品交互(playtime_log 加权) |
| `manifest.json` | — | 形状 / 构建参数 / 校验结果 |

**在线路径只做 mmap + 查表 + 点积**:不加载 sentence-transformers、不做 embedding 推理 ——
这就是方向 A "降算力" 的核心。大数组用 `mmap_mode="r"` 打开,常驻内存由 OS 页缓存管理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTENT_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 的输出维度

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_INDEX_DIR = DATA_DIR / "recommend"

CATALOG_FILE = "catalog.parquet"
CONTENT_EMB_FILE = "content_emb.npy"
CONTENT_TOPK_FILE = "content_topk.npz"
COLLAB_TOPK_FILE = "collab_topk.npz"
USER_EMB_FILE = "user_emb.npy"
USER_ITEM_FILE = "user_item_csr.npz"
MANIFEST_FILE = "manifest.json"


def cat_rows_of(cat_appids: np.ndarray, appids: np.ndarray) -> np.ndarray:
    """appid -> cat_idx 的**向量化**查找(找不到返回 -1)。

    用 `searchsorted` 而不是 Python dict:dict 不支持数组下标,而这里有 84 万次查找,
    循环版本会慢两个数量级。`cat_appids` 由 `np.union1d` 产生,天然有序。
    """
    pos = np.searchsorted(cat_appids, appids)
    pos = np.clip(pos, 0, len(cat_appids) - 1)
    return np.where(cat_appids[pos] == appids, pos, -1).astype(np.int64)


def topk_from_dense(
    scores: np.ndarray, k: int, *, exclude_self_offset: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """从 `[rows, n]` 的相似度矩阵按行取 top-k,返回 `(indices, values)` 降序。

    用 `argpartition` 而不是全排序:在 11k 列的规模下 O(n) 选择比 O(n log n) 快一个量级,
    之后只对选中的 k 个排序。

    `exclude_self_offset` 用于分块计算:传入该块起始的行下标,函数会把每行的
    "自身列"(`row + offset`)置为 -inf,避免物品把自己当成最近邻。
    """
    if scores.ndim != 2:
        raise ValueError(f"期望二维相似度矩阵,收到 shape={scores.shape}")
    rows, n = scores.shape
    k = max(1, min(k, n - (1 if exclude_self_offset is not None else 0)))

    work = np.nan_to_num(np.array(scores, dtype=np.float32, copy=True), nan=-np.inf)
    if exclude_self_offset is not None:
        work[np.arange(rows), np.arange(rows) + exclude_self_offset] = -np.inf

    part = np.argpartition(-work, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(work, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    indices = np.take_along_axis(part, order, axis=1).astype(np.int32)
    values = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
    return indices, values


def user_vector(
    item_emb: np.ndarray, item_rows: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """按权重对物品内容向量加权求和并 L2 归一化 —— 用户的"口味向量"。

    只累加有内容向量的物品:无文本的物品是零向量,天然不贡献任何方向。
    若某用户的物品全都没有内容向量,返回零向量(调用方需要能识别这种冷启动)。
    """
    if len(item_rows) != len(weights):
        raise ValueError("item_rows 与 weights 长度不一致")
    vector = np.zeros(item_emb.shape[1], dtype=np.float32)
    if len(item_rows) == 0:
        return vector
    vector = (item_emb[item_rows] * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)


def content_topk_from_embeddings(
    embeddings: np.ndarray, has_content: np.ndarray, k: int, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """内容向量的 item-item 余弦 TopK(只在有文本的物品之间算)。

    零向量与任何向量的余弦都是 0;若把无文本物品也放进来,它们会以 0 分挤掉
    真正的邻居,所以先筛掉。

    分块计算(`[chunk, d] @ [d, n_valid]`),峰值内存只有几十 MB,
    不必物化 11k×11k 的完整相似度矩阵。

    放在这里而不是构建脚本里,是因为 **A6 的嵌入配置对比必须重算它** ——
    换了 embedding 却不重算 TopK,比的就是"半新半旧"的管线。
    """
    n = embeddings.shape[0]
    idx_out = np.full((n, k), -1, dtype=np.int32)
    score_out = np.zeros((n, k), dtype=np.float32)

    valid = np.flatnonzero(has_content)
    valid_emb = embeddings[valid]
    for start in range(0, len(valid), chunk):
        stop = min(start + chunk, len(valid))
        scores = valid_emb[start:stop] @ valid_emb.T
        local_idx, local_val = topk_from_dense(scores, k, exclude_self_offset=start)
        idx_out[valid[start:stop]] = valid[local_idx]
        score_out[valid[start:stop]] = local_val
    return idx_out, score_out


@dataclass
class RecommendIndex:
    """一次性加载全部 A2 产物;大数组用 mmap,避免启动时读满内存。"""

    catalog: "object"                     # pandas.DataFrame
    content_emb: np.ndarray               # [N, 384] float32
    content_topk_idx: np.ndarray          # [N, K] int32
    content_topk_score: np.ndarray        # [N, K] float32
    collab_topk_idx: np.ndarray           # [N, K] int32
    collab_topk_score: np.ndarray         # [N, K] float32
    user_emb: np.ndarray                  # [U, 384] float32
    user_item: "object"                   # scipy.sparse.csr_matrix [N, U]
    manifest: dict
    path: Path
    _appid_to_row: dict = field(default_factory=dict, repr=False)

    @property
    def n_items(self) -> int:
        return int(len(self.catalog))

    @property
    def appids(self) -> np.ndarray:
        return self.catalog["appid"].to_numpy()

    def row_of(self, appid: int) -> int:
        """appid -> cat_idx;找不到返回 -1。"""
        return int(self._appid_to_row.get(int(appid), -1))

    def rows_of(self, appids) -> np.ndarray:
        return np.array([self.row_of(a) for a in appids], dtype=np.int64)

    def vectors_for(self, appids) -> np.ndarray:
        """取若干 appid 的内容向量(调用方自行处理 -1 / 零向量)。"""
        rows = self.rows_of(appids)
        rows = rows[rows >= 0]
        return self.content_emb[rows]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_DIR) -> "RecommendIndex":
        import json

        import pandas as pd
        import scipy.sparse as sp

        path = Path(path)
        required = [
            CATALOG_FILE, CONTENT_EMB_FILE, CONTENT_TOPK_FILE,
            COLLAB_TOPK_FILE, USER_EMB_FILE, USER_ITEM_FILE, MANIFEST_FILE,
        ]
        missing = [f for f in required if not (path / f).exists()]
        if missing:
            raise RuntimeError(
                f"推荐索引不完整,缺少 {missing};请先运行 scripts/build_recommend_index.py"
            )

        catalog = pd.read_parquet(path / CATALOG_FILE)
        content_topk = np.load(path / CONTENT_TOPK_FILE)
        collab_topk = np.load(path / COLLAB_TOPK_FILE)
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))

        appids = catalog["appid"].to_numpy()
        return cls(
            catalog=catalog,
            content_emb=np.load(path / CONTENT_EMB_FILE, mmap_mode="r"),
            content_topk_idx=content_topk["indices"],
            content_topk_score=content_topk["scores"],
            collab_topk_idx=collab_topk["indices"],
            collab_topk_score=collab_topk["scores"],
            user_emb=np.load(path / USER_EMB_FILE, mmap_mode="r"),
            user_item=sp.load_npz(path / USER_ITEM_FILE),
            manifest=manifest,
            path=path,
            _appid_to_row={int(a): i for i, a in enumerate(appids)},
        )
