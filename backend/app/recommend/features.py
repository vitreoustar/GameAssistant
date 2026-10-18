"""纯函数:好评率标签、销量下界、时长分档。

放在这里而不是 `rag/store.py`,是因为 rag 侧依赖 chromadb / sentence-transformers,
而推荐算法的离线建表与数据管线不应该被迫加载 torch。
`rag/store.py` 反过来从这里导入 `rating_text`,保证两边口径一致。
"""
from __future__ import annotations

import re

# 好评票数低于此值时不给评价标签(样本太少,百分比不可信)
MIN_RATING_VOTES = 50

# 物品文本里描述的截断长度
DESC_LIMIT = 800


def build_item_text(
    name: str, genres: str, categories: str, description: str = ""
) -> str:
    """物品侧的规范文本模板。

    **这是唯一实现**:`scripts/build_corpus.py`(喂给 ChromaDB 的语料)与
    `scripts/build_recommend_index.py`(A2 的内容向量)都调用它,保证两边文本
    逐字节一致 —— 否则 A6 拿新推荐器和现有 RAG 基线对比时,差异里会混进
    "文本模板不同"这个混淆因素。

    Publishers 之类的结构化字段**故意不放进文本**,而是作为 A4 排序器的显式特征:
    文本向量负责语义,结构化特征负责精确信号,混在一起反而不好归因。
    """
    parts = [name]
    if genres:
        parts.append(f"Genres: {genres}")
    if categories:
        parts.append(f"Categories: {categories}")
    if description:
        parts.append(description[:DESC_LIMIT])
    return "\n".join(parts)


def bayesian_rating(
    positive: int, negative: int, prior_mean: float, prior_strength: float = 50.0
) -> float:
    """好评率的贝叶斯平滑:`(pos + C·m) / (pos + neg + C)`。

    直接用好评率会让"1 好评 0 差评 = 100%"的冷门物品排在最前,`prior_strength`
    就是往全局均值 `prior_mean` 拉回多少票。票数越多,先验影响越小。
    """
    pos = positive or 0
    neg = negative or 0
    return (pos + prior_strength * prior_mean) / (pos + neg + prior_strength)


def rating_text(positive: int, negative: int) -> str:
    """Steam 好评率 -> 中文评价标签。"""
    total = (positive or 0) + (negative or 0)
    if total < MIN_RATING_VOTES:
        return "评价较少"
    pct = (positive or 0) / total * 100
    if pct >= 95:
        return "好评如潮"
    if pct >= 80:
        return "特别好评"
    if pct >= 70:
        return "多半好评"
    if pct >= 40:
        return "褒贬不一"
    if pct >= 20:
        return "多半差评"
    return "差评如潮"


def positive_ratio(positive: int, negative: int) -> float | None:
    """好评率(0~1)。没有票数时返回 None,交给下游用全局均值填充,而不是伪造成 0。"""
    total = (positive or 0) + (negative or 0)
    if total <= 0:
        return None
    return positive / total


def owners_lower_bound(value: object) -> int:
    """把 '500000 - 1000000' 这样的销量区间解析成下界整数;解析失败返回 0。

    取**下界**是有意的保守选择:上界容易被区间跨度放大,只适合做粗分位。

    入参声明为 `object`,因为经 pandas left-join 之后缺失值是 `NaN`(float),
    而 `NaN` 是真值,`nan or ""` 不会兜住它 —— 这里显式只接受 str。
    """
    if not isinstance(value, str):
        return 0
    match = re.match(r"\s*([\d,]+)", value)
    if not match:
        return 0
    return int(match.group(1).replace(",", ""))


def release_year(value: object) -> int:
    """从 'Nov 16, 2018' / '1 Nov, 2000' 这类日期串里抠出 4 位年份;失败返回 0。

    两个数据源的日期格式不同,所以不做完整解析,只取年份当排序特征。
    同样只接受 str,理由见 `owners_lower_bound`。
    """
    if not isinstance(value, str):
        return 0
    match = re.search(r"(\d{4})", value)
    return int(match.group(1)) if match else 0


def playtime_bucket(minutes: float | None) -> int:
    """游玩时长分档,供 LTR 当相关性等级(label)。

    0 = 极短(<1h) / 1 = 浅尝(1~10h) / 2 = 投入(10~50h) / 3 = 深度(>=50h)

    分档而不是直接用分钟数,是因为 playtime 极长尾(p99.9 ≈ 3,669 小时),
    直接回归会被极端值带偏(见 docs/ROADMAP.md §3-A1.1 的坑 3)。
    """
    value = minutes or 0
    if value < 60:
        return 0
    if value < 600:
        return 1
    if value < 3000:
        return 2
    return 3
