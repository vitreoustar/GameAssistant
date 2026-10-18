"""方向 A:混合(hybrid)推荐算法。

子模块规划(见 docs/ROADMAP.md §3):
  * `features.py` —— 纯函数(评价标签 / 销量 / 时长分档 / 年份),无第三方依赖
  * `dataset.py`  —— A1.2 产物的加载入口(mmap / numpy 视图)
  * `recall.py`   —— A3 多路召回(内容向量 / ItemKNN / 热门 / 发行商 / 社区)
  * `rank.py`     —— A4 LightGBM LTR 排序
  * `service.py`  —— A5 对外统一入口 `recommend(...)`
"""
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

__all__ = [
    "DESC_LIMIT",
    "MIN_RATING_VOTES",
    "bayesian_rating",
    "build_item_text",
    "owners_lower_bound",
    "playtime_bucket",
    "positive_ratio",
    "rating_text",
    "release_year",
]
