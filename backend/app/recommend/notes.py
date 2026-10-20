"""把推荐结果的诊断信息(`matched_count` / `relaxed` / `no_match`)转成给用户看的话。

**为什么单独成一个模块、且只依赖标准库**:

1. 这些判断(命中了几款、要不要补齐、要不要劝用户换方向)是产品里最容易被粉饰的
   地方,必须能被快速、独立地测。塞进 `graph.py` 里会被 `sentence-transformers`
   的导入拖成 14 秒的单文件测试。
2. **事实由代码算、由代码说**,LLM 只负责把开场白写得自然。否则模型很容易把
   "3/5 命中"写成"都符合你的口味",而这正是最误导用户的地方。
"""
from __future__ import annotations

from typing import Sequence


def constraint_note(result, unmapped: Sequence[str] = ()) -> str:
    """返回一句确定性的实话;没有要说的时候返回空串。

    `result` 是 `app.recommend.service.RecommendationResult`(鸭子类型,避免循环导入);
    `unmapped` 是用户说了、但词表里没有对应项的类型词 —— 它属于
    `PreferenceState` 而不是推荐结果,所以单独传进来。
    """
    terms = "、".join(result.prefer)
    extra = [t for t in unmapped if isinstance(t, str) and t.strip()]

    if result.no_match and terms:
        parts = [
            f"说实话,按「{terms}」筛了一遍你库里的 {result.pool_size} 个候选,一个都没对上。"
        ]
        if extra:
            parts.append(
                f"另外「{'、'.join(extra)}」这类标签我的数据里没有,"
                "只能按类型和玩法这些大类筛。"
            )
        parts.append("下面这几款是没按你的条件挑的、但和你游戏库最接近的,要不要先看看?")
        return " ".join(parts)

    if result.relaxed and terms:
        note = (
            f"严格符合「{terms}」的只有 {result.matched_count} 款,"
            f"剩下 {len(result.recommendations) - result.matched_count} 款是我按你的库补的相近选择。"
        )
        if extra:
            note += f"你说的「{'、'.join(extra)}」我没有对应标签,没法按它筛。"
        return note

    if result.exhausted:
        return "能推的我基本都推过一轮了,要不你换个方向,或者放宽一点条件?"

    # 说了偏好、但一个词都对不上词表 —— 这时候 `prefer` 是空的,如果不单独报一句,
    # 用户会以为自己的要求被采纳了,实际完全没影响排序。必须主动说明。
    if not terms and extra:
        return (
            f"「{'、'.join(extra)}」这类标签我的数据里没有(Steam 只分了 Action / RPG / "
            "Strategy / Co-op 这些大类,没有更细的主题标签),所以没法按它筛。"
            "下面还是按你的游戏库挑的,你看看方向对不对?"
        )
    return ""
