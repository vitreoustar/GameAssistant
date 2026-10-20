"""对话级偏好状态 —— 让推荐器在多轮对话里"记得"上一轮说了什么。

## 为什么需要这一层

`Recommender.recommend()` 是**游戏库的纯函数**:同一个库、同一份离线资产,
永远得到同一批候选。单轮调用时这是优点(可复现、可评测),但放进多轮对话就是 bug:

    用户:推荐几个游戏
    助手:《A》《B》《C》《D》《E》
    用户:再推荐点别的
    助手:《A》《B》《C》《D》《E》   <- 一模一样

因为"别的"这两个字从来没有进入排序器。用户顺口说的"我更喜欢策略类"也只被
LLM 拿去写推荐理由了,没有影响**排序**。

这一层补上两个缺失的状态:

1. **去重集合** `shown` —— 已经展示过的 appid,下一轮从候选里剔除。
2. **偏好约束** `genres` / `avoid` —— 用户逐轮说出的类型偏好,累积成过滤器。

## 状态存在哪

由**前端回传**(`ChatContext.recommend`),服务端不持有会话。取舍:

* 好处:服务无状态、可水平扩展、后端重启不丢、不需要 TTL 清理。
* 代价:刷新页面后去重记录会重置(会重新看到早先推荐过的游戏)。

对单用户本地部署这是可接受的,写在这里以免以后被当成缺陷。

## 词表为什么是封闭的

Steam 的类型(catalog `genres`)只有 26 个,且高度集中在
Indie(7,248)/Action(5,470)/Adventure(4,644)/Casual/Simulation/Strategy/RPG,
**没有 Horror / Stealth / Roguelike 这类主题标签**。所以这里**不允许**把用户说的
任何词直接当过滤条件 —— `normalise_terms()` 会把词表外的词移进 `unmapped`,
由上层如实告诉用户"这个标签我这没有",而不是假装筛过了。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

MAX_TERMS = 8          # 单个维度(正向/负向)最多累积多少个词
MAX_TERM_CHARS = 24
MAX_SHOWN = 400        # 去重集合上限;召回池只有 200 条,再多也用不上
MAX_UNMAPPED = 4       # 只保留最近几个"说了但筛不了"的词,用于提示用户

# ---------------------------------------------------------------- 封闭词表

#: 来自 catalog `genres` 列。顺序按 Steam 习惯排列,不是按频率。
GENRE_VOCAB: tuple[str, ...] = (
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Casual", "Indie",
    "Racing", "Sports", "Massively Multiplayer", "Free To Play", "Early Access",
    "Violent", "Gore", "Nudity",
)

#: 来自 catalog `categories` 列里**有主题含义**的那部分。
#: 该列共 60 个值,大部分是技术/无障碍项(Stereo Sound、Adjustable Text Size…),
#: 这里只挑用户会主动提的玩法维度。
FEATURE_VOCAB: tuple[str, ...] = (
    "Single-player", "Multi-player", "Co-op", "Online Co-op", "PvP", "Online PvP",
    "Shared/Split Screen", "MMO", "Steam Workshop", "In-App Purchases",
)

#: 可执行的过滤词表 = 类型 + 玩法。**只有**这些词会进入 `genres` / `avoid`。
FILTER_VOCAB: tuple[str, ...] = GENRE_VOCAB + FEATURE_VOCAB

_VOCAB_BY_LOWER: dict[str, str] = {v.lower(): v for v in FILTER_VOCAB}

#: 无 LLM(或 LLM 返回不可用)时的中文关键词兜底。
#: CJK 词按子串匹配,ASCII 词按词边界匹配(否则 "act" 会匹配到 "action" 里的任意位置)。
LEXICON: dict[str, str] = {
    "动作": "Action", "action": "Action",
    "冒险": "Adventure", "adventure": "Adventure",
    "角色扮演": "RPG", "rpg": "RPG", "回合制": "RPG",
    "策略": "Strategy", "战略": "Strategy", "战棋": "Strategy", "strategy": "Strategy",
    "模拟": "Simulation", "经营": "Simulation", "建造": "Simulation",
    "种田": "Simulation", "simulation": "Simulation",
    "休闲": "Casual", "轻松": "Casual", "casual": "Casual",
    "独立": "Indie", "indie": "Indie",
    "赛车": "Racing", "竞速": "Racing", "racing": "Racing",
    "体育": "Sports", "运动": "Sports", "sports": "Sports",
    "多人": "Multi-player", "联机": "Multi-player", "线上": "Multi-player",
    "单机": "Single-player", "单人": "Single-player",
    "合作": "Co-op", "双人": "Co-op", "联机合作": "Co-op",
    "对战": "PvP", "竞技": "PvP", "pvp": "PvP",
    "免费": "Free To Play", "白嫖": "Free To Play",
    "抢先体验": "Early Access", "创意工坊": "Steam Workshop",
    "内购": "In-App Purchases", "氪金": "In-App Purchases",
}

#: 常见但词表里**没有对应值**的词。仅用于无 LLM 时把话说清楚
#: ("恐怖"确实没有对应标签,不是我筛漏了)。有 LLM 时由模型自己判断。
UNSUPPORTED_HINTS: tuple[str, ...] = (
    "恐怖", "惊悚", "潜行", "魂系", "肉鸽", "roguelike", "开放世界",
    "像素", "沙盒", "生存", "解谜", "音游", "galgame", "二次元", "武侠", "修仙",
)

_NEGATIONS: tuple[str, ...] = (
    "不要", "不想", "不喜欢", "不想要", "不太想", "别给", "别推荐", "别要",
    "排除", "除了", "讨厌", "无感", "不玩", "不感兴趣",
)
_NEGATION_WINDOW = 6  # 否定词只在匹配点前这么多字符内生效


# ---------------------------------------------------------------- 规范化

def canonical(term: str) -> str | None:
    """把任意写法的词映射到词表里的规范写法;不在词表里返回 `None`。"""
    if not isinstance(term, str):
        return None
    key = term.strip().lower()
    return _VOCAB_BY_LOWER.get(key)


def normalise_terms(terms: Any) -> tuple[list[str], list[str]]:
    """把模型/关键词给出的词分成 `(可执行, 不可执行)` 两组。

    可执行的那组已去重、长度受限、大小写归一;顺序保留(越早说的越重要,
    上层截断时优先保留)。
    """
    ok: list[str] = []
    bad: list[str] = []
    if isinstance(terms, str):
        terms = [terms]
    if not isinstance(terms, (list, tuple, set)):
        return ok, bad
    for raw in terms:
        if not isinstance(raw, str):
            continue
        text = raw.strip()[:MAX_TERM_CHARS]
        if not text:
            continue
        term = canonical(text)
        if term is None:
            if text not in bad:
                bad.append(text)
        elif term not in ok:
            ok.append(term)
    return ok[:MAX_TERMS], bad[:MAX_UNMAPPED]


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _find_all(haystack: str, needle: str) -> list[int]:
    """返回 `needle` 在 `haystack` 里的所有起点。

    ASCII 词用**仅限 ASCII 字母数字**的边界判断,不能用 `\\b`:Python 的 `\\w`
    包含中日韩字符,`\\brpg\\b` 在「推荐点rpg」里不会匹配(前一个字符"点"也算 word)。
    CJK 词按子串匹配。
    """
    if not needle:
        return []
    if _is_ascii(needle):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
        return [m.start() for m in re.finditer(pattern, haystack)]
    positions: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return positions
        positions.append(idx)
        start = idx + len(needle)
    return positions


def _negated(text: str, position: int) -> bool:
    window = text[max(0, position - _NEGATION_WINDOW):position]
    return any(marker in window for marker in _NEGATIONS)


def parse_message(message: str) -> dict[str, list[str]]:
    """无 LLM 时的确定性偏好抽取。返回 `{"genres": [], "avoid": [], "unmapped": []}`。

    只做"词表内关键词 + 否定词在前"这一件事。它不追求准确 —— 有 DeepSeek 时
    走 LLM 抽取,这里只保证"LLM 不可用时不是完全没反应"。
    """
    text = (message or "").lower()
    if not text.strip():
        return {"genres": [], "avoid": [], "unmapped": []}

    genres: list[str] = []
    avoid: list[str] = []
    for word, term in LEXICON.items():
        for position in _find_all(text, word.lower()):
            target = avoid if _negated(text, position) else genres
            if term not in target:
                target.append(term)
            break  # 同一个词出现多次只算一次

    # 同一个词不能既是正项又是负项:后者(更靠后的说法)不进,负向优先
    genres = [g for g in genres if g not in avoid]

    unmapped: list[str] = []
    for hint in UNSUPPORTED_HINTS:
        if _find_all(text, hint.lower()) and hint not in unmapped:
            unmapped.append(hint)

    return {"genres": genres[:MAX_TERMS], "avoid": avoid[:MAX_TERMS],
            "unmapped": unmapped[:MAX_UNMAPPED]}


def _as_list(value: Any) -> list:
    """把"可能是列表/可能是单个字符串"的外部输入统一成列表。

    直接 `list("abc")` 会得到 `['a','b','c']` —— 实测踩过:前端/模型偶尔把
    `unmapped` 传成字符串,结果每个字母都被当成了一个类型词。
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


# ---------------------------------------------------------------- 状态

@dataclass
class PreferenceState:
    """一次对话里累积下来的推荐约束。

    * `genres`  —— 用户明说想要的类型/玩法
    * `avoid`   —— 用户明说不要的
    * `unmapped`—— 用户说了、但词表里没有的词(只用于提示,不参与过滤)
    * `shown`   —— 已经展示过的 appid(去重)
    * `turn`    —— 已经推荐过几次,仅用于文案与调试
    """

    genres: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    shown: list[int] = field(default_factory=list)
    turn: int = 0

    # ---------------------------------------------------------- 序列化

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "PreferenceState":
        """从前端回传的字典恢复。**对脏数据完全容错** —— 这是外部输入。"""
        if not isinstance(raw, Mapping):
            return cls()
        genres, bad_g = normalise_terms(raw.get("genres"))
        avoid, bad_a = normalise_terms(raw.get("avoid"))
        # 与 `merge()` 同一条规则:同一个词出现在两边时以负向为准
        genres = [g for g in genres if g not in avoid]
        shown: list[int] = []
        for value in _as_list(raw.get("shown")):
            try:
                appid = int(value)
            except (TypeError, ValueError):
                continue
            if appid > 0 and appid not in shown:
                shown.append(appid)
        turn = raw.get("turn")
        try:
            turn = max(0, int(turn)) if turn is not None else 0
        except (TypeError, ValueError):
            turn = 0
        unmapped: list[str] = []
        for value in _as_list(raw.get("unmapped")) + bad_g + bad_a:
            if isinstance(value, str):
                text = value.strip()[:MAX_TERM_CHARS]
                if text and text not in unmapped:
                    unmapped.append(text)
        return cls(
            genres=genres,
            avoid=avoid,
            unmapped=unmapped[:MAX_UNMAPPED],
            shown=shown[:MAX_SHOWN],
            turn=turn,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "genres": list(self.genres),
            "avoid": list(self.avoid),
            "unmapped": list(self.unmapped),
            "shown": list(self.shown),
            "turn": self.turn,
        }

    # ---------------------------------------------------------- 累积

    def merge(self, delta: Mapping[str, Any] | None, *, restart: bool = False) -> "PreferenceState":
        """把本轮抽到的偏好并入状态,**返回新对象**(不改 self)。

        `restart=True` 表示用户推翻了之前的说法("我说错了,重新来"),
        此时清空三个偏好列表,但**保留 `shown`** —— 否则用户说一句"重新来"
        就会立刻看到上一轮刚看过的游戏,又变回原来的 bug。
        """
        genres = [] if restart else list(self.genres)
        avoid = [] if restart else list(self.avoid)
        unmapped = [] if restart else list(self.unmapped)

        if isinstance(delta, Mapping):
            new_genres, bad_g = normalise_terms(delta.get("genres"))
            new_avoid, bad_a = normalise_terms(delta.get("avoid"))
            for term in new_genres:
                if term not in genres:
                    genres.append(term)
            for term in new_avoid:
                if term not in avoid:
                    avoid.append(term)
            raw_unmapped = delta.get("unmapped")
            for term in bad_g + bad_a + [t for t in _as_list(raw_unmapped) if isinstance(t, str)]:
                text = term.strip()[:MAX_TERM_CHARS]
                if text and text not in unmapped:
                    unmapped.append(text)

        # 同一个词同时出现在两边时,以**负向**为准:用户说"不要策略"比
        # 早前说"喜欢策略"更晚、更明确,而且宁可少推也不要推错。
        genres = [g for g in genres if g not in avoid]

        return PreferenceState(
            genres=genres[:MAX_TERMS],
            avoid=avoid[:MAX_TERMS],
            unmapped=unmapped[:MAX_UNMAPPED],
            shown=list(self.shown),
            turn=self.turn,
        )

    def with_shown(self, appids: Sequence[int]) -> "PreferenceState":
        """记录本轮展示过的 appid,并让 `turn` +1。返回新对象。"""
        shown = list(self.shown)
        seen = set(shown)
        for appid in appids:
            try:
                value = int(appid)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in seen:
                seen.add(value)
                shown.append(value)
        return PreferenceState(
            genres=list(self.genres),
            avoid=list(self.avoid),
            unmapped=list(self.unmapped),
            shown=shown[-MAX_SHOWN:],
            turn=self.turn + 1,
        )

    # ---------------------------------------------------------- 便利

    def is_empty(self) -> bool:
        return not (self.genres or self.avoid or self.shown)


# ---------------------------------------------------------------- 词表匹配

def term_columns(terms: Sequence[str], vocab: Sequence[str] = FILTER_VOCAB) -> np.ndarray:
    """把词换成矩阵列号;词表外的词直接丢弃(不报错,调用方应已过滤过)。"""
    index = {v.lower(): i for i, v in enumerate(vocab)}
    cols = [index[t.lower()] for t in terms if isinstance(t, str) and t.lower() in index]
    return np.array(sorted(set(cols)), dtype=np.int64)


def compile_term_matrix(
    genres: Sequence[Any],
    categories: Sequence[Any],
    vocab: Sequence[str] = FILTER_VOCAB,
) -> np.ndarray:
    """把 catalog 的 `genres` + `categories` 预编译成 `[n_items, n_terms]` 布尔矩阵。

    一次构建(11,444 × 25 = 286 KB),之后每轮匹配只做一次列切片 + `any`,
    比逐行拆字符串快两个数量级 —— 而推荐是在请求路径上的。
    """
    index = {v.lower(): i for i, v in enumerate(vocab)}
    n = len(genres)
    matrix = np.zeros((n, len(vocab)), dtype=bool)
    for row in range(n):
        for column in (genres[row], categories[row]):
            if not isinstance(column, str):
                continue
            for token in column.split(","):
                col = index.get(token.strip().lower())
                if col is not None:
                    matrix[row, col] = True
    return matrix


def match_mask(matrix: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """`rows` 里哪些物品命中 `columns` 中的任一词。空列 → 全 False。"""
    rows = np.asarray(rows, dtype=np.int64)
    if len(columns) == 0 or len(rows) == 0:
        return np.zeros(len(rows), dtype=bool)
    return matrix[np.ix_(rows, np.asarray(columns, dtype=np.int64))].any(axis=1)
