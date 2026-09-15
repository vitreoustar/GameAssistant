"""LangGraph Agent:意图路由 -> (DLC 整理 | DLC 展开 | 游戏推荐 | 游戏评价 | 闲聊)。

StateGraph 结构:
    START -> route -> {dlc | dlc_detail | recommend | game_ask | chat} -> END
"""
from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import (
    DLC_SUMMARY_PROMPT,
    EXTRACT_GAME_PROMPT,
    GAME_ASK_PROMPT,
    RECOMMEND_REASONS_PROMPT,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
)
from app.config import settings
from app.rag import store as rag_store
from app.steam import client as steam_client
from app.steam.dlc import build_dlc_report


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    library: list[dict]
    intent: str
    dlc_report: dict | None
    profile: dict | None
    candidates: list[dict]
    game_analysis: dict | None
    game_candidates: list[dict]
    context_candidates: list[dict]


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0.3,
        timeout=30,
        max_retries=1,
    )


async def _llm_stream_text(messages: list[BaseMessage]) -> str:
    """用 astream 逐个 token 生成并拼接,便于上层捕获 token 流。"""
    if not settings.deepseek_api_key:
        raise ValueError("LLM 未配置")
    llm = get_llm()
    parts: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            parts.append(str(chunk.content))
    return "".join(parts)


# ---------- 路由 ----------

async def route_node(state: AgentState) -> dict:
    message = str(state["messages"][-1].content)
    if not settings.deepseek_api_key:
        return {"intent": _keyword_route(message)}
    try:
        llm = get_llm().bind(response_format={"type": "json_object"})
        resp = await llm.ainvoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=message),
            ]
        )
        intent = "chat"
        try:
            data = json.loads(resp.content)
            intent = data.get("intent", "chat")
        except (json.JSONDecodeError, AttributeError):
            pass
    except Exception:
        intent = _keyword_route(message)

    if intent not in {"dlc", "dlc_detail", "recommend", "game_ask", "chat"}:
        intent = "chat"
    return {"intent": intent}


def _keyword_route(message: str) -> str:
    """LLM 不可用时的关键词兜底路由。"""
    m = message.lower()
    if "dlc" in m:
        if any(k in m for k in ("展开", "更多", "全部", "详细", "还有", "完整", "有哪些", "多少")):
            return "dlc_detail"
        return "dlc"
    if re.search(r"第\s*[0-9一二三四五]+\s*[个款]", m):
        return "game_ask"
    if any(k in m for k in ("怎么样", "好不好玩", "好玩吗", "值得买", "值得玩", "评价", "如何")):
        return "game_ask"
    if any(k in m for k in ("推荐", "recommend", "喜欢", "好玩", "类似")):
        return "recommend"
    return "chat"


def _route_after_intent(state: AgentState) -> str:
    return state["intent"]


# ---------- DLC ----------

async def dlc_node(state: AgentState) -> dict:
    games = state.get("library") or []
    if not games:
        return {"messages": [AIMessage(content="先连接你的游戏库,我才能帮你整理 DLC 哦~")]}

    report = await build_dlc_report(games, top_games=20, max_dlc_per_game=8)
    summary = _summarize_dlc(report)
    try:
        summary = await _llm_stream_text(
            [
                SystemMessage(content=DLC_SUMMARY_PROMPT),
                HumanMessage(content=_dlc_context(report)),
            ]
        )
    except Exception:
        pass  # LLM 失败时退回确定性摘要

    return {"dlc_report": report, "messages": [AIMessage(content=summary)]}


def _dlc_context(report: dict) -> str:
    """把 DLC 报告转成自然语言上下文,避免 LLM 念出字段名。"""
    lines: list[str] = []
    if report.get("capped"):
        lines.append(f"分析范围:你最常玩的 {report['processed_games']} 款(总共 {report['total_games']} 款)")
    else:
        lines.append(f"分析范围:你的全部 {report['total_games']} 款游戏")
    lines.append(f"总共缺 {report['total_missing_dlc']} 个 DLC")
    price = report.get("total_missing_price_cents", 0)
    if price:
        lines.append(f"已列出的未拥有 DLC 合计 ${price / 100:.2f}")
    for r in report.get("rows", [])[:8]:
        flag = "(列表有截断)" if r.get("truncated") else ""
        names = "、".join(x["name"] for x in r["missing_dlc"][:3])
        lines.append(f"- {r['name']}:缺 {r['missing_count']} 个{flag};比如 {names}")
    return "\n".join(lines)


def _summarize_dlc(report: dict) -> str:
    """LLM 不可用时的确定性兜底摘要。"""
    n = report.get("total_missing_dlc", 0)
    price = report.get("total_missing_price_cents", 0)
    lines: list[str] = []
    if report.get("capped"):
        lines.append(f"先帮你看了最常玩的 {report['processed_games']}/{report['total_games']} 款游戏~")
    lines.append(f"这些游戏一共缺 {n} 个 DLC。")
    if price:
        lines.append(f"已列出的未拥有 DLC 合计约 ${price / 100:.2f}。")
    for r in report.get("rows", [])[:5]:
        if r.get("missing_dlc"):
            names = "、".join(x["name"] for x in r["missing_dlc"][:2])
            if r.get("truncated"):
                lines.append(f"- 《{r['name']}》缺 {r['missing_count']} 个,比如 {names}…想细看可以再问我")
            else:
                lines.append(f"- 《{r['name']}》缺 {r['missing_count']} 个:{names}")
    return "\n".join(lines)


async def dlc_detail_node(state: AgentState) -> dict:
    """用户追问时,展开某款游戏的完整 DLC 列表(支持中文/英文游戏名)。"""
    games = state.get("library") or []
    if not games:
        return {"messages": [AIMessage(content="先连接你的游戏库哦~")]}

    message = str(state["messages"][-1].content)
    appids = await _resolve_game_appids(message, games)
    if not appids:
        by_play = sorted(games, key=lambda g: g.get("playtime_forever") or 0, reverse=True)[:3]
        examples = "、".join(f"《{g['name']}》" for g in by_play)
        return {
            "messages": [
                AIMessage(
                    content=f"我好像没在你库里找到这款游戏,再说一下游戏名试试?比如 {examples}"
                )
            ]
        }

    report = await build_dlc_report(
        games, top_games=None, max_dlc_per_game=None, only_appids=appids
    )
    summary = _summarize_dlc_detail(report)
    return {"dlc_report": report, "messages": [AIMessage(content=summary)]}


def _find_games_mentioned(message: str, games: list[dict]) -> list[int]:
    m = message.lower()
    return [g["appid"] for g in games if (g.get("name") or "").lower() in m]


async def _resolve_game_appids(message: str, games: list[dict]) -> list[int]:
    """解析用户提到的游戏:先英文名子串匹配,再 LLM 提取 + 商店搜索(支持中文名)。"""
    appids = _find_games_mentioned(message, games)
    if appids:
        return appids

    name = await _extract_game_name(message)
    if not name:
        return []

    library_ids = {g["appid"] for g in games}
    try:
        lang = "schinese" if _has_cjk(name) else "english"
        items = await steam_client.search_store(name, lang=lang)
    except Exception:
        return []
    for it in items:
        if it.get("id") in library_ids:
            return [it["id"]]
    return []


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


async def _extract_game_name(message: str) -> str:
    if not settings.deepseek_api_key:
        return ""
    try:
        llm = get_llm()
        resp = await llm.ainvoke(
            [
                SystemMessage(content=EXTRACT_GAME_PROMPT.format(message=message)),
                HumanMessage(content=message),
            ]
        )
        name = str(resp.content or "").strip()
        if name.lower() in {"", "无", "空", "null", "none"}:
            return ""
        return name
    except Exception:
        return ""


def _summarize_dlc_detail(report: dict) -> str:
    if not report.get("rows"):
        return "没有找到相关游戏的 DLC 信息。"
    lines: list[str] = []
    for r in report["rows"]:
        total = r["owned_count"] + r["missing_count"]
        lines.append(f"《{r['name']}》一共有 {total} 个 DLC:")
        for x in r["missing_dlc"]:
            price = f" ${x['price_cents'] / 100:.2f}" if x["price_cents"] is not None else "(免费)"
            lines.append(f"  · 未拥有:{x['name']}{price}")
        if r["owned_dlc"]:
            lines.append(f"  · 已拥有 {r['owned_count']} 个,比如 {r['owned_dlc'][0]['name']}")
    return "\n".join(lines)


# ---------- 用户画像 & 推荐 ----------

async def _build_profile(games: list[dict]) -> dict:
    """从用户库统计类型/玩法偏好,生成画像。"""
    ids = [g["appid"] for g in games[:30]]
    details = await steam_client.get_many_app_details(ids)

    genre_counter: dict[str, int] = {}
    cat_counter: dict[str, int] = {}
    for d in details.values():
        if not d:
            continue
        for item in d.get("genres") or []:
            desc = item.get("description") if isinstance(item, dict) else item
            if desc:
                genre_counter[desc] = genre_counter.get(desc, 0) + 1
        for item in d.get("categories") or []:
            desc = item.get("description") if isinstance(item, dict) else item
            if desc:
                cat_counter[desc] = cat_counter.get(desc, 0) + 1

    top_genres = sorted(genre_counter, key=genre_counter.get, reverse=True)[:6]
    top_cats = sorted(cat_counter, key=cat_counter.get, reverse=True)[:6]
    summary = f"偏爱类型:{'、'.join(top_genres) or '暂无'};常见玩法:{'、'.join(top_cats) or '暂无'}"
    return {"top_genres": top_genres, "top_categories": top_cats, "summary": summary}


def _build_queries(profile: dict, games: list[dict]) -> list[str]:
    queries: list[str] = []
    if profile.get("top_genres"):
        queries.append(f"Games in genres: {', '.join(profile['top_genres'][:3])}")
    by_play = sorted(games, key=lambda x: x.get("playtime_forever") or 0, reverse=True)
    for g in by_play[:3]:
        queries.append(f"Games similar to {g['name']}")
    return queries[:4]


async def recommend_node(state: AgentState) -> dict:
    games = state.get("library") or []
    if not games:
        return {"messages": [AIMessage(content="先连接你的游戏库,我才能推荐哦~")]}

    profile = await _build_profile(games)
    owned = {g["appid"] for g in games}

    candidates: dict[int, dict] = {}
    for q in _build_queries(profile, games):
        for r in rag_store.search(q, k=8, exclude=owned):
            candidates[r["appid"]] = r
    cand_list = sorted(candidates.values(), key=lambda x: x["distance"])[:5]

    intro, reasons = await _generate_reasons(profile, cand_list)
    for c in cand_list:
        c["reason"] = reasons.get(c["appid"]) or _fallback_reason(c, profile)

    text = intro or _fallback_recommend(cand_list)
    return {"profile": profile, "candidates": cand_list, "messages": [AIMessage(content=text)]}


async def _generate_reasons(profile: dict, candidates: list[dict]) -> tuple[str, dict[int, str]]:
    """LLM 生成开场白 + 每款候选的推荐理由(JSON)。"""
    if not settings.deepseek_api_key:
        return "", {}
    lines = [f"- {c['appid']} | {c['name']} | {c['genres']}" for c in candidates]
    human = f"用户偏好:\n{profile['summary']}\n\n候选游戏:\n" + "\n".join(lines)
    try:
        llm = get_llm().bind(response_format={"type": "json_object"})
        resp = await llm.ainvoke(
            [SystemMessage(content=RECOMMEND_REASONS_PROMPT), HumanMessage(content=human)]
        )
        data = json.loads(str(resp.content or ""))
        intro = str(data.get("intro", ""))
        reasons: dict[int, str] = {}
        for r in data.get("reasons", []):
            if r.get("appid"):
                reasons[int(r["appid"])] = str(r.get("reason", ""))
        return intro, reasons
    except Exception:
        return "", {}


def _fallback_reason(candidate: dict, profile: dict) -> str:
    genres = candidate.get("genres", "")
    top = "、".join(profile.get("top_genres", [])[:2]) or "你的偏好"
    return f"类型「{genres}」和 {top} 相近。"


def _fallback_recommend(candidates: list[dict]) -> str:
    if not candidates:
        return "暂时没找到合适的推荐,确认已连接游戏库再试?"
    names = "、".join(f"《{c['name']}》" for c in candidates[:5])
    return f"根据你的喜好,我挑了这几款 🎮:{names}。"


# ---------- 游戏评价(game_ask) ----------

async def game_ask_node(state: AgentState) -> dict:
    games = state.get("library") or []
    message = str(state["messages"][-1].content)

    # 1) "第x个" 从上次推荐上下文解析
    idx = _extract_ordinal(message)
    ctx_cands = state.get("context_candidates") or []
    if idx is not None and 0 < idx <= len(ctx_cands):
        appid = ctx_cands[idx - 1]["appid"]
        return await _single_game_analysis(appid, games)

    name = await _extract_game_name(message)
    if not name:
        return {"messages": [AIMessage(content="你想问哪款游戏?把名字告诉我,比如「赛博朋克2077怎么样」")]}

    items = await _search_store(name)
    if not items:
        return {"messages": [AIMessage(content=f"没找到《{name}》,换个名字试试?")]}

    game_items = await _filter_games(items[:8])
    if not game_items:
        return {"messages": [AIMessage(content=f"没找到《{name}》,换个名字试试?")]}

    if len(game_items) > 1:
        text, cards = await _multi_result_cards(name, game_items[:5], games)
        return {"game_candidates": cards, "messages": [AIMessage(content=text)]}

    return await _single_game_analysis(game_items[0]["id"], games)


def _extract_ordinal(message: str) -> int | None:
    """从消息里解析「第x个/第x款」中的序数。"""
    m = re.search(r"第\s*([0-9一二三四五]+)\s*[个款]", message)
    if not m:
        return None
    num = m.group(1)
    cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    if num in cn:
        return cn[num]
    if num.isdigit():
        return int(num)
    return None


async def _filter_games(items: list[dict]) -> list[dict]:
    """过滤商店搜索结果,只保留 type == game 的本体游戏。"""
    result: list[dict] = []
    for it in items:
        d = await steam_client.get_app_details(it["id"])
        if d and d.get("type") == "game":
            result.append(it)
    return result


async def _search_store(name: str) -> list[dict]:
    try:
        lang = "schinese" if _has_cjk(name) else "english"
        return await steam_client.search_store(name, lang=lang)
    except Exception:
        return []


async def _single_game_analysis(appid: int, games: list[dict]) -> dict:
    details = await steam_client.get_app_details(appid)
    if not details:
        return {"messages": [AIMessage(content="没查到这款游戏的详细信息。")]}

    profile = (
        await _build_profile(games)
        if games
        else {"top_genres": [], "top_categories": [], "summary": "暂无游戏库"}
    )

    game_genres = {g.get("description", "") for g in details.get("genres") or []}
    match_genres = sorted(game_genres & set(profile.get("top_genres") or []))
    owned = appid in {g["appid"] for g in games}

    meta = rag_store.get_game(appid)
    rating = meta["rating"] if meta else "评价较少"
    header = (details.get("header_image") or "") or (meta.get("header_image") if meta else "")

    reviews = await steam_client.get_app_reviews(appid, num=3)
    game_info = _format_game_info(details, meta, match_genres, owned, reviews)

    text = "我先看看这款游戏~"
    try:
        text = await _llm_stream_text(
            [
                SystemMessage(content=GAME_ASK_PROMPT),
                HumanMessage(content=f"游戏信息:\n{game_info}\n\n用户偏好:\n{profile['summary']}"),
            ]
        )
    except Exception:
        text = _fallback_game_ask(details, meta, match_genres, owned)

    price = (details.get("price_overview") or {}).get("final")
    analysis = {
        "appid": appid,
        "name": details.get("name", ""),
        "header_image": header,
        "genres": ", ".join(g.get("description", "") for g in details.get("genres") or []),
        "categories": ", ".join(c.get("description", "") for c in details.get("categories") or []),
        "price_cents": price,
        "positive": meta.get("positive", 0) if meta else 0,
        "negative": meta.get("negative", 0) if meta else 0,
        "rating": rating,
        "metacritic_score": meta.get("metacritic_score", 0) if meta else 0,
        "match_genres": match_genres,
        "owned": owned,
    }
    return {"game_analysis": analysis, "messages": [AIMessage(content=text)]}


async def _multi_result_cards(
    name: str, items: list[dict], games: list[dict]
) -> tuple[str, list[dict]]:
    """多结果时,返回说明文字 + 候选卡片列表(最多 5 个)。"""
    profile = (
        await _build_profile(games)
        if games
        else {"top_genres": [], "top_categories": [], "summary": "暂无游戏库"}
    )
    top = set(profile.get("top_genres") or [])

    cards: list[dict] = []
    lines = [f"搜到好几个和「{name}」相关的结果,你看看是哪个 👀"]
    for i, it in enumerate(items, 1):
        d = await steam_client.get_app_details(it["id"])
        if not d:
            continue
        meta = rag_store.get_game(it["id"])
        rating = meta["rating"] if meta else "评价较少"
        header = (d.get("header_image") or "") or (meta.get("header_image") if meta else "")
        genres = ", ".join(g.get("description", "") for g in d.get("genres") or []) or "未知"
        gset = {g.get("description", "") for g in d.get("genres") or []}
        overlap = sorted(gset & top)
        cards.append(
            {
                "appid": it["id"],
                "name": d.get("name", ""),
                "header_image": header,
                "genres": genres,
                "categories": ", ".join(c.get("description", "") for c in d.get("categories") or []),
                "price_cents": (d.get("price_overview") or {}).get("final"),
                "rating": rating,
                "match_genres": overlap,
            }
        )
        lines.append(f"{i}. 《{d.get('name')}》")
    lines.append("告诉我是第几个,或者给个更准确的名字~")
    return "\n".join(lines), cards


def _format_game_info(
    details: dict,
    meta: dict | None,
    match_genres: list[str],
    owned: bool,
    reviews: dict | None = None,
) -> str:
    name = details.get("name", "")
    genres = ", ".join(g.get("description", "") for g in details.get("genres") or [])
    cats = ", ".join(c.get("description", "") for c in details.get("categories") or [])
    price = (details.get("price_overview") or {}).get("final")
    price_s = f"${price / 100:.2f}" if price is not None else "免费"
    rating = meta["rating"] if meta else "评价较少"
    desc = (details.get("short_description") or "").strip()[:300]
    lines = [f"游戏名:{name}", f"类型:{genres}", f"玩法/特性:{cats}", f"价格:{price_s}", f"评价:{rating}"]
    if desc:
        lines.append(f"简介:{desc}")
    lines.append(f"用户是否已拥有:{'是' if owned else '否'}")
    if match_genres:
        lines.append(f"与用户偏好重叠的类型:{'、'.join(match_genres)}")
    if reviews:
        for r in reviews.get("reviews", [])[:2]:
            vote = "👍" if r.get("voted_up") else "👎"
            rv = (r.get("review") or "").strip().replace("\n", " ")
            if rv:
                lines.append(f"玩家评价({vote}):{rv[:150]}")
    return "\n".join(lines)


def _fallback_game_ask(details: dict, meta: dict | None, match_genres: list[str], owned: bool) -> str:
    name = details.get("name", "")
    genres = ", ".join(g.get("description", "") for g in details.get("genres") or [])
    rating = meta["rating"] if meta else "评价较少"
    if match_genres:
        m = f"它和你的偏好挺合的,重叠类型有:{'、'.join(match_genres)}。"
    else:
        m = "它和你常玩的类型交集不大,可以观望一下。"
    own = "你已经拥有这款游戏啦。" if owned else ""
    return f"《{name}》类型:{genres}。评价:{rating}。{m}{own}"


# ---------- 闲聊 ----------

async def chat_node(state: AgentState) -> dict:
    message = state["messages"][-1].content
    try:
        text = await _llm_stream_text(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=str(message)),
            ]
        )
    except Exception:
        text = "抱歉,我现在无法回复(LLM 未配置或不可用)。"
    return {"messages": [AIMessage(content=text)]}


# ---------- 构建图 ----------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("dlc", dlc_node)
    g.add_node("dlc_detail", dlc_detail_node)
    g.add_node("recommend", recommend_node)
    g.add_node("game_ask", game_ask_node)
    g.add_node("chat", chat_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        _route_after_intent,
        {
            "dlc": "dlc",
            "dlc_detail": "dlc_detail",
            "recommend": "recommend",
            "game_ask": "game_ask",
            "chat": "chat",
        },
    )
    for n in ("dlc", "dlc_detail", "recommend", "game_ask", "chat"):
        g.add_edge(n, END)
    return g.compile()


agent = build_graph()
