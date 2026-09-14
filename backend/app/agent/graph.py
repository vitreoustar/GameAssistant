"""LangGraph Agent:意图路由 -> (DLC 整理 | 游戏推荐 | 闲聊)。

StateGraph 结构:
    START -> route -> {dlc | recommend | chat} -> END
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import (
    DLC_SUMMARY_PROMPT,
    RECOMMEND_PROMPT,
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
    candidates: list[dict]


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

    if intent not in {"dlc", "recommend", "chat"}:
        intent = "chat"
    return {"intent": intent}


def _keyword_route(message: str) -> str:
    """LLM 不可用时的关键词兜底路由。"""
    m = message.lower()
    if "dlc" in m:
        return "dlc"
    if any(k in m for k in ("推荐", "recommend", "喜欢", "好玩", "类似")):
        return "recommend"
    return "chat"


def _route_after_intent(state: AgentState) -> str:
    return state["intent"]


# ---------- DLC ----------

async def dlc_node(state: AgentState) -> dict:
    games = state.get("library") or []
    if not games:
        return {"messages": [AIMessage(content="我还不知道你的游戏库,请先在右侧连接你的 Steam 库。")]}

    report = await build_dlc_report(games)
    summary = _summarize_dlc(report)
    try:
        summary = await _llm_stream_text(
            [
                SystemMessage(content=DLC_SUMMARY_PROMPT),
                HumanMessage(content=json.dumps(report, ensure_ascii=False)),
            ]
        )
    except Exception:
        pass  # LLM 失败时退回确定性摘要

    return {"dlc_report": report, "messages": [AIMessage(content=summary)]}


def _summarize_dlc(report: dict) -> str:
    n = report.get("total_missing_dlc", 0)
    total = report.get("total_missing_price_cents", 0)
    games = report.get("games_with_dlc", 0)
    lines = [f"你的游戏库中有 {games} 款游戏含 DLC,共缺 {n} 个 DLC。"]
    if total:
        lines.append(f"补齐这些 DLC 约需 ${total / 100:.2f}。")
    for r in report.get("rows", [])[:5]:
        if r.get("missing_dlc"):
            names = "、".join(x["name"] for x in r["missing_dlc"][:2])
            lines.append(f"- 《{r['name']}》缺 {len(r['missing_dlc'])} 个 DLC,如 {names}。")
    return "\n".join(lines)


# ---------- 推荐 ----------

async def recommend_node(state: AgentState) -> dict:
    games = state.get("library") or []
    if not games:
        return {"messages": [AIMessage(content="请先连接你的游戏库,我才能推荐。")]}

    queries = await _build_queries(games)
    owned = {g["appid"] for g in games}

    candidates: dict[int, dict] = {}
    for q in queries:
        for r in rag_store.search(q, k=8, exclude=owned):
            candidates[r["appid"]] = r
    cand_list = sorted(candidates.values(), key=lambda x: x["distance"])[:15]

    text = _fallback_recommend(cand_list)
    try:
        text = await _llm_stream_text(
            [
                SystemMessage(content="你是游戏推荐助手,输出用中文。"),
                HumanMessage(
                    content=RECOMMEND_PROMPT.format(
                        library=_format_library(games),
                        candidates=_format_candidates(cand_list),
                    )
                ),
            ]
        )
    except Exception:
        pass

    return {"candidates": cand_list, "messages": [AIMessage(content=text)]}


async def _build_queries(games: list[dict]) -> list[str]:
    """从用户库构建检索查询:热门类型 + 代表游戏相似查询。"""
    owned_ids = [g["appid"] for g in games[:20]]
    details = await steam_client.get_many_app_details(owned_ids)

    genre_counter: dict[str, int] = {}
    for d in details.values():
        if not d:
            continue
        for item in d.get("genres") or []:
            desc = item.get("description") if isinstance(item, dict) else item
            if desc:
                genre_counter[desc] = genre_counter.get(desc, 0) + 1

    top_genres = sorted(genre_counter, key=genre_counter.get, reverse=True)[:3]
    queries: list[str] = []
    if top_genres:
        queries.append(f"Games in genres: {', '.join(top_genres)}")

    by_play = sorted(games, key=lambda x: x.get("playtime_forever") or 0, reverse=True)
    for g in by_play[:3]:
        queries.append(f"Games similar to {g['name']}")
    return queries[:4]


def _format_library(games: list[dict]) -> str:
    by_play = sorted(games, key=lambda x: x.get("playtime_forever") or 0, reverse=True)
    return ", ".join(g["name"] for g in by_play[:15])


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        price = f"${c.get('price_cents', 0) / 100:.2f}"
        lines.append(f"- {c['name']} | {c.get('genres', '')} | {price} | {c.get('positive', 0)} 好评")
    return "\n".join(lines)


def _fallback_recommend(candidates: list[dict]) -> str:
    if not candidates:
        return "暂时没有找到合适的推荐,请确认已连接游戏库。"
    names = "、".join(f"《{c['name']}》" for c in candidates[:5])
    return f"根据你的游戏库,推荐你试试:{names}。"


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
    g.add_node("recommend", recommend_node)
    g.add_node("chat", chat_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        _route_after_intent,
        {"dlc": "dlc", "recommend": "recommend", "chat": "chat"},
    )
    g.add_edge("dlc", END)
    g.add_edge("recommend", END)
    g.add_edge("chat", END)
    return g.compile()


agent = build_graph()
