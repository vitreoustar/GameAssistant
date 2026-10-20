# SteamAssistant 项目文档

> 本文档用于项目交接与后续开发。换设备后,先看「快速开始」跑通,再按「核心模块」理解代码,最后看「路线图」继续开发。
> 深化开发的完整计划见 [ROADMAP.md](ROADMAP.md)(方向选型、清理清单、里程碑);本文档描述**现状**。

---

## 1. 项目概述

SteamAssistant 是一个面向游戏玩家的 AI 助手 Agent:

- 读取用户 Steam 游戏库(SteamID 导入)
- 整理已拥有游戏的 **DLC 信息**(已拥有/未拥有/价格)
- 基于用户画像做**个性化游戏推荐**(RAG 语义检索 + LLM 生成理由)
- 分析「某游戏怎么样」(与用户偏好匹配度 + 真实玩家评价)

技术形态:**LangGraph 多节点 Agent + RAG + 全栈(React + FastAPI)+ SSE 流式**。

---

## 2. 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 19 + Vite 8 + TypeScript 6 |
| 后端 | FastAPI + uvicorn + httpx(asyncio) |
| Agent | LangGraph + langchain-openai |
| LLM | DeepSeek `deepseek-flash`(OpenAI 兼容) |
| RAG | ChromaDB + sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` |
| 数据处理 | pyarrow(建库脚本)+ pandas / numpy(推荐算法数据管线) |
| 语料 | FronkonGames/steam-games-dataset(parquet),过滤后 5362 条 |
| 推荐数据 | 《Steam_November2025_Dataset》(CC-BY-4.0),去重后 840,274 交互 / 7,731 用户 / 9,935 物品 |

---

## 3. 架构

```
React 前端(聊天 + 卡片 + 侧栏)
        │ POST /api/chat(SSE) / POST /api/library / POST /api/dlc
        ▼
FastAPI
        │
        ▼
LangGraph Agent(StateGraph)
  ├─ route        意图路由 + **偏好抽取**(LLM JSON / 关键词兜底)
  ├─ dlc          DLC 整理
  ├─ dlc_detail   DLC 追问展开
  ├─ recommend    个性化推荐(本地混合召回 + LTR,**按对话去重与偏好过滤**)
  ├─ game_ask     游戏评测/匹配分析
  └─ chat         闲聊
        │
        ├─ SteamClient(Web API / Store API)
        ├─ 本地推荐器(离线索引 + LightGBM,零推理)
        ├─ ChromaDB(5362 款游戏语料,降级兜底)
        └─ DeepSeek LLM(写文案,不参与排序)
```

**数据流(一次"推荐")**:
1. 前端把 `{message, library, context{candidates, recommend, history}}` POST 到 `/api/chat`
2. route 节点判断 intent=recommend,并顺带从这句话里抽出**偏好增量**(想要/不要哪些类型)
3. 推荐节点:本地混合召回 200 条 → LTR 精排 → 剔除已展示过(含同名)/不想要的 →
   命中明说偏好的排前面 → top5 → LLM 只写开场白 + 逐条理由
4. 通过 `astream_events` 把 node 事件 + token 流 + 最终卡片数据 + **新的 `recommend_state`** 推给前端
5. 前端在对话里内联渲染卡片,并把 `recommend_state` 存起来下一轮回传

---

## 4. 目录结构

```
SteamAssistant/
├─ frontend/
│  ├─ vite.config.ts          # /api 代理到 127.0.0.1:8000
│  └─ src/
│     ├─ App.tsx              # 状态管理、SSE 事件处理、卡片渲染
│     ├─ api.ts               # importLibrary / streamChat(fetch 流式解析)
│     ├─ types.ts             # 全部 TS 类型
│     ├─ index.css            # 主题变量 + zoom 1.2
│     ├─ App.css              # Steam 蓝黑主题 + 布局 + 卡片
│     └─ components/
│        ├─ ChatPanel.tsx     # 聊天消息 + 内联卡片 + 输入
│        ├─ GameCard.tsx      # 游戏卡片(封面/标签/评价/价格/理由)
│        ├─ LibraryPanel.tsx  # 游戏库导入
│        ├─ SteamIdGuide.tsx  # 获取 SteamID 引导弹窗
│        └─ DlcPanel.tsx      # DLC 报告表
├─ backend/
│  ├─ app/
│  │  ├─ main.py              # FastAPI 入口 + CORS + /healthz + 推荐索引预热(lifespan)
│  │  ├─ config.py            # Settings(读 .env)、HF_ENDPOINT 导出
│  │  ├─ schemas.py           # Game 等 Pydantic 模型
│  │  ├─ routers/
│  │  │  ├─ library.py        # POST /api/library
│  │  │  ├─ dlc.py            # POST /api/dlc
│  │  │  ├─ recommend.py      # POST /api/recommend(本地推荐器,不调 LLM,~9ms)
│  │  │  └─ chat.py           # POST /api/chat(SSE 流式 + 多轮状态回传)
│  │  ├─ agent/
│  │  │  ├─ graph.py          # ★ LangGraph 图、各节点、画像、推荐、评测
│  │  │  ├─ stream.py         # ★ astream_events → SSE 事件
│  │  │  ├─ prompts.py        # 全部提示词
│  │  │  └─ __init__.py       # 导出 agent
│  │  ├─ rag/
│  │  │  ├─ embedder.py       # 本地 embedding 加载(注意 import 顺序!)
│  │  │  └─ store.py          # ★ ChromaDB 建库/检索/按 id 查/评级
│  │  ├─ recommend/           # ★ 方向 A:混合推荐算法(见 ROADMAP.md §3)
│  │  │  ├─ features.py       # 纯函数:评价标签/好评率/销量下界/年份/时长分档
│  │  │  ├─ index.py          # ★ A2 产物加载(mmap)+ 数组助手
│  │  │  ├─ recall.py         # ★ A3 五路召回 + 配额轮转融合
│  │  │  ├─ rank.py           # ★ A4 特征构造(训练/线上共用)+ 排序器
│  │  │  ├─ evaluate.py       # 留一切分 + 排序指标 + 两套评测协议
│  │  │  ├─ preferences.py    # ★ A7 对话偏好状态机 + 封闭词表 + 关键词兜底
│  │  │  ├─ notes.py          # ★ A7 "命中了几款"的确定性文案
│  │  │  └─ service.py        # ★ A5 对外门面 + 懒加载单例(含多轮去重/偏好分桶)
│  │  └─ steam/
│  │     ├─ client.py         # ★ SteamClient(缓存/令牌桶限速/并发/搜索/评价)
│  │     ├─ dlc.py            # DLC 计算(拥有状态求交、上限、截断)
│  │     ├─ importer.py       # 游戏库解析(仅 steamid 模式)
│  │     └─ __init__.py       # 共享 client 单例
│  ├─ scripts/
│  │  ├─ build_corpus.py      # 下载 parquet → 清洗 → corpus.jsonl
│  │  ├─ build_index.py       # corpus.jsonl → 嵌入 → ChromaDB
│  │  ├─ build_interactions.py  # 交互数据集 + corpus → 推荐算法用表(A1.2)
│  │  ├─ build_recommend_index.py # → content_emb / 两路 TopK / 用户向量(A2)
│  │  ├─ train_ranker.py      # LightGBM lambdarank 训练 + 评测(A4)
│  │  ├─ eval_recall.py       # 两套协议的召回评测 + 嵌入配置对比(A3.1/A6)
│  │  ├─ eval_recall_channels.py # 单通道消融 / 边际贡献 / 配额扫描(A3.2)
│  │  ├─ eval_rag_baseline.py # 现有 RAG 方案对照(A6)
│  │  └─ eval_embedding_config.py # 嵌入配置端到端裁决(A6)
│  ├─ tests/                  # 离线单测(unittest)+ 手工联调脚本
│  ├─ data/                   # gitignored:parquet/corpus/chroma/interactions/recommend
│  ├─ requirements.txt
│  └─ .env / .env.example
├─ docs/
│  ├─ PROJECT.md              # 本文档(现状:架构、模块、API、踩坑)
│  ├─ ROADMAP.md              # 深化路线图(方向选型、Phase 0、方向 B 计划)
│  └─ RECOMMENDER.md          # ★ 推荐算法(方向 A)实现全记录 + 可优化点
├─ Steam_November2025_Dataset/ # gitignored:公开交互数据集(A1.1)
├─ notes/                     # gitignored:面试相关(INTERVIEW/RESUME/知识点详解)
└─ README.md
```

> 标 ★ 的是最核心、后续开发最常改的文件。

---

## 5. 快速开始(新设备上手)

### 0. 依赖

- Python 3.11+(开发用 3.13)、Node.js 20+
- 能访问 Steam 的网络(国内需代理,见 `.env` 的 `STEAM_PROXY`)
- HuggingFace 访问(国内用 `HF_ENDPOINT=https://hf-mirror.com`)

### 1. 后端

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp .env.example .env    # 填入 DeepSeek key、Steam key、steamid
```

`.env` 关键项:

```env
DEEPSEEK_API_KEY=sk-xxxx          # LLM
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
STEAM_API_KEY=xxxx                # Steamworks Web API key
STEAM_ID=7656119xxxxxxxxxx        # 你的 steamid64(可选,页面也能输)
STEAM_PROXY=http://127.0.0.1:12450  # 国内访问 Steam 的本地代理
STEAM_RATE_LIMIT=10               # Steam 请求限速(每秒请求数,全局令牌桶)
STEAM_RATE_BURST=5                # 允许的突发请求数
HF_ENDPOINT=https://hf-mirror.com   # 国内下载模型/数据集镜像
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

构建数据(一次性):

```bash
python scripts/build_corpus.py   # 下载数据集(192MB)→ data/corpus.jsonl(5362 条)
python scripts/build_index.py    # 嵌入 → data/chroma

# 方向 A(推荐算法)需要:把公开交互数据集 + corpus 落成推荐算法用表
# 先把 Steam_November2025_Dataset/(两个 csv)放到仓库根目录,然后:
python scripts/build_interactions.py   # → data/interactions/{interactions,users,items}.parquet + stats.json
```

启动:

```bash
uvicorn app.main:app --reload    # http://127.0.0.1:8000/docs
```

### 2. 前端

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

### 3. 测试

```bash
cd backend
# 离线单测(不需要任何 key / 网络)
.venv/Scripts/python -m unittest tests.test_recommend_features tests.test_recommend_index \
  tests.test_recommend_evaluate tests.test_recommend_recall tests.test_recommend_rank \
  tests.test_recommend_service tests.test_recommend_preferences tests.test_steam_client

# 多轮对话整图测试(离线,但会 import app.agent → 加载 torch,约 23s)
.venv/Scripts/python -m unittest tests.test_multiturn_agent

# 手工联调脚本(需要真实 key + 网络)
.venv/Scripts/python tests/test_recommend.py
.venv/Scripts/python tests/test_agent.py
.venv/Scripts/python tests/test_stream.py
.venv/Scripts/python tests/test_resolve.py
.venv/Scripts/python tests/test_frontend_plan.py
```

> ⚠️ 目前**真单测共 135 个**(离线 129 + 多轮整图 6);剩下几个是脚本式的流程验证,没有断言化。
> 补齐 pytest 结构在 [ROADMAP.md](ROADMAP.md) 的 Phase 0 里。
> 💡 只跑离线单测请**指定模块**(如 `python -m unittest tests.test_recommend_features`):`unittest discover` 会 import 全部 `test_*.py`,连带加载 torch + sentence-transformers,实测要 22 秒(`-m unittest discover` 的 22.1s vs 单模块 0.2s)。

---

## 6. 核心模块详解

### 6.1 Agent 图(`agent/graph.py`)

**状态 `AgentState`**(TypedDict):

```python
messages: Annotated[list, add_messages]  # 对话,add_messages reducer 追加
library: list[dict]                      # 用户游戏库
intent: str                              # 路由结果
dlc_report / profile / candidates / game_analysis / game_candidates / context_candidates
history: list[dict]                      # 最近几轮对话,仅供路由消解省略与指代
recommend_state: dict | None             # 累积偏好 + 已展示 appid(前端回传,见 §6.7)
```

**节点与职责**:

| 节点 | 职责 |
|---|---|
| route | LLM(JSON)判断 intent **并顺带抽取偏好增量**;失败走 `_keyword_route` 关键词兜底 |
| dlc | `build_dlc_report(top20, cap8)` + LLM 总结 → 侧栏表 |
| dlc_detail | 追问展开某游戏完整 DLC(解析游戏名) |
| recommend | 本地混合召回 → LTR 精排 → **去重 + 偏好分桶** → top5 → LLM 只写 intro+reasons |
| game_ask | 解析游戏名(第x个/中英文搜索)→ 详情+真实评价 → 匹配分析 |
| chat | 直接 LLM 回复 |

**关键函数**:

- `get_llm()`:LLM 唯一入口(替换本地 LLM 就改这里)
- `_llm_stream_text()`:流式生成并拼接(节点内用 astream 才有 token 流)
- `_top_games_by_playtime(games, n)`:按游玩时长降序取 Top N。**画像与查询都必须走它** —— `library` 的顺序由 Steam 决定(近似按 appid),直接切片会取到与偏好无关的随机游戏
- `_build_profile()`:取时长 Top30 → 统计 genres/categories 频率 → 画像(有本地推荐器时走 `_profile_for`,零网络请求)
- `_parse_router()`:**一次 LLM 调用同时得到 intent 与偏好增量**(多花一次调用要 300~600ms,而两者都是"读一句话做分类")
- `_constraint_note()`:把"这轮命中了几款"写成确定性实话(实现在 `recommend/notes.py`)
- `_generate_reasons()`:LLM 返回 JSON {intro, reasons};候选行带 `pref_match` / `because_of` 作证据
- `_extract_game_name()`:LLM 从消息抽取游戏名
- `_resolve_game_appids()` / `_search_store()` / `_filter_games()`:游戏名解析(英文子串 → LLM 抽取 → 中文商店搜索 → 过滤 type==game)
- `_extract_ordinal()`:解析"第x个/第x款";**没有可用候选时直接给提示**,不再把「第2个」当游戏名去搜商店

### 6.2 流式(`agent/stream.py`)

用 `agent.astream_events(input, version="v2")` 产出三类事件:

- `on_chain_start/end` → node 事件(节点状态)
- `on_chat_model_stream` → token 事件(按 `metadata.langgraph_node` 过滤,route 的 JSON 不外泄)
- 根链 `on_chain_end` → 最终 state(取 done 事件)

done 事件包含:text、intent、dlc_report、profile、candidates、game_analysis、
game_candidates、**recommend_state**。最后一个必须由前端原样回传,否则下一轮的去重记录归零。

### 6.3 RAG(`rag/`)

- `embedder.py`:加载本地 sentence-transformers。**注意:必须先 import app.config 再 import sentence_transformers**(否则 HF_ENDPOINT 镜像不生效)。
- `store.py`:
  - `build_index()`:corpus.jsonl → 嵌入 → ChromaDB
  - `search(query, k, exclude)`:语义检索,返回 appid/name/header_image/genres/price/positive/negative/rating/distance
  - `get_game(appid)`:按 id 查单款元数据
  - `rating_text(positive, negative)`:好评率 → 中文评价(好评如潮/特别好评/褒贬不一/差评如潮)

### 6.4 Steam 客户端(`steam/client.py`)

`SteamClient`(async httpx),内置:

- `TTLCache`(3600s 内存缓存)
- `RateLimiter`:**全局令牌桶**(`rate` 每秒请求数 + `burst` 突发额度),所有请求都必须经过它。早期版本允许 `use_limiter=False` 让并发批处理绕过限速,是真实的 429 风险点,已移除该旁路
- in-flight 去重:同一 URL 的并发请求只发一次网络请求(避免并发重复拉同一 appid)
- `_client()`:带锁的懒加载,避免并发下重复创建 `AsyncClient`
- `get_owned_games(steamid)`:GetOwnedGames(需 key + 资料公开)
- `get_app_details(appid)` / `get_many_app_details(appids, concurrency=5)`:商店详情(并发只决定"同时发出",总量仍由令牌桶约束)
- `search_store(term, lang)`:商店搜索(中文用 `schinese`)
- `get_app_reviews(appid, num=3)`:真实玩家评价(免费)

### 6.5 DLC 计算(`steam/dlc.py`)

`build_dlc_report(games, top_games=20, max_dlc_per_game=8, only_appids=None)`:

1. 按 `playtime_forever` 排序取 top N 游戏(性能优化)
2. 拉 appdetails,取 `type=="game"` 的本体 + 其 `dlc` 列表
3. **拥有状态 = 用户 appid 集合 ∩ dlc 列表**(纯集合运算,不请求)
4. 只拉「要展示的 DLC」详情(上限 8 个,截断标记 truncated)
5. 追问展开:`only_appids=[某游戏]` + `max_dlc_per_game=None`

### 6.6 前端(`frontend/src/`)

- `App.tsx`:持有 library/messages/steps/dlcReport/lastCandidates/**recommendState** 状态;SSE done 事件把 candidates/game_analysis/game_candidates 统一 `toCard()` 成 `GameCard[]`,挂到 assistant 消息上(`cardPosition: before/after`)。
- `api.ts`:`streamChat` 用 fetch ReadableStream 逐块解析 `data:` 事件。
- `lastCandidates`:上次推荐候选,作为 `context.candidates` 传给后端,用于解析"第x个"。
- `recommendState`:多轮推荐状态(偏好 + 已展示 appid),由 done 事件给出、下一轮原样回传。**这一轮哪怕没有候选也要更新**,否则去重记录回退。
- 历史对话用 `useRef` 而非 state 读取:依赖里没有 `messages` 的 `useCallback` 会读到过期闭包,而 ref 在调用瞬间拿到的正是本次发送之前的消息。
- 换游戏库时重置 `recommendState`(旧 `shown` 是上一份库的记录,留着会让新库没东西可推)。
- UI:`#root { zoom: 1.2 }` 整体放大;Steam 蓝黑主题(变量在 index.css)。

### 6.7 推荐算法(`recommend/`)

方向的完整实现在 **[RECOMMENDER.md](RECOMMENDER.md)**;这里只列模块职责:

| 模块 | 职责 |
|---|---|
| `features.py` | 纯函数:评价标签 / 好评率 / 销量下界 / 年份 / 时长分档 / **文本模板(与 build_corpus 共用)** / 贝叶斯平滑 |
| `index.py` | A2 产物加载(mmap)+ 数组助手(`topk_from_dense` / `cat_rows_of` / `content_topk_from_embeddings`) |
| `recall.py` | 5 路通道(内容画像 / 内容 kNN / 协同 kNN / 质量热度 / 发行商)+ 配额轮转融合 |
| `rank.py` | 26 维 `FeatureBuilder`(**训练与线上共用**,避免训练-服务偏斜)+ `Ranker` |
| `evaluate.py` | 按用户留一切分 + 两套评测协议(全库排序 / 热度匹配)+ 排序指标 |
| `preferences.py` | **A7 多轮**:`PreferenceState` 状态机 + 25 词封闭词表 + 关键词兜底抽取 |
| `notes.py` | **A7 多轮**:"命中了几款"的确定性文案(纯标准库,便于秒级测试) |
| `service.py` | 对外门面 `Recommender` + `get_recommender()` 懒加载单例;冷启动与可解释性 |

要点:
- **在线零推理** —— `load()` 后 `torch` / `sentence_transformers` 都未被 import(实测)
- **离线资产 73 MB**,`data/recommend/`(已 gitignore),由 5 个脚本构建(见 RECOMMENDER.md §9.1)
- **索引缺失时自动降级**到原来的 RAG 路径,服务不会起不来
- **多轮约束施加在排序之后**(不是之前):`merge_rank` 特征依赖候选在召回池里的位置,
  先过滤再排序会让特征分布偏离训练时口径。详见 RECOMMENDER.md §11.4

---

## 7. API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/api/library` | `{mode:"steamid", value}` → `{count, games}` |
| POST | `/api/dlc` | `{games:[...]}`(推荐)或 `{mode:"steamid", value}` → DLC 报告 |
| POST | `/api/recommend` | `{library, top_k, exclude_appids?, prefer?, avoid?}` → 结构化候选(**不调 LLM**,P50 ~9ms) |
| POST | `/api/chat` | `{message, library, context?}` → SSE 流 |

`/api/recommend` 返回每条候选的 `score` / `channels`(由哪几路召回)/
`because_of`("因为你在玩《X》")/ `match_genres`(与**长期画像**重叠的类型)/
`pref_match`(与**本轮明说偏好**重叠的类型),以及 `cold_start`、
`unmatched_in_library`、`matched_count`、`relaxed`、`no_match`、`exhausted`、
`skipped_shown` 这些状态标志。

`/api/chat` 的 `context`(前端持有,服务端不存会话):

| 字段 | 用途 |
|---|---|
| `candidates` | 上次推荐候选,解析「第 x 个」 |
| `recommend` | `{genres, avoid, unmapped, shown, turn}`,多轮去重与偏好过滤 |
| `history` | 最近 8 轮(每轮截 400 字),让路由理解「好」「换一批」这类省略 |

SSE 事件格式(每条 `data: {json}\n\n`):

```json
{"type":"node","data":{"node":"recommend","status":"start"}}
{"type":"token","node":"recommend","data":"这"}
{"type":"done","data":{"text":"...","intent":"recommend","candidates":[...],"game_analysis":null,"game_candidates":null,"recommend_state":{...}}}
{"type":"error","data":"..."}
```

---

## 8. 关键设计决策与取舍

| 决策 | 理由 |
|---|---|
| 意图路由 + 确定性节点(而非 function calling) | 意图有限、要可控;路由式更稳、可单测 |
| 本地 embedding(而非 API) | DeepSeek 无 embedding 接口;免费离线;多语言模型适配中英 |
| RAG 语义检索做推荐(而非传统推荐算法) | MVP 阶段冷启动无行为数据;可解释(LLM 给理由);无需训练。**注意:这已是待替换的临时方案** —— 深化阶段会引入公开交互数据集做混合推荐,详见 [ROADMAP.md](ROADMAP.md) §3 |
| 多查询检索 | 单查询召回差,多角度查询合并去重提升召回 |
| DLC 限 top20 + 单游戏上限 8 | 大库性能优化;追问再展开 |
| LLM 降级兜底(关键词路由 + 确定性文案) | LLM 不可靠,保证最差可用 |
| SSE(而非 WebSocket) | 单向推送够用、更简单 |
| 只保留 SteamID 一种导入方式 | 主页抓取依赖社区页面结构、脆弱且易被限流;名称匹配准确率不足以支撑推荐质量 |
| Steam 请求用全局令牌桶限速 | 既能并发发出(不逐条等 0.3s),又有总量上限(不触发 429) |
| 画像/查询按游玩时长取样 Top N | 库顺序由 Steam 决定,直接切片等于随机取样,画像会失真 |
| 前端 zoom 1.2 | 用户要求 UI 放大;注意与 `height: calc(100vh/1.2)` 配合防整页滚动 |

---

## 9. 踩坑记录(后续开发避坑)

1. **DeepSeek 无 embedding** → 本地 sentence-transformers。
2. **Valve 下线 GetAppList 端点**(404)→ 改用 `storesearch`。
3. **数据集字段错位**:CSV 的"Tags"其实是类型、真 Tags 缺失 → 用 parquet 的 genres+categories+描述。
4. **U+2028 Unicode 行分隔符**导致 `splitlines()` 切坏 JSON → 用 `split("\n")`。
5. **huggingface_hub import 时冻结 HF_ENDPOINT** → 先 import config 再 import sentence_transformers。
6. **空 API key 时 LLM 调用挂起** → 无 key 直接走关键词路由 + timeout=30/max_retries=1。
7. **中文名 vs 英文库**("神之天平" vs "ASTLIBRA Revision")→ LLM 抽取 + 中文商店搜索。
8. **Windows 端口占用报 WinError 10013** → `netstat -ano | findstr :8000` + `taskkill /F /PID`。
9. **`app/steam/__init__.py` 里的单例叫 `client`,与子模块 `app.steam.client` 同名**:所以 `from app.steam import client` 拿到的是**实例**,`import app.steam.client as c` 拿到的也是**实例**。要看模块本身得用 `importlib.import_module("app.steam.client")`。改动 Steam 客户端时注意这个坑。
10. **ChromaDB `delete_collection` 不会删掉旧的 HNSW segment 目录** → 反复 `build_index.py` 会在 `data/chroma/` 下留下孤儿目录,需要手工清理。
11. **"目录下标"与"候选池内位置"是两个索引空间** → 同名去重踩过这个坑:按 `cat_idx` 排的数组被用池内位置索引,表现是**随机失效**(小目录下位置常与 `cat_idx` 撞对,单测反而通过)。用前先 `arr[rows]` 重排。
12. **`app/agent/__init__.py` 会 `import graph`** → 任何 `app.agent.*` 的导入都会拉起 torch + sentence-transformers(14s)。纯函数别放在 `app/agent/` 下,除非愿意为它等 14 秒。
13. **用 `SimpleNamespace` 造假结果测不出字段漂移** → 假对象恰好带着真对象没有的字段,测试永远绿,线上直接 `AttributeError`。要么用真 dataclass,要么加一条整图测试。

---

## 10. 路线图

**文档分工**:

| 文档 | 内容 |
|---|---|
| 本文档 | **现状**:架构、模块、API、踩坑 |
| [ROADMAP.md](ROADMAP.md) | **计划**:方向选型、Phase 0、方向 A 各阶段、方向 B 计划、WSL2 迁移 |
| [RECOMMENDER.md](RECOMMENDER.md) | **推荐算法(方向 A)实现全记录**:数据 / 索引 / 评测协议 / 召回 / 排序 / 在线服务 / 消融 / **已知局限与可优化点** |

### 方向 A(混合推荐算法)—— ✅ 已完成

端到端结果:**recall@10 从改造前的 0.80% 提升到 24.00%(约 30 倍)**,NDCG@10 0.0031 → 0.1430,
在线 P50 **9.1ms**(约束路径 9.4ms)、常驻 73MB、纯 CPU 零推理。
完整实现细节、10 个 bug 的根因、以及 **21 条可优化点(按预期收益排序)** 见
[RECOMMENDER.md](RECOMMENDER.md) §10。

**A7 多轮对话(2025 追加)**:修复"再推荐点别的"原样返回上一批的问题。
补上两样缺失状态 —— **见过哪些**(按 appid + 同名去重)与**说过什么**(25 词封闭词表的偏好过滤),
并把"命中了几款"写成确定性文案(命中不足 / 完全没命中 / 池子耗尽三档)。
连续 5 轮换一批零重复,延迟增量 **+0.24ms**。实现记录见 RECOMMENDER.md §11。

### 方向 B(本地 LLM 推理与调优)—— 计划中

技术路线见 [ROADMAP.md](ROADMAP.md) §4:WSL2 + vLLM + 3B/4B 模型 + QLoRA 微调 + 手搓推理。

### 已确认的方向决策

| 决策点 | 结论 |
|---|---|
| 主方向 | **B 主干(本地 LLM 推理 + 调优)+ A 做全** → **A 已完成** |
| **A 的数据来源** | `corpus.jsonl` **只有物品侧特征,零个 user/interaction 字段**,无法单独支撑推荐算法 → 改用公开交互数据集(Zenodo《Steam_November2025_Dataset》,CC-BY-4.0);`corpus.jsonl` 转为 **item 特征侧** → 做混合推荐 |
| **A 的嵌入配置** | 端到端裁决为 **保持 `cur128`,不重建**(三配置打平,seed 噪声比配置差异大一个数量级) |
| **B 的模型与算力** | 开源预训练小模型 **3B/4B**(Qwen3-4B / Qwen2.5-3B 一类)+ **QLoRA**,**全程本地**部署 |
| **B0 环境** | **WSL2 Ubuntu 22.04 + CUDA + vLLM**(用户已有验证过的 CUDA / PyTorch 环境) |
| **微调数据** | **自蒸馏**(用 DeepSeek 批量生成训练目标)+ 少量通用中文指令数据防遗忘;held-out 评测集绝不入训练 |
| **工作区位置** | 迁移进 WSL2 的 **ext4**(不要放 `/mnt/d`,性能差) |
| `url` / `names` 导入模式 | **彻底删除,不做兜底**(`/api/library` 只支持 steamid) |

### 遗留待办(来自早期 TODO)

- [ ] **游戏俗称/简称识别**:例「龙信」→《龙之信条》,「MHWS」→《怪物猎人荒野》。需强化游戏名检索(俗称/简称 → 正式名),可考虑别名表 + 向量检索。

### 已知限制(现状)

- **无服务端会话记忆**:没有 checkpointer。多轮上下文(已推荐过哪些、用户说过什么偏好)靠前端回传 `context`(见 §7)。**代价:刷新页面后去重记录重置**,会重新看到早先推荐过的游戏
- **偏好只能落在 25 个词里**:Steam 类型只有 26 个,没有 Horror / Stealth / Roguelike,所以「恐怖」「潜行」这类要求只能如实告诉用户"没有这个标签"(RECOMMENDER.md §11.3)
- **「换一批」只保证不重复,不保证更不一样**:候选池固定 200 条,没有多样性重排(RECOMMENDER.md §11.11)
- **无结构化日志**:大量 `except Exception: pass` 吞异常,线上不可观测(P0.2 未做)
- **LLM 侧仍无评测集/基线数字**:推荐算法侧已建立(P0.4 只完成了方向 A 需要的部分)
- **`tests/` 仍是平铺结构**:`unittest discover` 会连带加载 torch,单轮 22s(见 RECOMMENDER.md §9.3)

---

## 11. 注意事项

- `backend/data/`、`notes/`、`.env` 均已 gitignore,换设备需重新构建数据、重新配 key。
- 首次推荐会加载 embedding 模型(10~20s),之后常驻内存很快。
- `notes/` 下是面试相关私有内容(INTERVIEW/RESUME/知识点详解/简历建议),不要上传。
- 当前 **Windows 侧**装的是 CPU 版 torch(`2.14.0+cpu`),embedding 在 CPU 上跑。本地 LLM 路线走 **WSL2 Ubuntu 22.04**(已验证 CUDA + PyTorch),需在 Linux 侧建独立 venv,别污染 Windows venv;工作区迁移见 [ROADMAP.md](ROADMAP.md) §7(**不要放在 `/mnt/d`**)。

### Steamworks Web API Key 申请指南

1. 用 Steam 账号登录 <https://steamcommunity.com/dev/apikey>
2. 域名随便填(如 `localhost`,仅作记录),勾选同意条款 → 注册
3. 拿到 32 位 hex key → 填入 `backend/.env` 的 `STEAM_API_KEY`
4. 同时确保 Steam 设置 → 隐私 → 「我的个人资料」与「游戏详情」设为**公开**,否则 GetOwnedGames 返回空

### 提交前安全自查(项目会上传 GitHub)

- 真实 API key 只放 `backend/.env`,**`.gitignore` 必须忽略 `.env`**;仓库只提供 `.env.example` 占位模板
- 文档、README、代码注释中不得出现真实 key
- 提交前自查:`git ls-files | grep -iE '(env|key|secret)'` 应只匹配到 `.env.example`
- ⚠️ 若 key 曾在聊天/截图/提交历史中明文出现过,请到 DeepSeek 开放平台**重新生成并作废旧 key**
