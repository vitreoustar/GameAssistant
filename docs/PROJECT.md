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
| 数据处理 | pyarrow(仅建库脚本用) |
| 语料 | FronkonGames/steam-games-dataset(parquet),过滤后 5362 条 |

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
  ├─ route        意图路由(LLM JSON + 关键词兜底)
  ├─ dlc          DLC 整理
  ├─ dlc_detail   DLC 追问展开
  ├─ recommend    个性化推荐(RAG)
  ├─ game_ask     游戏评测/匹配分析
  └─ chat         闲聊
        │
        ├─ SteamClient(Web API / Store API)
        ├─ ChromaDB(5362 款游戏语料)
        └─ DeepSeek LLM
```

**数据流(一次"推荐")**:
1. 前端把 `{message, library, context}` POST 到 `/api/chat`
2. route 节点判断 intent=recommend
3. recommend 节点:画像(类型统计)→ 多查询 RAG 检索 → 过滤已拥有 → top5 → LLM 生成开场白 + 逐条理由(JSON)
4. 通过 `astream_events` 把 node 事件 + token 流 + 最终卡片数据以 SSE 推给前端
5. 前端在对话里内联渲染卡片(封面/标签/评价/价格/理由)

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
│  │  ├─ main.py              # FastAPI 入口 + CORS + /healthz
│  │  ├─ config.py            # Settings(读 .env)、HF_ENDPOINT 导出
│  │  ├─ schemas.py           # Game 等 Pydantic 模型
│  │  ├─ routers/
│  │  │  ├─ library.py        # POST /api/library
│  │  │  ├─ dlc.py            # POST /api/dlc
│  │  │  └─ chat.py           # POST /api/chat(SSE 流式)
│  │  ├─ agent/
│  │  │  ├─ graph.py          # ★ LangGraph 图、各节点、画像、推荐、评测
│  │  │  ├─ stream.py         # ★ astream_events → SSE 事件
│  │  │  ├─ prompts.py        # 全部提示词
│  │  │  └─ __init__.py       # 导出 agent
│  │  ├─ rag/
│  │  │  ├─ embedder.py       # 本地 embedding 加载(注意 import 顺序!)
│  │  │  └─ store.py          # ★ ChromaDB 建库/检索/按 id 查/评级
│  │  └─ steam/
│  │     ├─ client.py         # ★ SteamClient(缓存/令牌桶限速/并发/搜索/评价)
│  │     ├─ dlc.py            # DLC 计算(拥有状态求交、上限、截断)
│  │     ├─ importer.py       # 游戏库解析(仅 steamid 模式)
│  │     └─ __init__.py       # 共享 client 单例
│  ├─ scripts/
│  │  ├─ build_corpus.py      # 下载 parquet → 清洗 → corpus.jsonl
│  │  └─ build_index.py       # corpus.jsonl → 嵌入 → ChromaDB
│  ├─ tests/                  # 离线单测(unittest)+ 手工联调脚本
│  ├─ data/                   # gitignored:parquet/corpus.jsonl/chroma
│  ├─ requirements.txt
│  └─ .env / .env.example
├─ docs/
│  ├─ PROJECT.md              # 本文档(现状)
│  └─ ROADMAP.md              # ★ 深化路线图(待办与方向选型)
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
.venv/Scripts/python -m unittest discover -s tests -v

# 手工联调脚本(需要真实 key + 网络)
.venv/Scripts/python tests/test_recommend.py
.venv/Scripts/python tests/test_agent.py
.venv/Scripts/python tests/test_stream.py
.venv/Scripts/python tests/test_resolve.py
.venv/Scripts/python tests/test_frontend_plan.py
```

> ⚠️ 目前只有 `tests/test_steam_client.py` 是真单测(11 个用例,全离线);其余是脚本式的流程验证,没有断言化。补齐 pytest 结构在 [ROADMAP.md](ROADMAP.md) 的 Phase 0 里。

---

## 6. 核心模块详解

### 6.1 Agent 图(`agent/graph.py`)

**状态 `AgentState`**(TypedDict):

```python
messages: Annotated[list, add_messages]  # 对话,add_messages reducer 追加
library: list[dict]                      # 用户游戏库
intent: str                              # 路由结果
dlc_report / profile / candidates / game_analysis / game_candidates / context_candidates
```

**节点与职责**:

| 节点 | 职责 |
|---|---|
| route | LLM(JSON)判断 intent;失败走 `_keyword_route` 关键词兜底 |
| dlc | `build_dlc_report(top20, cap8)` + LLM 总结 → 侧栏表 |
| dlc_detail | 追问展开某游戏完整 DLC(解析游戏名) |
| recommend | 画像 → 多查询 RAG → top5 → LLM 生成 intro+reasons(JSON)→ 内联卡片 |
| game_ask | 解析游戏名(第x个/中英文搜索)→ 详情+真实评价 → 匹配分析 |
| chat | 直接 LLM 回复 |

**关键函数**:

- `get_llm()`:LLM 唯一入口(替换本地 LLM 就改这里)
- `_llm_stream_text()`:流式生成并拼接(节点内用 astream 才有 token 流)
- `_top_games_by_playtime(games, n)`:按游玩时长降序取 Top N。**画像与查询都必须走它** —— `library` 的顺序由 Steam 决定(近似按 appid),直接切片会取到与偏好无关的随机游戏
- `_build_profile()`:取时长 Top30 → 统计 genres/categories 频率 → 画像
- `_build_queries()`:画像 → 多条查询(类型 + "Games similar to X")
- `_generate_reasons()`:LLM 返回 JSON {intro, reasons}
- `_extract_game_name()`:LLM 从消息抽取游戏名
- `_resolve_game_appids()` / `_search_store()` / `_filter_games()`:游戏名解析(英文子串 → LLM 抽取 → 中文商店搜索 → 过滤 type==game)
- `_extract_ordinal()`:解析"第x个/第x款";**没有可用候选时直接给提示**,不再把「第2个」当游戏名去搜商店

### 6.2 流式(`agent/stream.py`)

用 `agent.astream_events(input, version="v2")` 产出三类事件:

- `on_chain_start/end` → node 事件(节点状态)
- `on_chat_model_stream` → token 事件(按 `metadata.langgraph_node` 过滤,route 的 JSON 不外泄)
- 根链 `on_chain_end` → 最终 state(取 done 事件)

done 事件包含:text、intent、dlc_report、profile、candidates、game_analysis、game_candidates。

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

- `App.tsx`:持有 library/messages/steps/dlcReport/lastCandidates 状态;SSE done 事件把 candidates/game_analysis/game_candidates 统一 `toCard()` 成 `GameCard[]`,挂到 assistant 消息上(`cardPosition: before/after`)。
- `api.ts`:`streamChat` 用 fetch ReadableStream 逐块解析 `data:` 事件。
- `lastCandidates`:上次推荐候选,作为 `context.candidates` 传给后端,用于解析"第x个"。
- UI:`#root { zoom: 1.2 }` 整体放大;Steam 蓝黑主题(变量在 index.css)。

---

## 7. API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/api/library` | `{mode:"steamid", value}` → `{count, games}` |
| POST | `/api/dlc` | `{games:[...]}`(推荐)或 `{mode:"steamid", value}` → DLC 报告 |
| POST | `/api/chat` | `{message, library, context?}` → SSE 流 |

SSE 事件格式(每条 `data: {json}\n\n`):

```json
{"type":"node","data":{"node":"recommend","status":"start"}}
{"type":"token","node":"recommend","data":"这"}
{"type":"done","data":{"text":"...","intent":"recommend","candidates":[...],"game_analysis":null,"game_candidates":null}}
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

---

## 10. 路线图

**深化开发的完整计划见 [ROADMAP.md](ROADMAP.md)**,包含:

- 代码盘点(正确性/工程化问题清单)与**多余文件清理清单**
- **Phase 0 共同地基**:可观测性、LLM 抽象层、评测集、pytest/CI、会话记忆
- **方向 A**:混合(hybrid)推荐算法 —— 离线预计算 + 协同/内容/社区多路召回 + LightGBM LTR 排序 + 离线评测
- **方向 B**:本地 LLM 推理服务与调优(AI Infra)—— WSL2 + vLLM + 3B/4B 模型 + QLoRA 微调 + 手搓推理
- **WSL2 工作区迁移**步骤与踩坑

### 已确认的方向决策

| 决策点 | 结论 |
|---|---|
| 主方向 | **B 主干(本地 LLM 推理 + 调优)+ A 可做全** |
| **A 的数据来源** | `corpus.jsonl` **只有物品侧特征,零个 user/interaction 字段**,无法单独支撑推荐算法 → 改用公开交互数据集(首选 Zenodo《Steam_November2025_Dataset》,CC-BY-4.0,7,731 用户 / 9,935 游戏 / 100 万+ 交互);`corpus.jsonl` 转为 **item 特征侧** → 做混合推荐 |
| **B 的模型与算力** | 开源预训练小模型 **3B/4B**(Qwen3-4B / Qwen2.5-3B 一类)+ **QLoRA**,**全程本地**部署 |
| **B0 环境** | **WSL2 Ubuntu 22.04 + CUDA + vLLM**(用户已有验证过的 CUDA / PyTorch 环境) |
| **微调数据** | **自蒸馏**(用 DeepSeek 批量生成训练目标)+ 少量通用中文指令数据防遗忘;held-out 评测集绝不入训练 |
| **工作区位置** | 迁移进 WSL2 的 **ext4**(不要放 `/mnt/d`,性能差) |
| `url` / `names` 导入模式 | **彻底删除,不做兜底**(`/api/library` 只支持 steamid) |

### 遗留待办(来自早期 TODO)

- [ ] **游戏俗称/简称识别**:例「龙信」→《龙之信条》,「MHWS」→《怪物猎人荒野》。需强化游戏名检索(俗称/简称 → 正式名),可考虑别名表 + 向量检索。

### 已知限制(现状)

- **无会话记忆**:没有 checkpointer,多轮上下文靠前端回传 `context.candidates`
- **无评测集/无基线数字**:无法量化"换模型/换算法"的好坏
- **无结构化日志**:大量 `except Exception: pass` 吞异常,线上不可观测
- **测试未断言化**:除 `test_steam_client.py` 外,`tests/` 下的脚本只打印不校验

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
