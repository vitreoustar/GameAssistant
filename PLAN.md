# SteamAssistant — AI 游戏助手 Agent(面试项目)实施计划

## Context

为 AI 开发实习生面试准备一个完整可演示的项目:游戏助手 Agent。
核心能力:
1. 读取用户 Steam 游戏库(API key / 主页抓取 / 手动粘贴三种方式)
2. 整理已拥有游戏的 DLC 信息(已拥有 / 未拥有 / 价格)
3. 基于用户游戏库的定制化游戏推荐(RAG 检索层实现)

技术栈:React + FastAPI + LangGraph + ChromaDB。
时间线:**2~3 周内完成,MVP 优先,支持"边做边投简历"**。当前目录为空,全新项目。

## 技术决策(已确认)

| 决策点 | 结论 | 说明 |
|---|---|---|
| LLM | DeepSeek 官方 API,model=`deepseek-flash` | 已核实为官方现行模型名(旧名 `deepseek-chat` 将于 2026-07-24 退役);OpenAI 兼容接口,支持 tool calls + JSON 输出 |
| Embedding | **DeepSeek 官方无 embedding 接口**(已核实)→ 本地 `sentence-transformers`,`paraphrase-multilingual-MiniLM-L12-v2` | 免费、离线、中英混合语料可用;下载慢可设 `HF_ENDPOINT=https://hf-mirror.com` |
| RAG 语料(方案B) | HuggingFace `FronkonGames/steam-games-dataset` 的 `games.csv`(约 12 万行,含 Name/Genres/Tags/About the game/评分/销量估计/DLC count) | 过滤至 **3000~5000 条**(About 非空 + 评分/销量门槛)→ 清洗 → chunk → embed → ChromaDB 持久化 |
| Steam 库读取 | ① `IPlayerService/GetOwnedGames`(需 Steamworks key + 资料公开) ② 抓取 `steamcommunity.com/id/<id>/games/?tab=all`(无 key,同样需公开) ③ 粘贴游戏名列表(语料库模糊匹配 appid) | 三种模式覆盖所有演示场景 |
| DLC 拥有状态 | Store API `appdetails?appids=X` 的 `dlc` 字段列出全部 DLC → 与用户拥有的 appid 集合求交 = 已拥有;未拥有 DLC 逐个查价格 | 无需 GetDLCForUser 也能完整实现 |
| UI 形式 | 聊天窗口 + 右侧面板(游戏库概览 / DLC 表格 / 推荐卡片) | 聊天流式展示 agent 中间步骤 |
| 加分项 | **仅做:流式输出展示 agent 中间步骤**(思考过程折叠面板) | 状态图可视化、RAG 评估脚本、Docker 均不做 |

## Security(项目会上传 GitHub,重要)

- 真实 API key 只放 `backend/.env`,**`.gitignore` 必须忽略 `.env`**;仓库提供 `.env.example` 占位模板
- 计划文件、README、代码注释中不得出现真实 key
- ⚠️ 本次 key 已在聊天中明文出现,建议尽快在 DeepSeek 开放平台重新生成并作废旧 key(新 key 同样只写入本地 .env)
- 提交前自查:`git ls-files | grep -i env` 确认无泄露

## Steamworks Web API Key 申请指南(给不熟悉流程的用户)

1. 用 Steam 账号登录 <https://steamcommunity.com/dev/apikey>
2. 域名随便填(如 `localhost`,仅作记录),勾选同意条款 → 注册
3. 拿到 32 位 hex key → 填入 `backend/.env` 的 `STEAM_API_KEY`
4. 同时确保 Steam 设置 → 隐私 → 「我的个人资料」与「游戏详情」设为**公开**,否则 GetOwnedGames / 主页抓取都返回空
5. (备选路径)partner.steamgames.com 注册合作伙伴 → User & Permissions → Create Web API Key;通常 dev/apikey 页面即可,key 申请不阻塞开发(有 ②③ 模式兜底)

## 架构

```
React (Vite) 前端 — 聊天窗口 + 右侧面板(库/DLC/推荐卡片)
        │ POST /chat (SSE 流: agent 节点事件 + token 流)
        │ GET /library /dlc /recommend(非聊天直接调用)
FastAPI 后端
        │
LangGraph Agent
  ├─ 意图路由(LLM 结构化输出: dlc / recommend / chat)
  ├─ 工具节点:get_owned_games(3 模式)、get_dlc_report、rag_search、get_game_detail
  ├─ 推荐节点:RAG 检索(用户库画像 → 候选) → LLM 生成推荐+理由
  └─ 回复节点(流式 token)
        │
  ├─ ChromaDB 向量库(3000~5000 款游戏语料,data/ 持久化)
  ├─ Steam Web API / Store API / 主页抓取(httpx + 本地缓存 + 限速)
  └─ DeepSeek API(deepseek-flash)+ 本地 embedding
```

## 核心流程设计

**DLC 整理流**:steamid → 获取库(3 模式其一)→ 过滤掉 DLC 类条目 → 对每款游戏查 `appdetails.dlc` → 与用户 appid 集合求交得出"已拥有" → 未拥有的查 `price_overview` → 汇总表 + LLM 生成摘要("你有 X 款游戏,共 Y 个 DLC 未拥有,总价约 Z")。

**推荐流**:取游玩时长 Top N 的代表游戏 → 汇总其 tags/genres/类型 → 构造多条查询向量检索(过滤已拥有)→ 按销量/评分元数据重排 → LLM 结合用户画像生成个性化推荐+理由 → 推荐卡片(名称/标签/价格/相似理由)。

**意图路由**:LLM JSON 输出 `{intent, args}` 三类意图,闲聊直接进回复节点。

## 项目结构

```
SteamAssistant/
├─ frontend/               # React + Vite(TypeScript)
│  └─ src/components/      # ChatPanel、LibraryPanel、DlcTable、RecommendCards、AgentSteps
├─ backend/
│  ├─ app/
│  │  ├─ main.py           # FastAPI 入口 + CORS
│  │  ├─ routers/          # chat(SSE)、library、dlc、recommend
│  │  ├─ agent/            # graph.py、nodes/、tools/、prompts.py
│  │  ├─ rag/              # corpus.py(数据加载过滤)、embed.py、store.py(retriever)
│  │  └─ steam/            # client.py(API+抓取+缓存)、dlc.py、importer.py(粘贴匹配)
│  ├─ scripts/             # build_corpus.py(下载数据集→建库)、demo_data.py
│  ├─ data/                # chroma 持久化 + 语料缓存(.gitignore 或存小样例)
│  ├─ .env / .env.example
│  └─ requirements.txt
├─ demo/                   # 内置演示用示例游戏库 JSON(面试兜底)
├─ .gitignore
└─ README.md               # 架构图(mermaid)+ 启动步骤 + 面试演示脚本
```

## 工具准备清单

**账号/凭证**(不阻塞开发,②③ 模式可兜底):
- [ ] DeepSeek API key(已有,建议重新生成)
- [ ] Steamworks Web API key(按上文指南申请)
- [ ] GitHub 仓库(公开,README 写好)

**环境**:
- [ ] Python 3.11+(Windows 直接装,venv 虚拟环境)
- [ ] Node.js 20+ / npm
- [ ] Git
- [ ] 可选:HuggingFace 访问不畅时用 `HF_ENDPOINT=https://hf-mirror.com`(数据集与 embedding 模型下载都要)

**后端依赖**(requirements.txt):`fastapi uvicorn httpx langgraph langchain-openai langchain-core chromadb sentence-transformers pandas beautifulsoup4 pydantic python-dotenv`

**前端依赖**:`vite + react + typescript`;样式用纯 CSS 或 Tailwind(可选,出卡片效果更快)

## 里程碑(2~3 周,MVP 优先)

**Week 1 — 骨架 + 数据层(做完即可演示"读库/看 DLC")**
- D1:仓库初始化、.gitignore/.env、前后端骨架、README 骨架
- D2:Steam 数据层:GetOwnedGames + 主页抓取 + 粘贴导入三模式,本地缓存
- D3:DLC 信息:appdetails 批量查询 + 拥有状态求交 + 价格;提供 `/library` `/dlc` REST 端点

**Week 2 — RAG + Agent(做完 = MVP:可聊天推荐)**
- D4-5:下载 games.csv → 过滤清洗 3000~5000 条 → 本地 embedding → ChromaDB 建库
- D6-7:LangGraph 图:意图路由 + DLC 工具 + RAG 检索 + 推荐生成(deepseek-flash)
- D8:FastAPI `/chat` SSE 流式端点(node 事件 + token 流),Swagger 验证

**Week 3 — 前端 + 打磨(做完 = 面试可用)**
- D9-10:聊天 UI + SSE 解析 + 中间步骤折叠面板(流式展示是加分项)
- D11:游戏库面板、DLC 表格、推荐卡片
- D12-13:错误处理、限速/缓存、demo 兜底数据、README 架构图 + 面试演示脚本、GitHub 上传自查

> 每个周末的产出都是可演示状态:即使投简历约到面试,随时能讲能跑。

## Steps(执行清单)

- [x] 1. 仓库初始化:目录结构、.gitignore(.env/data/缓存)、.env.example、git init
- [x] 2. 后端骨架:FastAPI + 健康检查 + CORS,requirements.txt
- [x] 3. Steam 客户端:`GetOwnedGames`(key 模式)、主页 HTML 解析(bs4)、游戏名匹配导入;httpx 缓存(TTL)+ 限速
- [x] 4. DLC 服务:appdetails 批量拉取、拥有状态求交、价格聚合、`/dlc` 端点
- [x] 5. 语料管线 `scripts/build_corpus.py`:下载 games.csv → 过滤 → 字段拼接(Name+Genres+Tags+About 截断)→ 生成 chunk 文档
- [x] 6. 向量库:`sentence-transformers` 嵌入 + ChromaDB 持久化 + `as_retriever`(元数据:价格/评分/销量)
- [x] 7. Agent 图:路由节点(结构化输出)、get_owned_games/get_dlc_report/rag_search 工具节点、推荐节点(多查询检索→重排→LLM 生成)、回复节点
- [x] 8. 流式集成:`astream_events` 转发 node 事件与 token,`/chat` SSE 端点
- [x] 9. 前端:聊天窗(SSE 读取、markdown 渲染、思考过程折叠)、库面板、DLC 表格、推荐卡片、模式选择(key/URL/粘贴)
- [x] 10. 打磨:demo 数据兜底、限速缓存调优、README(架构图+演示脚本+key 申请指南)、GitHub 泄露自查

## Verification

- 后端:`uvicorn` 启动 → Swagger 依次验证 `/library`(三模式)、`/dlc`、`/chat`(SSE 逐事件)
- RAG:手工抽查 5~10 条查询(如"类似文明6的策略游戏")的 top-5 相关性;检查已拥有游戏被过滤
- 端到端:输入真实 steamid(或 demo 数据)→ 聊天触发「整理我的DLC」「推荐我喜欢的游戏」→ 确认流式中间步骤与最终卡片
- 前端:Vite 联调后端,CORS 正常、SSE 断线重连、空库/私密资料报错文案友好
- 上传前:`git ls-files | grep -iE '(env|key|secret)'` 为空或仅 .env.example
