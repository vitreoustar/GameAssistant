# SteamAssistant 深化路线图（初步计划）

> 本文档基于对现有代码的完整通读写成，用于把已完成的 MVP 做深。
> 结论先行：**以「本地 LLM 推理服务 + 调优」为主干（方向 B，AI Infra）**；方向 A（推荐算法）原先因缺交互数据准备放弃，现已找到可用的公开数据集，因此**保留并做全，形态为"混合（hybrid）推荐"**。两者都建立在同一个「工程化地基 Phase 0」之上，且共享同一份数据工作。

## 已确认决策

| 决策点 | 结论 | 状态 |
|---|---|---|
| 主方向 | **B 主干（本地 LLM 推理 + 调优）+ A 保留并落地** | ✅ 已定 |
| **A 的数据来源** | `corpus.jsonl` **不能**单独支撑推荐算法（已实测：只有物品侧特征，零个 user/interaction 字段，见 §3-A0）。改用**公开交互数据集**：《Steam_November2025_Dataset》（CC-BY-4.0）为首选，UCSD `steam-200k` 为备选。`corpus.jsonl` 转为 **item 特征侧** → 做成真正的**混合（hybrid）推荐** | ✅ 已定，**A1.1 已验证通过** |
| **B 的模型与算力** | 开源预训练小模型 **3B/4B**（Qwen3-4B / Qwen2.5-3B 一类）+ **QLoRA**，**全程本地部署**，不租云、不做 7B | ✅ 已定 |
| **B0 环境路径** | **WSL2 Ubuntu 22.04 + CUDA + vLLM**（用户已有验证过的 CUDA / PyTorch 环境）；~~Windows 原生 Ollama~~ 降为可选兜底 | ✅ 已定 |
| **工作区位置** | 迁移进 WSL2 的 ext4 文件系统（**不要放在 `/mnt/d`**，见 §7） | ✅ 已定 |
| **B 的微调数据** | **自蒸馏**（用 DeepSeek 批量生成 `(prompt, 目标输出)` 对）+ 少量通用中文指令数据混合防遗忘；held-out 评测集绝不入训练。**数据不是瓶颈**，见 §4-B4 | ✅ 已定 |
| `url` / `names` 导入模式 | 彻底删除，不做任何兜底；`/api/library` 只支持 `mode=steamid` | ✅ 已执行 |
| Phase 0 | 先做清理 + 真 bug 修复；可观测性 / LLM 抽象层 / 评测集 / pytest / 会话记忆待定 | ✅ 已执行清理部分 |

### 已完成（本轮）

- 删除：`routers/recommend.py`（占位）、`frontend/src/assets/{react.svg,vite.svg,hero.png}`、`frontend/public/icons.svg`、`frontend/dist/`、`frontend/README.md`、`PLAN.md`、`TODO.md`
- 剥离死代码：`steam/client.py` 的主页抓取（`scrape_profile_games` / `_parse_games_html`）、名称匹配（`match_games_by_name` / `_norm`）及 `beautifulsoup4` 依赖
- 收敛导入模式到 `steamid`；`/api/recommend` 注销
- `requirements.txt`：移除 `pandas`（全项目未使用）与 `beautifulsoup4`，补版本下限
- 修复 P1 真 bug：**画像按游玩时长取样**（原 `games[:30]` 取到随机 30 款）、**并发绕过限速**（改为全局令牌桶 + in-flight 去重）、**序数越界回退**（原会把「第2个」当游戏名搜商店）
- `tests/test_steam_client.py` 重写为 11 个离线单测（覆盖令牌桶限速、TTL 缓存、DLC 纯函数）
- 清理 ChromaDB 孤儿 HNSW segment、全部 `__pycache__`
- 刷新 `docs/PROJECT.md`（同步现状 + 并入 PLAN.md 的 key 申请指南与安全自查、TODO.md 的"问题六"）
- 验证：11 个单测全绿；`tsc -p` 两个 project exit 0；`vite build` 产物文件名与删除前**逐字节一致**（证明删的都是零引用文件）
- **A1.1 数据集验证已完成**（见 §3-A1.1）：item id 确认为 Steam appid、`user_idx`/`game_idx` 与 id 严格 1:1、去重后 840,274 交互 / 7,731 用户 / 9,935 物品 / 密度 1.094%、**77.3% 的交互有 corpus 内容特征** → 方向 A 可按完整 hybrid 推进，无需半合成兜底

---

## 0. 环境事实（决定方案可行性，先摆出来）

| 项 | 实测值 | 影响 |
|---|---|---|
| GPU | NVIDIA RTX 4060 Laptop, **8188 MiB**, compute cap 8.9 (Ada) | 8 GB 是硬约束；**已定 3B/4B 模型**，8 GB 下余量充足（见 §4-B2） |
| **WSL2** | **用户已确认：Ubuntu 22.04，内含验证过的 CUDA + PyTorch** | ✅ **B0 已定走 WSL2 + vLLM**。（本机 `wsl --status` 曾返回 `E_ACCESSDENIED`，那是沙箱限制，不代表 WSL 缺失） |
| Windows 侧 torch | **2.14.0+cpu**（仅 Windows venv） | 现有 embedding 在 CPU 上跑；方向 B 在 WSL2 里用 CUDA 版，**不动 Windows venv** |
| CUDA toolkit / nvcc | Windows 侧未安装 | WSL2 侧已有可用 CUDA；llama.cpp 需要时再补 |
| Docker | 29.3.1 已装（Windows） | 仅作备选容器载体；WSL2 原生跑 vLLM 更简单 |
| 磁盘 | C: 42.9 GB / D: 49.8 GB 空闲（Windows） | **迁移到 WSL2 前需先看 WSL 虚拟盘的剩余空间**；3B-4bit ≈ 2.5 GB + CUDA torch ≈ 3 GB + 数据集 ≈ 1 GB，压力远小于原 7B 方案 |
| Python | Windows venv 3.13.12；WSL2 内版本待确认 | **Linux 侧建议 3.10/3.11**（vLLM / bitsandbytes 对 3.13 支持较新，3.10~3.11 最稳） |
| 已装关键库 | chromadb 1.5.9, sentence-transformers 6.0.1, transformers 5.17.0, langgraph 1.2.11, langchain-core 1.6.3, numpy 2.5.3, pyarrow 25.0.1 | Windows 侧无 vllm / peft / bitsandbytes / lightgbm；WSL2 侧待盘点 |
| 物品数据 | `corpus.jsonl` **5362** 条；chroma.sqlite3 87 MB | 只含**物品侧**特征（见 §3-A0），作为 hybrid 推荐的特征侧复用 |
| 交互数据 | **已有**：`Steam_November2025_Dataset/`（1,009,745 行 → 按 (user,appid) 去重后 **840,274 交互** / 7,731 用户 / 9,935 物品），已加入 `.gitignore` | 方向 A 的关键前置**已解决**，见 §3-A1.1 |
| 测试 | 11 个离线单测通过；仍无 pytest | Phase 0 需要补 |

---

## 1. 代码现状盘点（通读结论）

### 1.1 规模与结构

后端 ~1500 行（不含测试）、前端 ~1000 行 TS/CSS，6 个 commit。架构本身是干净的：`routers → agent(graph/stream/prompts) → {steam, rag}` 分层清晰，`get_llm()` 是唯一 LLM 入口（这一点对方向 B 非常有利），`rag/embedder.py` 有正确的 import 顺序注释。

### 1.2 真实问题清单（不只是"多余文件"）

**P1 — 正确性 / 性能（建议 Phase 0 修）**

| # | 位置 | 问题 |
|---|---|---|
| 1 | `agent/graph.py:269` | `_build_profile` 取 `games[:30]`——但 library 是 GetOwnedGames 的返回顺序（近似按 appid），**不是按游玩时长排序**。而 `_build_queries`（:296）却按 playtime 排序。两处口径不一致，"用户画像"实际取样自任意 30 款游戏，而非核心 30 款。**这是真 bug。** |
| 2 | `agent/graph.py:314` | 多查询召回合并后只按 `distance` 升序取 top5。没有质量先验（好评率/销量）、没有多样性约束，"最像某款已玩游戏"的候选通吃，推荐同质化。 |
| 3 | `agent/graph.py:267,270` | `_build_profile` 每次调用拉 30 次 `appdetails`；`recommend_node` 与 `_single_game_analysis` 各调一次 → 单次会话可能拉 90 次。仅有 1h TTL 缓存兜底，无跨调用记忆。 |
| 4 | `agent/graph.py:73,331` | `bind(response_format={"type":"json_object"})` 在路由和理由生成里都用；本地小模型若不支持该参数会抛异常，被 `except` 吞掉 → **静默降级**（路由变 chat、理由全空）。这是接本地 LLM 的最大风险点，必须显式处理。 |
| 5 | `agent/graph.py:326` | `_generate_reasons` 在无 key 时直接返回空，且 `data.get("reasons", [])` 假定是 list——模型返回 dict 时会静默失败。 |
| 6 | `agent/stream.py:49` | 靠 `name == "LangGraph" and not parent_ids` 抓最终 state，对 langgraph 1.x 的事件结构很脆弱；升级即可能拿不到 `done` 的 state。应改用 `astream(stream_mode=["updates","messages"])` 或稳定的 root 输出通道。 |
| 7 | `steam/client.py:175` | `get_many_app_details` 并发 5，但内部调 `get_app_details(use_limiter=False)` **绕过限速器** → 无节流 5 并发，易触发 Steam 429。且无 in-flight 去重（并发同一 appid 会重复打）。 |
| 8 | `steam/client.py:36-54` | `TTLCache` 无容量上限（长跑进程内存持续增长）。 |
| 9 | `steam/dlc.py:100-101` | `total_missing_dlc` 是全部数量，但 `total_missing_price_cents` 只累加"已展示"的（受 `max_dlc_per_game` 截断）→ 文案"已列出的合计"有歧义，容易误导用户。 |
| 10 | `agent/graph.py:366-374` | `game_ask_node`：`_extract_ordinal` 命中但 `context_candidates` 为空/越界时，会 fallthrough 到 `_extract_game_name("第2个游戏")` → 抽出垃圾名字再搜商店。应直接给明确提示。 |
| 11 | 全局 | **无会话记忆**：没有 checkpointer，多轮上下文靠前端回传 `context.candidates`。一个"Agent"项目没有 memory 是明显短板。加 `AsyncSqliteSaver` + `thread_id` 成本很低、收益很高。 |
| 12 | 全局 | 无结构化日志、无请求追踪；大量 `except Exception: pass` 吞异常，线上不可观测——**这直接妨碍"调优"**（没有基线数字）。 |

**P2 — 工程化 / 文档**

| # | 问题 |
|---|---|
| 13 | 无 pytest（`unittest` + 5 个手写脚本混用）；live-network 测试与单测混放；`tests/test_frontend_plan.py` 名字与内容不符（测的是后端行为） |
| 14 | 无 CI、Python 侧无 lint/format 配置（前端有 oxlint） |
| 15 | `requirements.txt` 无版本约束；`pandas` 全项目未 import（只有 pyarrow）；`beautifulsoup4` 仅服务于下方死代码 |
| 16 | `backend/data/chroma/` 下有**两个** HNSW segment 目录（各 8.9 MB）→ `build_index` 只 `delete_collection`，遗留孤儿索引目录 |
| 17 | 前端 `types.ts` 的 `Candidate` / `GameAnalysis` / `GameCard` 三份近乎重复，靠 `App.tsx:49 toCard()` 抹平；后端也没有统一 card schema |
| 18 | `App.css` 用 `zoom: 1.2` + `calc(100vh/1.2)` 做整体放大，脆弱 |
| 19 | `docs/PROJECT.md` 与代码不同步：称 `graph.py` 498 行（实际 599）、提到不存在的 `BUG.md`、说"三种导入模式"（UI 只剩 SteamID）、`/api/dlc` 仍写 mode+value |

### 1.3 多余文件清理清单

| 文件 / 目录 | 判定 | 依据 |
|---|---|---|
| `backend/app/routers/recommend.py` | **删除**（并在 `main.py:20` 注销） | 12 行占位，`/api/recommend` 永远返回 `not_implemented`；方向 A 落地时再写真的 |
| `frontend/src/assets/react.svg` | **删除** | Vite 脚手架残留，全项目零引用 |
| `frontend/src/assets/vite.svg` | **删除** | 同上 |
| `frontend/src/assets/hero.png` | **删除** | 零引用（grep 全前端 tsx/ts/css/html/json 无匹配） |
| `frontend/public/icons.svg` | **删除** | 零引用；`favicon.svg` 被 `index.html:5` 引用，保留 |
| `frontend/dist/` | **删除** | 构建残留（3 个文件），已 gitignore |
| `frontend/README.md` | **删除** | Vite 模板默认文档，与根 README 重复 |
| `PLAN.md`（根） | **合并后删除** | 初始计划稿，与 `docs/PROJECT.md` 重复且多处过期（三种导入模式、`demo/`、RecommendCards、"3000~5000 条"）。把其中仍有价值的 **Steamworks key 申请指南** 与 **密钥泄露自查** 并入 `docs/PROJECT.md` 后删除 |
| `TODO.md`（根） | **合并后删除** | 问题一~五全部完成，仅剩"问题六 俗称/简称识别"；并入 `docs/PROJECT.md` 路线图后删除，避免根目录三份 md |
| `steam/client.py`：`scrape_profile_games` / `_parse_games_html` / `_norm` / `match_games_by_name`（:203-321） | **删除** | UI 已只保留 SteamID 导入（TODO.md 问题一）；主页抓取脆弱、易被限流。连带 `beautifulsoup4` 依赖移除 |
| `backend/tests/test_steam_client.py` | **重写** | 原文只测上面那段死代码;已改为 11 个离线单测(令牌桶限速 / TTL 缓存 / DLC 纯函数) |
| `backend/data/chroma/10417c04-*` 等孤儿 segment | **删除** | 索引遗留;活跃 segment 是 `9ead40c0-*`(可从 `chroma.sqlite3` 的 `segments` 表确认) |
| 所有 `__pycache__/` | 清理 | 非源码 |
| `notes/`(INTERVIEW/RESUME/知识点详解/简历建议) | **保留** | 已 gitignore,确认不入库即可 |
| `backend/data/`(corpus/parquet/chroma) | **保留** | 已 gitignore,本地数据 |

> ✅ 以上清理**本轮已执行完毕**。
> ⚠️ 删除 `url` / `names` 导入模式后**不再做兜底**(已确认决策):演示完全依赖 Steam 网络与 API key。若将来需要离线演示,再补 `backend/samples/demo_library.json` + `mode=demo`。

---

## 2. Phase 0：共同地基（两个方向都必须先做，约 1 周）

**为什么必须先做**：两个方向的成败都取决于"能不能量化对比"。方向 B 的"调优"没有评测集就是玄学；方向 A 的"算法更好"没有离线指标就是自说自话。现在项目里一个评测集、一条基线数字都没有。

| 任务 | 产出 | 说明 |
|---|---|---|
| **P0.1 清理与重构** | 干净的仓库 | 执行 1.3 清单；`requirements.txt` 去 `pandas`/`bs4` 并加版本约束 |
| **P0.2 可观测性** | `app/obs.py` | 结构化日志 + 每次请求记录 `prompt_tokens / completion_tokens / latency / cache_hit`。LLM 调用统一走一个 wrapper。**这是后面所有"优化了多少"数字的来源** |
| **P0.3 LLM 抽象层** | `app/llm/{provider,client,registry}.py` | `.env` 加 `LLM_PROVIDER=deepseek\|openai_compat\|local`；统一构造 `ChatOpenAI`；**能力探测**：该 provider 是否支持 `response_format` / guided decoding，不支持则走 prompt 约束 + 校验重试（解决 P1-4）；把散落的 `get_llm()` / `_llm_stream_text` 收进来。`graph.py` 的 `get_llm()` 保留为薄封装以兼容现有测试 |
| **P0.4 评测集与评测脚本** | `backend/eval/` | ① 路由：60~100 条标注样本 → intent accuracy / JSON 合法率 ② 游戏名抽取：40 条（中/英/简称）→ 精确匹配率 ③ 推荐理由：20 条 → LLM-as-judge（相关性/无编造/长度合规）+ 人工抽检 ④ 离线推荐指标（Phase 1 定义）⑤ 输出 `eval/reports/*.json` 并与基线 diff |
| **P0.5 测试与 CI** | pytest + ruff + GH Actions | 测试分 `unit` / `live` 两个 marker，并**拆目录**（`tests/unit/` 与 `tests/live/`）；`test_frontend_plan.py` 重命名归位；CI 只跑 unit。<br>**实测依据**：`unittest discover` 要 **22.1s**，只因它会 import 那 3 个脚本式测试文件（`test_frontend_plan.py` / `test_recommend.py` / `test_stream.py`），连带加载 torch + sentence-transformers；而真正跑那 11 个单测只要 **0.2s**。拆开目录能把开发循环从 22s 降到 0.2s —— 这是"为什么要有测试结构"最直接的证据 |
| **P0.6 会话记忆** | `AsyncSqliteSaver` + `thread_id` | 顺手补上 P1-11 的短板（对方向 A 的"多轮偏好澄清"、方向 B 的 prefix caching 都有用） |
| **P0.7 文档刷新** | `docs/PROJECT.md` 同步 + README 减肥 | 本文件 `docs/ROADMAP.md` 成为唯一路线图 |

**Phase 0 验收**：`pytest -m "not live"` 全绿；`python -m eval.run --suite route` 能打印基线数字；一次推荐请求的 token/延迟在日志里可见；`git ls-files` 里再无多余文件。

---

## 3. 方向 A：混合（hybrid）推荐算法（离线预计算 + 交给 Agent 调用）

**深度在哪**：只做"用户向量余弦"就是查表，简历上说不响。要做深，必须补齐推荐系统的完整链条：**数据 → 多路召回 → 排序(LTR) → 离线评测 → 在线服务 → Agent 集成**。

### A0. `corpus.jsonl` 能不能直接用？——**不能**，但它不是没用了

实测结论（证据：`backend/data/corpus.jsonl`，5362 条）：

字段只有 13 个，**全是物品侧**：
`appid / name / text / genres / categories / price_cents / positive / negative / metacritic_score / release_date / dlc_count / estimated_owners / header_image`

逐项探测 `user` / `steamid` / `review` / `playtime` / `interaction` / `rating` / `session` / `timestamp` 在字段名中**全部不存在**；唯一含 `date` 的是 `release_date`（发行日期，不是行为时间）。字段非空率也不是问题所在——问题是**维度缺失**：

| 想做的事 | `corpus.jsonl` 能否支撑 |
|---|---|
| 内容相似度 / 用户向量（现有 RAG 已在做） | ✅ 可以 |
| **离线预计算 item-item 相似度矩阵**（降算力的核心） | ✅ 可以 —— 但它是**性能优化**，不是新算法 |
| ItemKNN 协同过滤（"喜欢 A 的人也喜欢 B"） | ❌ **不行**：没有共现 |
| ALS / BPR-MF 矩阵分解 | ❌ 不行：没有 user-item 矩阵 |
| LightGBM LTR 排序（需要标签） | ❌ 不行：没有可构造的相关性标签 |
| Recall@K / NDCG@K 离线评测 | ❌ 不行：没有可留出的交互 |

> **结论**：`corpus.jsonl` 单独用，只能做"内容推荐 + 预计算加速"——和现有 RAG 没有本质区别，**不构成一个值得做的方向**。
> 但它是极好的 **item 特征侧**（genres / categories / 描述 / 评分 / 销量 / 价格 / 发行年份）。只要补上**交互数据**，就能做真正的**混合（hybrid）推荐**。这就是下面 A1 要解决的事。

### A1. 交互数据：公开数据集【✅ A1.1 验证已通过，方向 A 保留】

数据集已下载到 `Steam_November2025_Dataset/`（**已被 `.gitignore` 的 `*.csv` 覆盖，不会入库**）：

| 文件 | 大小 | 结构 |
|---|---|---|
| `zenodo_interactions_dataset.csv` | 116.7 MB | **1,009,745** 行；列：`user_id, group_name, appid, game_name, playtime_forever, playtime_normalized, is_played, playtime_user_deviation, playtime_game_deviation, user_idx, game_idx` |
| `zenodo_games_dataset.csv` | 3.1 MB | **14,736** 款；列：`appid, name, genres, is_free, release_date, recommendations, categories, **publishers**` |

#### A1.1 验证结论（✅ 已完成，实测数据）

**验证 1：item id 是 Steam appid —— ✅ 完全通过**
`220`=Half-Life 2、`730`=Counter-Strike 2、`292030`=The Witcher 3,均为真实 appid;两张表都以 `appid` 为键,可直接与 `corpus.jsonl` join。且 `user_idx` / `game_idx` 与字符串 id **严格 1:1**(已实测),可直接当矩阵下标用,省掉一次映射。

**验证 2：覆盖率 —— ✅ 足够支撑完整 hybrid**

| 指标 | 实测值 |
|---|---|
| 用户 / 物品 / 社群群组 | **7,731** / **9,935** / **15** |
| 原始交互行数 | 1,009,745 |
| **去重后的真实交互数**(见下方⚠️1) | **840,274** |
| 矩阵密度 | **1.094%** |
| 交互/用户 | min 4,median 63~69,max 4,935 |
| 交互/物品 | min 5,median 21,max 7,746 |
| **交互物品 ∩ corpus** | **3,853**(占交互物品 38.8%) |
| **交互记录被 corpus 内容特征覆盖** | **649,443 / 840,274 = 77.3%** |
| games-csv ∩ corpus | 4,137(占 games csv 28.1%) |
| corpus 中无交互的冷 item | 1,509(28.1% of 5,362)→ 内容路仍能召回 |
| 数据集中无 corpus 特征的 item | 6,082 → 只能走协同路 |

**评测切分可行性（leave-one-out 完全够用）**

| 用户交互数 ≥ | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| 用户数 | 7,727 | **7,495** | 6,768 | 5,996 | 4,579 | 2,507 |

**LTR 训练池**：物品交互数 ≥20 有 4,821 个(其中在 corpus 内 2,779)、≥50 有 2,830 个(内 1,896)、≥100 有 1,684 个。

**额外收获:R4「同发行商」通道不再需要抓 Steam API** —— `publishers` 已在 games csv 里(6,201 个不同发行商,Top: Ubisoft 127 / THQ Nordic 115 / SEGA 111 / EA 111 / Square Enix 103 / Devolver 101)。原来估的"5362 次请求、9 分钟"可以直接划掉。

#### ⚠️ 四个必须处理的坑（实测发现）

1. **必须按 `(user_id, appid)` 去重**:数据集的真实粒度是 `(user_id, group_name, appid)` —— `(user,appid)` 上有 **169,471 行重复**,因为 975 个用户属于多个群组(最多 8 个)。两种粒度的去重结果:1,009,745 → **840,274**。只有 393 对(0.05%)playtime 冲突,所以取 `max(playtime)` 合并即可。**不去重会污染训练集。**
2. **`unobserved ≠ 负样本`(MNAR)**:`is_played` 恒为 `1`,且 `playtime_forever` 最小 10 分钟 → 这个数据集是"**玩过 ≥10 分钟的游戏**",**不是完整拥有列表**(所以人均只有 63~69 款)。缺失是非随机的。**影响**:绝对指标会偏乐观;**算法之间的相对比较仍然成立**。必须在文档/README 里显式声明这一点。
3. **playtime 极长尾,必须压缩**:p50=4.2h、p90=53h、p99=831h、p99.9=**3,669h**(最长 597,266 分钟 ≈ 414 天)。直接当回归目标会被极端值带偏 → 用对数或分档。数据集已提供 `playtime_normalized`,**实测确认它等于 `log1p(playtime_forever)`**(逐行验证通过),可直接用。
4. **357 个交互 item 不在 games csv 里**(无名称/类型,如 431960 Wallpaper Engine、322330 Don't Starve Together)→ 名称/类型需从 `corpus.jsonl` 补,或按需补抓;两者都缺的只能走协同路。

#### 落盘格式（A1.2，半天）

统一成 `backend/data/interactions/`：
- `interactions.parquet`：`(user_idx, game_idx, appid, playtime_minutes, playtime_log, group_id)` —— **按 (user,appid) 去重后** 840,274 行
- `items.csv`：合并 `corpus.jsonl` 与 `zenodo_games_dataset.csv` 的物品侧特征(appid, name, genres, categories, publishers, recommendations, 评分, 价格, 发行年份…),标注每行的特征来源
- `stats.json`：本节的统计指标,供 README / 面试材料引用

> **结论:方向 A 按完整 hybrid 推进,不需要退回半合成协议。** 下面 A2~A6 全部保留。

### A2. 离线特征与预计算矩阵（2~3 天）

- **item 内容向量**：复用现有 sentence-transformers，把 5362 款的 embedding **离线落盘**（`items.npy` + `appids.npy`）→ **在线彻底不加载模型、不做推理**，只做点积。这是"降算力"的核心。
- **item-item 相似度**：`S = normalize(E) @ normalize(E).T`，TopK 稀疏化（每 item 200 邻居）→ `itemknn_topk.npz`；另一路来自 A1 共现（cosine / Jaccard），两路互补。
- **用户向量**：`log1p(playtime)` 加权 + 近期权重 + 归一化；按 `hash(library)` 缓存。
- **质量先验**：好评率贝叶斯平滑、metacritic、owners 分位。
- 体量估算：5362×384 float32 ≈ 8 MB；item-item TopK200 ≈ 4.3M 边 ≈ 17 MB。**可全部 mmap 常驻，CPU 毫秒级**。

### A3. 召回层（多路，离线可算，在线纯查表）（3~4 天）

| 通道 | 内容 | 配额示例 |
|---|---|---|
| R1 | 内容向量 ANN（现有 RAG，改读预计算矩阵） | 50 |
| R2 | ItemKNN 共现/相似（"喜欢 A 的人也喜欢 B"，来自 A1） | 50 |
| R3 | 热门 / 高口碑先验（冷启动兜底） | 20 |
| R4 | 同发行商 / 同系列 | 20 |
| R5 | 社区共现（A1 数据集自带 Steam 社区群组维度，同群组用户共现） | 20 |

合并去重后按路配额混合，保证覆盖率与多样性。

> ✅ R4 需要的 `publishers` **已在 `zenodo_games_dataset.csv` 里**(6,201 个不同发行商),**不需要额外抓 Steam API** —— 原估的"5362 次请求、9 分钟"已划掉。
> ⚠️ 但只有 3,853 个交互物品在 `corpus.jsonl` 内、4,137 个在 games csv ∩ corpus 内;R4 只对这些 item 生效,其余 item 该通道为空(由其他路兜底)。

### A4. 排序层（真正的"算法"）（4~5 天）

> A1 拿到真实交互数据后，这一层**可以做**（原计划因缺数据而准备跳过）。

- **特征**：各路召回分、item 向量余弦、ItemKNN 分、genre/category overlap 数、Jaccard、好评率、owners 分位、价格、发行年份、playtime 加权亲和、用户在该 genre 上的历史时长占比。
- **模型**：LightGBM `lambdarank`（或 XGBoost `rank:ndcg`）。
- **标签与切分**：隐式反馈来自 A1。按**用户**做 leave-one-out（或按时间切分，若数据集带时间戳）；`playtime/hours` 分档为相关性等级（如 <1h=0、1~10h=1、>10h=2）。**必须按用户分组切分，不能随机切分**，否则同一用户的信息泄漏到测试集。
- **负采样**：训练时对每个正样本采 50~100 个未交互 item（popularity-based 负采样）。
- **基线对比**：随机 / 流行度 / ItemKNN / 纯向量余弦 /（可选）双塔。
- **指标**：Recall@10,20、NDCG@10、MAP、HitRate、Coverage、Intra-list Diversity、**P99 延迟**。
- **产出**：`train_ranker.py` → 模型文件 + 特征重要性报告（面试可直接放图）。

### A5. 在线服务与 Agent 集成（3 天）

- 新建 `app/recommend/`：`index.py`（mmap 加载预计算）、`recall.py`、`rank.py`、`service.py`（`recommend(user_appids, playtime, k, filters) -> list[Candidate]`）。
- **两种集成方式都做，展示 Agent 设计能力**：
  1. **节点内替换式**（最省算力）：`recommend_node` 的召回从 `rag_store.search` 换成 `recommend.service`，LLM 只写理由 → 在线零 embedding 推理。
  2. **工具调用式**（更"Agent"）：用 `bind_tools` 暴露 `recommend_games(...)`，由 LLM 决定何时调用、传参 → 顺带解决方向 B 的 function-calling 适配问题。
- **冷启动**：库为空/过小 → 回退 R3 + LLM 引导询问偏好，把回答当作临时用户向量。
- 新增 `POST /api/recommend`（占位桩已删除，这里是真实现），返回结构化候选 + 可解释特征（"因为你喜欢 X"）。

### A6. 评测与消融（2 天）

与现有纯 RAG 基线的正面对比表；消融：去排序层 / 去 ItemKNN / 去内容路（纯协同）/ 换用户向量策略；在线 P50/P99、内存占用、是否需 GPU。

**里程碑**：~~A1.1 接入验证~~ ✅ 已完成 → **A1.2 落盘标准化(0.5d)** → A2 预计算矩阵(2~3d) → A3 多路召回(3~4d) → A4 排序层(4~5d) → A5 集成(3d) → A6 评测消融(2d) ≈ **2.5 周**

**风险**：
- ✅ ~~数据集 item id 与 appid 对不上~~ → **已排除**（实测就是 Steam appid，且 `user_idx`/`game_idx` 与 id 严格 1:1）
- ⚠️ **MNAR（缺失非随机）**：数据集是"玩过 ≥10 分钟"而非完整拥有列表 → 绝对指标偏乐观。**对策**：只用它做算法间相对比较 + 在 README 显式声明；生产侧仍用用户真实完整库
- ⚠️ **协同路只覆盖 3,853/9,935 个物品的内容特征** → 交集外物品靠协同+流行度兜底，消融实验要单独看聚合+内容路的增益
- A4 在小数据上过拟合 → 按用户分组切分 + 早停 + 简化特征集

---

## 4. 方向 B：本地 LLM 推理服务 + 调优（AI Infra）

**深度在哪**：**推理服务化的全链路** —— 模型选型/量化 → 服务框架 → 并发与显存调优 → 结构化输出约束 → 微调 → 端到端评测。
**硬约束**：RTX 4060 Laptop **8 GB VRAM**。**已定选 3B/4B 小模型**，8 GB 下余量充足（见 B2）。

### B0. 环境路径（✅ 已定：WSL2 + CUDA + vLLM）

用户已有 **Ubuntu 22.04 的 WSL2 环境，内含验证过的 CUDA + PyTorch** —— 这直接解决了 vLLM 不官方支持 Windows 的问题，路径确定：

| 路径 | 定位 | 说明 |
|---|---|---|
| **WSL2 (Ubuntu 22.04) + CUDA + vLLM** | ✅ **主路径** | Linux 原生支持，PagedAttention / continuous batching / prefix caching / guided decoding 全都能用，AI Infra 叙事最硬 |
| Windows 原生 llama.cpp / Ollama | 可选兜底 | 只在需要"一键演示"或 CPU offload 实验时用；不作为主线 |

**环境就绪清单（B0 的交付物，1 天）**：
1. 在 WSL2 内建**独立** venv（**不动 Windows 侧 venv**，避免污染已跑通的 MVP）；建议 **Python 3.10/3.11**（vLLM / bitsandbytes 对 3.13 支持较新）。
2. 确认 `nvidia-smi` 在 WSL2 内可见 GPU、`torch.cuda.is_available()` 为 True，并记录 CUDA / 驱动 / torch 版本组合。
3. 装 vLLM 并起一个 hello-world 服务，用 `curl` 打通 OpenAI 兼容端点。
4. 检查 WSL2 内存/磁盘：在 `%UserProfile%\.wslconfig` 里按需设 `memory=`、`swap=`；确认 WSL 虚拟盘剩余空间（3B-4bit ≈ 2.5 GB + CUDA torch ≈ 3 GB + 数据集 ≈ 1 GB）。
5. **工作区迁移见 §7**。

### B1. 基线画像（2~3 天，先测量后优化）

用 P0.2 的统计，量化现状：各类意图的 prompt/completion tokens、延迟分布、每千次请求成本 → 折算"本地化能省多少钱"（面试讲钱最有说服力）。同时用 P0.4 建立在线基线：路由 JSON 合法率、意图准确率、理由质量分、端到端 P50/P99。

### B2. 模型选型与量化（2~3 天）

**已定：开源预训练小模型 3B/4B + QLoRA，全程本地。** 8 GB 下这个尺寸余量很舒服，也让 B5 手搓推理能真跑起来。

| 候选 | 定位 | 备注 |
|---|---|---|
| **Qwen3-4B**（Instruct） | **首选** | 中文强、生态成熟（vLLM / Unsloth / QLoRA 配方齐全），fp16 都能放进 8 GB |
| **Qwen2.5-3B-Instruct** | **KV cache 最省** | GQA 仅 2 个 KV head → KV 开销极小，并发与长上下文余量最大 |
| Llama-3.2-3B-Instruct / Phi-3.5-mini(3.8B) | 消融对照 | 中文与 JSON 服从性需实测，作为"为什么选 Qwen"的对照证据 |

- **量化**：AWQ/GPTQ（vLLM 原生）vs GGUF Q4_K_M/Q5（llama.cpp）。3B/4B 在 4-bit 下权重仅 **~2~2.5 GB**，其余 5~6 GB 全留给 KV cache。
- **KV cache 预算必须算清并写进文档**（这就是 AI Infra 的"算得清"）：

  ```
  每 token 字节 = 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
  ```

  - Qwen2.5-3B（36 层，**2** 个 KV head，head_dim 128）≈ **36 KB/token**（fp16）→ 8k 上下文 ≈ 0.3 GB/序列
  - Qwen3-4B（36 层，**8** 个 KV head，head_dim 128）≈ **144 KB/token**（fp16）→ 8k 上下文 ≈ 1.2 GB/序列

  > ⚠️ 上表数字是估算，**开工第一件事是从各模型自己的 `config.json` 读出真实的 `num_hidden_layers` / `num_key_value_heads` / `head_dim` 重算**，做成表格。这个"先算再跑、算完再用实测验证"的动作本身就是面试可讲的点。
  > 可见 **KV cache 效率本身就是选型依据**：3B 的 GQA 更激进，同样显存下并发数差 3~4 倍 —— 这正是"量化三角"之外的第 4 个维度。
- **量化的质量代价必须实测**：同一评测集上对比 fp16 vs AWQ-4bit 的意图准确率 / JSON 合法率（并入 B3 的"量化三角"）。
- 产出：选型对比表（显存 / 质量 / 吞吐）+ KV cache 预算表 → 结论落到 `.env`。

### B3. 服务化与推理优化（5~7 天，核心）

vLLM 起服务（3B/4B 下可以把上下文和并发都开大）：

```bash
vllm serve <model> --quantization awq --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --max-num-seqs 32 --enable-chunked-prefill
```

**开 guided decoding（xgrammar / outlines）强制 router 输出 JSON schema** → 直接消灭现在的"JSON 解析失败静默降级"（P1-4）。

> 3B/4B 的好处：显存不再捉襟见肘，**并发可以真的扫到 16~32**，所以下面这些曲线会比 7B 方案有说服力得多——"同样的卡，换小模型 + 调参后能扛多少并发"是很好的面试素材。

**每个优化项都要有前后对比数字**，产出 `docs/bench/` 报告 + 图表（`vllm bench serve` 或自写 asyncio 压测）：

1. **Prefix caching**：系统提示词固定 → 命中后 TTFT 大幅下降（本项目 prompt 结构非常适合）
2. **Continuous batching**：并发 1→32 的吞吐曲线（tokens/s）与单请求延迟的权衡
3. `max-num-seqs` / `gpu-memory-utilization` / `max-model-len` 三方扫描 → 显存-吞吐-上下文 的帕累托图
4. **Chunked prefill**：长 prompt 与 decode 混跑，降低 TTFT 抖动（本项目的 DLC / 画像 prompt 很长，正好能体现）
5. **量化三角（+1）**：fp16 vs AWQ-4bit 的 质量/显存/吞吐；再加上 KV cache 效率（GQA head 数）作为第四维
6. **流式链路对齐**：vLLM 的 OpenAI 兼容流式 → 现有 SSE 链路不变（`langchain-openai` 直连 `localhost:8000/v1` 即可，这是好消息）

### B4. 微调（"调优"的实质化）（5~7 天，全程本地）

**先澄清一个问题：微调的数据不是瓶颈。**

"数据集不好搞"针对的是**人工标注**。本项目走 **自蒸馏（self-distillation）** 路线：用强模型（现有 DeepSeek API）**批量生成**训练目标，再蒸馏进本地 3B/4B。这不需要任何人工标注，成本只是几十元 API 费，而且"用强模型造数据训小模型"本身就是标准且好讲的做法。真正难的是**评测**（怎么证明微调有用），那个由 P0.4 解决。

数据配方（目标 5k~15k 条 SFT 样本）：

| 任务 | 生成方式 | 目标量 | 备注 |
|---|---|---|---|
| 意图路由 | 真实/合成用户消息 → 期望 JSON | 2k | 必须覆盖边界样例（含混意图、口语、错别字） |
| 游戏名抽取 | 含中/英/俗称的消息 → 正式名 | 1.5k | 顺带解决 `PROJECT.md` 遗留的"俗称/简称识别" |
| 推荐理由 | 用户画像 + 候选 → `{intro, reasons}` JSON | 3k | **用户侧从 A1 数据集的 7,731 个真实库采样**，比手编画像真实得多 |
| DLC 摘要 | DLC 报告 → 自然语言摘要 | 1.5k | 复用 `_dlc_context` 生成确定性上下文 |
| 通用中文指令 | 公开小数据集（BelleGroup 一类） | 1~2k | **只占 10~20%**，用于防灾难性遗忘 |

- **数据质量三道过滤**：① JSON schema 校验 → ② 规则校验（意图合法、抽取名非空、无编造游戏名）→ ③ 去重 + 长度分布检查。
- **held-out 评测集（300~500 条）单独切分，绝不入训练** —— 这是 B4 结论可信的唯一前提。
- **方法**：QLoRA（peft + bitsandbytes，4-bit NF4 + 双量化）+ LoRA rank 16~32。**3B/4B 在 8 GB 上：seq 2048 + batch 1~2 + gradient checkpointing + paged AdamW → 有余量，不需要租云 GPU。**
- **训练重点**：**结构化输出能力** —— 让 3B/4B 稳定产出 router / 理由的 JSON，从而**减少对 guided decoding 的依赖**。这是与 B3 呼应的闭环，也是最能体现"微调确实有用"的指标。
- **评测对比**：base vs LoRA，在 held-out 集上看 ① JSON 合法率 ② 意图准确率 ③ 理由质量（LLM-as-judge + 人工抽检）④ 长度合规 ⑤ 通用中文能力小测（查灾难性遗忘）。
- 产出：LoRA adapter + 训练脚本 + loss 曲线 + 对比表 + merge 后的 vLLM 服务配置 + **失败案例手记**（面试最容易被追问的部分）。

### B5. 手搓推理（Level 2，深水区，加分巨大，可与 B3 并行）

`backend/lab/minigpt/`，纯 torch、单文件、可单测，**每步都有"与 transformers/vLLM 数值一致"的测试**：

1. 加载 HF 权重 → 逐层 forward（对照 transformers 输出）
2. **KV cache** + 自回归解码（对照 greedy 输出一致）
3. 采样：temperature / top-k / top-p / repetition penalty
4. **手写 paged KV cache**（分页管理，讲清 PagedAttention 的动机与实现）
5. **连续批处理调度器**（iteration-level scheduling）+ 流式输出
6. int8 weight-only 反量化 matmul

用 0.5B~1.5B 小模型在 8 GB 上纯验证实现正确性 + 吞吐/显存对比。**这是最能体现 AI Infra 功底的部分。**

### B6. 端到端切换与降级（2 天）

- `LLM_PROVIDER=local` 一键切换（P0.3 已铺好）；保留 DeepSeek 作 fallback（`provider=auto` + 健康检查）。
- 本地小模型服从性差的兜底：缩短 prompt、加 few-shot、确定性规则回退（现有 `_keyword_route` 正合适）。
- 终局对比表：本地化前后 意图准确率 / 理由质量 / P50/P99 / 单请求成本 → 一张图讲完"省了多少钱、掉了多少点"。

**里程碑**：B0 环境就绪(1d) → B1 基线(2~3d) → B2 选型量化(2~3d) → B3 服务化调优(5~7d) → B4 QLoRA 微调(5~7d) → B6 切换降级(2d) ≈ **3.5 周**（B5 与 B3 并行）

**风险**：
- vLLM 在 WSL2 的安装坑（预留 1~2 天，主要是 **CUDA 版本 / 驱动 / torch 版本三者对齐**）
- WSL2 默认内存上限可能偏小 → 用 `.wslconfig` 调 `memory` / `swap`（vLLM 起服务时会预分配 KV cache）
- **自蒸馏数据的质量直接决定微调上限** → 用 P0.4 的评测集把关，宁可少而精；严格过滤"编造游戏名"的样本
- 小模型在"理由质量"上大概率明显掉点 → 用 guided decoding + 规则兜底，并**如实报告掉点幅度**（掉点多少是结论，不是失败）
- 3B/4B 的**中文能力**需实测确认（Qwen 系最稳，Llama/Phi 可能不达标）→ 所以 B2 要留对照实验

---

## 5. 对比与推荐

| 维度 | A 混合推荐算法 | B 本地 LLM 推理 |
|---|---|---|
| 深度来源 | 推荐系统全链路 + 离线评测 | 推理服务 + 显存/并发调优 + QLoRA 微调 + 手搓推理 |
| **主要风险** | **数据集 id 能否与 appid 对齐**（已找到可用数据集，风险从"没有数据"降为"能否 join"） | **环境搭建**（WSL2 + vLLM 版本对齐）+ 自蒸馏数据质量 |
| 硬件依赖 | 低（CPU 毫秒级召回，训练用 LightGBM 不吃 GPU） | 高（必须 GPU；3B/4B 下 8 GB 够用） |
| 与"AI Infra"目标契合 | 中（偏算法/工程） | **高** |
| 面试可讲性 | 指标/消融/特征重要性/冷启动/hybrid 融合 | 显存计算/KV cache/量化/吞吐曲线/QLoRA |
| 现有代码复用度 | 高（corpus 作 item 特征侧 + embedding 复用） | 中（只换 LLM 后端 + prompt 适配） |
| 能证明的"省算力" | 在线无 embedding 推理、CPU 毫秒级 | 零 API 成本 + 数据不出本地 |

### 我的建议

**仍然以 B 为主干，但 A 从"切片"升级为"可做全"。** A1 的数据调查已经落地（找到 CC-BY-4.0 的真实交互数据集），原先"放弃 A"的理由——没有交互数据——已经不成立，所以 A4 排序层重新变成可选项。

- **B 做全（B0→B1→B2→B3→B5→B4）**：目标明确（AI Infra）、WSL2 环境现成、3B/4B 让显存不再是瓶颈（可以把并发/上下文开大做更漂亮的曲线）、QLoRA 全程本地。`get_llm()` 是唯一 LLM 入口，改造面很小。
- **A 做全（A1→A6）**：先用**半天**做 A1 的两个验证（appid 对齐 + 覆盖率）。结论好 → 按 A2→A6 走完整 hybrid 流水线，A4 用 LightGBM LTR；结论差 → 退回半合成协议并如实声明，或放弃。
- **两者合起来才是完整故事**：
  > **混合推荐离线预计算（协同 + 内容 + 社区多路召回、LightGBM 排序，CPU 毫秒级、零 embedding 推理）+ 本地 3B/4B 小模型只做最后的理由生成（零 API 成本、数据不出本地、guided decoding 保证 JSON 合法）= 单次推荐端到端零 API 成本、召回毫秒级、全流程可离线。**
- **两条方向共享同一份数据工作**：A1 的 7,731 个真实用户库，既是推荐算法的交互矩阵，也是 B4 微调数据里"用户画像"的采样池。这是把它俩一起做最大的杠杆。

**如果时间只够一个**：选 B。B 内部优先级 **B3（服务化+调优）> B5（手搓推理）> B4（微调）** —— B3/B5 是"AI Infra"最硬的部分；B4 一定放在 P0.4 评测集之后做，否则无从判断微调是否有效。

---

## 6. 建议的执行顺序（总览）

```
Phase 0  地基（1 周）
   ├─ P0.1 清理 + 真 bug 修复 + 文档刷新      ✅ 已完成
   └─ P0.2~P0.7 可观测性 → LLM 抽象层 → 评测集 → pytest/CI → 会话记忆   ⏳ 待定
   │
   ├─ §7 工作区迁移到 WSL2（0.5 天，可与 P0.2 并行）
   │
   ├── 方向 B 主干（3.5 周）  B0 环境就绪 → B1 基线 → B2 选型量化 → B3 服务化调优 ─┬─→ B6 切换降级
   │                                                                             └─→ B5 手搓推理（并行）
   │                          B4 QLoRA 微调（必须在 P0.4 评测集之后）
   │
   └── 方向 A（3 周）         A1 接入验证(半天，**先出结论**) → A2 预计算矩阵 → A3 多路召回
                             → A4 LightGBM 排序 → A5 接入 → A6 评测消融
```

> ⚠️ **P0.2 / P0.4 是方向 B 的前置条件**：没有 token/延迟统计（B1 基线）与评测集（B2/B4 的对比），B3 的"优化了多少"和 B4 的"微调有没有用"都无法量化。建议在开始 B2 之前先补这两项。
> ⚠️ **A1 的半天验证要尽早做**：它的结论（数据集 id 能否与 appid 对齐）决定 A 是"完整 hybrid / 以协同为主 / 放弃"，越早知道越好，而且这半天不占 GPU。

**关键决策点**

1. ~~**主方向**~~ → ✅ **B 主干 + A 可做全**（见文首「已确认决策」）
2. ~~**`url` / `names` 导入模式**~~ → ✅ **彻底删除，不做兜底**
3. ~~**vLLM 路径**~~ → ✅ **WSL2 Ubuntu 22.04 + CUDA + vLLM**（B0）
4. ~~**微调算力**~~ → ✅ **全程本地，3B/4B + QLoRA，不租云**
5. ~~**是否现在执行 Phase 0 清理**~~ → ✅ **已执行清理 + 三个真 bug 修复**
6. **A 的数据集最终形态** → ⏳ 待 A1 的半天验证结论（appid 对齐 + 覆盖率）
7. **是否现在做 P0.2 / P0.4**（可观测性 + 评测集）→ ⏳ 待定，是 B 的前置条件

---

## 7. 工作区迁移到 WSL2（0.5 天，建议尽早做）

### 7.1 关键前提：不要放在 `/mnt/d`

WSL2 通过 9p 协议访问 Windows 盘（`/mnt/c`、`/mnt/d`），**IOPS 与元数据性能远低于 ext4**。而本项目恰好是 IO 密集型的：

- ChromaDB（`chroma.sqlite3` 87 MB + HNSW 索引）反复读写
- `sentence-transformers` 加载模型、`pip` 安装 CUDA 版 torch（几千个小文件）
- QLoRA 训练时 checkpoint 反复落盘
- 推荐算法的 `items.npy` / `itemknn_topk.npz` 要 mmap 常驻

放在 `/mnt/d` 的典型后果：`build_index.py` 从几分钟变十几分钟，模型加载和数据读取经常卡顿。

> **结论**：仓库放进 WSL2 的 ext4，例如 `~/projects/SteamAssistant`（Windows 侧仍可用资源管理器或 VS Code 通过 `\\wsl$\Ubuntu\home\<user>\projects\SteamAssistant` 打开）。

### 7.2 迁移步骤

```bash
# 1) 在 WSL2 里从 git 克隆（比整目录拷贝干净）
mkdir -p ~/projects && cd ~/projects
git clone <repo-url> SteamAssistant

# 2) 把 .gitignore 掉的本地资产从 Windows 侧拷过去（这些不走 git）
SRC=/mnt/d/Homework/SteamAssistant
cp $SRC/backend/.env                                ~/projects/SteamAssistant/backend/.env
cp $SRC/backend/data/corpus.jsonl                   ~/projects/SteamAssistant/backend/data/
cp $SRC/backend/data/steam_games.parquet            ~/projects/SteamAssistant/backend/data/   # 可选，省一次 192MB 下载

# 3) 只在 Linux 侧建 venv（Windows 的 .venv 完全不可复用）
cd ~/projects/SteamAssistant/backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4) 重建向量索引（跨平台拷 chroma 的 sqlite 不保证可用，重建最稳）
python scripts/build_index.py

# 5) 验证
python -m unittest discover -s tests -v
```

### 7.3 迁移后必须处理的坑

| 坑 | 说明 | 处理 |
|---|---|---|
| **`STEAM_PROXY` 的 `127.0.0.1` 语义变了** | 若代理跑在 Windows 上（如 `127.0.0.1:12450`），在 WSL2 里 `127.0.0.1` 指 WSL 自己，**连不到 Windows 的代理** | 三选一：① Windows 11 开镜像网络（`.wslconfig` 里 `networkingMode=mirrored`），之后 `localhost` 直接可用；② 用 Windows 主机 IP（`ip route show default \| awk '{print $3}'`）；③ 在 WSL 内自建代理 |
| **CRLF 换行符** | Windows 检出的文件是 CRLF，搬过去可能让脚本 / lint 报错 | 加 `.gitattributes`（`* text=auto eol=lf`）后重新 checkout；或 `git config core.autocrlf input` |
| **`.venv/` 必须重建** | Windows 的 venv 是 `Scripts/*.exe`，Linux 是 `bin/`，无法复用 | 见步骤 3 |
| **`data/chroma/` 建议重建** | `chroma.sqlite3` 跨平台拷贝不保证可用；重建只需几分钟 | 见步骤 4 |
| **Python 版本** | vLLM / bitsandbytes 对 3.13 支持较新 | Linux 侧用 **3.10 / 3.11** |
| **两边不要同时改** | D 盘那份留着当备份可以，但**同时编辑会产生分叉** | 明确一个主副本，建议 WSL 侧为主 |
| **前端无需额外配置** | WSL2 的端口会自动转发到 Windows，浏览器直接开 `localhost:5173` | —— |
| **`HF_ENDPOINT` 继续用镜像** | WSL2 内下载 HF 模型同样受限 | 保留 `.env` 里的 `HF_ENDPOINT=https://hf-mirror.com` |

### 7.4 迁移的额外收益

- 方向 B 的 vLLM / bitsandbytes / QLoRA 全在 Linux 侧，**工作区与推理环境同处一个文件系统**，避免跨 `/mnt` 反复读写 checkpoint。
- 与后续 GitHub Actions（Ubuntu runner）环境一致，CI 能复现的问题本地也能复现。
- 8 GB 显存的 CUDA 环境已经在 WSL2 里验证过，省掉最大的环境风险。

### 7.5 迁移后 DSH 的工作区怎么改（已查证）

**结论:当前这个会话无法移动到别的 DSH 工作区,迁移后必须在新工作区开新会话。**

依据(查证 DSH 安装内的官方文档与本地存储)：

1. **工作区是持久实体**:`C:\Users\UMA\.dsh\storages\workspace.json`(领域 v2)里存着每个工作区的 `path`、`title`、`sessionIds[]`。当前会话 `session-c504dbee-…` 属于工作区 `fae0ec8d-…`(path = `D:\Homework\SteamAssistant`);会话日志落在 `C:\Users\UMA\.dsh\sessions\--D-Homework-SteamAssistant--\<会话id>\session.v4.jsonl.zstd` —— **目录名由工作区路径派生**,即会话与工作区是绑定的。
2. **官方文档明确禁止跨目录移入**:`@deepseek-ai/dsh-workspace` 的 README 写着「一个会话只能属于一个项目」,以及「**来自其他目录的会话无法移入**」;没有可解析目录的会话保持 Ungrouped。移除项目也不会删数据,那些会话只是变成 Ungrouped。
3. 工作区路径的校验规则:**必须是已存在的绝对目录**;相对路径、`C:work` 这类盘符相对路径、不存在的路径、以及文件,都会被拒绝。

**操作方式**:在 GUI 里把 WSL 里的项目目录**新增为一个工作区**(DSH 有原生目录选择器:`dsh-client-ui-directory-picker-native` / `dsh-host-directory-picker-native`),然后在那个工作区里开新会话。

**交接上下文**:会话不会自动带过去,`goal` 也不会。最省事的做法是迁移前让当前会话产出一份**交接摘要**(已完成/待办/关键决策/文件清单),在新会话里作为第一条消息贴进去。想搬会话日志文件的话理论上可以拷 `session.v4.jsonl.zstd`,但成员资格由「记录的 cwd」决定,拷过去也只会是 Ungrouped,不值得。

#### 关于「工作区直接指向 `\\wsl$` 路径」的四个坑

DSH 的路径工具**语法层是支持 UNC 的**(`dsh-util-workspace-path` 明确识别 POSIX 绝对路径、Windows 盘符路径和 UNC 路径),但有四个坑:

| 坑 | 说明 | 对策 |
|---|---|---|
| **Windows ACL 沙箱在 UNC 上不工作** | `dsh-sandbox-windows-acl` 靠写 NTFS ACL 来授权写访问(本会话第一条命令就撞到 `SetNamedSecurityInfoW failed (Win32 5): grantWrite(...)`)。WSL 的 9p 共享不支持修改 ACL | 这种工作区要用**完全权限(full access)**模式,别用受限模式 |
| **唯一性靠 `fs.realpath`** | 每个规范路径一条记录。`\\wsl$\Ubuntu\…` 与 `\\wsl.localhost\Ubuntu\…` 可能被当成两个不同目录,产生两条工作区记录 | **只用一种写法**,建议 `\\wsl.localhost\…` |
| **9p 文件系统很慢** | 跨 `/mnt` / UNC 的元数据操作远慢于 ext4,DSH 的目录扫描与文件观察会更明显 | 能放 ext4 就放 ext4;别在 UNC 工作区里跑构建 |
| **⚠️ 最关键:shell 仍然是 Windows 的** | DSH 的 shell 工具在这里是 `pwsh`。工作区指向 `\\wsl$` **只让 DSH 能读写 Linux 文件,不会让你用上 WSL 的 shell** —— 也就是说跑不了 WSL 里的 CUDA / python / vLLM | 见下面的方案对比 |

#### 三种方案对比

| 方案 | 做法 | 适用 | 代价 |
|---|---|---|---|
| **A. 在 WSL2 里跑 DSH**(推荐) | 在 WSL2 内安装并启动 DSH,工作区用原生 Linux 路径 `~/projects/SteamAssistant`,GUI 在 Windows 浏览器里开 | **想真正用上 WSL 的 CUDA / vLLM** —— 也就是本项目迁移的初衷 | 需要 Linux 版 DSH(见下);WSL2 内需装 Node/Electron 或走 web 模式 |
| **B. Windows DSH + `\\wsl.localhost\…` 工作区** | 当前这套 DSH 不动,只是把工作区新增到 WSL 路径 | 只想用 DSH 的编辑/搜索能力去改 WSL 里的代码 | shell 仍是 Windows;沙箱需全权限;9p 慢;**跑不了 WSL 的 Python/CUDA** |
| **C. 项目留 Windows,只把运行时搬进 WSL** | 代码在 D 盘,`.venv`/模型/CUDA 在 WSL,你自己在 WSL 终端里跑训练与推理 | 最省事,不想折腾 DSH | 两边容易分叉;DSH 无法直接驱动训练/推理 |

> **为什么推荐 A**:DSH 本身是跨平台设计的 —— 包清单里有 `dsh-bash-local` / `dsh-tool-bash`(Linux/macOS 路径)与 `dsh-pwsh-local`(Windows 路径),只有 `dsh-sandbox-windows-acl` 是 Windows 专有的。所以「在 Linux 里跑」是被支持的形态。
> ⚠️ **待确认**:本机装的这份是 Windows Electron 发行版(`D:\program\DSH\DeepSeek Harness.exe`,`dsh` CLI 走 `dsh-desktop-host/lib/cli.js`),**是否提供 Linux 安装方式需要单独确认**(比如是否有 Linux 安装包,或 `@deepseek-ai/dsh` 能否用 npm 在 WSL 里装)。这一条不确认,方案 A 就无法落地。
> 折中:如果 A 暂时走不通,就先按 **B** 把代码搬过去(配合完全权限),需要跑训练/推理时自己在 WSL 终端操作。

---

## 附：本次盘点使用的验证方式

- 通读全部后端 Python（`app/` 14 个模块、`scripts/` 2 个、`tests/` 6 个）与前端 TS/TSX/CSS
- `git ls-files` 核对入库文件；`grep` 确认 `pandas` / `bs4` / `assets/*` / `icons.svg` 的真实引用情况
- 实测环境：`nvidia-smi`、`torch.cuda.is_available()`（Windows 侧为 `+cpu`）、`pip list`、磁盘、`wsl --status`（沙箱拒绝，后经用户确认真实存在）、`docker --version`
- **`corpus.jsonl` 字段实测**：5362 条、13 个字段，逐项探测 `user / steamid / review / playtime / interaction / rating / session / timestamp` 全部不存在 → 得出 §3-A0 的结论
- **A1.1 数据集实测**（用 stdlib `csv` 流式遍历 1,009,745 行）：验证 appid 为真实 Steam appid、`user_idx`/`game_idx` 与 id 严格 1:1、`playtime_normalized == log1p(playtime_forever)`、按 (user,group,appid) 无重复而按 (user,appid) 有 169,471 重复、与 corpus 的 77.3% 记录覆盖率、leave-one-out 切分可行性
- **DSH 工作区机制查证**：本地存储 `~/.dsh/storages/workspace.json`（领域 v2）+ `~/.dsh/sessions/--<path-slug>--/`；安装内官方文档 `@deepseek-ai/dsh-workspace`（「来自其他目录的会话无法移入」）、`@deepseek-ai/dsh-util-workspace-path`（识别 UNC）、`dsh-sandbox-windows-acl`（ACL 授权）；`dsh --help` 确认 profile 与 `--resume`
- 联网核实交互数据集：Zenodo [10.5281/zenodo.20013577](https://zenodo.org/records/20013577)（CC-BY-4.0，7,731 用户 / 9,935 游戏 / 100 万+ 交互）、UCSD `steam-200k`（[CSE258 作业](https://cseweb.ucsd.edu/classes/fa20/cse258-a/files/assignment1.pdf)在用）
- 清理后验证：11 个离线单测通过；`tsc -p` 两个 project exit 0；`vite build` 产物文件名与删除前逐字节一致
