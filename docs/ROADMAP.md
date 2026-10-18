# SteamAssistant 深化路线图（初步计划）

> 本文档基于对现有代码的完整通读写成，用于把已完成的 MVP 做深。
> 结论先行：**以「本地 LLM 推理服务 + 调优」为主干（方向 B，AI Infra）**；方向 A（推荐算法）原先因缺交互数据准备放弃，现已 **✅ 完成**，形态为"混合（hybrid）推荐"。两者都建立在同一个「工程化地基 Phase 0」之上。
>
> 📄 **方向 A 的完整实现记录**（数据 / 索引 / 评测协议 / 召回 / 排序 / 在线服务 / 消融 / **18 条可优化点**）
> 见 **[RECOMMENDER.md](RECOMMENDER.md)**。本文档 §3 只保留计划视角的摘要与决策记录。

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
- **A1.2 数据落盘已完成**：新增 `backend/scripts/build_interactions.py` 与 `backend/app/recommend/`；产出 `interactions/users/items.parquet` + `stats.json`（12/12 不变式通过）；`rating_text` 从 `rag/store.py` 抽到 `recommend/features.py` 消重；新增 12 个纯函数单测（`tests/test_recommend_features.py`，离线单测总数 11 → 23）
- **A2 离线索引已完成**：新增 `backend/scripts/build_recommend_index.py` + `backend/app/recommend/index.py`；产出 11,444 物品目录 / 两路 TopK(200) / 用户向量 / 交互稀疏矩阵（共 ~70 MB，10/10 校验通过）；**实测确认在线路径零推理**（load 后 torch/sentence_transformers 未被 import），全库打分 **0.30 ms**；新发现 **embedding 被 128 token 静默截断**（97.5% 文本超限，现有 RAG 同样受影响）
- **A3.1 离线评测框架已完成**：新增 `backend/app/recommend/evaluate.py` + `backend/scripts/eval_recall.py`；实现**两套协议**（全库排序 / 热度匹配候选集）——发现全库排序协议被热度主导（pop 25% vs content 0.45%≈随机），热度匹配协议才让内容信号显形（**1.34x** 于随机）；据此产出嵌入配置的三 seed 对比（recall@50：desc256 57.84% vs cur128 55.22%），**结论暂缓**，留待端到端指标；新增 15 个评测单测
- **A3.2 多路召回层已完成**：新增 `backend/app/recommend/recall.py`（5 路通道 + 配额轮转融合）+ `backend/scripts/eval_recall_channels.py`（单通道消融 / 边际贡献 / 配额扫描）；关键结果 —— **协同单路 27.47%@20 超过朴素热门基线 25.07%**；融合以 1.4pp recall 换 3.5x 覆盖率；**内容配额 50/50 → 20/20**（依据扫描）；**修掉一个让消融虚高 20 倍的跨通道泄漏 bug**；新增 8 个召回单测
- **A4 LTR 排序层已完成**：新增 `backend/app/recommend/rank.py`（26 维特征 + `Ranker`）+ `backend/scripts/train_ranker.py`；**排序层全面胜出** —— recall@10 24.27% vs 融合基线 19.64%（**+23.6%**）、NDCG@10 0.1466 vs 0.1181（**+24.1%**）、MRR 0.127 vs 0.105；特征重要性显示用户侧交叉特征（发行商亲和 / 内容余弦 / 类型时长亲和）进前 7，说明模型在学个性化而非复读热度；**修掉共享可变状态导致基线串台的 bug**（已给引擎加 `config=` 逐次覆盖参数）；新增 10 个特征单测
- **A6 评测与消融已完成**：新增 `eval_embedding_config.py` + `eval_rag_baseline.py`；**与改造前对比 —— recall@10 从 0.80% 提升到 24.00%（约 30 倍），NDCG@10 0.0031 → 0.1430**，且旧 RAG 方案甚至不如"直接推热门"（17.55%）；消融确认排序层净收益 +4.36pp；在线 **P50 9.12ms / P99 36.64ms、常驻 72.8MB、纯 CPU 零推理**；**嵌入配置端到端裁决为「保持 cur128 不重建」**——三配置打平，seed 噪声比配置差异大一个数量级
- **A5 在线服务与 Agent 集成已完成（方向 A 收尾）**：新增 `app/recommend/service.py`（`Recommender` + 懒加载单例）、`app/routers/recommend.py`（`POST /api/recommend` 真实现）、`main.py` 启动预热、11 个服务层单测；`recommend_node` 改为调用本地推荐器（RAG 路径保留为降级兜底），**画像改读离线目录，砍掉每次推荐 30 次 Steam 请求**；可选解释字段 `because_of` / `match_genres` 同时喂给 LLM 提示词与确定性兜底文案；**Agent 推荐节点实测 26ms、单次推荐 P50 7.4ms**；冷启动三条路径均有单测

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

#### A1.2 落盘标准化【✅ 已完成】

产出脚本 `backend/scripts/build_interactions.py`(离线、可重复运行),输出到 `backend/data/interactions/`:

| 文件 | 大小 | 内容 |
|---|---|---|
| `interactions.parquet` | 6.9 MB | **840,274** 行 × 7 列:`user_idx, game_idx, user_id, appid, playtime_minutes(int32), playtime_log, playtime_bucket` |
| `users.parquet` | 0.16 MB | 7,731 行:`user_idx, user_id, n_items, playtime_minutes, group_ids` |
| `items.parquet` | 0.75 MB | 9,935 行 × 21 列:合并两路的物品侧特征 + `in_corpus` / `in_dataset` / `n_interactions` |
| `stats.json` | — | 全部统计指标(机器可读,供 README / 面试材料直接引用) |

`items.parquet` 的列:`game_idx, appid, name, genres, categories, publishers, is_free, recommendations, price_cents, positive, negative, positive_ratio, rating, metacritic_score, estimated_owners_lower, release_year, dlc_count, header_image, in_corpus, in_dataset, n_interactions`

两处结构决策:
- **只给两路都有的列加后缀**(`name_corpus` / `name_dataset` …),数值列保持原名 —— 否则下游到处要写 `_corpus`。
- **`app/recommend/` 已建立**:`features.py` 收纳纯函数(`rating_text` / `positive_ratio` / `owners_lower_bound` / `release_year` / `playtime_bucket`);`rag/store.py` 改为从这里导入 `rating_text`,消掉重复实现,同时让数据管线不必为了一个字符串函数去加载 torch。

**实测校验(12/12 不变式通过)**:`user_idx` / `game_idx` 连续、无重复 `(user,item)`、行数与 `users.n_items` 和 `items.n_interactions` 两侧求和都相等、`playtime_log == log1p(minutes)`、`is_free` 为 `bool`、`recommendations` 为 `int64`、无票数物品的 `positive_ratio` 为空(null 而非伪造 `0`)。

**过程中踩到并修掉的两个坑**(已写进代码注释与单测):
1. **pandas 3.x 的 `NaN` 陷阱**:left-join 之后缺失值是 float `NaN`,而 **`NaN` 在 Python 里是真值**,`value or ""` 兜不住它 → `re.match` 抛 `TypeError`。已把 `owners_lower_bound` / `release_year` 的入参改为 `object` 并显式 `isinstance(value, str)` 判定,单测覆盖。
2. **join 后的 dtype 退化**:`recommendations` 会变成 `float64`(伪"整数"列)、`is_free` 会变成 `object`(伪"布尔"列)。已在合并后显式 `fillna` + `astype` 定型。

**顺带验证**:357 个"不在 games csv 里"的交互物品中,有 **65 个能从 `corpus.jsonl` 补到名称/类型** → 真正两路都无特征的只剩 **292** 个(与脚本报告一致;如 `322330` Don't Starve Together、`431960` Wallpaper Engine 都靠 corpus 补上了名字)。

> **结论:方向 A 按完整 hybrid 推进,不需要退回半合成协议。** 下面 A2~A6 全部保留。

### A2. 离线特征与预计算矩阵【✅ 已完成，但发现一个待裁决的嵌入配置问题】

产出脚本 `backend/scripts/build_recommend_index.py` + 运行时加载器 `backend/app/recommend/index.py`,输出到 `backend/data/recommend/`:

| 文件 | 大小 | 形状 / 内容 |
|---|---|---|
| `catalog.parquet` | 0.88 MB | **11,444** 个物品:`cat_idx -> appid` + 全部结构化特征与标志位 |
| `content_emb.npy` | 17.58 MB | `[11444, 384]` float32,L2 归一化(无文本的 292 个为零向量) |
| `content_topk.npz` | 18.31 MB | `[11444, 200]` 内容相似邻居 |
| `collab_topk.npz` | 18.31 MB | `[11444, 200]` 协同共现邻居(playtime_log 加权余弦) |
| `user_emb.npy` | 11.87 MB | `[7731, 384]` 训练集用户画像向量(离线预计算) |
| `user_item_csr.npz` | 3.31 MB | `[11444, 7731]` 交互稀疏矩阵(840,274 非零) |
| `manifest.json` | — | 形状 / 构建参数 / 校验结果 / 泄漏警告 |

**物品空间取并集**(corpus 5,362 ∪ 交互 9,935 = **11,444**):交互空间是 A4 训练/评测所必需的;corpus 独有的 1,509 个物品虽无交互数据,线上仍可作为冷物品由内容通道推荐。

**校验 10/10 通过**:两路 TopK 均无自环、下标都指向合格物品、分数降序、`content_emb` 与 `user_emb` 均已 L2 归一化、无内容物品向量为零、`user_item` 非零数 == 交互行数。

**核心承诺已实测兑现**:
- `RecommendIndex.load()` 之后 **`torch` / `sentence_transformers` / `transformers` 全部未被 import** —— 在线路径确实零推理
- 全库 11,444 个物品打分 **0.30 ms**、TopK 查表 **<0.001 ms**、`load()` 871 ms
- 大数组用 `np.load(mmap_mode="r")`,常驻内存交给 OS 页缓存

#### ⚠️ 待裁决:embedding 被 128 token 静默截断

诊断发现 **`paraphrase-multilingual-MiniLM-L12-v2` 的 `max_seq_length = 128`**(而 `max_position_embeddings = 512`),但语料文本 **97.5% 超过 128 token**(p50 = 241),平均丢掉 116 token:

```
模板:  name(9) + "Genres:"(12) + "Categories:"(29) = 50 token
       只剩 78 token 给描述,而描述本身需要 194 token  → 描述被砍掉大半
```

**这同时是现有 RAG(ChromaDB)的缺陷** —— chroma 用同一模板、同一截断限制建的索引,所以"语义检索"实际主要在匹配 name + genres + categories。

对照实验(3,000 条语料):

| 配置 | 类型重合@10 | 随机基线 | 名字词共享率 |
|---|---|---|---|
| **A. 现模板 + 128(当前线上)** | **90.3%** | 65.7% | **20.2%** |
| B. 现模板 + 256 | 85.1% | 66.2% | 10.9% |
| C. 描述前置 + 256 | 84.3% | 65.9% | 10.7% |
| D. 描述前置 + 384 | 84.2% | 66.2% | 10.2% |

**结果反直觉**:放宽截断后"类型重合"反而**下降**(90.3% → 84~85%),但"名字词污染"**减半**(20.2% → ~10%)。原因是 128 token 时文本几乎只剩 name+Genres+Categories,embedding 等于"类型词袋",**类型重合率高有一部分是循环论证**(类型本来就写在文本里)。

**结论:类型重合率不能用来裁决这个配置**,必须用真实交互上的 `Recall@K`。已并入 A3 第一步完成,见下。

#### ✅ 裁决结果（A3.1 评测,3 个 seed）

用**热度匹配候选集**协议（见 A3.1,随机命中率 9.9%）测内容通道,三个 seed 的均值:

| 配置 | recall@10 | recall@20 | **recall@50** | **全库 recall@50** |
|---|---|---|---|---|
| cur128（当前线上） | **13.47%** | 24.51% | 55.22% | 0.75% |
| cur256 | 11.86% | 23.63% | 55.61% | 0.93% |
| **desc256（描述前置 + 256）** | 11.87% | **25.37%** | **57.84%** | **1.36%** |

三个 seed 之间不重叠（desc256 的 recall@50 = 57.96 / 58.19 / 57.38,cur128 = 55.22 / 54.69 / 55.76）,差异是稳定的。

**关键判断:该看哪个指标。** 内容通道在 hybrid 架构里是**候选生成器**（配额 50,交棒给 A4 的 LTR 精排）,
**不是最终排序器**。所以决策口径应该是 **recall@50**（能不能把目标捞进候选池）,
而不是 recall@10（那是排序指标）。

按这个口径:**`desc256` 全面胜出** —— recall@50 高 2.6pp、recall@20 高 0.9pp、
全库 recall@50 高出 **1.8 倍**（1.36% vs 0.75%）。cur128 只在 recall@10 上领先。

> 换个角度理解这个"反直觉"结果:128 token 截断后文本≈类型词袋,对"精确命中前 10"有利;
> 把描述放进来会**拓宽**语义匹配面 —— 对候选召回（recall@50）有利,对精确排序不利。
> 这正是"候选生成"和"精排"该分开设计的原因,也让 A4 的存在有了实证依据。

**决定（用户裁决）：暂不改动，保持 `cur128`，把配置裁决推迟到 A3.2 + A4 完成之后，用端到端指标定。**

理由：内容通道是**候选生成器**，它的价值最终体现在"多路召回 + 精排"之后的端到端 `recall@N` 上。
现在按单通道 recall@50 下结论虽然三个 seed 一致，但**换个口径就翻转**（cur128 在 recall@10 领先 1.6pp），
说明 2.6pp 这个量级还不足以支撑"重建两套索引 + 改动已提交 RAG 行为"的代价。

届时候选的改动已记录待用（三处）：
1. `build_item_text` 把描述提到 `Genres`/`Categories` **之前**（让描述在截断里活下来）
2. `app/rag/embedder.py` 显式设 `max_seq_length = 256`
3. 重跑 `build_corpus.py` → `build_index.py` → `build_recommend_index.py`

评测脚本已有 `data/recommend/config_emb/` 缓存，届时对比**不需要重新编码**（省 5 分钟/配置）。

#### ✅ 最终裁决（A6 端到端,3 配置 × 3 seed）→ **保持 cur128,不重建**

把每套 embedding（含其派生的 content TopK）装进**完整管线**重训排序器后：

| 配置 | recall@10 | recall@20 | recall@50 | NDCG@10 | NDCG@20 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|---|
| **cur128（当前线上）** | **24.00%** | 31.94% | 42.90% | **0.1430** | **0.1632** | **0.1850** | **0.123** |
| cur256 | 23.85% | **32.53%** | **43.02%** | 0.1392 | 0.1611 | 0.1820 | 0.119 |
| desc256（描述前置 + 256） | 23.64% | 31.76% | 42.76% | 0.1402 | 0.1608 | 0.1826 | 0.121 |

**三者在端到端上基本打平。** 而且要看清楚噪声量级 —— 单看 cur128 的 recall@10 在三个 seed 之间是
**24.27 / 22.22 / 25.51**（跨度 **3.3pp**），而配置之间只差 **0.36pp**：
**seed 噪声比配置差异大一个数量级**，任何"某配置更好"的说法在这个量级上都站不住。

**结论：不改动,保持 `cur128`。** 理由：
1. 换了没有可测量的收益；
2. 换的成本是重建 corpus + chroma + A2 索引（约 30 分钟），并**改变已提交的 RAG 行为**；
3. cur128 在 recall@10 / NDCG / MRR 上还略微领先。

> **这个结果反过来验证了"推迟裁决"是对的。** A3.1 单通道口径下 desc256 在 recall@50 上领先 2.6pp
> 且三 seed 一致，看起来很像真信号；但装进完整管线后，**排序层把这个差异吸收掉了**。
> 单通道指标是"没有下游时的代理指标"，用它下架构级结论会误导 —— 这是 A6 最值得记录的一条经验。
>
> 那三处候选改动（`build_item_text` 描述前置 / `max_seq_length=256` / 重跑三个脚本）
> 仍然记录在此备用，`config_emb/` 与 `config_topk/` 缓存也保留着，将来若换模型或换语料可几分钟内复测。

> 另外两个已记录但不阻塞的问题:
> - **薄/富文本分簇**:数据集独有物品(6,082 个,只有 name+Genres+Categories)的最近邻 **96.3%** 仍在同类内(富文本物品为 67.8%),即嵌入空间被"文本长度/风格"切成了两簇。物品表已带 `in_corpus` / `has_content` 标志,召回层可按需限制候选池。
> - **用户向量召回偏冷门**:对一个 2,143 款的用户做加权平均后,Top-10 全是"评价较少"的冷门薄文本物品(余弦都挤在 0.90~0.92,判别力弱)。这是"加权平均向量趋近全局质心"的经典问题,**正是要靠 A3 多路召回 + A4 质量先验排序来纠正** —— 也说明单靠内容分不能当推荐用。

### A3. 离线评测框架（✅ 已完成）+ 多路召回层（进行中）

#### A3.1 两套评测协议 —— 为什么必须有两套

新增 `backend/app/recommend/evaluate.py`（切分 + 指标）与 `backend/scripts/eval_recall.py`（CLI）。
按**用户**留一（每个合格用户藏起 1 个物品，7,495 个用户参与），**绝不能随机切分交互对**，
否则同用户的其他物品会同时出现在训练与评测两侧。

**协议 A：全库排序 + 随机留一**

| 通道 | recall@10 | recall@20 | recall@50 | NDCG@20 | MRR | 中位名次 | 覆盖率@20 |
|---|---|---|---|---|---|---|---|
| random | 0.19% | 0.36% | 0.65% | 0.001 | 0.001 | 5542 | 100% |
| popular | 17.55% | **25.07%** | 35.92% | 0.127 | 0.100 | 125 | 1.0% |
| content | 0.29% | 0.45% | 0.80% | 0.002 | 0.002 | 6126 | 8.6% |
| collab | 15.86% | 21.49% | 32.02% | 0.101 | 0.075 | 158 | 8.3% |

**⚠️ 这套协议基本在考"能不能猜中热门游戏"**，而不是"懂不懂这个用户"：一条朴素的热门基线就拿到 **25%**，
而内容通道只有 **0.45%**（≈随机 0.36%）。原因是随机留一的目标物品本身就大概率是热门游戏。
用它做决策会得出"内容通道没用"的错误结论。

**协议 B：热度匹配候选集** —— 每个测试物品配 100 个**同热度桶**（训练集热度等频分 10 桶）的负样本，
即 1 正 + 100 负，随机命中率 ≈ 9.9%。热度先验在桶内拿不到优势，剩下的差异才反映个性化：

| 通道 | recall@10 | recall@20 | lift@10 |
|---|---|---|---|
| random | 9.70% | 19.85% | 0.98x |
| popular | 49.77% | 63.95% | 5.03x |
| **content** | **13.25%** | **24.35%** | **1.34x** |
| collab | 49.06% | 65.38% | 4.95x |

**协议 B 才让内容通道的信号显形**：从"约等于随机"变成高于随机 **1.34 倍**。
`random` 落在 9.70% ≈ 理论 9.9%，说明协议实现正确。

> `popular` 在协议 B 下仍有 5x —— 因为 10 个热度桶还太粗，桶内热度差异仍可利用。
> 想要更纯的个性化度量可把桶数提到 30~50（`evaluate_bucketed(n_buckets=...)`）。
>
> **覆盖率是对比里的关键补充**：`popular` 的 cov@20 只有 **1.0%**（给所有人推同样 100 款），
> 而 `content` / `collab` 是 **8.6% / 8.3%**。多路召回的价值就在于把长尾捞出来，
> 单看 recall 会低估这一点。

#### A3.2 多路召回层【✅ 已完成】

新增 `backend/app/recommend/recall.py`(召回引擎)+ `backend/scripts/eval_recall_channels.py`(消融评测)。
五路通道全部是**离线预计算的查表/点积/bincount**,在线不加载模型。

**单通道消融 + 融合(7,495 用户,候选生成口径)**

| 运行 | recall@20 | recall@50 | recall@100 | cov@20 | cov@50 | cov@100 |
|---|---|---|---|---|---|---|
| only:content_user | 0.48% | 0.84% | 1.41% | 6.3% | 11.1% | 16.2% |
| only:content_itemknn | 1.05% | 2.44% | 4.26% | 14.2% | 24.7% | 36.6% |
| **only:collab_itemknn** | **27.47%** | **39.37%** | **44.44%** | 5.8% | 8.7% | 12.2% |
| only:quality(热度/质量先验) | 21.79% | 32.15% | 40.76% | **1.0%** | 1.7% | 1.7% |
| only:publisher | 5.02% | 8.11% | 10.58% | 23.5% | 31.1% | 35.8% |
| **merge:quota(默认)** | 26.10% | 37.16% | 40.88% | 10.3% | 19.7% | **42.2%** |
| merge:score | 5.06% | 15.40% | 27.40% | 9.7% | 15.2% | 19.4% |

**四个结论:**

1. **`collab_itemknn` 是最强的单通道(27.47%@20),而且超过了朴素热门基线(25.07%)** ——
   这是"传统推荐算法确实有用"的直接证据,不是靠热度堆出来的。
2. **`merge:quota` 用 1.4pp 的 recall@20 换来了 3.5 倍覆盖率**(cov@100:42.2% vs 协同单路的 12.2%)。
   对**候选生成器**来说这是正确的取舍 —— 候选池要广,排序交给 A4。
3. **`merge:score`(各路归一化后加权求和)明显更差(5.06%@20)**。
   "按最大值归一化再相加"会让强通道以未校准的尺度垄断名额。**配额轮转是对的融合方式。**
4. **内容两路贡献很小**(0.48% / 1.05%),`quality` 覆盖率极低(1.0%,给所有人推同一批)。
   但这**不代表内容路没用** —— 它是唯一能覆盖**冷物品**(无交互的物品)和 corpus 独有物品的通道,
   LOO 任务本身奖励"猜中热门",会系统性低估内容路的长期价值。边际贡献消融见下。

**边际贡献消融(去掉某一路后融合结果怎么变)**

| 运行 | recall@20 | recall@50 | recall@100 | cov@20 | cov@50 | cov@100 |
|---|---|---|---|---|---|---|
| merge:quota(基准) | 26.10% | 37.16% | 40.88% | 10.3% | 19.7% | **42.2%** |
| quota_no_content | **27.53%** | **39.67%** | **45.19%** | 8.2% | 18.0% | 30.2% |
| quota_no_collab | 1.36% | 2.72% | 26.52% | 20.1% | 32.4% | **47.3%** |
| quota_no_publisher | 26.10% | 37.16% | 40.85% | 10.2% | 19.3% | 36.5% |

- **去掉内容路反而提升 recall**(+1.4pp @20、+4.3pp @100),代价是 cov@100 从 42.2% 掉到 30.2%
  —— 说明内容路的配额在**挤占**更强的候选。
- **去掉协同路灾难性**(recall@20 从 26.10% 掉到 1.36%)—— 协同通道承载了几乎全部排序能力。
- **去掉发行商路 recall 完全不变**,只掉一点覆盖率 —— 这是**最划算的一路**(纯覆盖,零 recall 成本)。

**内容配额扫描(据此定默认值)**

| content_user/itemknn | r@20 | cov@20 | r@50 | cov@50 | r@100 | cov@100 |
|---|---|---|---|---|---|---|
| 50 / 50 | 26.10% | 10.3% | 37.16% | 19.7% | 40.88% | **42.2%** |
| 30 / 30 | 26.23% | 10.1% | 37.45% | 20.5% | 41.99% | 41.6% |
| **20 / 20（新默认）** | 26.35% | 9.9% | **37.69%** | **22.2%** | 43.26% | 39.3% |
| 10 / 10 | 26.74% | 10.0% | 38.00% | 22.2% | 44.16% | 36.5% |
| 0 / 0 | **27.53%** | 8.2% | 39.67% | 18.0% | **45.19%** | 30.2% |

**默认配额已从 50/50 调到 20/20**:在 recall@50 与 cov@50 上同时更好;
继续降到 0 虽然 recall 更高,但覆盖率掉得快 —— 而**内容路是唯一能覆盖冷物品**
(无交互记录 / corpus 独有物品)的通道,不能砍掉。**这是"内容路的价值在冷启动与长尾,
而不在 LOO recall"的量化依据。**

> ⚠️ 诚实说明:LOO 协议奖励"猜中已经热门的游戏",会**系统性低估内容路**。
> 内容路真正的考卷是 A5 的冷启动路径(库为空 / 库里全是数据集外的新游戏)与
> A6 的覆盖率/多样性指标,而不是这张表。

**在线性能**:融合 200 个候选时,7,495 个用户跑完一轮约 40s → **单用户 <5ms**,全程无模型推理。

**踩到并修掉的一个真 bug（已加回归测试）**：`_merge_by_quota` 的"补齐剩余名额"阶段
最初遍历了**所有**通道（而不是只遍历启用通道），导致 `only:X` 的消融**悄悄混进了其他通道的候选** ——
`only:content_user` 一度测出 **9.25%@20**，真实值只有 **0.48%**，**虚高约 20 倍**。
发现方式是横向对比"全库排序"与"引擎候选"两种口径的逐用户判定，**37/400 用户不一致**。
修复后两种口径一致（0.48% vs 0.45%）。回归测试：`tests/test_recommend_recall.py`。

> **配额语义澄清**：配额是**下限**不是上限 —— 补齐阶段可以在**启用通道内**继续取，
> 但绝不跨通道；`total` 才是上限。这个语义差异正是上面那个 bug 的温床。

> ✅ R4 需要的 `publishers` **已在 `zenodo_games_dataset.csv` 里**(6,201 个不同发行商),**不需要额外抓 Steam API** —— 原估的"5362 次请求、9 分钟"已划掉。
> ⚠️ 但只有 3,853 个交互物品在 `corpus.jsonl` 内、4,137 个在 games csv ∩ corpus 内;R4 只对这些 item 生效,其余 item 该通道为空(由其他路兜底)。

### A4. LightGBM LTR 排序层【✅ 已完成】

新增 `backend/app/recommend/rank.py`(`FeatureBuilder` + `Ranker`)+ `backend/scripts/train_ranker.py`。

**26 维特征**,分四组:

| 组 | 特征 |
|---|---|
| 召回证据 | 5 路的归一化分数与名次(`content_user` / `content_itemknn` / `collab` / `quality` / `publisher`)、命中几路、融合名次 |
| 用户×物品 | 内容余弦、类型重合数、类型 Jaccard、**用户在该类型上的时长占比**、**发行商亲和**、库大小 |
| 物品静态 | 贝叶斯好评率、好评率、销量、Metacritic、价格、发行年份、DLC 数 |
| 热度 | **训练集**交互计数(泄漏红线,见下) |

**关键设计:训练与线上共用同一个 `FeatureBuilder`** —— 它放在 `app/` 而不是训练脚本里,
否则会出现训练-服务偏斜(train/serve skew):离线指标好看,线上却不是同一个分布。

**结果(测试集 1,125 用户,与 A3 同一套留一协议)**

| 方法 | recall@10 | recall@20 | recall@50 | NDCG@10 | NDCG@20 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|---|
| popular_only | 16.89% | 22.93% | 34.04% | 0.0978 | 0.1131 | 0.1349 | 0.085 |
| recall_merge(基线) | 19.64% | 27.47% | 39.47% | 0.1181 | 0.1379 | 0.1618 | 0.105 |
| collab_only(最强单路) | 21.09% | 28.51% | 41.02% | 0.1268 | 0.1455 | 0.1706 | 0.112 |
| **ltr_lambdarank** | **24.27%** | **32.44%** | **43.02%** | **0.1466** | **0.1674** | **0.1885** | **0.127** |

**排序层全面胜出**:相对召回融合基线 recall@10 **+23.6%**(19.64% → 24.27%)、
NDCG@10 **+24.1%**(0.1181 → 0.1466)、MRR **+21%**(0.105 → 0.127);
也超过最强单路协同(21.09% → 24.27%)。这是"多路召回 + 精排"两段式相对"单路召回"的净收益。

**特征重要性(gain)**:`collab_score`(63017)>> `train_popularity`(19427)>
`collab_rank`(14752)> `merge_rank`(6784)> `user_publisher_affinity`(4464)>
`content_cosine`(3728)> `user_genre_affinity`(2987)。
**用户侧的交叉特征进了前 7** —— 说明模型确实在学个性化,而不只是复读热度与协同分。

**训练规模**:5,246 训练用户 / 1,048,804 行 / 2,439 正样本;early stopping 落在 88 轮;
数据构造 32s(逐用户召回+特征)。切分是**按用户**(5,246/1,124/1,125),用户之间零重叠。

**⚠️ 泄漏边界（如实记录）**
- `train_popularity` 只统计**训练用户**的交互 —— 这一项是干净的,且有单测守卫;
- 协同的 item-item 矩阵、内容 TopK 仍在**全量交互**上预计算(含各用户测试边),
  会让协同类特征略微偏乐观。这是业界常见做法,但跨方法比较时要记得;
- 留一的正样本是**随机**挑的,噪声较大 —— 对所有方法同等加噪,相对比较成立,
  绝对数字不能当线上预期(见 `evaluate.py` 的 MNAR 说明)。

**踩到并修掉的第二个真 bug(已加回归测试思路)**:`train_ranker.py` 最初用
"临时改 `engine.config` → 存起闭包 → 恢复 config" 的写法切换单通道基线,
结果恢复得太早,`collab_only` / `popular_only` / 基线跑出**完全相同的数字**。
根因是**共享可变状态**。已改成把配置**绑进闭包**、并给 `RecallEngine` 的
`channel_outputs` / `recall_rows` 加 `config=` 逐次覆盖参数(顺带让引擎可并发调用)。

### A5. 在线服务与 Agent 集成【✅ 已完成】

新增:
- `backend/app/recommend/service.py` —— `Recommender` 门面 + `get_recommender()` 懒加载单例
- `backend/app/routers/recommend.py` —— `POST /api/recommend`(**P0.1 删掉的占位桩,这里是真实现**)
- `backend/app/main.py` —— 加 `lifespan` 启动预热(冷加载 0.9s 不落在首个用户请求上)
- `backend/tests/test_recommend_service.py` —— 11 个单测(含冷启动与解释来源)

**节点内替换式集成**(方案 1,已落地):`recommend_node` 的召回从
`rag_store.search` 换成 `Recommender`,LLM 只负责写开场白与理由。
本地索引不可用时**自动退回原 RAG 路径**,保证最差仍可用。

**顺带砍掉 30 次 Steam 请求**:`_profile_for` 优先用离线目录算画像
(类型/玩法直接从 `catalog.parquet` 读),拿不到时才退回原来的 `appdetails` 版本。
画像还因此更稳定(不受 Steam 限流影响)。

**可解释性接进了 LLM 提示词**:候选行现在带 `because_of`(来自 TopK 邻居的"你在玩《X》")
与 `match_genres`(与画像重叠的类型),理由能落到具体依据上;
**LLM 不可用时的确定性理由也用上了这两个字段** —— 不再是泛泛的"类型相近":

> 《Portal 2》—— 你在玩《PAYDAY 2》,这款和它路子很像,类型上有 Action、Adventure 和你口味重合。

**冷启动三条路径**(都有单测):

| 场景 | 行为 |
|---|---|
| 库为空 | 只走质量/热度通道,给出兜底候选 |
| 库里的游戏全不在目录内 | 同上,并在文案里说明"先按口碑和热度挑几款" |
| 库已覆盖整个候选池 | 返回空 + 诚实文案("已经覆盖了所有相关游戏"),**不误标成冷启动** |

> 最后一条是单测逼出来的语义修正:早期实现把"库能识别但没剩下可推的"也标成
> `cold_start=True`,会让上层吐出"你的库里还没有我能识别的游戏"这种错误文案。

**实测**(`user_idx=1`,184 款库):

| 指标 | 值 |
|---|---|
| `POST /api/recommend` 端到端 | 200,5 条候选 + 画像 + 冷启动标志 |
| Agent `recommend_node`(不调 LLM) | **26 ms** |
| 单次推荐 P50 / P99(184 款库) | **7.4 ms / 8.4 ms** |
| 单次推荐 P50(2,143 款大库) | 42.9 ms |
| 冷进程首次加载 | 910 ms |

> 2,143 款大库的 42.9ms 是**优化后**的数字:解释器早期用 Python 双层循环
> (库 × TopK = 20 万级条目)要 232ms,已改成 numpy 排序取最大值,快 5.4 倍。

**未做的部分(如实记录)**:工具调用式集成(方案 2,`bind_tools`)没有落地 ——
它属于"让 LLM 自主决定何时召回"的形态,更适合与方向 B 的 function-calling 适配一起做,
留到那边统一处理。

### A6. 评测与消融【✅ 已完成】

新增 `backend/scripts/eval_embedding_config.py`（嵌入配置端到端对比）+
`backend/scripts/eval_rag_baseline.py`（现有 RAG 方案对照）。

#### A6.1 与"改造前"的 RAG 方案正面对比

忠实复刻 `app/agent/graph.py` 的多查询 RAG 逻辑（画像 top-30 → top-6 类型 → 1 条类型查询 + top-3 游戏的
"Games similar to X" → chroma 语义检索 → 按 distance 合并去重）。
**为公平起见把单查询 `k` 从应用里的 8 提到 50**，让它拿到约 200 个候选 —— 与新方案召回层同预算。
这个调整对旧方案**有利**，所以下面的差距只会被低估、不会被夸大。

| 方案 | recall@10 | recall@20 | recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| 随机 | 0.19% | 0.36% | 0.65% | 0.001 | 0.001 |
| 热门基线 | 17.55% | 25.07% | 35.92% | 0.127 | 0.100 |
| **现有 RAG 多查询（改造前）** | **0.80%** | **1.16%** | **1.60%** | **0.0031** | **0.002** |
| **混合召回 + LTR（改造后）** | **24.00%** | 31.94% | 42.90% | **0.1430** | **0.123** |

**recall@10 从 0.80% 提升到 24.00%（约 30 倍）**，NDCG@10 从 0.0031 到 0.1430。
更值得注意的是：**现有 RAG 方案（0.80%@10）甚至远不如"直接推最热门"（17.55%@10）**。

> ⚠️ **这个对比只衡量"检索相关性"，不等同于产品体验**。旧方案的卖点是 LLM 生成推荐理由与可解释性，
> 那是另一条价值线，recall@K 测不到。但就"把用户真正会玩的游戏捞进候选"这件事而言，
> **纯语义检索在这份真实交互上几乎等于随机** —— 这与 A3.1 里 `content_user` 通道 0.48%@20 的结论一致，
> 也解释了为什么必须引入协同信号。

#### A6.2 消融汇总（每一项都在 A3.2 / A4 里单独做过）

| 消融 | recall@10 | recall@20 | cov@100 | 结论 |
|---|---|---|---|---|
| 去排序层（只用召回融合序） | 19.64% | 27.47% | 42.2% | **排序层净收益 +4.36pp**（→24.00%） |
| 去协同路 | — | — | — | recall@20 从 26.10% 崩到 **1.36%**，协同承载几乎全部排序能力 |
| 去内容两路 | — | — | 30.2%（原 42.2%） | recall 反而升、覆盖率降 → 内容路的价值在**覆盖率与冷启动**，不在 LOO recall |
| 纯热门 | 16.89% | 22.93% | 1.7% | 覆盖率极低（给所有人推同一批） |

#### A6.3 在线性能（纯 CPU，零模型推理）

| 阶段 | P50 | P95 | P99 |
|---|---|---|---|
| 召回（5 路 + 融合，200 候选） | 4.79 ms | 8.22 ms | 17.58 ms |
| 特征（26 维） | 3.85 ms | 7.78 ms | 18.83 ms |
| LightGBM 打分 | 0.34 ms | 0.60 ms | 0.80 ms |
| **端到端（召回 + 精排）** | **9.12 ms** | **16.77 ms** | **36.64 ms** |

- 索引加载（11,444 物品，mmap）**107 ms**
- 常驻数组合计 **72.8 MB**（content_emb 17.6 + content_topk 18.3 + collab_topk 18.3 + user_emb 11.9 + csr 6.8）
- **需要 GPU：否**
- 对比改造前：旧方案每次推荐要**跑一次 embedding 推理**（首次还要加载模型 10~20s）+ 4 次 chroma 查询

#### A6.4 嵌入配置裁决 → **保持 cur128，不重建**

结论与依据见 [A2 的最终裁决](#) 一节：三配置端到端打平，**seed 噪声（3.3pp）比配置差异（0.36pp）大一个数量级**。
单通道指标（A3.1 里 desc256 领先 2.6pp）被排序层吸收了 —— **代理指标不能用来下架构级结论**。

**里程碑**：~~A1.1~~ ✅ → ~~A1.2~~ ✅ → ~~A2~~ ✅ → ~~A3.1~~ ✅ → ~~A3.2~~ ✅ → ~~A4~~ ✅ → ~~A5~~ ✅ → ~~A6~~ ✅ → **方向 A 完成**（下一步：方向 B —— 本地 LLM 推理与调优）

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
