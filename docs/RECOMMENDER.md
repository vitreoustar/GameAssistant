# 推荐算法（方向 A）实现全记录

> 本文档记录 SteamAssistant **混合推荐算法**从零到在线服务的完整实现过程：数据、离线索引、
> 评测框架、召回、排序、在线服务、端到端消融，以及**踩过的所有坑与已知局限**。
> 面向两类读者：想接手继续优化的人，和面试时想讲清"为什么这么设计"的人。
>
> 代码入口：`backend/app/recommend/`、`backend/scripts/build_*` + `eval_*` + `train_ranker.py`
> 计划与背景：`docs/ROADMAP.md` §3；项目现状：`docs/PROJECT.md`

---

## 0. 最终结果（先说结论）

| 指标 | 改造前（多查询 RAG） | 改造后（混合召回 + LTR） |
|---|---|---|
| **recall@10** | 0.80% | **24.00%**（≈30 倍） |
| **NDCG@10** | 0.0031 | **0.1430** |
| MRR | 0.002 | **0.123** |
| 在线延迟 P50 | 每次要跑一次 embedding 推理 | **7.4 ms**（184 款库） |
| 在线延迟 P99 | 首次还要加载模型 10~20s | **8.4 ms** |
| 需要 GPU | 否（但 CPU 跑 embedding） | **否**（零推理） |
| 常驻内存 | — | 72.8 MB |
| 每次推荐的 Steam 请求 | 30 次 `appdetails` | **0** |

对照基线（同一套留一协议）：

| 方法 | recall@10 | recall@20 | NDCG@10 |
|---|---|---|---|
| 随机 | 0.19% | 0.36% | 0.001 |
| 热门 | 17.55% | 25.07% | 0.127 |
| **改造前 RAG** | **0.80%** | **1.16%** | **0.0031** |
| 协同单路（最强单通道） | 21.09% | 28.51% | 0.1268 |
| **完整管线（召回+精排）** | **24.00%** | **31.94%** | **0.1430** |

> **一句话故事**：原来那套"多查询语义检索"在真实交互上几乎等于随机，甚至**不如直接推最热门**；
> 换成"多路召回 + LightGBM 精排"，并把 embedding 与 Steam 调用全部离线化之后，
> recall@10 提升约 30 倍，在线延迟降到毫秒级、零模型推理。

**A7 多轮对话**（§11）在此之上补了两样状态：**见过哪些**（去重）与**说过什么**（偏好过滤）。
修复前用户说"再推荐点别的"会原样拿到上一批 —— 因为推荐器是游戏库的纯函数，
对话从未进入排序器。

| 多轮能力 | 实测 |
|---|---|
| 连续 5 轮「换一批」 | 25 款零重复、零同名卡片 |
| 补充「我更喜欢策略类的」 | 5/5 命中并排在前面 |
| 偏爱词在池里不存在 | `no_match` + 诚实文案 + 替代候选 |
| 延迟增量 | **+0.24 ms**（P50 9.13 → 9.37 ms） |

---

## 1. 数据层（A1）

### 1.1 为什么不能只用 `corpus.jsonl`

项目原有的 `backend/data/corpus.jsonl`（5,362 条）**只有物品侧特征**，实测字段仅 13 个：

```
appid name text genres categories price_cents positive negative
metacritic_score release_date dlc_count estimated_owners header_image
```

逐项探测 `user / steamid / review / playtime / interaction / rating / session / timestamp`
**全部不存在**。因此：

| 想做的事 | `corpus.jsonl` 能否支撑 |
|---|---|
| 内容相似度 / 用户向量 | ✅ |
| 离线预计算 item-item 相似度 | ✅（但只是**性能优化**，不是新算法） |
| ItemKNN 协同过滤 | ❌ 没有共现 |
| ALS / BPR-MF 矩阵分解 | ❌ 没有 user-item 矩阵 |
| LightGBM LTR 排序 | ❌ 没有可构造的相关性标签 |
| Recall@K / NDCG@K 离线评测 | ❌ 没有可留出的交互 |

**结论**：只用它做出来和现有 RAG 没有本质区别，不构成一个方向。需要外部交互数据。

### 1.2 引入的公开数据集

《Steam_November2025_Dataset》（Zenodo `10.5281/zenodo.20013577`，**CC-BY-4.0**，可自由使用与再分发）：

| 文件 | 大小 | 内容 |
|---|---|---|
| `zenodo_interactions_dataset.csv` | 116.7 MB | 1,009,745 行 |
| `zenodo_games_dataset.csv` | 3.1 MB | 14,736 款物品侧元数据 |

**关键验证（这决定了整个方向能否成立）**：
- `appid` 是**真实 Steam appid**（`220`=Half-Life 2、`730`=Counter-Strike 2、`292030`=The Witcher 3）
- `user_idx` / `game_idx` 与字符串 id **严格 1:1**，可直接当矩阵下标
- `playtime_normalized` 实测**逐行等于 `log1p(playtime_forever)`**

### 1.3 清洗决策与统计（`scripts/build_interactions.py`）

**决策 1：必须按 `(user_id, appid)` 去重。**
数据集的真实粒度是 `(user_id, group_name, appid)` —— `(user,appid)` 上有 **169,471 行重复**，
因为 975 个用户属于多个群组（最多 8 个）。去重后 1,009,745 → **840,274**。
只有 393 对（0.05%）playtime 冲突，取 `max` 合并。**不去重会污染训练集。**

**决策 2：playtime 极长尾，必须压缩。**
p50=252 分钟、p90=3,172、p99=49,694、p99.9=**219,543**（最长 597,266 分钟 ≈ 414 天）。
落盘时同时给 `log1p` 与 4 档分级两列。

**决策 3：物品空间取并集。**
`corpus.jsonl`（5,362）∪ 交互物品（9,935）= **11,444**。
交互空间是训练/评测所必需；corpus 独有的 1,509 个物品虽无交互数据，线上仍可作冷物品由内容通道推荐。

**最终统计（`data/interactions/stats.json`）**

| 项 | 值 |
|---|---|
| 原始 / 去重后交互 | 1,009,745 / **840,274** |
| 用户 / 物品 / 群组 | 7,731 / 9,935 / 15 |
| 矩阵密度 | **1.0940%** |
| 交互/用户（min/中位/max） | 3 / **63** / 4,075 |
| 交互/物品（min/中位/max） | 1 / **18** / 6,613 |
| 可评测用户（≥10 交互） | **7,495** |
| LTR 池（物品 ≥20 / ≥50） | 4,821 / 2,830 |
| 不同发行商 | 4,139 |
| **交互记录被 corpus 覆盖** | **77.3%**（649,443 / 840,274） |

**⚠️ 四个必须记住的坑**

1. **去重**（上面决策 1）。
2. **`unobserved ≠ 负样本`（MNAR）**：`is_played` 恒为 `1`，playtime 最小 10 分钟
   → 数据集是"**玩过 ≥10 分钟**"而非完整拥有列表（人均只有 63 款）。缺失是非随机的。
   **绝对指标偏乐观，只可用于方法间相对比较。**
3. **playtime 长尾**（上面决策 2）。已核实数据集自带的 `playtime_normalized == log1p(playtime_forever)`。
4. **357 个交互物品不在 games csv 里**（无名称/类型）→ 其中 **65 个能从 `corpus.jsonl` 补到**，
   真正两路都无特征的只剩 **292** 个，只能走协同路。

**产出**：`interactions.parquet`（840,274×7）、`users.parquet`（7,731）、`items.parquet`（9,935×21）、`stats.json`

---

## 2. 离线索引（A2，`scripts/build_recommend_index.py`）

**目标**：把"在线要算的东西"全部提前算好，让在线只剩查表与点积。

| 产物 | 形状 / 大小 | 内容 |
|---|---|---|
| `catalog.parquet` | 11,444 行 / 0.84 MB | `cat_idx -> appid` + 全部结构化特征与标志位 |
| `content_emb.npy` | `[11444, 384]` / 16.8 MB | 内容向量，L2 归一化（无文本的 292 个为零向量） |
| `content_topk.npz` | `[11444, 200]` / 17.5 MB | 内容相似邻居（`indices` + `scores`） |
| `collab_topk.npz` | `[11444, 200]` / 17.5 MB | 协同共现邻居（playtime_log 加权余弦） |
| `user_emb.npy` | `[7731, 384]` / 11.3 MB | ⚠️ **目前是死资产**，见 §10.1 |
| `user_item_csr.npz` | 840,274 非零 / 3.2 MB | 交互稀疏矩阵 |
| `manifest.json` | — | 形状 / 参数 / 校验结果 / 泄漏警告 |

### 2.1 实现要点

- **物品空间取并集**（见 §1.3 决策 3），`cat_idx` 是唯一的全局物品下标。
  ⚠️ 项目里因此存在**两个物品空间**：`items.parquet` 的 `game_idx`（0..9,934）与
  `catalog.parquet` 的 `cat_idx`（0..11,443）。混用会静默错位 ——
  这个坑实际踩到过（表现为 scipy 报 "index exceeds matrix dimension"）。
  `app/recommend/index.py: cat_rows_of()` 负责向量化换算。
- **文本模板只有一个实现**：`app/recommend/features.py: build_item_text()`，
  由 `build_corpus.py`（喂 ChromaDB）与 `build_recommend_index.py`（A2 内容向量）共用。
  这样 A6 与旧 RAG 对比时，差异里不会混进"文本模板不同"这个混淆因素。
  **Publishers 等结构化字段故意不放进文本** —— 文本负责语义、结构化特征负责精确信号，
  混在一起不好归因。
- **corpus 物品直接复用 `corpus.jsonl` 里的 `text`**，保证与 ChromaDB 嵌入的文本逐字节一致。
- **矩阵分块计算**：11k×11k 相似度不整体物化，按 chunk 做 `[chunk, d] @ [d, N]` 再逐块取 TopK，
  峰值内存只有几十 MB。函数在 `app/recommend/index.py: content_topk_from_embeddings()`
  （放在 `app/` 而非脚本里，是因为 A6 要重算它）。
- **协同相似度用 `playtime_log` 加权**（不是二值），行 L2 归一化后点积即余弦。
- **用户画像向量**用稀疏矩阵一次算完：`W [U,N] @ E [N,384] -> [U,384]`。

### 2.2 核心承诺的实测验证

```
RecommendIndex.load() 之后:
  torch                 imported: False
  sentence_transformers imported: False
  transformers          imported: False
全库 11,444 个物品打分:  0.30 ms
TopK 查表(50 邻居):     <0.001 ms
```

**在线路径确实零推理** —— 这是"降算力"叙事的硬证据。

**校验 10/10 通过**：两路 TopK 无自环、下标指向合格物品、分数降序、
`content_emb` / `user_emb` 已 L2 归一化、无内容物品向量为零、`user_item` 非零数 == 交互行数。

**编码耗时**：11,152 条文本在 CPU 上 **195.3s**（batch 64，模型
`paraphrase-multilingual-MiniLM-L12-v2`，384 维）。

### 2.3 embedding 的 128 token 静默截断（重要发现）

`paraphrase-multilingual-MiniLM-L12-v2` 的 **`max_seq_length = 128`**，而
`max_position_embeddings = 512`。但语料文本 **97.5% 超过 128 token**（p50 = 241），
平均丢掉 116 token：

```
模板:  name(9) + "Genres:"(12) + "Categories:"(29) = 50 token
       只剩 78 token 给描述,而描述本身需要 194 token  → 描述被砍掉大半
```

**这同时是现有 RAG（ChromaDB）的缺陷** —— chroma 用同一模板、同一限制建的索引，
所以"语义检索"实际主要在匹配 name + genres + categories。

对照实验（3,000 条语料）：

| 配置 | 类型重合@10 | 随机基线 | 名字词共享率 |
|---|---|---|---|
| **A. 现模板 + 128（当前线上）** | **90.3%** | 65.7% | **20.2%** |
| B. 现模板 + 256 | 85.1% | 66.2% | 10.9% |
| C. 描述前置 + 256 | 84.3% | 65.9% | 10.7% |
| D. 描述前置 + 384 | 84.2% | 66.2% | 10.2% |

**结果反直觉**：放宽截断后"类型重合"反而下降，但"名字词污染"减半。
原因是 128 token 时文本几乎只剩 name+Genres+Categories，embedding 等于"类型词袋"，
**类型重合率高有一部分是循环论证**（类型本来就写在文本里）。

→ 因此改用真实 `Recall@K` 裁决，最终结论见 §8.4。

---

## 3. 评测框架（A3.1，`app/recommend/evaluate.py`）

### 3.1 协议设计

**按用户留一（leave-one-out）**：每个合格用户藏起 1 个物品，用剩下的当画像把它找回来。
7,495 个用户参与。

**为什么不能随机切分交互对**：同一用户的物品之间高度相关，随机切分会让
"同用户的其他物品"同时出现在训练与评测两侧，指标虚高到没有参考价值。

**为什么 `train_user_item` 只含训练边**：评测时要把用户"已经玩过的"排除掉。
如果混进测试边，目标物品会先被判为负样本再被找回来，等于自己泄题给自己。

### 3.2 两套协议 —— 以及为什么必须有两套

**协议 A：全库排序 + 随机留一**

| 通道 | recall@10 | recall@20 | recall@50 | NDCG@20 | MRR | 中位名次 | 覆盖率@20 |
|---|---|---|---|---|---|---|---|
| random | 0.19% | 0.36% | 0.65% | 0.001 | 0.001 | 5542 | 100% |
| popular | 17.55% | **25.07%** | 35.92% | 0.127 | 0.100 | 125 | 1.0% |
| content | 0.29% | 0.45% | 0.80% | 0.002 | 0.002 | 6126 | 8.6% |
| collab | 15.86% | 21.49% | 32.02% | 0.101 | 0.075 | 158 | 8.3% |

**⚠️ 这套协议基本在考"能不能猜中热门游戏"**：一条朴素热门基线就拿到 25%，
而内容通道只有 0.45%（≈随机 0.36%）。原因是随机留一的目标物品本身就大概率是热门游戏。
**用它做决策会得出"内容通道没用"的错误结论。**

**协议 B：热度匹配候选集**
每个测试物品配 100 个**同热度桶**负样本（训练集热度等频分 10 桶），即 1 正 + 100 负，
随机命中率 ≈ 9.9%。热度先验在桶内拿不到优势，剩下的差异才反映个性化：

| 通道 | recall@10 | recall@20 | lift@10 |
|---|---|---|---|
| random | 9.70% | 19.85% | 0.98x |
| popular | 49.77% | 63.95% | 5.03x |
| **content** | **13.25%** | **24.35%** | **1.34x** |
| collab | 49.06% | 65.38% | 4.95x |

**协议 B 才让内容通道的信号显形**：从"约等于随机"变成高于随机 1.34 倍。
`random` 落在 9.70% ≈ 理论 9.9%，说明协议实现正确。

> 覆盖率是对比里的关键补充：`popular` 的 cov@20 只有 **1.0%**（给所有人推同样 100 款），
> 而 content/collab 是 8.6%/8.3%。多路召回的价值在长尾覆盖率上，单看 recall 会低估。

### 3.3 指标实现细节

- **用 `(scores > target).sum()` 拿精确名次**，不真的排序 —— 7,495 用户的全库排序是浪费
- **分块评测**：7,495 × 11,444 的打分矩阵一次性物化要 343 MB，按 1024 行分块后峰值只有 ~47 MB
- 指标：Recall@K、NDCG@K、MRR、中位名次、Coverage@K

---

## 4. 召回层（A3.2，`app/recommend/recall.py`）

### 4.1 五路通道

| 通道 | 信号 | 来源 | 在线耗时 |
|---|---|---|---|
| `content_user` | 用户画像向量 @ 内容向量 —— 整体口味 | `content_emb` | 点积 |
| `content_itemknn` | 库内物品的内容邻居 —— "因为你玩了 X" | `content_topk` | 查表 |
| `collab_itemknn` | 库内物品的协同邻居 —— "喜欢 X 的人也喜欢 Y" | `collab_topk` | 查表 |
| `quality` | 质量/热度先验 —— 冷启动兜底 | 调用方注入 | O(1) |
| `publisher` | 同发行商 —— 系列/厂商偏好 | `catalog` | 倒排 |

**为什么两路 item-kNN 单独设通道**：用户向量是对整库的加权平均，库一大就趋近全局质心
（实测：2,143 款的用户召回 Top-10 全是冷门物品，余弦都挤在 0.90~0.92）。
item-kNN 走"逐个物品找邻居再累加"，不会质心塌缩，两者互补。

**⚠️ `quality` 的先验必须由调用方注入**：评测时传**训练集**统计，线上传全量。
写在构造函数里会导致评测泄漏。

### 4.2 融合：配额轮转（quota）而非分数求和（score）

| 运行 | recall@20 | recall@50 | recall@100 | cov@20 | cov@50 | cov@100 |
|---|---|---|---|---|---|---|
| only:content_user | 0.48% | 0.84% | 1.41% | 6.3% | 11.1% | 16.2% |
| only:content_itemknn | 1.05% | 2.44% | 4.26% | 14.2% | 24.7% | 36.6% |
| **only:collab_itemknn** | **27.47%** | **39.37%** | **44.44%** | 5.8% | 8.7% | 12.2% |
| only:quality | 21.79% | 32.15% | 40.76% | **1.0%** | 1.7% | 1.7% |
| only:publisher | 5.02% | 8.11% | 10.58% | 23.5% | 31.1% | 35.8% |
| **merge:quota（默认）** | 26.10% | 37.16% | 40.88% | 10.3% | 19.7% | **42.2%** |
| merge:score | 5.06% | 15.40% | 27.40% | 9.7% | 15.2% | 19.4% |

- **`merge:score` 明显更差**（5.06%）："按最大值归一化再相加"会让强通道以未校准的尺度垄断名额。
- **`merge:quota` 用 1.4pp recall@20 换来了 3.5 倍覆盖率**（cov@100 42.2% vs 协同单路 12.2%）。
  对**候选生成器**来说这是正确的取舍 —— 候选池要广，排序交给 A4。

### 4.3 边际贡献消融

| 运行 | recall@20 | recall@50 | cov@100 |
|---|---|---|---|
| merge:quota（基准） | 26.10% | 37.16% | 42.2% |
| quota_no_content | **27.53%** | **39.67%** | 30.2% |
| quota_no_collab | **1.36%** | 2.72% | **47.3%** |
| quota_no_publisher | 26.10% | 37.16% | 36.5% |

- **去掉协同路灾难性**（26.10% → 1.36%）—— 协同承载几乎全部排序能力
- **去掉发行商路 recall 完全不变**，只掉一点覆盖率 —— **最划算的一路**
- **去掉内容路反而提升 recall**，代价是覆盖率

### 4.4 内容配额扫描 → 默认值定为 20/20

| content_user/itemknn | r@20 | r@50 | cov@50 | cov@100 |
|---|---|---|---|---|
| 50 / 50（旧） | 26.10% | 37.16% | 19.7% | **42.2%** |
| 30 / 30 | 26.23% | 37.45% | 20.5% | 41.6% |
| **20 / 20（新默认）** | 26.35% | **37.69%** | **22.2%** | 39.3% |
| 10 / 10 | 26.74% | 38.00% | 22.2% | 36.5% |
| 0 / 0 | **27.53%** | 39.67% | 18.0% | 30.2% |

20/20 在 recall@50 与 cov@50 上同时更好；降到 0 虽然 recall 更高，但覆盖率掉得快 ——
而**内容路是唯一能覆盖冷物品**（无交互记录）的通道，所以保留。

**配额语义澄清**：配额是**下限**不是上限 —— 补齐阶段可以在**启用通道内**继续取，
但绝不跨通道；`total` 才是上限。这个语义差异是 §9 那个"消融虚高 20 倍"bug 的温床。

---

## 5. 排序层（A4，`app/recommend/rank.py` + `scripts/train_ranker.py`）

### 5.1 26 维特征

| 组 | 维数 | 特征 |
|---|---|---|
| **召回证据** | 12 | 5 路的归一化分数(5)与名次(5)、命中几路(`n_channels`)、融合名次(`merge_rank`) |
| **用户×物品** | 6 | `content_cosine`、`genre_overlap`、`genre_jaccard`、**`user_genre_affinity`**、**`user_publisher_affinity`**、`user_size_log` |
| **物品静态** | 7 | `bayes_rating`、`positive_ratio`、`log_owners`、`metacritic_norm`、`price_norm`、`release_year_norm`、`dlc_count_norm` |
| **热度** | 1 | `train_popularity`（训练集交互计数，见 §5.6 泄漏红线） |

> 完整顺序见 `app/recommend/rank.py: FEATURE_NAMES`（训练脚本按它取特征重要性名次，
> 顺序错了会静默毁掉模型，所以有单测锁定维度与顺序）。

### 5.2 关键设计：训练与线上共用同一个 `FeatureBuilder`

它放在 `app/` 而不是训练脚本里 —— 否则会出现**训练-服务偏斜**：
离线指标好看，线上却不是同一个分布。这是"离线模型不能直接上线"最常见的坑。

### 5.3 训练设置

| 项 | 值 |
|---|---|
| 切分 | **按用户** 5,246 / 1,124 / 1,125（用户之间零重叠） |
| 训练样本 | **1,048,804 行 × 26 列** |
| 正样本 | 训练 2,439 / 验证 511 |
| 目标函数 | LightGBM `lambdarank`，`ndcg_eval_at=[10]`，`truncation_level=30` |
| 超参 | `num_leaves=31`、`min_data_in_leaf=50`、`learning_rate=0.05`（**未调优，见 §10**） |
| 早停 | valid NDCG@10，patience 60 → best_iter **88** |
| 数据构造耗时 | 32s（逐用户召回 + 特征） |

**标签设计**：隐式反馈来自留一。被藏物品的相关性等级由它的 **playtime 分档**给出
（`playtime_bucket + 1`，即 1~4），其余候选为 0。
用分档而不是 0/1，是给 LTR 更多信息；不用原始分钟数是因为长尾会把梯度带偏。

### 5.4 结果（测试集 1,125 用户）

| 方法 | recall@10 | recall@20 | recall@50 | NDCG@10 | NDCG@20 | NDCG@50 | MRR |
|---|---|---|---|---|---|---|---|
| popular_only | 16.89% | 22.93% | 34.04% | 0.0978 | 0.1131 | 0.1349 | 0.085 |
| recall_merge（基线） | 19.64% | 27.47% | 39.47% | 0.1181 | 0.1379 | 0.1618 | 0.105 |
| collab_only（最强单路） | 21.09% | 28.51% | 41.02% | 0.1268 | 0.1455 | 0.1706 | 0.112 |
| **ltr_lambdarank** | **24.27%** | **32.44%** | **43.02%** | **0.1466** | **0.1674** | **0.1885** | **0.127** |

**排序层全面胜出**：相对召回融合基线 recall@10 **+23.6%**、NDCG@10 **+24.1%**、MRR **+21%**；
也超过最强单路协同（21.09% → 24.27%）。

### 5.5 特征重要性（gain）

| 排名 | 特征 | gain |
|---|---|---|
| 1 | `collab_score` | 63017 |
| 2 | `train_popularity` | 19427 |
| 3 | `collab_rank` | 14752 |
| 4 | `merge_rank` | 6784 |
| 5 | **`user_publisher_affinity`** | 4464 |
| 6 | **`content_cosine`** | 3728 |
| 7 | **`user_genre_affinity`** | 2987 |
| 8 | `publisher_score` | 2756 |
| 9 | `release_year_norm` | 2651 |
| 10 | `user_size_log` | 2503 |

**用户侧的交叉特征进了前 7** —— 模型确实在学个性化，而不只是复读热度与协同分。

### 5.6 泄漏边界（如实记录）

- `train_popularity` 只统计**训练用户**的交互 —— 这一项干净，且有单测守卫
- 协同的 item-item 矩阵、内容 TopK 仍在**全量交互**上预计算（含各用户测试边），
  会让协同类特征略微偏乐观。这是业界常见做法，但跨方法比较时要记得
- 留一的正样本是**随机**挑的，噪声较大 —— 对所有方法同等加噪，相对比较成立，
  绝对数字不能当线上预期（MNAR，见 §1.3）

---

## 6. 在线服务（A5，`app/recommend/service.py`）

### 6.1 结构

- `Recommender` 门面：`recommend(games, top_k)` / `build_profile(rows, weights)` / `_explain(...)`
- `get_recommender()` **懒加载单例**（带锁）。加载失败只记录一次原因，不重试 ——
  缺文件时每次请求都去尝试加载会拖慢服务
- FastAPI `lifespan` **启动预热**：冷加载 0.9s 不落在第一个用户请求上
- `POST /api/recommend`：不调 LLM，纯查表 + 精排

### 6.2 冷启动三条路径（都有单测）

| 场景 | 行为 |
|---|---|
| 库为空 | 只走质量/热度通道，给出兜底候选 |
| 库里的游戏全不在目录内 | 同上，文案说明"先按口碑和热度挑几款" |
| 库能识别但已覆盖整个候选池 | 返回空 + 诚实文案，**不误标成冷启动**（§9 bug 4） |

### 6.3 可解释性

`_explain()` 为每个候选找"最像它"的库内游戏，产出 **`because_of`**（"因为你在玩《X》"）；
同时算 **`match_genres`**（与画像重叠的类型）。两者同时喂给：
1. **LLM 提示词**（让理由落到具体依据）
2. **确定性兜底文案**（LLM 不可用时）：

> 《Portal 2》—— 你在玩《PAYDAY 2》，这款和它路子很像，类型上有 Action、Adventure 和你口味重合。

`channels` 字段保留"由哪几路召回"，供调试与未来的前端展示。

### 6.4 性能

| 指标 | 值 |
|---|---|
| Agent `recommend_node`（不调 LLM） | **26 ms** |
| 单次推荐 P50 / P95 / P99（184 款库） | **7.4 / 7.8 / 8.4 ms** |
| 单次推荐 P50（2,143 款大库） | **42.9 ms** |
| 冷进程首次加载 | 910 ms |
| 加载拆解 | import 378ms + 索引 83ms + LightGBM 8ms + 引擎 28ms + 特征构造器 20ms |

> 2,143 款大库的 42.9ms 是**优化后**数字：`_explain` 早期用 Python 双层循环
> （库 × TopK = 20 万级条目）要 **232ms**，改为 numpy 排序取最大值后快 **5.4 倍**。

### 6.5 Agent 集成

`recommend_node` 改为调用 `Recommender`，**原 RAG 路径保留为降级兜底**
（索引缺失时自动退回，服务不会起不来）。

**顺带砍掉 30 次 Steam 请求**：`_profile_for` 优先用离线目录算画像
（类型/玩法直接从 `catalog.parquet` 读），拿不到时才退回原来的 `appdetails` 版本。

多轮去重与偏好过滤是在这一层之上加的（`exclude_appids` / `prefer` / `avoid` 三个参数），
详见 §11。

---

## 7. 端到端评测与消融（A6）

### 7.1 与"改造前"的 RAG 方案正面对比

忠实复刻 `app/agent/graph.py` 的多查询 RAG 逻辑（画像 top-30 → top-6 类型 →
1 条类型查询 + top-3 游戏的 `Games similar to X` → chroma 检索 → 按 distance 合并去重）。
**为公平起见把单查询 `k` 从应用里的 8 提到 50**，让它拿到约 200 个候选 ——
与新方案召回层同预算。**这个调整对旧方案有利**，所以差距只会被低估。

| 方案 | recall@10 | recall@20 | recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| 随机 | 0.19% | 0.36% | 0.65% | 0.001 | 0.001 |
| 热门基线 | 17.55% | 25.07% | 35.92% | 0.127 | 0.100 |
| **现有 RAG（改造前）** | **0.80%** | **1.16%** | **1.60%** | **0.0031** | **0.002** |
| **混合召回 + LTR（改造后）** | **24.00%** | **31.94%** | **42.90%** | **0.1430** | **0.123** |

> ⚠️ 这**只衡量检索相关性**，不等同于产品体验。旧方案的卖点是 LLM 生成理由与可解释性，
> 那条价值线 recall@K 测不到。但就"把用户真正会玩的游戏捞进候选"而言，
> 纯语义检索在这份真实交互上几乎等于随机。

### 7.2 消融汇总

| 消融 | 结论 |
|---|---|
| 去排序层 | recall@10 19.64% → 24.00%，**排序层净收益 +4.36pp** |
| 去协同路 | recall@20 从 26.10% 崩到 **1.36%** |
| 去内容两路 | recall 反而升、cov@100 从 42.2% 掉到 30.2% → 价值在覆盖率与冷启动 |
| 纯热门 | recall@20 22.93% 但 cov@20 只有 1.7% |

### 7.3 在线延迟与内存（纯 CPU）

| 阶段 | P50 | P95 | P99 |
|---|---|---|---|
| 召回（5 路 + 融合，200 候选） | 4.79 ms | 8.22 ms | 17.58 ms |
| 特征（26 维） | 3.85 ms | 7.78 ms | 18.83 ms |
| LightGBM 打分 | 0.34 ms | 0.60 ms | 0.80 ms |
| **端到端** | **9.12 ms** | **16.77 ms** | **36.64 ms** |

- 索引加载 **107 ms**（mmap）
- 常驻数组 **72.8 MB**（content_emb 17.6 + content_topk 18.3 + collab_topk 18.3 + user_emb 11.9 + csr 6.8）
- **不需要 GPU**

### 7.4 嵌入配置裁决 → **保持 cur128，不重建**

把每套 embedding（含其派生的 content TopK）装进**完整管线**重训排序器后：

| 配置 | recall@10 | recall@20 | recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| **cur128（当前线上）** | **24.00%** | 31.94% | 42.90% | **0.1430** | **0.123** |
| cur256 | 23.85% | **32.53%** | **43.02%** | 0.1392 | 0.119 |
| desc256（描述前置 + 256） | 23.64% | 31.76% | 42.76% | 0.1402 | 0.121 |

**三者在端到端上基本打平。** 关键是要看噪声量级 —— 单看 cur128 的 recall@10 在三个 seed
之间是 **24.27 / 22.22 / 25.51**（跨度 **3.3pp**），而配置之间只差 **0.36pp**：
**seed 噪声比配置差异大一个数量级**。

**决定：不改动。** 理由是没收益、换的成本是重建三个索引（约 30 分钟）并改变已提交的 RAG 行为。

> **这个结果反过来验证了"推迟裁决"是对的。** §2.3 单通道口径下 desc256 在 recall@50 上
> 领先 2.6pp 且三 seed 一致，看起来很像真信号；但装进完整管线后**排序层把差异吸收掉了**。
> **单通道指标是"没有下游时的代理指标"，用它下架构级结论会误导。**
>
> 候选改动与 `config_emb/` / `config_topk/` 缓存都保留着，将来换模型或换语料可几分钟内复测。

---

## 8. 工程踩坑清单

实现过程中抓到 **10 个真 bug**，其中 5 个是"数据/指标/现象看起来不对"逼出来的，不是读代码看出来的。

| # | 阶段 | 现象 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | P0.1 | 画像与时序无关 | `_build_profile` 用 `games[:30]`，但库顺序由 Steam 决定（近似按 appid） | 统一走 `_top_games_by_playtime` |
| 2 | P0.1 | 并发批处理无节流 | `get_many_app_details` 传 `use_limiter=False` **绕过唯一限速器** | 改全局令牌桶 + in-flight 去重 |
| 3 | P0.1 | 「第2个」当游戏名搜商店 | 序数越界后 fallthrough 到名字抽取 | 直接给明确提示 |
| 4 | A1.2 | 构建脚本 `TypeError: expected string, got float` | **pandas 3.x left-join 后缺失值是 float `NaN`，而 `NaN` 在 Python 里是"真值"**，`value or ""` 兜不住 | 入参改 `object` + 显式 `isinstance(v, str)`，加单测 |
| 5 | A1.2 | parquet 里出现 `float64` 的"整数"列与 `object` 的"布尔"列 | join 引入 NaN 后 dtype 退化 | 合并后显式 `fillna` + `astype` |
| 6 | A3.2 | **`only:content_user` 测出 9.25%@20，真实值 0.48% —— 虚高 20 倍** | `_merge_by_quota` 的"补齐名额"阶段遍历了**所有**通道而非仅启用通道 | 补齐只看启用通道；**发现方式**：横向对比"全库排序"与"引擎候选"两种口径，37/400 用户判定不一致；已加回归测试 |
| 7 | A4 | `collab_only` / `popular_only` / 基线跑出**完全相同的数字** | 用"临时改 `engine.config` → 存闭包 → 恢复"切换配置，**恢复得太早**，共享可变状态串台 | 配置绑进闭包；`RecallEngine` 加 `config=` 逐次覆盖参数（顺带可并发） |
| 8 | A5 | 库覆盖整个候选池时，文案说"你的库里还没有我能识别的游戏" | "库能识别但没剩下可推的"被误标成 `cold_start=True` | 拆开两个语义；单测覆盖 |
| 9 | A7 | **同名去重"偶尔"失效**，`Counter-Strike: Condition Zero` 连推两次 | `_name_keys` 按 `cat_idx` 索引，却被用**召回池内位置**取用 —— 两个索引空间混用 | 先 `self._name_keys[rows]` 按池内位置重排再取；回归测试在旧实现下**必红**（已双向验证） |
| 10 | A7 | 多轮文案直接 `AttributeError: 'RecommendationResult' object has no attribute 'unmapped'` | `unmapped` 属于 `PreferenceState` 而不是推荐结果，写文案时想当然当成结果字段 | 显式传参；并补一条**用真的 dataclass** 而非 `SimpleNamespace` 的回归测试（`SimpleNamespace` 测不出字段漂移） |

**两条方法论修正（比 bug 更值钱）**：

1. **代理指标不能下架构级结论**（§7.4）：单通道 recall@50 的 2.6pp 差异在端到端被排序层吸收，
   而 seed 噪声是它的 9 倍。**先用真实目标指标量，再决定要不要动架构。**
2. **常规评测协议可能测的不是你想测的东西**（§3.2）：全库排序协议被热度主导，
   一度让"内容通道"看起来等于随机。加一套**热度匹配**协议后才显形。

> bug 9 与 10 都是被**整图测试**（`tests/test_multiturn_agent.py`）抓出来的，
> 而不是被单元测试 —— 单元测试里我用 `SimpleNamespace` 造了"假结果"，
> 它恰好带着 `unmapped` 字段，于是永远测不出真实契约的漂移。**假对象的形状要和真的一样**。

---

## 9. 复现指南

### 9.1 完整构建链（从零开始）

```bash
cd backend

# 1. RAG 语料与向量库(项目原有)
python scripts/build_corpus.py        # 下载 parquet(192MB) → corpus.jsonl(5362 条)
python scripts/build_index.py         # 嵌入 → data/chroma          ~4 min

# 2. 方向 A 数据层
#    先把 Steam_November2025_Dataset/(两个 csv) 放到仓库根目录
python scripts/build_interactions.py  # 去重 + 两路特征合并 → data/interactions/
                                      #   实测 12/12 不变式通过

# 3. 离线索引
python scripts/build_recommend_index.py
#   编码 11,152 条文本在 CPU 上 ~195s;总计 ~4 min
#   10/10 校验通过 → data/recommend/ (~73 MB)

# 4. 排序器
python scripts/train_ranker.py --num-boost-round 600 --early-stopping 60
#   数据构造 32s + 训练 ~40s → data/recommend/ranker.txt
```

### 9.2 评测与实验脚本

| 脚本 | 用途 | 耗时 |
|---|---|---|
| `eval_recall.py` | 两套协议的通道基线 + 嵌入配置单通道对比 | 通道 ~40s；带 `--configs` 每个配置 ~3min 编码（有缓存） |
| `eval_recall_channels.py` | 单通道消融 / 边际贡献 / 配额扫描 | 每配置 ~20-40s（7,495 用户） |
| `eval_rag_baseline.py` | 改造前 RAG 方案对照 | ~1 min |
| `eval_embedding_config.py` | 嵌入配置端到端裁决（完整管线重训） | 每配置×seed ~70s |

### 9.3 测试

```bash
cd backend
# 指定模块(推荐):全部离线,~0.8s
python -m unittest tests.test_recommend_features tests.test_recommend_index \
  tests.test_recommend_evaluate tests.test_recommend_recall \
  tests.test_recommend_rank tests.test_recommend_service \
  tests.test_recommend_preferences tests.test_steam_client
# 共 129 个用例

# 多轮对话的整图测试(会 import app.agent,连带加载 torch,约 25s)
python -m unittest tests.test_multiturn_agent
```

> ⚠️ 不要用 `unittest discover` 做快速迭代：它会 import 全部 `test_*.py`，
> 连带加载 torch + sentence-transformers（那 3 个脚本式测试文件），实测 **22.1s**，
> 而真正跑完只要 **0.2s**。这也是路线图 Phase 0 要拆 `tests/unit` 与 `tests/live` 的原因。
>
> 相关教训：`app/agent/__init__.py` 会 `import graph`，所以**任何** `app.agent.*`
> 的导入都会拉起整条重依赖链。`constraint_note` 因此被挪到
> `app/recommend/notes.py` —— 一个纯字符串函数不值得让测试等 14 秒。

---

## 10. 已知局限与可优化点

> 这一节按**预期收益**排序，每条都标注了证据来源。带 🔍 的是我实际查证过的。

### 10.1 高价值（建议优先看）

**🔍 O1. `user_emb.npy` 是死资产（11.3 MB）**
实测确认：该文件只被**写入、加载、校验**，管线里**没有任何消费方** ——
`content_cosine` 特征是从库内物品现算的加权和。
原设计意图是"线上缓存用户向量"，但实现时走了现算路径。
→ **要么用起来**（按 `hash(library)` 做 LRU 缓存，重复请求省掉一次 384 维加权），
**要么删掉**（少 11.3 MB 与一次无用的 mmap）。
顺带：`RecommendIndex.load()` 里 `sp.load_npz` 也会白白解压一份 3.2 MB。

**🔍 O2. 热度特征的训练/服务归一化口径不一致（train/serve skew）**
训练用**训练用户**计数（569,684 / max 4,441），线上用**全量**计数（840,274 / max 6,613），
两边各自 min-max 归一化。实测偏差：中位 **0.024**、热门物品 ~0.001，
均值上线上偏大 8.2%。**量级不大**（`log1p` 压缩了顶部），但这是**形式上的偏斜**：
LightGBM 的分裂阈值是在训练尺度上学到的。
→ 修法很便宜：改用**固定归一化常数**（如 `log1p(c) / log1p(全局最大)`），
训练与线上都传原始计数，不要在构造器里各自归一化。

**O3. 内容通道的配额**无法用现有数据辩护**
LOO 下内容两路贡献几乎为零（0.48% / 1.05%），去掉它们 recall 反而升。
保留它们的理由（唯一能覆盖冷物品）**从未被评测过** —— 现有协议里每个测试用户都有
10+ 条交互，根本测不到冷启动。
→ 建议补一套**冷启动评测**：用"库内物品全部替换为数据集外的新 appid"构造用户，
看内容通道能否给出合理候选。只有测出来，配额才站得住。

**O4. 特征里缺几个强信号**
- **系列/续作关系**：`Mount & Blade II → Mount & Blade: Warband`、
  `Shadow of the Tomb Raider → Tomb Raider` 是**最强的推荐信号之一**，
  目前只能靠 publisher + 协同间接表达。可加"名字前缀/词元重叠"或 Steam franchise 数据。
- **Steam 用户标签**：现在只有 13 个粗粒度 `genres`（Action/Indie/…），
  而 Steam 有几百个细标签（Roguelike / Metroidvania / City Builder…）。区分度差一个量级。
  （注意：项目早已记录过"数据集里的 Tags 其实是类型"这个坑。）
- **语言支持**：对中文用户，**是否支持简体中文**是硬门槛，现在完全没建模。
- **物品平均时长**：`playtime` 在物品侧的中位数（反映"这游戏能不能留住人"），
  现在只用了热度计数。

**O5. 融合的"补齐"阶段语义可疑**
排序层（A4）已经会重排候选，所以**融合阶段唯一的目标是最大化召回**，
不需要用分数挑候选。但现在 `_merge_by_quota` 的补齐阶段用"各路归一化分数的 max"
来填剩余名额 —— 这引入了没必要且未校准的分数比较。
→ 补齐应按**各路的原始顺序轮流取**（纯轮转），或干脆只做"并集 + 各路 top-N"。

**O6. 没有多样性 / 同系列去重**
实测输出里出现过 `Tomb Raider GOTY` 与 `Shadow of the Tomb Raider`
（同一系列）同时进 top-5；`Terraria` 与 `Don't Starve Together` 也高度同质。
→ top-k 阶段加 **MMR** 或简单的"同系列/同发行商最多 N 个"约束，
指标上应补 **Intra-list Diversity**（路线图里列了但没实现）。

### 10.2 中价值

**O7. LightGBM 超参完全没调** —— `num_leaves=31`、`min_data_in_leaf=50`、
`learning_rate=0.05` 都是凭直觉写的。值得做一次小网格（leaves ∈ {15,31,63}、
min_data ∈ {20,50,100}、lr ∈ {0.03,0.05,0.1}）。

**O8. 早停很噪** —— best_iteration 在三个 seed 上是 **88 / 41 / 113**，
说明验证集（1,124 用户 / 511 正样本）太小。→ 用 5 折交叉的均值定轮数，
或扩大验证集、或对多个 seed 的模型做平均。

**O9. 候选池与 TopK 深度没扫过** —— `total=200`、`knn_depth=100`、`K=200` 都是拍的。
候选池越大召回上限越高（但延迟与内存线性增长）。值得扫 {100, 200, 400, 800}。

**O10. 协同矩阵含测试边（轻微泄漏）** —— §5.6 已记录。若要严格评测，
需按训练折重建 item-item 矩阵（成本：N 折 × 一次分块矩阵乘，可接受）。

**O11. MNAR 的根本限制** —— 数据集是"玩过 ≥10 分钟"，未观测 ≠ 负样本。
真正的修法是拿**拥有 vs 玩过**的区分，可以用 `GetOwnedGames` 采一批真实公开库
（项目已有 Steam 客户端与令牌桶限速，采 500~2000 个用户是可行的）。
那时才能把"排序"和"召回"分开评估，也才能做时间切分。

**O12. A3.2 的通道消融只用了一个 seed** —— §4.2/§4.3 的表都是 seed 0。
A3.1 与 A6 都做了多 seed，这里应该补齐，否则"边际贡献"的结论可能是噪声。

### 10.3 低价值 / 清理项

- **🔍 O13. `_quality_channel` 里的 `quota * 4` 是魔法数字**（`recall.py`）——
  意图是"多取一些留余量"，但 4 倍没有依据。改成显式参数或按 `total` 派生。
- **O14. `engine._publisher_channel` 每次调用分配一个 `np.zeros(n_items)`**（46 KB）——
  高频调用下是纯浪费，可以复用缓冲区。
- **O15. 前端没有渲染 `because_of` / `channels`** —— `/api/recommend` 与 SSE 的 `done`
  事件都已经带了这两个字段，但 `GameCard.tsx` 没展示。"因为你在玩《X》"是很好的
  可解释性卖点，现在白算了。
- **O16. 融合结果没有请求级缓存** —— 同一个库连续问两次会完整重算一遍
  （7~43ms，不算贵，但重复请求场景下可省）。
- **O17. 测试目录没有拆分** —— 见 §9.3，`unittest discover` 要 22s。
- **O18. `user_item_csr.npz` 同时被 A4 的特征与协同通道使用**，
  但它是 A2 就落盘的静态产物；如果将来支持增量更新，需要明确它的刷新语义。
- **O19. 多轮偏好没有强度**（§11.11）—— "更偏向"与"只要"共用同一种硬分桶。
  做成软约束需要线上反馈来校准权重，而现有留一协议没有对话数据可用。
- **O20. 「换一批」只是换了位置，没换池子** —— 候选池固定 200 条，
  5 轮之后就只剩长尾。真正的多样性需要 MMR / 同系列去重（与 O6 同源）。
- **O21. `parse_message` 的中文关键词表是手工维护的** —— 有 LLM 时不走它，
  但它决定了"没有 API key 时的体验下限"，且没有测试覆盖所有词条。

---

## 11. 多轮对话（A7）

### 11.1 现象与根因

用户实测发现：

```
用户：推荐几个游戏
助手：《A》《B》《C》《D》《E》
用户：再推荐点别的
助手：《A》《B》《C》《D》《E》     ← 一字不差
```

根因不是"忘了传参数"，而是**单轮设计放进多轮场景的结构性缺陷**：

1. `Recommender.recommend()` 是**游戏库的纯函数** —— 输入只有 `games`，没有对话状态。
   "可复现"在单轮评测里是优点，在多轮对话里就是 bug。
2. 用户顺口说的"我更喜欢策略类的游戏"只被 LLM 拿去写**推荐理由**，
   从来没有影响**排序**。理由写着"因为策略类"，卡片却全是动作游戏。

所以这一节补的不是一个 if 判断，而是**两样缺失的状态**：见过哪些（去重）、说过什么（偏好）。

### 11.2 状态放在前端，不放服务端

| 方案 | 优点 | 代价 |
|---|---|---|
| 服务端 session（Redis/内存字典） | 前端无感 | 要 TTL、要清理、多进程不一致、重启丢失 |
| **前端回传（采用）** | 服务无状态、可水平扩展、重启不丢、零清理逻辑 | 刷新页面后去重记录重置 |
| 落库 | 跨会话记忆 | 需要用户体系，超出当前范围 |

单用户本地部署下，"刷新丢记录"是可接受的取舍，并且**写进文档而不是留成隐性缺陷**。

### 11.3 词表是封闭的 —— 说不出来就承认

Steam 的类型只有 **26 个**，且高度集中：

| 类型 | 物品数 | | 类型 | 物品数 |
|---|---|---|---|---|
| Indie | 7,248 | | Strategy | 2,517 |
| Action | 5,470 | | RPG | 2,394 |
| Adventure | 4,644 | | Free To Play | 788 |
| Casual | 3,264 | | Racing | 462 |
| Simulation | 2,593 | | Gore / Nudity | 6 / 2 |

**没有 Horror、没有 Stealth、没有 Roguelike、没有 Open World。** 这是方向 A 的
已知数据限制（§10.1 O4），多轮功能把它暴露到了用户面前 —— 用户会直接说"想要恐怖一点的"。

处理方式是**不给模型自由发挥的空间**：

```python
FILTER_VOCAB = GENRE_VOCAB (15) + FEATURE_VOCAB (10)   # 类型 + 有主题含义的玩法
```

* 路由 LLM 被要求**只能**从这 25 个词里挑，词表外的词（恐怖/潜行/魂系/开放世界）
  必须放进 `unmapped`；
* 服务端 `normalise_terms()` **再校验一遍**，词表外的一律移进 `unmapped`
  （大小写归一：`rpg` → `RPG`）；
* 于是 `prefer` 里**永远只可能出现能真正执行的条件**。
  **宁可说"我没有这个标签"，也不能假装筛过了。**

玩法维度取自 `categories` 列（共 60 个值，大部分是 `Stereo Sound`、`Adjustable Text Size`
这类技术/无障碍项，只保留 10 个用户会主动提的：`Single-player` / `Multi-player` /
`Co-op` / `Online Co-op` / `PvP` / `Online PvP` / `Shared/Split Screen` / `MMO` /
`Steam Workshop` / `In-App Purchases`）。

### 11.4 约束施加在**排序之后**，不是之前

这是整节最关键的设计决定。

直觉做法是"先按类型过滤候选，再排序"。**这会破坏训练/服务一致性**：
26 维特征里的 `merge_rank` 是候选在 200 条召回池里的**位置**。
先过滤再排序，送进模型的池子变成另一个分布（长度不同、位置整体前移），
`merge_rank` 的取值就和训练时不一致 —— 典型的 train/serve skew，离线指标不会报警。

实际流程：

```
召回 200 条 ──> LightGBM 给【完整池子】打分 ──> 按"是否命中明说偏好"分桶
                                                    │
                          ┌─────────────────────────┴──────────────────────┐
                    matched 桶（命中）                                rest 桶（未命中）
                   桶内保持精排顺序，先取                              不够 top_k 时再补
```

一句话概括语义：**用户明说的偏好决定"要不要"，游戏历史决定"要哪个"。**
桶内的顺序仍然是 LTR 分数，所以"结合用户游戏历史"这个要求没有被牺牲掉。

### 11.5 分桶选品的四条规则与四个标志

| 规则 | 说明 |
|---|---|
| 1. 剔除 `exclude`（已展示 appid） | 解决"再推荐点别的" |
| 2. 剔除命中 `avoid` 的 | 用户说"不要恐怖"就是不要 |
| 3. 剔除**同名**的 | 见 §11.6 |
| 4. matched 优先，不够用 rest 补 | 桶内保持精排顺序 |

产出的四个诊断标志（`RecommendationResult`）：

| 标志 | 含义 | 用于 |
|---|---|---|
| `matched_count` | 最终列表里命中明说偏好的条数 | 决定文案说"命中几款" |
| `relaxed` | 命中了但不够，补了不完全符合的 | `True` 时如实说明"只有 N 款严格符合" |
| `no_match` | **一个都没命中** | `True` 时坦白并给替代方案 |
| `exhausted` | 池子被去重清空了 | `True` 时提示"能推的都推过了" |

`relaxed` 与 `no_match` 必须分开：一个 5 条里命中 3 条，和一个都没命中，
是完全不同的处境，糊成一句话就是在骗人。

### 11.6 同名去重：目录里有 319 组同名不同 appid

实测 `catalog.parquet` 里 **319 组**物品名字完全相同但 appid 不同，例如：

| cat_idx | appid | name |
|---|---|---|
| 7 | 80 | Counter-Strike: Condition Zero |
| 8 | 100 | Counter-Strike: Condition Zero |

只按 appid 去重的话，第二轮的候选池里还有另一个 appid 的同名条目，
用户看到的就是**两张一模一样的卡**。所以去重键是 `(appid, name_key)`：

```python
def name_key(name) -> str:          # 小写 + 去掉 ™®© + 折叠空白
    ...
```

**这里踩了一个极难定位的 bug（§8 bug 9）**：`_name_keys` 是按 `cat_idx` 排的数组，
而 `picked` 里存的是**池内位置**。两者混用时，"同名去重"只在位置恰好等于 `cat_idx`
时生效 —— 表现为**随机失效**，单测里因为小目录下位置与 `cat_idx` 常常撞对而通过。
修法是在使用前先 `self._name_keys[rows]` 按池内位置重排一次，并加了一条
**在旧实现下必红**的回归测试（已双向验证：改回旧代码 → 2 条测试失败）。

### 11.7 诚实回退：事实由代码说，不由 LLM 说

`constraint_note()`（`app/recommend/notes.py`）是"实话"的唯一来源：

| 处境 | 文案 |
|---|---|
| 一个都没命中 | 「说实话,按「X」筛了一遍你库里的 200 个候选,一个都没对上。另外「潜行」这类标签我的数据里没有,只能按类型和玩法这些大类筛。下面这几款是没按你的条件挑的、但和你游戏库最接近的,要不要先看看?」 |
| 部分命中 | 「严格符合「X」的只有 3 款,剩下 2 款是我按你的库补的相近选择。」 |
| **说了但一个词都对不上词表** | 「「恐怖」这类标签我的数据里没有(Steam 只分了 Action / RPG / Strategy / Co-op 这些大类,没有更细的主题标签),所以没法按它筛。下面还是按你的游戏库挑的,你看看方向对不对?」 |
| 全部命中 | （不额外说，避免噪音） |
| 池子耗尽 | 「能推的我基本都推过一轮了,要不你换个方向,或者放宽一点条件?」 |

> 第三行是**实际踩出来的缺口**：用户说"想找恐怖一点的"，路由正确地把「恐怖」放进了
> `unmapped`、`prefer` 为空 ⇒ `no_match` 也是 `False` ⇒ 一句提示都没有。
> 用户完全不会知道自己的要求被忽略了。**"没有反馈"和"假装满足"一样糟糕**，
> 所以单独加了一个 `prefer` 为空但 `unmapped` 非空的分支。

**为什么不让 LLM 写这段话**：LLM 很擅长把"5 条里命中 3 条"润色成"都符合你的口味"。
数字由代码算、由代码说；LLM 只负责把**开场白**写得自然，并且提示词里明确要求
"候选行里没有『对上你说的偏好』就是没对上，理由要如实写"。

同时给每条候选加了 **`pref_match`** 字段（与用户**明说偏好**的重叠），
和原有的 `match_genres`（与**长期画像**的重叠）区分开 —— 这两个是不同的东西，
混在一起既写不清理由，也没法排查"为什么推了这条"。

### 11.8 路由顺带抽取偏好：不多花一次 LLM 调用

多轮功能需要"从用户这句话里读出偏好"。单独加一个 LLM 节点最直观，但**每条消息都会多
300~600ms**，而"判断意图"和"抽偏好"都是"读一句话做分类"，没有理由跑两遍。

所以 `ROUTER_PROMPT` 被扩成同时返回：

```json
{"intent": "recommend",
 "pref": {"genres": ["Strategy"], "avoid": [], "unmapped": ["潜行"], "restart": false}}
```

并给了三条判断规则（写进提示词，每条都对应一个实测场景）：

* 「换一批」/「再推荐点别的」只是要新的一批，**不是**否定之前的偏好；
* 用户只说偏好、没说要推荐 → 意图用 `recommend`（对应"用户一开始说自己更喜欢某某类型"）；
* 结合历史对话理解「好」这类省略 —— 上一轮助手问"要不要看看其他类型"，用户答"好"，
  应判为 `recommend` 且 `restart=true`。

服务端还有一条**兜底升级规则**：即使路由判成 `chat`，只要抽到了具体类型偏好，
就升级为 `recommend` —— 用户说"我更喜欢策略类的"时，他要的显然是推荐，不是闲聊。

LLM 不可用时走 `parse_message()`：中文关键词表 + 否定词窗口（"不要/不想/别推荐/除了…"）。
它不追求准确，只保证"没有 API key 也不是完全没反应"。

### 11.9 接线与契约

```
前端 App.tsx ──context{candidates, recommend, history}──> POST /api/chat
                                                            │
                                                    AgentState.recommend_state
                                                            │
                                          route_node   抽 intent + pref + 合并状态
                                                            │
                                          recommend_node  去重 + 偏好过滤 + 写回新状态
                                                            │
       前端保存 <──done.recommend_state──────────────────────┘
```

| 字段 | 用途 |
|---|---|
| `context.candidates` | 上次推荐候选，解析「第 x 个」 |
| `context.recommend` | `{genres, avoid, unmapped, shown, turn}`，多轮状态 |
| `context.history` | 最近 8 轮（每轮截 300 字），只为让路由消解省略与指代 |

两个容易翻车的细节：

1. **`recommend_state` 必须无条件更新**，哪怕这一轮一个候选都没给 ——
   否则下一轮前端拿到的 `shown` 会**回退**，去重记录整体丢失。
   这一条有专门的单测（`test_state_survives_a_turn_with_no_candidates`）。
2. **历史用 `useRef` 而不是 state** 读取。`send` 的 `useCallback` 依赖列表里没有
   `messages`（闭包会过期），而 `useEffect` 同步的 ref 在调用瞬间读到的恰好是
   **本次发送之前**的消息，正是需要的。

换游戏库时要手动重置状态（`handleLibraryLoaded`）—— 旧的 `shown` 是上一份库的推荐记录，
留着会让新库一上来就没东西可推。

### 11.10 实测（真实索引，180 款库）

| 场景 | 结果 |
|---|---|
| 连续 5 轮「换一批」 | 25 款全部互不重复，同一轮内无同名卡片，无已拥有游戏 |
| 补充「我更喜欢策略类的」 | `matched=5`，5 款全部带 `pref_match=["Strategy"]` |
| 补充「不要策略」 | `avoid=["Strategy"]`，结果里 0 条含 Strategy |
| 偏好一个池里几乎没有的词（Nudity） | `no_match=True` + 诚实文案 + 5 款替代候选 |
| 延迟增量（交错 A/B，各 200 次） | 无约束 P50 **9.13ms** → 去重+偏好 P50 **9.37ms**，即 **+0.24ms（2.6%）** |

> 延迟增量几乎为零，因为约束全部是 `np.isin` / 布尔索引 / 一次 200 元素的循环，
> 相比 LightGBM 打分的占比可以忽略。

### 11.11 仍然没解决的（如实列出）

* **偏好只能落在 25 个词里。** "恐怖/潜行/魂系/开放世界"确实筛不了。
  根治要去拿 Steam 的**细标签**（`appdetails` 的 `tags` 字段或 SteamSpy tags），
  属于 §10.1 O4 的一部分。
* **「换一批」只保证"不一样"，不保证"更不一样"。** 候选池仍是同样那 200 条，
  换个顺序而已。真正的"探索"需要**多样性重排**（MMR / 同系列去重），
  对应 §10.1 O6，目前没做 —— 实测输出里 `Tomb Raider GOTY` 与
  `Shadow of the Tomb Raider` 仍会同时出现。
* **偏好没有强度。"** 更偏向策略"和"只要策略"现在是同一种硬分桶。
  更自然的做法是把偏好做成**软约束**（在 LTR 分数上加权），但加权系数需要
  用线上反馈校准 —— 现有离线协议（留一）无法评估"多轮偏好"这件事，
  因为数据集里没有对话。
* **状态在客户端，刷新即丢。** 单用户场景可接受，多用户产品需要 session。
* **没有跨会话记忆。** "上次你说喜欢策略类，这次还按这个来吗" 需要用户体系。

---

## 12. 文件与产物索引

### 代码

| 文件 | 职责 |
|---|---|
| `app/recommend/features.py` | 纯函数：评价标签 / 好评率 / 销量下界 / 年份 / 时长分档 / 文本模板 / 贝叶斯平滑 |
| `app/recommend/index.py` | A2 产物加载（mmap）+ 数组助手（`topk_from_dense` / `cat_rows_of` / `content_topk_from_embeddings`） |
| `app/recommend/recall.py` | 5 路通道 + 配额轮转融合（`RecallEngine`） |
| `app/recommend/rank.py` | 26 维 `FeatureBuilder`（训练/线上共用）+ `Ranker` |
| `app/recommend/evaluate.py` | 留一切分 + 两套协议 + 排序指标 |
| `app/recommend/service.py` | 对外门面 `Recommender` + 懒加载单例（含多轮去重与偏好分桶） |
| `app/recommend/preferences.py` | A7 对话偏好状态机 + 封闭词表 + 关键词兜底抽取 |
| `app/recommend/notes.py` | A7 约束文案（"命中了几款"的确定性实话，纯标准库） |
| `app/routers/recommend.py` | `POST /api/recommend` |
| `scripts/build_interactions.py` | A1.2 数据落盘 |
| `scripts/build_recommend_index.py` | A2 离线索引 |
| `scripts/train_ranker.py` | A4 训练与评测 |
| `scripts/eval_recall.py` | A3.1 两套协议 + 嵌入配置单通道对比 |
| `scripts/eval_recall_channels.py` | A3.2 单通道消融 / 边际贡献 / 配额扫描 |
| `scripts/eval_rag_baseline.py` | A6 改造前 RAG 对照 |
| `scripts/eval_embedding_config.py` | A6 嵌入配置端到端裁决 |

### 产物（均在 `backend/data/`，已 gitignore）

```
data/interactions/     interactions.parquet(6.9MB) users.parquet items.parquet stats.json
data/recommend/        catalog.parquet content_emb.npy content_topk.npz collab_topk.npz
                       user_emb.npy user_item_csr.npz manifest.json ranker.txt
                       eval_recall.json eval_recall_channels.json eval_rank.json
                       eval_rag_baseline.json eval_embedding_config.json
                       config_emb/  config_topk/     # A6 复测缓存(可删,~100MB)
```

### 测试

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_recommend_features.py` | 20 | 评价标签 / 好评率 / 销量 / 年份 / 时长分档 / 贝叶斯平滑 / 文本模板 |
| `tests/test_recommend_index.py` | 12 | `topk_from_dense` / `user_vector` |
| `tests/test_recommend_evaluate.py` | 15 | 留一切分 / 指标 / 热度匹配协议 |
| `tests/test_recommend_recall.py` | 8 | 通道隔离 / 配额语义 / 拥有排除 / 权重语义 |
| `tests/test_recommend_rank.py` | 10 | 特征维度与顺序 / 哨兵值 / 类型与发行商特征 / **泄漏守卫** |
| `tests/test_recommend_service.py` | 11 | 冷启动三路径 / 解释来源 / 前端契约 |
| `tests/test_recommend_preferences.py` | 42 | 偏好状态机 / 封闭词表 / 关键词抽取 / **多轮去重** / 偏好分桶 / **同名去重** / 约束文案 |
| `tests/test_multiturn_agent.py` | 6 | 整图多轮:两轮不重复 / 偏好进入排序 / `shown` 不丢失（**会加载 torch，约 23s**） |
| `tests/test_steam_client.py` | 11 | 令牌桶限速 / TTL 缓存 / DLC 纯函数 |
| **合计** | **129** | 前 7 行全部离线，~0.8s |
