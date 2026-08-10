# 面向超图自适应 RAG 的问题集 v2.1 冻结设计

## 1. 文档状态

- 状态：理论与指标复核后重新冻结，进入实现前准备
- 日期：2026-07-29
- 当前领域：神经医学语料
- 实验设定：冻结知识库下的闭世界问答
- Pilot 规模：80 道候选题
- Pilot 用途：开发、校准与流程验收，不进入最终 locked test

本文档固化问题集 v2.1 的构造原则和可执行流程。v2.1 重点修正“固定策略等级不等于真实成本顺序”的理论问题，并将 Router 能力阶梯实验与预算匹配的超图归因实验分离。实现阶段若需要修改核心定义、策略配置、成功判据、成本配置或数据切分规则，必须更新本文档并生成新的数据集版本。

配套机器可读契约：`docs/question_set_v2_contract.yaml`（contract_version: `question-set-v2.1-contract-v1`），把 P0–P4 策略、标签语义、成本定义、证据与指标、稳定性与 Judge 规则、Trace 字段冻结为可校验、版本化的实验契约；两个 JSON schema 位于 `docs/schema/`（`question_item_v2.schema.json`、`retrieval_trace_v2.schema.json`）。修改上述任一冻结项必须同步升级契约版本号并重新生成数据集版本。

## 2. 研究目标

问题集的第一目标是评价：

> 在给定模型、知识库和检索系统的条件下，Router 能否选择一条稳定正确、资源使用合理且不过度的回答策略。

问题集的第二目标是解释：

- 问题为什么需要检索；
- 普通 chunk 检索是否足够；
- 是否需要高阶超图关系；
- 失败发生在检索、上下文截断、证据使用还是答案生成阶段。

正式标注不再用一个标签同时表达“策略等级最低”和“真实成本最低”，而是保存：

```text
successful_route_set
minimum_sufficient_route_by_policy
route_cost_vectors
pareto_optimal_routes
minimum_cost_route_by_profile
```

其中 `minimum_sufficient_route_by_policy` 是预定义策略阶梯中第一个稳定成功配置；`minimum_cost_route_by_profile` 只有在明确部署成本配置后才能计算。两者不得混用。

四维难度画像用于解释，不作为 Router 真值的直接来源：

```text
D_reason
D_retrieval
D_hyper
D_model
```

## 3. 基本原则

### 3.1 系统相对性

难度、策略充分性、成功路径集合和成本最优结果必须绑定系统快照：

```text
D(q | S)
S = (answer model, corpus, chunking, hypergraph, retrievers, prompts, routes)
```

更换回答模型、知识库、索引、检索算法或路径参数后，静态题目可以保留，但动态路径标签必须重新生成。

### 3.2 闭世界设定

Gold truth 由冻结语料快照定义。本研究评价系统是否忠实回答当前知识库中的问题，不讨论知识库冻结后的现实医学事实变化。

### 3.3 原文优先

证据可信度优先级为：

```text
原始 source chunk
> 经原文验证的实体和超边
> LLM 生成的答案、摘要或解释
```

实体或超边描述不能单独成为医学事实真值。所有必要答案点必须回溯到原始 source evidence span。

### 3.4 Evidence-first 构造

正式问题按以下顺序构造：

```text
选取原始证据或图 motif
→ 验证 source evidence spans
→ 定义必要答案点
→ 定义必要证据组和子超图
→ 生成问题
→ 审核与改写
→ shortcut audit
→ P_gold 准入测试
→ 固定策略全路径与预算匹配实验
```

现有问题集仅作为候选题池。只有重新完成上述标注和审核后，旧题才能进入 v2。

## 4. 问题与答案形式

### 4.1 语言

v2 主问题集统一使用英文：

- question、gold answer 和 answer units 使用英文；
- source evidence 保持原文；
- 中文仅用于项目文档和人工审核备注；
- 跨语言问答放入未来独立挑战集。

### 4.2 问题形式

主格式为受约束的开放式问答：

- 不使用选择题；
- 不使用宽泛综述题；
- 明确回答范围；
- 每题包含 1 至 5 个原子答案点；
- 允许同义表述；
- 必要限定条件不得遗漏。

### 4.3 医学范围

在没有医学专家审核的条件下，Pilot 仅包含原文可直接验证的生物医学知识问答。

允许：

- 解剖、定义、分类和机制；
- 疾病、症状、药物、基因之间的文献内关系；
- 多段原文的比较、聚合和条件提取；
- 能够由冻结语料直接支持的事实。

排除：

- 个体化诊断；
- 治疗推荐和剂量决策；
- 风险收益权衡和预后判断；
- 依赖文档外临床常识的关键推理；
- 无法可靠裁决的证据冲突。

## 5. Gold 标注结构

### 5.1 原子答案点

每个问题必须提供：

```json
{
  "question_id": "qv2-...",
  "question": "...",
  "gold_answer": "...",
  "answer_units": [
    {
      "unit_id": "AU1",
      "claim": "...",
      "required": true
    }
  ]
}
```

### 5.2 精确证据片段

每个必要答案点必须绑定精确 evidence span：

```json
{
  "unit_id": "AU1",
  "evidence_spans": [
    {
      "chunk_id": "chunk-...",
      "start_char": 418,
      "end_char": 672,
      "quote": "..."
    }
  ]
}
```

要求：

- `chunk_id` 可重建检索上下文；
- `start_char` 和 `end_char` 可程序校验原文位置；
- `quote` 用于人工审核与 Judge；
- span 必须真实蕴含对应答案点。

### 5.3 必要证据组

同一事实可能由多个 chunk 等价支持，因此证据要求采用“组内 OR，组间 AND”：

```json
{
  "evidence_requirements": [
    {
      "requirement_id": "ER1",
      "answer_unit_ids": ["AU1"],
      "alternative_chunk_ids": ["chunk-A", "chunk-B"]
    },
    {
      "requirement_id": "ER2",
      "answer_unit_ids": ["AU2"],
      "alternative_chunk_ids": ["chunk-C"]
    }
  ]
}
```

路径证据覆盖成功要求每个必要证据组至少命中一个有效替代来源。

### 5.4 必要证据子超图

保存可重建结构，而不只保存统计数：

```json
{
  "required_vertices": ["..."],
  "required_hyperedges": [
    {
      "hyperedge_id": "he-...",
      "entity_set": ["..."],
      "source_chunk_ids": ["chunk-..."]
    }
  ],
  "answer_unit_links": {
    "AU1": ["he-1"],
    "AU2": ["he-1", "he-2"]
  },
  "topology_metrics": {
    "edge_count": 2,
    "max_arity": 5,
    "path_depth": 2,
    "branch_width": 1
  }
}
```

`hyperedge_id` 应由规范化实体集合与来源信息生成稳定哈希，同时保留原始实体集合。

若存在多套同样充分的结构，分别保存：

```text
primary_evidence_subgraph
alternative_evidence_subgraphs
```

## 6. 固定策略配置与成本语义

### 6.1 预定义策略阶梯

系统保留以下策略顺序：

```text
P0 llm
< P1 naive
< P2 hyper_lite
< P3 hyper_standard
< P4 hyper_expanded
```

该顺序是研究者预先定义的能力与资源策略阶梯，不是假定的真实成本全序。固定顺序中第一个满足运行规则的配置称为：

```text
minimum_sufficient_route_by_policy
```

不得将其直接称为 `minimum_cost_route`，也不得仅依据该标签声称某题必须使用超图。

### 6.2 固定策略参数

| 路径 | chunk VDB top-k | entity VDB top-k | relation VDB top-k | Entity description cap | Relation description cap | Source text cap | Final context hard cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 `llm` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P1 `naive` | 20 | 0 | 0 | 0 | 0 | 4000 | 4000 |
| P2 `hyper_lite` | 0 | 25 | 0 | 150 | 800 | 2000 | 7000 |
| P3 `hyper_standard` | 0 | 30 | 30 | 250 | 1200 | 3000 | 12000 |
| P4 `hyper_expanded` | 0 | 50 | 50 | 350 | 1800 | 4000 | 15000 |

共同参数：

```text
answer model = qwen-27b-int4
temperature = 0.1
max_response_tokens = 3000
LLM max length = 24576
Router = disabled
type-aware weighting = disabled
```

P0 至 P4 的最终回答 prompt 必须统一。唯一允许的差异是上下文是否存在，以及上下文的检索来源和结构格式。

### 6.3 Budget 的单位与生效阶段

各项 budget 不处于同一层级，不要求简单相加等于 Final context。执行顺序必须固定为：

```text
VDB candidate top-k
→ 各检索线按 description/source token cap 预截断
→ 实体线与关系线合并
→ 按稳定 ID 去重
→ 解析 Entities / Relationships / Sources
→ section-aware allocation
→ Qwen tokenizer 下的 final context hard cap
```

参数分为三类：

```text
candidate budget:
  chunk_vdb_top_k
  entity_vdb_top_k
  relation_vdb_top_k

pre-merge caps, per active retrieval line:
  entity_description_cap
  relation_description_cap
  source_text_cap

post-merge hard control:
  section_allocation_ratios
  final_context_hard_cap
```

P2 的 `relation_vdb_top_k=0` 表示不主动查询 Relation VDB，但实体种子仍会扩展邻接超边，因此 `relation_description_cap=800` 有效。实现、Trace 和论文表格必须使用上述完整参数名，避免把“关系检索候选数”和“最终关系区预算”混为一谈。

### 6.4 Section 分配

Pilot 固定使用：

```text
Sources        55%
Relationships  30%
Entities       15%
```

该比例只在 post-merge 总上下文需要截断时作为初始分配；未使用额度会按确定性规则重分配。比例不是三类 pre-merge cap 的解释，也不要求与它们相加对应。

不得根据 gold evidence 为单题动态分配。Pilot 记录每个 section 在 raw、pre-merge、post-merge 和 final context 阶段的实际 token、截断量和证据损失。若需调整比例，只能使用开发集统一调整后重新冻结。

### 6.5 Token 校准

当前 `tiktoken(gpt-4o-mini)` 只能作为近似计数。Pilot 开始前必须比较：

- tiktoken 计数；
- Qwen 实际 tokenizer 计数；
- API 返回的 `usage.prompt_tokens`，若可用。

正式 hard cap 以 Qwen tokenizer 或 API usage 校准。若无法取得真实计数，使用 Pilot 观测到的最坏误差设置安全系数。所有预算匹配实验均按实际 Qwen context token 计数，而不是只比较配置值。

### 6.6 成本向量与 Pareto 路径

每条路径每次运行保存成本向量：

```text
input_tokens
output_tokens
llm_calls
embedding_calls
retrieved_candidate_count
graph_lookup_count
reranker_calls
latency_ms
gpu_seconds_or_api_cost, if available
timeout_count
retry_count
```

对每道题派生：

```text
successful_route_set
minimum_sufficient_route_by_policy
route_cost_vectors
pareto_optimal_routes
minimum_cost_route_by_profile
```

`pareto_optimal_routes` 包含所有不存在另一条“同样稳定成功且所有成本分量均不高于它、至少一项更低”的路径。Benchmark 不规定一个通用加权成本函数。

如需单一成本最优路径，必须声明并版本化部署配置，例如：

```text
token_first
latency_first
local_gpu
api_billing
```

配置需包含成本归一化方法和显式权重。`cost_oracle(profile)`、成本 regret 和 `minimum_cost_route_by_profile` 都必须带 profile 名称。

## 7. 预算匹配的超图归因实验

不同预算的 P0 至 P4 用于 Router 策略实验，不能用于单独证明超图结构贡献。超图有效性必须通过预算匹配实验回答。

### 7.1 检索机制消融

先使用两个统一预算档位：

```text
B_low  = 7000 Qwen context tokens
B_high = 12000 Qwen context tokens
```

每个档位运行：

| 组别 | 候选检索机制 | 送入回答模型的上下文 |
|---|---|---|
| `N-source@B` | naive chunk VDB | 仅 Sources |
| `HL-source@B` | entity VDB + 邻接超边回溯 | 仅最终 source chunks |
| `H-source@B` | entity/relation 双线 + 超图回溯 | 仅最终 source chunks |

三组必须使用相同的：

- Qwen final context hard cap；
- Source chunk 格式；
- 最终回答 prompt；
- max response tokens；
- 温度、重复规则和 Judge；
- evidence-group 检索指标。

内部检索单位和计算成本允许不同，但必须完整记录。跨架构的公平共同指标是最终 source evidence group coverage。

若 `H-source@B` 在相同 source budget 下优于 `N-source@B`，才能归因于超图引导的证据发现，而不是更长上下文或 Entity/Relation CSV。

### 7.2 结构表示消融

固定同一问题、同一批 source IDs、相同 Source 内容和顺序，再构造：

| 组别 | Source 区 | 结构信息区 |
|---|---|---|
| `source_only` | 相同 | 无 |
| `flat_facts` | 相同 | 将同一实体/关系事实确定性线性化为普通文本 |
| `hyper_structured` | 相同 | Entities/Relationships 结构化表示 |

`flat_facts` 与 `hyper_structured` 必须：

- 包含完全相同的事实字段；
- 不使用 LLM 重新摘要；
- 使用相同 Qwen total token hard cap；
- 使用相同 Source 区；
- 只改变事实组织形式。

`flat_facts > source_only` 表示增加图中事实有价值；`hyper_structured > flat_facts` 才能支持“高阶结构化组织本身有价值”的结论。

### 7.3 Budget-response 曲线

在 Pilot 的结构分层子集上，对 Naive、Hyper-lite 和 Hyper 运行：

```text
4000 / 7000 / 12000 / 15000 Qwen context tokens
```

绘制以下指标随实际 context tokens 的变化：

- operational stable answer success；
- ER-Recall；
- CompleteEvidenceHit；
- Answer Unit Recall；
- unsupported claim rate；
- evidence token density；
- latency 与调用成本。

Pilot 后根据饱和点决定正式消融保留哪些预算档，不因单题结果动态修改。

### 7.4 结论边界

- P3/P4 胜过 P1，只能说明完整高资源配置更有效；
- 相同 source budget 下 Hyper-source 胜过 Naive-source，才能说明图引导检索有增益；
- 相同事实和 token 下 Hyper-structured 胜过 Flat-facts，才能说明结构表示有增益；
- 若预算匹配后增益消失，论文不得声称超图结构独立带来提升。

## 8. 路径真值生成

### 8.1 禁止循环标注

真值运行必须完全绕过 Router：

- 不读取 `expected_complexity`；
- 不读取 Router 的 `focus_types`；
- 不按问题动态调整参数；
- 不允许 Router 预测结果影响路径执行。

### 8.2 运行稳定性规则

运行判定规则命名为：

```text
operational_stable_success_v1
```

每条候选路径先独立运行 3 次：

- `3/3` 通过：按工程规则判定成功；
- `0/3` 或 `1/3` 通过：按工程规则判定失败；
- `2/3` 通过：补跑 2 次，累计 `4/5` 才判定成功。

该规则用于构造可执行标签，不声称统计上证明真实成功概率大于某个阈值。每条路径同时报告：

```text
success_count / trial_count
empirical_success_rate
Wilson interval or Beta posterior interval
```

任一次出现严重事实错误时，该路径进入人工复核。locked test 中影响主标签的边界路径补充至 10 次运行；其他路径保持最多 5 次。若未来要声明 `P(success) >= tau`，必须另行设计满足统计功效的重复次数，不能用 4/5 直接代替。

正式标签运行必须禁用 LLM response cache。每次 repeat 独立执行：

```text
关键词抽取
→ 检索
→ 上下文组装
→ 最终回答
```

索引保持冻结。若接口支持，为 repeat 记录可复现 sampling seed。
### 8.3 双成功门槛

P1 至 P4 的单次成功定义为：

```text
route_success
= source_evidence_coverage
AND answer_correctness
```

其中：

- final context 必须覆盖所有必要证据组；
- 最终回答必须覆盖所有必要答案点；
- 不得包含与原文冲突的事实；
- 实体或超边描述命中不能替代 source evidence 命中。

图结构命中但原文证据未进入上下文时，记录：

```text
graph_hit = true
source_evidence_hit = false
```

P0 没有证据门槛，只评价答案正确性。

### 8.4 P_gold 准入路径

在 P0 至 P4 之前运行诊断路径 `P_gold`：

- 直接向回答模型提供完整 gold evidence；
- 使用与正式路径相同的回答 prompt；
- 按稳定成功规则重复运行。

`P_gold` 稳定失败的题不进入五路径 Router 主集，应检查：

- gold answer 或证据是否有误；
- 问题是否歧义；
- 当前回答模型是否存在推理能力上限。

确认题目有效但模型仍失败时，可进入独立 `reasoning_limit` 分析集。

### 8.5 候选池与正式集运行方式

原始候选预筛：

- 可以按 P0 到 P4 顺序运行；
- 第一条 operational success 出现后可早停；
- 早停结果只用于降低明显不合格候选的筛选成本；
- 早停结果不能生成 `successful_route_set`、Pareto 路径或成本 oracle。

所有进入 accepted Pilot、dev 或 locked test 的题：

- 必须完整运行 P0 至 P4；
- 保留所有路径的重复结果；
- 记录路径非单调现象；
- 生成 `successful_route_set`；
- 生成 `minimum_sufficient_route_by_policy`；
- 根据真实成本向量生成 `pareto_optimal_routes`；
- 仅在声明成本 profile 后生成 `minimum_cost_route_by_profile`。

## 9. 四维难度与正式指标协议

四维难度尽量由行为与结构测量产生，而不是由 LLM 直接主观打分。

### 9.1 模型认知难度

```text
D_model = P0 的 empirical failure rate、置信区间与错误类型
```

### 9.2 推理难度

```text
D_reason = P_gold 的 empirical failure rate、答案单元错误与置信区间
```

### 9.3 检索难度

检索难度由必要证据组在候选、重排和最终上下文中的排名、覆盖与噪声共同描述。跨 Naive 与 Hyper 的共同评价单位统一为最终原始 source evidence groups，而不是直接比较 chunk、vertex 和 hyperedge 的内部排名。

### 9.4 超图结构难度

由验证后的最小必要子超图计算：

- 必要实体数；
- 必要超边数；
- 最大关系元数；
- 超路径深度；
- 分支宽度；
- 候选结构歧义；
- 证据连通性。

可以将连续值映射为 `low / medium / high` 用于展示，但原始测量值必须保留。

### 9.5 Evidence-group 检索指标

设问题有 `m` 个必要证据组 `G_1 ... G_m`，每组包含能够替代支持同一要求的 source chunk IDs；`R_k` 是某检索阶段排序前 k 的 source chunk IDs。

```text
ER-Recall@k
= (1 / m) * sum_j I(R_k intersects G_j)
```

```text
CompleteEvidenceHit@k
= I(for every j, R_k intersects G_j)
```

```text
CompleteEvidenceRank
= min k such that CompleteEvidenceHit@k = 1
```

若到最大候选数仍未完整覆盖，`CompleteEvidenceRank = null` 并记录 `complete_hit=false`。

`AnswerUnitRecall@k` 定义为前 k 个 source chunks 能支持的必要 answer units 数量占全部必要 answer units 的比例。默认报告：

```text
k = 1, 3, 5, 10, 20
```

项目不得使用非标准名称 `Call@k` 表示召回率，统一使用 `Recall@k` 或 `ER-Recall@k`。

### 9.6 排序与相关性指标

辅助报告：

- `Precision@k`：前 k 个已判定结果中直接支持或相关的比例；
- `Recall@k`：相对于完整 qrels 的相关 source chunk 召回；
- `MRR`：第一个直接支持结果的 reciprocal rank，仅作为单证据或首次命中辅助指标；
- `nDCG@k`：使用分级 relevance 的排序质量；
- `Judged@k`：前 k 个结果中已完成人工/审核判定的比例。

多证据问题不得将 MRR 作为主指标，因为首次命中不能表示完整证据形成。

### 9.7 Relevance pooling

Precision、Recall 和 nDCG 所需 qrels 通过 pooling 构造：

1. 合并 Naive、Entity、Relation、Hyper-lite、Hyper 及必要 baseline 的 top-20 或 top-50 source 结果；
2. 按稳定 source chunk ID 去重；
3. LongCat 预标 `0=无关、1=相关背景、2=直接支持 answer unit`；
4. 人工依据 evidence spans 复核；
5. 未审核结果保持 `unjudged`，不能默认视为无关；
6. 发现新的等价支持证据时，补充 gold evidence group 并增加数据版本。

Pooling 配置、检索器版本、最大深度和 qrels 版本必须写入 system snapshot。

### 9.8 分阶段检索与上下文指标

每条路径在以下阶段计算统一指标：

```text
candidate_ER_recall
post_rerank_ER_recall
post_assembly_ER_recall
final_context_ER_recall
```

正式定义：

```text
TruncationLoss
= max(0, post_assembly_ER_recall - final_context_ER_recall)
```

```text
ContextPrecision
= final Sources 中 relevance>=1 的 chunk 数 / final Sources chunk 总数
```

```text
EvidenceTokenDensity
= final context 中 gold evidence span tokens / final context tokens
```

同时记录：

- redundancy rate；
- gold evidence position；
- source、entity、relation 各 section token 占比；
- 检索命中但在图扩展、重排、组装或截断阶段丢失的 evidence group IDs。

### 9.9 超图内部指标

以下指标只用于分析 Hyper 路径内部行为，不直接与 Naive 的 chunk ranking 做单位等价比较：

- `vertex_recall@k`；
- `hyperedge_recall@k`；
- `source_provenance_recall@k`；
- `subgraph_complete_hit@k`；
- gold vertex/hyperedge rank；
- seed-to-source lineage coverage。

跨架构主比较仍使用最终 source evidence group coverage。

### 9.10 图构建质量

全量自动检查：

- document ingestion coverage；
- chunk provenance coverage；
- hyperedge provenance coverage；
- duplicate entity rate；
- duplicate hyperedge rate；
- unsupported-provenance rate。

分层抽样人工审核：

- entity extraction precision；
- entity linking accuracy；
- hyperedge factual precision；
- hyperedge-to-source-span entailment。

没有完整人工 Gold 图时，不得声称测得全语料 entity/hyperedge extraction recall。Recall 只能在单独人工标注的 source subset 上报告。

### 9.11 上下文位置敏感性

在 Pilot 的分层子集上固定上下文内容，仅扰动 gold evidence 的位置和 section 顺序，比较：

```text
gold-first
gold-middle
gold-last
within-section shuffled
```

该实验用于检测 lost-in-the-middle 和固定 `Entities → Relationships → Sources` 排列的影响，不参与主路径标签生成。

## 10. Pilot 候选构造

### 10.1 双入口抽样

- 单 chunk / 单实体控制题：source-first；
- 高元超边、链、分支和消歧题：graph-first；
- 所有 graph-first motif 必须回溯验证原始 evidence spans。

不允许将语义相关但不存在可验证连接的材料任意拼成多证据题。

### 10.2 证据结构配额

| 结构类型 | 候选数 |
|---|---:|
| 单 chunk / 单实体事实 | 20 |
| 单个高元超边 | 15 |
| 两个或以上连续超边 | 20 |
| 多分支聚合或比较 | 15 |
| 相似子图消歧 | 10 |

### 10.3 检索表达配额

| 检索表达类型 | 候选数 |
|---|---:|
| 与原文术语直接匹配 | 25 |
| 医学同义词、简称或规范名转换 | 20 |
| 隐式关系表达 | 20 |
| 相似实体或关系消歧 | 15 |

同义词必须由可靠术语映射或原文别名验证。不得为制造难度而使用不自然表达。

### 10.4 推理操作配额

| 推理操作 | 候选数 |
|---|---:|
| 单一事实读取 | 20 |
| 多证据聚合 | 20 |
| 比较、区分或排除 | 15 |
| 条件或因果关系推导 | 15 |
| 顺序依赖或中间结论 | 10 |

不得把相关性改写为因果关系。

### 10.5 主题覆盖

- 基于章节、核心实体和证据语义形成医学主题簇；
- 单一主题不超过 Pilot 的约 20%；
- 同一核心医学事实在主集中最多出现一次；
- 同一必要 evidence cluster 在主集中最多生成一道题；
- 结构和路径覆盖优先于疾病类别严格等量。

### 10.6 候选生成与审核模型

模型角色固定为：

```text
候选题初稿        qwen-27b-int4
被测回答模型      qwen-27b-int4
题目审核与改写    meituan-longcat/LongCat-2.0
路径成功主 Judge  meituan-longcat/LongCat-2.0
```

正式构建前必须验证 LongCat：

- API 可调用；
- 所需上下文长度可用；
- 结构化 JSON 输出稳定；
- 超时和重试策略可控。

## 11. Shortcut Audit

题目构造时的结构标签只能称为：

```text
intended_structure
```

只有通过 shortcut audit 后，才能生成：

```text
verified_structure
```

每道超图题必须检查：

1. 在全知识库搜索高相关 chunk；
2. 是否存在单 chunk 完整支持全部答案点；
3. 是否存在单超边完整支持多超边题的答案；
4. 分别向回答模型提供单个证据，检查是否可独立稳定回答；
5. 问题措辞是否泄漏中间结论；
6. 是否存在更短的替代证据子图。

发现捷径后：

- 降级结构标签；
- 或改写后重新审核；
- 不得保留名义上的多跳标签。

## 12. Judge 协议与校准

### 12.1 结构化输出

LongCat Judge 使用温度 0，并返回严格 JSON：

```json
{
  "verdict": "pass | fail | uncertain",
  "answer_units": [
    {
      "unit_id": "AU1",
      "status": "supported | missing | contradicted"
    }
  ],
  "unsupported_claims": [],
  "critical_error": false,
  "evidence_sufficient": true
}
```

规则：

- 不向 Judge 暴露路径名称；
- 不要求或保存冗长思维链；
- JSON 解析失败自动重试一次；
- `uncertain`、unsupported claim、critical error 或判定矛盾进入人工复核；
- Judge 输出保存模型名、服务、prompt hash、qrels 版本和时间戳；
- 当前五维评分保留为辅助质量指标，不决定路径成功。

### 12.2 Pilot 校准集

在正式路径标签生成前，从 Pilot 路径输出中分层抽取约 100 个样本，覆盖：

- pass、fail 和 uncertain；
- P0 至 P4；
- 单答案点与多答案点；
- 证据完整、证据缺失、矛盾和 unsupported claim；
- 不同问题结构与检索难度。

人工审核者在不知道路径名称和模型输出来源的条件下，对照 answer units 与精确 evidence spans 给出 source-grounded 判定。由于没有医学专家，该校准只证明对冻结语料的忠实性，不宣称临床专家有效性。

### 12.3 校准指标与冻结门槛

至少报告：

- answer-unit precision、recall、F1；
- verdict accuracy；
- false-pass rate；
- false-fail rate；
- uncertain rate；
- Cohen's kappa 或可获得的一致率；
- 按问题结构和答案点数量分层的错误率。

Judge prompt、解析器和阈值只能使用 Pilot/dev 修改。进入 locked test 前必须冻结：

```text
judge_model
judge_prompt_hash
json_schema_version
adjudication_rule_version
retry_policy
```

false-pass 属于高风险错误，优先降低。边界样本可以使用本地 Qwen 作为第二意见，但最终由 source-grounded 人工复核裁决，不能用两个模型投票替代 Gold 审核。

## 13. 硬淘汰条件

以下候选题不得进入正式问题集：

- 标准答案存在多种合理解释且问题未限定；
- 任一必要答案点无法定位原文；
- 关键推理依赖语料外知识；
- 超边描述与 source chunk 不一致；
- 题目声称高结构难度但存在未处理捷径；
- `P_gold` 无法稳定成功；
- LongCat 审核发现事实扩大、因果扩大或术语错误；
- 与已有题共享核心答案事实或高度近似；
- 问题表达不自然；
- 无法明确哪些证据真正必要；
- 涉及当前审核能力无法覆盖的临床决策。

所有淘汰题写入 `rejected_candidates.jsonl`，保存淘汰原因和审核记录。

## 14. 数据集拆分、规模与统计功效

### 14.1 主集与挑战集

分为：

- `route_benchmark`：仅包含知识库中证据充分、P_gold 可按运行规则成功的问题；
- `unanswerable_challenge`：知识缺失、结构断裂、证据冲突或问题不可判定；
- `retrieval_robustness`：同一证据的直接表达与困难表达配对；
- `reasoning_limit`：完整证据下当前回答模型仍无法稳定处理的有效问题。

不可回答题不能标成 P4。当前 Router 没有 abstain 路径时，`unanswerable_challenge` 不计入 P0 至 P4 主指标。

### 14.2 Evidence cluster

以下问题归入同一 evidence cluster：

- 使用相同 gold chunk；
- 使用相邻或有 overlap 的 chunk；
- 共享必要超边；
- 核心答案实体相同且证据语义高度相似；
- 同一问题的直接与困难改写版本。

同一 evidence cluster 只能进入一个 split。不得简单按超图一跳连通聚类，以免形成无法切分的巨型连通分量。所有置信区间和 bootstrap 以 evidence cluster 为重采样单位，不能把同簇改写题视为独立样本。

### 14.3 Pilot

Pilot 80 题用于：

- prompt 和 schema 调试；
- 路径参数与 budget-response 校准；
- Judge 校准；
- Router 开发；
- relevance pooling 流程验证；
- 候选分布和配对差异估计；
- locked-test power analysis。

Pilot 可以进入最终 dev，但不能进入 locked test，也不用于正式显著性结论。

### 14.4 双测试集

正式测试拆为：

| 集合 | 抽样方式 | 主要用途 |
|---|---|---|
| `balanced_diagnostic_test` | 按 `minimum_sufficient_route_by_policy` 分层 | Macro-F1、confusion matrix、各路径诊断 |
| `corpus_sampled_prevalence_test` | 按冻结后的语料抽样流程，不按路径重平衡 | 总体稳定成功率、成本节省和策略分布 |

没有真实用户查询日志时，第二个集合不得称为 production-natural distribution。候选生成器产生的分布只能解释为当前 corpus-sampling procedure 下的 prevalence。

### 14.5 规模与功效

正式规模不在 Pilot 前强行冻结。Pilot 完成后使用以下观测量做 paired power analysis：

- baseline stable answer success；
- Router 与 baseline 的配对分歧率；
- 目标最小可检测差异，默认 5 个百分点；
- `alpha=0.05`；
- `power>=0.80`；
- evidence-cluster 数量和平均簇大小。

初始资源规划为：

- 先构造约 400 至 600 道高质量候选题；
- balanced diagnostic 暂按每类约 50 题、合计约 250 题规划；
- corpus-sampled prevalence 暂按 300 至 500 题规划；
- unanswerable challenge 额外 20 至 30 题。

最终题数由 power analysis 决定。如果资源限制使 locked test 仍只有约 150 题，论文必须将 2% 至 5% 的差异描述为探索性结果，并报告较宽置信区间，不得仅凭点估计下结论。

### 14.6 统计报告

正式比较至少使用：

- evidence-cluster bootstrap 95% confidence interval；
- paired bootstrap；
- 适用时的 McNemar test；
- effect size；
- 多重比较校正，若同时检验大量 Router/ablation；
- 成功率、成本和答案指标的 paired per-question difference。

正式路径标签必须在题目构造完成后通过实际运行得到，再进行分层抽样。不得先指定某题必须属于某路径。

## 15. Router 数据隔离

Router 输入文件仅包含：

```text
question_id
question
必要的语言或任务类型信息
```

隐藏标注文件包含：

- gold answer；
- answer units；
- evidence groups、qrels 和 spans；
- verified subgraph；
- 四维测量；
- P0 至 P4 的全路径运行结果；
- successful route set；
- minimum sufficient route by policy；
- route cost vectors；
- Pareto-optimal routes；
- profile-specific minimum-cost routes。

基于试探检索的 Router 可以访问知识库并自行生成运行时信号，但不能读取 gold evidence、qrels 或任何派生路径标签。

## 16. 系统快照

每次路径标签生成必须绑定 `system_snapshot_id`，至少记录：

- 数据集、Gold schema 和 qrels 版本；
- 实验契约版本 `question-set-v2.1-contract-v1`（docs/question_set_v2_contract.yaml）与 trace schema 版本；
- full docs、chunks、实体库、关系库和超图哈希；
- chunk 大小和 overlap；
- 回答模型、embedding 模型和 Judge 模型；
- API 服务与模型标识；
- P0 至 P4 的 candidate budget、pre-merge caps、section allocation 和 final hard cap；
- 预算匹配实验配置；
- cost profile、归一化方法和权重版本；
- retriever、reranker 和 score-normalization 版本；
- prompt hashes；
- temperature 与 sampling seed；
- tokenizer 与 token 校准信息；
- operational stability rule 版本；
- Judge/adjudication 版本；
- 代码 commit SHA；
- 运行时间。

## 17. Trace 与检索分数协议

### 17.1 运行级字段

每次路径 repeat 记录：

```text
question_id
system_snapshot_id
route_id
repeat_id
keyword extraction result/hash
context hash
answer text
answer-unit judgments
Judge raw JSON
latency breakdown
LLM / embedding / graph / reranker call counts
input/output token usage
timeout and retry counts
```

### 17.2 Retriever 级字段

每个 retriever 独立记录：

```text
retriever_id
retriever_version
query_embedding_hash
score_semantics
score_direction
score_normalization_method
score_normalization_version
candidate_count
top1_score
top1_top2_gap
score_mean
score_variance
score_entropy
score_skewness
```

每个候选项记录：

```text
candidate_id
candidate_type
rank
raw_retrieval_score
normalized_score
reranker_score or null
parent_seed_id or null
source_provenance_ids
```

不同 retriever 的 raw score 不可直接比较。Trace 必须保存距离/相似度方向和归一化版本；跨 retriever Router 只能使用明确校准后的特征。

由图扩展产生、没有直接 VDB score 的 vertex、hyperedge 或 source chunk，必须保存 seed-to-expanded lineage 和确定性的 score-propagation method，不能伪造原始检索分数。当前没有 reranker 时，`reranker_score=null`。

### 17.3 阶段与证据字段

同时保存：

```text
pre_rerank candidate IDs and scores
post_rerank IDs and scores
post_graph_expansion IDs and lineage
post_assembly source IDs
final_context source IDs
gold evidence-group coverage at every stage
section token counts at every stage
context tokens before/after truncation
lost evidence-group IDs
```

原始上下文优先通过 chunk IDs 重建，避免在主问题文件中重复保存大段文本。Trace schema 由单一版本化模块拥有，生成、比较、Router 和分析脚本不得各自解析私有字段定义。

## 18. Router 评价指标

### 18.1 一级指标

- Operational Stable Answer Success Rate；
- Under-routing Failure Rate；
- Cost Saving vs Fixed P4；
- Quality Regret vs `quality_oracle`；
- Cost Regret vs `cost_oracle(profile)`；
- Cost per Successful Answer；
- Pareto Hit Rate：Router 选择是否属于该题的 Pareto 成功路径集合。

其中：

- `quality_oracle`：所有路径中答案质量最高的路径；
- `policy_oracle`：`minimum_sufficient_route_by_policy`；
- `cost_oracle(profile)`：给定成本配置下所有 operational-success 路径中成本最低者。

所有 oracle 名称必须显式使用，不得使用无定义的通用 `oracle`。

### 18.2 二级分类指标

- Exact Policy Route Accuracy；
- Macro-F1；
- P0 至 P4 confusion matrix；
- Under-routing Rate；
- Over-routing Rate；
- Mean Policy Route Distance；
- Successful-route-set Hit Rate。

Exact Policy Route Accuracy 只比较预定义策略等级，不代表真实成本最优。预测 P3、policy label 为 P2 与预测 P1、policy label 为 P2 的业务后果不同，必须结合实际回答成功和欠路由失败解释。

### 18.3 工程指标

- input/output tokens；
- LLM、embedding、retrieval、graph 和 reranker calls；
- P50/P95 retrieval latency；
- P50/P95 end-to-end latency；
- timeout rate；
- retry rate；
- GPU seconds 或 API cost，若可获得。

论文主结论应围绕：

```text
稳定回答率保持程度
+ 相比固定 P4/Hyper 的成本节省
+ 欠路由造成的质量失败
+ Router 相对 Pareto 与 profile-specific cost oracle 的 regret
```

分类指标用于诊断，不作为唯一主结论。

## 19. Pilot 执行流程

### Step 0：合同与理论口径冻结

1. 固化 `minimum_sufficient_route_by_policy`、成功路径集合、Pareto 路径和 profile-specific cost oracle 的定义。
2. 固化 candidate budget、pre-merge caps、section allocation 和 final hard cap 的单位及数据流。
3. 固化 evidence-group 指标公式、qrels 和 relevance pooling 协议。
4. 固化 Trace schema、score semantics、归一化和 lineage 协议。
5. 固化 operational stability、Judge 校准和人工裁决规则。
6. 为所有 schema、配置和协议分配版本号。

### Step 1：实验基础设施

1. 测试 LongCat API、上下文长度和结构化输出。
2. 校准 tiktoken、Qwen tokenizer 和 API usage。
3. 统一 P0 至 P4 回答 prompt。
4. 拆分 chunk/entity/relation VDB top-k。
5. 明确并实现 pre-merge 与 post-merge budget 字段。
6. 支持正式运行禁用 LLM cache 和独立 repeat seed。
7. 建立 P_gold 与 P0 至 P4 固定 executor。
8. 扩展 Trace，保存 raw/normalized scores、reranker scores 和 graph lineage。
9. 建立统一的 schema owner，供 runner、Judge、分析和 Router 共同读取。
10. 冻结初始 system snapshot。

### Step 2：候选证据与 Gold

1. 按结构配额抽取 source chunks 和 graph motifs。
2. 回溯并验证原始证据。
3. 提取精确 evidence spans。
4. 定义 1 至 5 个原子答案点。
5. 建立证据组、qrels 和必要子超图。
6. 建立 evidence cluster 并检查主题覆盖。
7. 使用本地 Qwen 生成英文问题初稿。
8. 使用 LongCat 审核和必要改写。
9. 人工对照原文逐题确认。

### Step 3：质量、图审计与 Judge 校准

1. 去重和近重复检查。
2. shortcut audit。
3. 硬淘汰规则。
4. P_gold operational success 准入测试。
5. 建立多检索器 relevance pool。
6. 完成图 provenance 全量检查和分层抽样 factual audit。
7. 从路径输出构造约 100 条 Judge 校准样本。
8. 计算 Judge answer-unit F1、false-pass、false-fail 和 uncertain rate。
9. 冻结 Judge prompt、schema 和裁决规则。
10. 写入 accepted 或 rejected candidates。

### Step 4：固定策略全路径标注

1. 禁用 LLM response cache。
2. accepted Pilot 每题完整运行 P0 至 P4。
3. 应用 `operational_stable_success_v1` 和边界补跑规则。
4. 同时检查 evidence-group coverage 与 answer correctness。
5. 保存全阶段 Trace、检索分数、成本向量和 Judge 结果。
6. 生成 successful route set。
7. 生成 minimum sufficient route by policy。
8. 生成 Pareto-optimal routes。
9. 仅对已声明 profile 生成 minimum-cost route 与 cost regret。

### Step 5：预算匹配与敏感性实验

1. 在 `B_low=7K` 和 `B_high=12K` 运行 Naive-source、Hyper-lite-source 和 Hyper-source。
2. 运行 source-only、flat-facts 和 hyper-structured 表示消融。
3. 在分层子集上运行 4K/7K/12K/15K budget-response 曲线。
4. 在分层子集上运行 gold-first/middle/last 与 section-order 扰动。
5. 记录 matched-budget 的实际 Qwen tokens 和所有共同证据指标。

### Step 6：Pilot 分析与功效评估

1. 检查 P0 至 P4 operational-success 分布和非单调现象。
2. 检查成本向量、Pareto 集合和各 cost profile 的稳定性。
3. 检查 token 安全余量和各阶段 budget 使用率。
4. 检查 ER-Recall、CompleteEvidenceHit、TruncationLoss 和 evidence density。
5. 检查图构建审计和 unsupported hyperedge 风险。
6. 检查 Judge 校准结果与人工复核工作量。
7. 检查预算匹配后超图检索和结构表示是否仍有增益。
8. 使用 paired difference 和 evidence-cluster 数量做 power analysis。
9. 确定 balanced diagnostic 与 corpus-sampled prevalence 的正式规模。

### Step 7：Locked-test 前冻结

1. 冻结问题构造、Gold、qrels 和 evidence-cluster 协议。
2. 冻结 P0 至 P4、预算匹配消融和成本 profiles。
3. 冻结 Judge、Trace、指标和统计分析代码。
4. 生成与 Pilot evidence clusters 不重叠的 locked-test 候选。
5. 记录冻结版本和代码 commit SHA。

## 20. Pilot 验收标准

Pilot 至少满足：

- 80 道候选题均有完整状态记录；
- 每个 accepted item 均有 gold answer、answer units、qrels 和精确 evidence spans；
- 所有结构题均有 verified evidence subgraph；
- 所有超图题均通过 shortcut audit；
- 所有 route-benchmark 题均通过 P_gold 准入；
- P1 至 P4 的 operational success 均满足 source evidence 和 answer 双门槛；
- accepted Pilot 每题均完整运行 P0 至 P4；
- 正式路径运行未命中 LLM response cache；
- Qwen token 计数经过校准，final hard cap 在真实 tokenizer 下成立；
- budget 各阶段单位、使用量和截断量可由 Trace 复算；
- Trace 包含 raw score、normalized score、score semantics 和 graph lineage；
- relevance pool、qrels 和 evidence-group 指标通过人工样例复算；
- LongCat Judge 有校准报告，false-pass 与 uncertain 样本均完成复核；
- 图构建 provenance 完整，分层抽样 factual audit 有结果；
- successful route set、minimum sufficient policy route 和 Pareto routes 均可生成；
- cost regret 只在显式 profile 下计算；
- 完成至少一个 low/high matched-budget 检索对照；
- 完成结构表示消融的可运行性验证；
- Pilot 不与未来 locked test evidence clusters 重叠；
- 无临床决策型问题进入主集；
- 淘汰题及原因完整保留；
- 完成 locked-test 规模的 power analysis。

## 21. 冻结结论

问题集 v2.1 不再以题面 stage、LLM 主观复杂度或人为策略顺序充当绝对成本真值。其核心输出是：

```text
原文可验证的 Gold
+ 可重建的必要证据结构
+ 固定策略的独立全路径重复实验
+ 证据与答案双成功门槛
+ 真实成本向量
= successful_route_set
+ minimum_sufficient_route_by_policy
+ pareto_optimal_routes
+ minimum_cost_route_by_profile
```

Router 的任务不是预测抽象的“简单、中等、复杂”，而是在冻结系统条件下选择一条能够 operationally success、不过度消耗资源并尽量接近部署成本最优的实际策略。

不同预算的 P0 至 P4 只用于评价策略阶梯。超图结构贡献必须由 source budget、事实内容和 final context tokens 匹配的独立消融实验支持。
