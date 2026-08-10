# Phase 2 Adaptive 实现计划

> 目标：在已经完成 `neurology_chunk1000` baseline 的基础上，进入 Query-Adaptive Domain-Aware Hyper-RAG 创新实现。  
> 当前策略：先实现可评测的 `adaptive v1`，证明 query-aware + type-aware 检索能改善原版 `hyper` 的短板，再逐步加入更复杂的扩散和上下文组织模块。

重要前置修正：

- 先修复实体线 Entity CSV 不受 `max_token_for_entity_context` 控制的问题，否则 adaptive 预算表不能真实生效。
- `adaptive v1` 不通过放大 text budget 获益。complex 的 `max_token_for_text_unit` 先与 `hyper` baseline 保持一致，即 4000。

## 1. 当前基线状态

`neurology_chunk1000` baseline 已完成核心闭环：

| 项目 | 状态 |
|---|---|
| chunk1000 建库 | 完成 |
| naive 回答 | 43/43，错误 0 |
| hyper 回答 | 43/43，错误 0 |
| naive 五维评分 | 完成 |
| hyper 五维评分 | 完成 |
| pairwise selection | 暂缓 |

当前五维评分：

| 指标 | Naive | Hyper |
|---|---:|---:|
| Comprehensiveness | 91.02 | 81.93 |
| Diversity | 72.98 | 83.00 |
| Empowerment | 83.63 | 73.77 |
| Logical | 94.16 | 92.00 |
| Readability | 94.70 | 96.21 |
| Overall | 87.30 | 85.38 |

阶段判断：

- `hyper` 已经能够稳定完成回答，不再因 context 超限大量失败。
- `hyper` 在 Diversity / Readability 上有优势。
- `hyper` 的主要短板是 Comprehensiveness / Empowerment，说明结构化关系信息更丰富，但直接证据召回和答案支撑不足。
- 后续创新应优先解决“复杂问题召回不足”和“证据支撑不够强”的问题。

## 2. 总体路线

不要一次性实现模块2-6。先做一个最小可评测版本：

```text
adaptive v1 = Query Router + adaptive retrieval budget/top_k + type-aware soft weighting
```

对应论文模块：

| 模块 | adaptive v1 是否覆盖 | 说明 |
|---|---:|---|
| 模块2：Query 感知路由器 | 是 | 判断 simple / complex、query_type、target_entity_types |
| 模块3：类型感知检索 | 是 | 对 entity_type / edge_type 匹配结果进行 soft weighting |
| 模块4：自适应扩散 | 部分 | 先做 top_k / token budget / 关系保留策略，不急着做完整 2-hop |
| 模块5：超边质量排序 | 暂缓 | 等 v1 有结果后再加入，便于消融归因 |
| 模块6：结构化上下文 | 暂缓 | 后置，避免回答风格变化干扰检索效果判断 |

## 3. 实施顺序

### Step 0：修复实体线 Entity CSV token 截断

修改文件：

- `hyperrag/query_context.py`

问题：

- `_build_entity_query_context()` 中，`entities_vdb.query(query, top_k=query_param.top_k)` 返回的实体全部进入 `node_datas`。
- 这些实体随后全部写入 Entity CSV。
- 当前实体线没有使用 `query_param.max_token_for_entity_context` 做 token 截断。
- 因此当 adaptive complex 设置 `top_k=80` 时，Entity CSV 可能膨胀到数千 tokens，导致预算表失效，甚至再次触发 context overflow。

对照：

- 关系线 `_find_most_related_entities_from_relationships()` 已经使用 `truncate_list_by_token_size(..., max_token_size=query_param.max_token_for_entity_context)` 控制实体上下文。
- 实体线也应有类似控制。

推荐修法：

- 区分“检索扩散用实体”和“写入 prompt 的实体”。
- `seed_node_datas` 保留较多实体，用于寻找相关 text units 和 hyperedges，避免损失召回。
- `context_node_datas` 按 `max_token_for_entity_context` 截断，只用于 Entity CSV 和返回给 LLM 的结构化实体表。

示意：

```python
seed_node_datas = node_datas
context_node_datas = truncate_list_by_token_size(
    node_datas,
    key=lambda x: x.get("description", ""),
    max_token_size=query_param.max_token_for_entity_context,
)

use_text_units = await _find_most_related_text_unit_from_entities(
    seed_node_datas, query_param, text_chunks_db, knowledge_hypergraph_inst
)
use_relations = await _find_most_related_edges_from_entities(
    seed_node_datas, query_param, knowledge_hypergraph_inst
)

# Entity CSV 使用 context_node_datas
```

验收：

- `hyper` / `adaptive` 实体线 Entity CSV 受 `max_token_for_entity_context` 控制。
- 不因为截断 prompt 实体表而明显减少 text units / relations 的召回。
- `hyper` baseline 重跑少量问题不报错。

### Step 1：新增 adaptive mode 空壳

修改文件：

- `hyperrag/base.py`
- `hyperrag/hyperrag.py`
- `hyperrag/query_modes.py`
- `reproduce/Step_3_response_question.py`

目标：

- `QueryParam.mode` 支持 `"adaptive"`。
- `HyperRAG.aquery()` 能路由到 `adaptive_query()`。
- Step_3 可以执行：

```bash
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode adaptive
```

验收：

- adaptive mode 初始可退化为原 `hyper` 行为。
- 跑少量问题不报错。

### Step 2：实现 Query Router

新增文件：

- `hyperrag/query_router.py`

建议数据结构：

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class QueryRoute:
    complexity: Literal["simple", "complex"] = "complex"
    query_type: Literal["fact", "mechanism", "causal", "comparison", "multi-hop"] = "fact"
    target_entity_types: list[str] = field(default_factory=list)
```

实现方式：

- 第一版使用 LLM prompt 输出 JSON。
- 如果 JSON 解析失败，使用安全默认值：

```text
complexity = complex
query_type = fact
target_entity_types = []
```

路由标签建议：

| 字段 | 候选值 |
|---|---|
| complexity | simple / complex |
| query_type | fact / mechanism / causal / comparison / multi-hop |
| target_entity_types | DISEASE, SYMPTOM, SIGN, DRUG, TREATMENT, EXAMINATION, ANATOMICAL_STRUCTURE, PATHOLOGICAL_MECHANISM, GENE, PROTEIN, PATHWAY, RISK_FACTOR, DIAGNOSTIC_CRITERION, OTHER |

缓存建议：

- 优先复用 `global_config["llm_model_func"]` 调 router，因为该函数已经由 HyperRAG 包装并接入 LLM cache。
- 如果 router 直接调用 `openai_complete_if_cache`，则需要显式传入 `hashing_kv`。

验收：

- 对 43 条 2-stage 问题输出稳定 JSON。
- 人工抽查 10 条，复杂问题应大多判为 complex。

### Step 3：adaptive 参数控制

目标：

根据 `QueryRoute` 动态调整检索预算，而不是固定使用 `hyper` 的同一套参数。

建议第一版规则：

| complexity | top_k | max_token_for_text_unit | max_token_for_entity_context | max_token_for_relation_context |
|---|---:|---:|---:|---:|
| simple | 30 | 3000 | 200 | 1000 |
| complex | 80 | 4000 | 400 | 1800 |

注意：

- complex 的 text budget 第一版先与当前 hyper baseline 保持一致，即 4000。
- v1 的收益应主要来自 query route、type-aware weighting 和 adaptive top_k，而不是塞入更多原文上下文。
- 如果 4000 仍超限，应先检查 Entity CSV / Relation CSV 截断是否生效，再考虑降到 3000。
- simple 降低预算，用于证明自适应方法能减少噪声和成本。

验收：

- adaptive 跑 43 条不出现 context length error。
- 记录每题 route 和实际参数，方便后续分析。

### Step 4：类型感知 soft weighting

修改文件：

- `hyperrag/query_context.py`

目标：

实体和关系检索不只按 embedding 相似度排序，还结合 Query Router 的目标实体类型进行加权。

插入位置：

- 实体线：在 `entities_vdb.query()` 返回结果之后、回查 vertex 和后续排序/截断之前。
- 关系线：在 `relationships_vdb.query()` 返回结果之后、回查 hyperedge 和后续排序/截断之前。

注意：

- 当前 `NanoVectorDBStorage.query()` 返回字段主要是 `distance`，不是 `weight`。
- 建议新增 `adjusted_score` 字段，而不是覆盖原始 `distance`。
- 后续排序应基于 `adjusted_score` 或显式排序后的结果。

第一版建议：

```text
entity_type 命中 target_entity_types：adjusted_score = distance * 1.20
entity_type 未命中但 target_entity_types 非空：adjusted_score = distance * 0.95
UNKNOWN / OTHER：adjusted_score = distance * 0.90
```

关系线可先用 `edge_type` 做轻量加权：

```text
query_type = causal      -> CAUSES / MECHANISM_OF / RISK_FACTOR_FOR 加权
query_type = mechanism   -> MECHANISM_OF / REGULATES / PATHWAY_PROCESS 加权
query_type = comparison  -> DIFFERENTIAL_DIAGNOSIS / DIFFERENTIAL_GROUP 加权
query_type = fact        -> 不强加权
query_type = multi-hop   -> 高阶关系加权
```

原则：

- 第一版只做 soft weighting，不硬过滤。
- 避免因为 router 分类错误导致漏召回。

验收：

- adaptive 跑完 43 条。
- 与 hyper 对比五维评分。
- 重点看 Comprehensiveness / Empowerment 是否提升。
- 保存每题 route、预算和实际上下文 token 指标，用于解释效果。

## 4. 第一轮实验设计

### E1：Original Hyper-RAG

已完成：

```text
mode = hyper
data = neurology_chunk1000
```

### E3/E4：Adaptive v1

执行：

```bash
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode adaptive
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode adaptive --question-stage 2
```

输出：

```text
caches/neurology_chunk1000/response/adaptive_2_stage_result.json
caches/neurology_chunk1000/response/adaptive_2_stage_errors.json
caches/neurology_chunk1000/evalation/scoring_2_stage_question_adaptive.json
```

对比表：

| Method | Comprehensiveness | Diversity | Empowerment | Logical | Readability | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 91.02 | 72.98 | 83.63 | 94.16 | 94.70 | 87.30 |
| Hyper | 81.93 | 83.00 | 73.77 | 92.00 | 96.21 | 85.38 |
| Adaptive v1 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

成功标准：

- Adaptive v1 不低于 Hyper overall。
- Comprehensiveness 或 Empowerment 至少有一个明显提升。
- 错误数为 0。

## 5. 第二轮扩展计划

如果 Adaptive v1 有正向结果，再实现后续模块。

### Step 5：超边质量排序

新增文件：

- `hyperrag/edge_quality.py`

质量信号：

- 是否来自多个 source chunk
- 是否覆盖多个 query entities
- 是否包含目标实体类型
- 是否为高阶超边
- degree 是否过高
- 与 high-level keywords 是否相关

目标：

- 降低过泛化超边和 hub 关系的噪声。
- 提升复杂问题中的高阶证据质量。

### Step 6：更完整的自适应扩散

新增或修改：

- `hyperrag/adaptive_diffusion.py`
- `hyperrag/query_context.py`

扩展方向：

- complex query 允许更强关系保留。
- 尝试有限 2-hop 扩散。
- 对 hop distance 加惩罚，防止扩散过远。

### Step 7：结构化证据上下文

修改：

- `query_context.py`

目标上下文结构：

```text
Direct Evidence
Key Entities
High-order Relations
Supporting Sources
```

注意：

- 该模块会改变 LLM 最终回答风格，应放在检索模块稳定后再做。
- 做完后需要单独消融，避免和检索提升混淆。

## 6. 指标与记录

建议新增每题中间指标，保存为：

```text
caches/neurology_chunk1000/response/adaptive_2_stage_metrics.jsonl
```

字段建议：

```json
{
  "query": "...",
  "complexity": "complex",
  "query_type": "multi-hop",
  "target_entity_types": ["DISEASE", "EXAMINATION"],
  "top_k": 80,
  "max_token_for_text_unit": 4000,
  "entity_count": 60,
  "relation_count": 40,
  "text_unit_count": 6,
  "high_order_edge_count": 12,
  "configured_text_budget": 4000,
  "configured_entity_budget": 400,
  "configured_relation_budget": 1800,
  "actual_context_tokens": 18000,
  "latency_ms": 12345
}
```

这些指标用于论文解释：

- 为什么 adaptive 对复杂问题更有效；
- 是否减少 simple query 的无关上下文；
- 是否提升高阶关系命中；
- 是否控制了 context token 成本。

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| Router 输出 JSON 不稳定 | 加 JSON 提取兜底和默认 route |
| Adaptive complex 预算过大导致 context 超限 | 第一版 complex text budget 固定 4000，并先修实体线截断 |
| 类型加权导致漏召回 | 只 soft weighting，不 hard filter |
| 结构化上下文影响回答风格，难以归因 | 后置到 v1 之后单独消融 |
| Adaptive v1 效果不如 Hyper | 保留 route / 检索中间指标，分析是 router 误判还是检索策略问题 |

## 8. 最近可执行任务清单

1. 修复 `_build_entity_query_context()` 的实体线 Entity CSV token 截断。
2. 新建 `hyperrag/query_router.py`。
3. 在 `QueryParam.mode` 中加入 `"adaptive"`。
4. 在 `HyperRAG.aquery()` 中接入 `adaptive_query()`。
5. 在 `Step_3_response_question.py` 的 choices 中加入 `"adaptive"`。
6. 先让 `adaptive_query()` 退化为 `hyper_query()`，确认入口跑通。
7. 接入 Query Router，保存 route 信息。
8. 加 adaptive top_k / token budget，其中 complex text budget 先用 4000。
9. 加类型感知 soft weighting，使用 `adjusted_score` 排序。
10. 记录 `actual_context_tokens` 等真实指标。
11. 跑 43 条 2-stage adaptive 回答。
12. 跑 adaptive scoring。
13. 与 naive / hyper 做表格对比。

## 9. 阶段目标

短期目标：

```text
跑出 Adaptive v1 的 43 条回答和五维评分。
```

中期目标：

```text
证明 adaptive 在 Hyper 的短板指标 Comprehensiveness / Empowerment 上有改善。
```

论文目标：

```text
形成 E1 Hyper vs E7 Full Adaptive 的主线对比，并通过 E3-E6 消融说明各模块贡献。
```
