# 实验计划：Query-Adaptive Domain-Aware Hyper-RAG

> 生成时间：2026-07-04
> 数据集：neurology（12370 条 unique context）
> 评测方式：沿用论文五维打分（Comprehensiveness / Diversity / Empowerment / Logical / Readability）+ selection pairwise 对比
> 核心对比：**原版 Hyper-RAG (hyper) vs 我的创新方案 (adaptive)**

---

## 0. 实验总览

```
Phase 0  基础设施修复    → 保证流水线能跑通
Phase 1  Baseline 复现   → 跑出原版 hyper / naive 的成绩作为对照基准
Phase 2  创新实现        → 逐模块实现，每加一个模块做一次消融
Phase 3  消融实验        → 7 组对比，证明每个模块的贡献
Phase 4  分析与写作      → 中间指标 + 最终指标 + 结论
```

### 实验矩阵（Phase 3 最终产出）

| 编号 | 实验组 | mode 名称 | 模块1 | 模块2 | 模块3 | 模块4 | 模块5 | 模块6 | 说明 |
|------|--------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| E0 | Naive RAG | `naive` | - | - | - | - | - | - | 纯 chunk 向量检索 |
| E1 | Original Hyper-RAG | `hyper` | - | - | - | - | - | - | 论文原版，固定检索 |
| E2 | + 类型索引 | `hyper` | ✅ | - | - | - | - | - | 仅建库阶段增强 |
| E3 | + Query 路由 | `adaptive` | ✅ | ✅ | - | - | - | - | 路由但不调检索 |
| E4 | + 类型检索 | `adaptive` | ✅ | ✅ | ✅ | - | - | - | 类型加权 soft filter |
| E5 | + 自适应扩散 | `adaptive` | ✅ | ✅ | ✅ | ✅ | - | - | 核心创新模块 |
| E6 | + 超边排序 | `adaptive` | ✅ | ✅ | ✅ | ✅ | ✅ | - | 质量感知过滤 |
| E7 | Full Method | `adaptive` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 完整方法 |

> **E1 vs E7 是主线对比**（你指定的核心目标）。E2-E6 是消融实验，证明每个模块的边际贡献。

---

## Phase 0：基础设施修复（预计 1-2 轮调试）

### 0.1 删除旧 caches，重新建库

```bash
rm -rf caches/neurology/vdb_chunks.json
# 保留 caches/neurology/contexts/neurology_unique_contexts.json（Step_0 产物，可复用）
```

### 0.2 修复 Step_1 已知问题

**问题**：之前别机跑 Step_1 时 LLM 返回空 → hypergraph 0 vertices。

**排查清单**：
1. 确认 LLM 服务可达：`python scripts/test_llm_config.py`
2. 确认 embedding 维度：`python scripts/verify_embedding_dim.py`
3. 确认 prompt 格式：`entity_extraction` prompt 中的 `entity_types` / `relation_types` / `high_order_relation_types` 变量正确注入
4. 跑 5 条 context 的小规模测试，确认实体/超边抽取正常

### 0.3 Step_1 小规模验证

```bash
# 改造 Step_1 加 --limit 参数，只取前 50 条 context 做冒烟测试
python reproduce/Step_1.py --data-name neurology --limit 50
# 检查产出：
#   caches/neurology/hypergraph_chunk_entity_relation.hgdb  (要有 vertices > 0)
#   caches/neurology/vdb_entities.json
#   caches/neurology/vdb_relationships.json
#   caches/neurology/kv_store_text_chunks.json
#   caches/neurology/kv_store_full_docs.json
```

### 0.4 Step_1 全量建库

冒烟测试通过后跑全量 12370 条。预计 LLM 调用 864 chunk × (1 次抽取 + gleaning) ≈ 2000+ 次。

**产出文件清单（验收标准）**：
- [x] `caches/neurology/contexts/neurology_unique_contexts.json`
- [x] `caches/neurology/vdb_chunks.json` (864 条, dim=2048)
- [x] `caches/neurology/vdb_entities.json`
- [x] `caches/neurology/vdb_relationships.json`
- [x] `caches/neurology/kv_store_full_docs.json`
- [x] `caches/neurology/kv_store_text_chunks.json`
- [x] `caches/neurology/hypergraph_chunk_entity_relation.hgdb`
- [x] `caches/neurology/kv_store_llm_response_cache.json`（可选，加速用）

---

## Phase 1：Baseline 复现

### 1.1 生成评测问题（Step_2）

```bash
python reproduce/Step_2_extract_question.py --data-name neurology
```

产出：
- `caches/neurology/questions/2_stage.json`（二阶段问题，测试组合检索）
- `caches/neurology/questions/2_stage_ref.json`（参考原文，评测用）

> 建议同时生成 1_stage（单跳）和 3_stage（三跳），用于观察「复杂度越高，创新方法优势越明显」的结论。

### 1.2 跑 Naive RAG（E0）

```bash
python reproduce/Step_3_response_question.py --data-name neurology --mode naive
```

产出：`caches/neurology/response/naive_2_stage_result.json`

### 1.3 跑 Original Hyper-RAG（E1）

```bash
python reproduce/Step_3_response_question.py --data-name neurology --mode hyper
```

产出：`caches/neurology/response/hyper_2_stage_result.json`

### 1.4 评测 Baseline

```bash
# 五维打分
python evaluate/evaluate_by_scoring.py --data-name neurology --mode naive --question-stage 2
python evaluate/evaluate_by_scoring.py --data-name neurology --mode hyper --question-stage 2

# Pairwise 对比（naive vs hyper）
python evaluate/evaluate_by_selection.py --data-name neurology --mode-a naive --mode-b hyper --question-stage 2
```

**验收**：得到 E0 和 E1 的五维分数 + pairwise 胜率。确认 hyper > naive（与论文一致），否则建库质量有问题需回 Phase 0。

---

## Phase 2：创新模块实现

### 实现顺序与依赖关系

```
模块1 (类型索引) ──→ 模块3 (类型检索) ──→ 模块5 (超边排序)
                                          ↗
模块2 (Query路由) ──→ 模块4 (自适应扩散) ──→ 模块6 (上下文组装)
```

### 2.1 模块 1：领域类型增强索引

**现状**：prompt.py 已定义 15 类实体类型 + 关系类型 + 高阶关系类型。indexing.py 已在抽取时写入 `entity_type` / `edge_type`。

**待做**：
- 确认 entities_vdb upsert 时 content 拼接已包含 type（当前 indexing.py 第 246-254 行已做：`" | ".join([entity_type, entity_name, description])`）✅
- 确认 relationships_vdb 同理（第 262-272 行已做）✅
- **本模块基本已完成**，只需在 Phase 0 建库后验证 type 字段确实写入

**消融点**：E1 vs E2 的区别是建库时是否在 embedding content 中拼入 type。需要准备一个 `hyper_no_type` 版本（去掉 type 拼接）作为 E1，当前版本作为 E2。

### 2.2 模块 2：Query 感知路由器

**新建文件**：`hyperrag/query_router.py`

```python
@dataclass
class QueryRoute:
    complexity: Literal["simple", "complex"]
    query_type: Literal["fact", "mechanism", "causal", "comparison", "multi-hop"]
    target_entity_types: list[str]  # 如 ["DISEASE", "DRUG"]
```

**实现方式**：LLM prompt，输入用户问题，输出 JSON 标签。

**接入点**：`query_modes.py` 的 `hyper_query` 开头，调用 router 后将 `QueryRoute` 传入后续检索函数。

**消融验证**：路由准确率人工抽检 50 条，确保 simple/complex 分类合理。

### 2.3 模块 3：类型感知检索增强

**修改文件**：`query_context.py` 的 `_build_entity_query_context` / `_build_relation_query_context`

**改动**：
- `entities_vdb.query()` 返回结果后，根据 `QueryRoute.target_entity_types` 对每个结果加权
- 匹配类型的实体 weight × 1.5（soft filter，不删除不匹配的）
- 关系线同理，根据 `edge_type` 匹配加权

### 2.4 模块 4：自适应超图拓扑扩散（核心创新）

**修改文件**：`query_context.py` + 可能新增 `hyperrag/adaptive_diffusion.py`

**改动**：
```python
if route.complexity == "simple":
    top_k = 30        # 降低
    max_hop = 1
    keep_high_order = False
elif route.complexity == "complex":
    top_k = 80        # 提高
    max_hop = 2       # 允许 2-hop 扩散
    keep_high_order = True  # 优先保留高阶超边
```

**超边综合权重计算**：
```
weight = α × semantic_similarity
       + β × query_entity_coverage   # 超边覆盖了多少个 query 实体
       + γ × type_match_score        # 超边中实体类型与 target_types 匹配度
       + δ × order_bonus             # 高阶超边 bonus（complex 时才加）
       - ε × degree_penalty           # degree 过高的超边降权（防 hub 噪声）
       - ζ × hop_distance             # 距离起点越远降权
```

**消融验证**：对比 simple 问题的检索时间是否下降，complex 问题的召回是否提升。

### 2.5 模块 5：质量感知超边排序

**新增**：在模块 4 扩散得到候选超边后，过一层质量打分过滤。

**质量信号**：
- `multi_source`：超边来自 ≥2 个不同 source chunk → +分
- `multi_entity_coverage`：覆盖 ≥2 个 query entity → +分
- `type_relevance`：包含 target_entity_types → +分
- `keyword_similarity`：与 high_level_keywords 语义相似 → +分
- `over_generalization`：degree > 阈值 → -分

**实现**：新增 `hyperrag/edge_quality.py`，在 `query_context.py` 扩散后调用。

### 2.6 模块 6：结构化证据上下文组装

**修改文件**：`query_context.py` 的 `combine_contexts`

**改动**：不再简单拼接 Entities/Relationships/Sources，而是按证据角色组织：

```
=== Direct Evidence ===
[最相关的原文 chunk]

=== Key Entities ===
[实体表，simple 时精简，complex 时完整]

=== High-order Relations ===
[complex 时突出高阶超边，simple 时省略]

=== Supporting Sources ===
[补充来源]
```

### 2.7 新增 mode：`adaptive`

**修改文件**：`base.py` 的 `QueryParam.mode` 增加 `"adaptive"`；`query_modes.py` 新增 `adaptive_query` 函数。

```python
async def adaptive_query(query, ...):
    route = await query_router(query)          # 模块2
    entity_context = await _build_entity_query_context_adaptive(
        ..., route=route                        # 模块3+4
    )
    relation_context = await _build_relation_query_context_adaptive(
        ..., route=route                        # 模块3+4+5
    )
    context = combine_contexts_structured(       # 模块6
        entity_context, relation_context, route
    )
    return context
```

---

## Phase 3：消融实验执行

### 3.1 实验执行顺序

按依赖关系，每实现一个模块就跑一次评测：

| 步骤 | 实现 | 跑 Step_3 mode | 评测 |
|------|------|----------------|------|
| 1 | (Phase 1 已完成) | naive, hyper | E0, E1 baseline |
| 2 | 确认模块1已生效 | hyper（当前版本已含 type） | E2 |
| 3 | 实现模块2+3 | adaptive（仅路由+类型加权） | E3+E4 合并 |
| 4 | 实现模块4 | adaptive（+自适应扩散） | E5 |
| 5 | 实现模块5 | adaptive（+超边排序） | E6 |
| 6 | 实现模块6 | adaptive（完整） | E7 |

### 3.2 每组实验的执行命令模板

```bash
# Step_3 跑回答
python reproduce/Step_3_response_question.py --data-name neurology --mode <mode>

# 五维打分
python evaluate/evaluate_by_scoring.py --data-name neurology --mode <mode> --question-stage 2

# Pairwise 对比（E1 vs En）
python evaluate/evaluate_by_selection.py --data-name neurology --mode-a hyper --mode-b <mode> --question-stage 2
```

### 3.3 问题阶段扩展（可选但建议）

核心对比只在 2_stage 上做。如果时间允许，在 1_stage 和 3_stage 上也跑 E1 vs E7，验证：
- 1_stage（简单问题）：adaptive 不应弱于 hyper，且检索更快
- 3_stage（复杂问题）：adaptive 优势应更明显

---

## Phase 4：分析与写作

### 4.1 最终指标（论文表格）

**Table 1：五维打分（2_stage）**

| Method | Comprehensiveness | Diversity | Empowerment | Logical | Readability | Avg |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Naive (E0) | | | | | | |
| Hyper (E1) | | | | | | |
| +Type (E2) | | | | | | |
| +Route (E3) | | | | | | |
| +TypeRet (E4) | | | | | | |
| +AdaptDiff (E5) | | | | | | |
| +EdgeRank (E6) | | | | | | |
| **Full (E7)** | | | | | | |

**Table 2：Pairwise 胜率（E1 vs E7，8 维度）**

| Criterion | E1 Win | E7 Win | Tie |
|-----------|:-:|:-:|:-:|
| Comprehensiveness | | | |
| Empowerment | | | |
| Accuracy | | | |
| Relevance | | | |
| Coherence | | | |
| Clarity | | | |
| Logical | | | |
| Flexibility | | | |

### 4.2 中间指标（解释「为什么有效」）

在 `adaptive_query` 中埋点记录：

| 指标 | 说明 | 预期 |
|------|------|------|
| `retrieval_time` | 检索阶段耗时 | simple < hyper, complex ≈ hyper |
| `entity_recall` | 检索到的相关实体数 | complex 时 adaptive > hyper |
| `high_order_edge_ratio` | 高阶超边占候选比 | complex 时 adaptive > hyper |
| `source_chunk_hit` | 命中的 source chunk 数 | adaptive ≥ hyper |
| `context_token_count` | 最终上下文 token 数 | simple 时 adaptive < hyper |

> 需要在 Step_3 中加 `--log-metrics` 参数，把每个问题的中间指标写入 `response/<mode>_2_stage_metrics.jsonl`。

### 4.3 复杂度分层分析

将 2_stage 问题按 router 输出的 complexity 分组，分别统计 simple / complex 两组的分数：

**预期结论**：
- Simple 问题：adaptive ≈ hyper（不退步），但检索成本更低
- Complex 问题：adaptive > hyper（显著提升），高阶超边命中率更高

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| LLM 抽取返回空（Phase 0 老问题） | 小规模测试 + chunk 级重试已加，确认 prompt 变量正确注入 |
| LLM 评测打分不稳定 | 每组实验跑 2 次取平均；pairwise 评测顺序随机化 |
| Step_1 全量建库耗时过长 | 864 chunk × LLM 调用，预计数小时；可先跑 100 chunk 验证流程 |
| 路由器分类不准 | 先人工标注 50 条做校准；用 few-shot prompt 提升稳定性 |
| adaptive 模式不退步保证 | 模块3 用 soft filter 不硬删；模块4 simple 时退化为原 hyper 策略 |
