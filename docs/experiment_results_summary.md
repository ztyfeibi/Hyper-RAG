# Hyper-RAG Phase 2 实验结果汇总

> 生成时间: 2026-07-14
> 目的: 提供给其他 AI 阅读的完整上下文，包含实验设计、执行命令、结果数据和分析结论

---

## 1. 项目背景

Hyper-RAG 是一个基于超图（Hypergraph）的 RAG 框架，在传统向量检索基础上增加了实体关系超边检索。本项目在 Hyper-RAG 原版基础上实现 **Query-Adaptive Domain-Aware Hyper-RAG**，核心思想：让系统先理解 Query 类型，再动态决定检索范围、扩散深度和超边权重。

### 技术栈
- **LLM**: qwen-27b-int4 @ 10.65.1.110:8002 (vLLM 部署, max-model-len=24576)
- **Embedding**: qwen-8b-embed @ 10.65.1.110:8001, dim=4096
- **数据集**: neurology (12370 条 unique context, 864 chunks @ chunk_size=2400, 2317 chunks @ chunk_size=1000)
- **Python 环境**: conda hyperrag
- **评测**: 五维打分（Comprehensiveness / Diversity / Empowerment / Logical / Readability），每维 0-100 分

### 三种检索模式
1. **naive**: 纯向量检索，取 top-k 原文片段拼入 context
2. **hyper**: 双线检索 — 向量检索 + 超图实体/关系检索，合并为 Entities CSV + Relationships CSV + Sources CSV
3. **adaptive** (本项目创新): Router 先分类 query 复杂度 → 按档位动态调参 → 类型感知加权 → 区段感知预算截断 → 生成回答

---

## 2. Adaptive 管线架构

```
Query 输入
  │
  ▼
┌─────────────────────────────┐
│ Step 1: Query Router        │  LLM 分类: query_type / complexity / focus_types
│ (query_router.py)           │  输出: simple / medium / complex
└──────────┬──────────────────┘
           │ route result
           ▼
┌─────────────────────────────┐
│ Step 2: Adaptive Params     │  按复杂度档位选择参数组合
│ (adaptive_params.py)        │  simple:   top_k=30, budget=5000
│                             │  medium:   top_k=50, budget=7000
│                             │  complex:  top_k=70, budget=8000
└──────────┬──────────────────┘
           │ adapted QueryParam
           ▼
┌─────────────────────────────┐
│ Step 3: Type-Aware Weighting│  按 route.focus_types 软加权
│ (type_aware_weighting.py)   │  entity ×1.15, relation ×1.20, mechanism ×1.15
│                             │  加权后 rerank，不硬过滤
└──────────┬──────────────────┘
           │ re-ranked results
           ▼
┌─────────────────────────────┐
│ Step 4: Hyper Query         │  实体 VDB 检索 + 关系 VDB 检索 + 原文检索
│ (query_context.py)          │  合并为 Entities/Relationships/Sources CSV
└──────────┬──────────────────┘
           │ merged context
           ▼
┌─────────────────────────────┐
│ Step 5: Context Budget Fuse │  区段感知截断 (v2)
│ (context_budget.py)         │  Sources 55% / Relationships 30% / Entities 15%
│                             │  区段不足时盈余重分配
│                             │  优先保留 Sources 原文
└──────────┬──────────────────┘
           │ truncated context
           ▼
        LLM 生成回答
```

### Adaptive 参数档位

| 档位 | top_k | entity_tokens | relation_tokens | text_unit_tokens | total_budget |
|------|-------|---------------|-----------------|------------------|-------------|
| simple | 30 | 200 | 1200 | 2500 | 5000 |
| medium | 50 | 300 | 1600 | 3500 | 7000 |
| complex | 70 | 400 | 2200 | 4000 | 8000 |

### Router 策略 (Step 5.2 新增)

| 策略 | 路由来源 | 用途 |
|------|---------|------|
| `llm` | 调用 LLM 分类（默认） | 正式 adaptive 路径 |
| `fixed` | 强制固定档位 | 排除 Router，验证其余 adaptive 模块 |
| `oracle` | 读取 meta 的 expected_complexity | 测试 Router 理论上限，仅诊断 |

---

## 3. 实验设计

### 3.1 评测集构建

分层混合评测集 (mixed_stage)，83 题，按问题复杂度分层：

| Stage | 题数 | expected_complexity | 问题类型 |
|-------|------|---------------------|---------|
| Stage 1 | 20 | simple | 单跳事实查询（单个原文片段可回答） |
| Stage 2 | 43 | medium | 双跳递进查询（需要实体关系或一次证据连接） |
| Stage 3 | 20 | complex | 三跳递进查询（需组合多条独立证据） |

问题由 LLM 从原始 context 自动生成，Stage 2 复用了之前的 43 题，Stage 1/3 新生成。

### 3.2 实验组（5 组）

| 编号 | 名称 | mode | router_policy | 说明 |
|------|------|------|---------------|------|
| E0 | naive | naive | N/A | 纯向量检索基线 |
| E1 | hyper | hyper | N/A | 超图双线检索基线 |
| E2 | adaptive_v1 | adaptive | llm | 完整 adaptive（旧版尾截断 budget） |
| E3 | adaptive_fixed_medium | adaptive | fixed (medium) | 所有题走 medium 档，排除 Router |
| E4 | adaptive_oracle | adaptive | oracle | 按真实复杂度走对应档，Router 上限 |

### 3.3 执行的命令

```bash
# 所有命令在 conda activate hyperrag 环境下执行
# 工作目录: D:/projectes/lyw-rag

# === Phase 1: 生成分层问题 ===
python reproduce/Step_2_extract_question.py --data-name neurology_chunk1000 --stage 1 --max-cnt 20 --seed 101
python reproduce/Step_2_extract_question.py --data-name neurology_chunk1000 --stage 3 --max-cnt 20 --seed 103
# Stage 2 已有 43 题，补了 meta 文件后直接复用

# === Phase 2: 合并 mixed_stage ===
python scripts/build_mixed_questions.py --data-name neurology_chunk1000 --stages 1 2 3

# === Phase 3: 生成回答（每组约 15-30 分钟） ===
# 基线
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode naive --question-file mixed_stage
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode hyper --question-file mixed_stage

# Adaptive v1 (旧版尾截断 budget，已备份)
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage

# Adaptive fixed_medium (Step 5.2 新增，区段感知 budget)
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage --router-policy fixed --forced-complexity medium --output-suffix fixed_medium

# Adaptive oracle (Step 5.2 新增，按 stage 注入复杂度)
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage --router-policy oracle --output-suffix oracle

# === Phase 4: 五维打分 ===
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode naive --question-file mixed_stage
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode hyper --question-file mixed_stage
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage --output-suffix fixed_medium
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode adaptive --question-file mixed_stage --output-suffix oracle

# === Phase 5: Router 评估 ===
python scripts/evaluate_router_on_mixed.py --data-name neurology_chunk1000 --question-file mixed_stage
```

### 3.4 结果产物文件清单

```
caches/neurology_chunk1000/
├── questions/
│   ├── mixed_stage.json          # 83 题
│   ├── mixed_stage_ref.json      # 参考答案
│   └── mixed_stage_meta.json     # 每题 stage/expected_complexity
├── response/
│   ├── naive_mixed_stage_result.json
│   ├── hyper_mixed_stage_result.json
│   ├── adaptive_mixed_stage_result.json              # v1 (旧尾截断)
│   ├── adaptive_mixed_stage_fixed_medium_result.json  # fixed medium
│   └── adaptive_mixed_stage_oracle_result.json        # oracle
└── evalation/
    ├── scoring_mixed_stage_question_naive.json
    ├── scoring_mixed_stage_question_hyper.json
    ├── scoring_mixed_stage_question_adaptive.json
    ├── scoring_mixed_stage_fixed_medium_question_adaptive.json
    ├── scoring_mixed_stage_oracle_question_adaptive.json
    └── router_eval_mixed_stage.json

backups/
├── mixed_stage_baseline_v1_20260713/   # adaptive_v1 全量备份
└── adaptive_failed_tailcut_20260712/   # 早期 2_stage 失败结果备份
```

---

## 4. 实验结果

### 4.1 总体五维评分

| Mode | N | Overall | Comp | Div | Emp | Log | Read |
|------|---|---------|------|-----|-----|-----|------|
| **naive** | 83 | **85.0** | 87.8 | 71.1 | 79.3 | 92.8 | 93.9 |
| **hyper** | 82 | **81.2** | 76.8 | 78.1 | 68.2 | 88.0 | 94.8 |
| adaptive_v1 (旧尾截断) | 83 | **71.3** | 63.9 | 67.3 | 51.8 | 80.9 | 92.8 |
| adaptive_fixed_medium | 83 | **78.2** | 71.3 | 75.3 | 64.4 | 85.8 | 94.1 |
| adaptive_oracle | 82 | **72.1** | 64.0 | 67.3 | 54.9 | 80.9 | 93.3 |

> Overall = 五维均分。naive 最高，hyper 次之，adaptive 全面落后。

### 4.2 按 Stage 分层评分

| Mode | Stage 1 (simple, n=20) | Stage 2 (medium, n=43) | Stage 3 (complex, n=20) |
|------|----------------------|----------------------|------------------------|
| **naive** | 79.6 | 86.1 | 88.0 |
| **hyper** | 74.1 | 82.0 | 86.2 |
| adaptive_v1 | 69.6 | 72.6 | 70.4 |
| adaptive_fixed_medium | 77.7 | 79.5 | 75.8 |
| adaptive_oracle | 72.9 | 71.0 | 73.7 |

### 4.3 Router 评估结果

Router 使用 LLM (qwen-27b) 对 83 题 classify complexity，与 expected_complexity (stage 标签) 对比：

| 指标 | 值 |
|------|-----|
| 总体准确率 | 27.7% (23/83) |
| 过度分类率 | 71.1% (59/83) |
| 欠分类率 | 1.2% (1/83) |

| 类别 | 数量 | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| simple | 20 | 1.000 | **0.150** | 0.261 |
| medium | 43 | 0.056 | **0.023** | 0.033 |
| complex | 20 | 0.306 | **0.950** | 0.463 |

Router 预测分布: simple=3, medium=18, complex=62（实际: simple=20, medium=43, complex=20）

### 4.4 回答长度统计

| Mode | 平均回答长度 (chars) |
|------|---------------------|
| naive | 1667 |
| hyper | 3815 |
| adaptive_v1 | 3566 |
| adaptive_fixed_medium | 3470 |
| adaptive_oracle | 3429 |

> naive 回答最短（纯检索拼接），hyper/adaptive 回答更长（有实体关系上下文）。

---

## 5. 分析与结论

### 5.1 核心发现

**发现 1: adaptive 全面落后于 naive 和 hyper**

adaptive_v1 (71.3) 比 naive (85.0) 低 13.7 分，比 hyper (81.2) 低 9.9 分。在所有 5 个维度上均落后，Empowerment 维度差距最大（51.8 vs naive 79.3）。

**发现 2: 上下文截断是主要伤害源**

- adaptive_v1 使用旧版尾截断策略，截断时从尾部切掉 Sources（原文证据），导致 LLM 只有实体/关系 CSV 但没有原文
- adaptive_fixed_medium 使用新版区段感知截断（Sources 55% 优先保留），分数从 71.3 提升到 78.2，提升 6.9 分
- 这验证了"尾截断删除 Sources 是主要伤害"的假设

**发现 3: Router 严重过度分类**

- Router 将 62/83 题判为 complex（实际只有 20 题），medium 几乎不存在（recall 仅 2.3%）
- 过度分类率 71.1%，意味着大部分简单/中等题被强制走 complex 档（top_k=70），引入大量噪声

**发现 4: Oracle 模式反而不如 fixed_medium**

- oracle (72.1) < fixed_medium (78.2)，差 6.1 分
- 原因分析：oracle 在 Stage 3 (complex) 题上使用 top_k=70 + budget=8000，引入过多噪声实体关系，反而干扰回答
- 这说明当前 complex 档参数设计有问题 — 不是检索越多越好

**发现 5: naive 在所有 stage 上都最强**

- naive 在 simple (79.6)、medium (86.1)、complex (88.0) 三个 stage 上均领先
- hyper 在 simple 上 (74.1) 低于 naive，说明超图实体/关系检索对简单问题引入噪声
- hyper 在 complex 上 (86.2) 接近 naive (88.0)，说明超图检索对复杂问题有一定帮助但未能超越

### 5.2 消融对比逻辑

```
hyper (81.2)
  → adaptive_fixed_medium (78.2): -3.0  → adaptive 框架本身有轻微负面影响
  → adaptive_oracle (72.1): -9.1       → 复杂度自适应反而有害（complex 档参数太激进）
  → adaptive_v1 (71.3): -9.9          → Router + 旧截断双重伤害

fixed_medium (78.2) vs oracle (72.1):  oracle 反而更差
  → 复杂度自适应有潜力，但当前参数设计不对
  → complex 档 top_k=70 + budget=8000 引入太多噪声

adaptive_v1 (71.3) vs fixed_medium (78.2):  +6.9
  → 区段感知截断修复有效，但还不够追上 hyper
```

### 5.3 问题诊断

| 问题 | 证据 | 严重程度 |
|------|------|---------|
| Router 过度分类 | 62/83→complex, medium recall=2.3% | 高 — 导致大部分题走错误档位 |
| Context 尾截断删 Sources | v1→fixed_medium +6.9 分 | 高 — 已修复但影响 v1 结果 |
| Complex 档参数过激 | oracle < fixed_medium | 中 — top_k=70 噪声过大 |
| Type-Aware Weighting 无区分 | boost 70/70 题相同 | 中 — 所有结果获得相同 boost |
| Hyper 在简单题上不如 naive | hyper S1=74.1 vs naive S1=79.6 | 低 — 超图检索在简单题上引入噪声 |

### 5.4 下一步方向

1. **Step 5.3-5.4: 重写 Router**
   - 从"问题看起来复杂吗"改为"回答最少需要几层独立证据"
   - 加入硬规则：医学术语多不等于 complex；单个原文片段可回答必须判 simple
   - 目标：medium recall 从 2.3% 提升到 50%+

2. **调整 Complex 档参数**
   - 降低 top_k（70→50 或 40）
   - 降低 total_budget（8000→6000）
   - 复杂题不需要更多检索，需要更精准的检索

3. **Type-Aware Weighting 需要改进**
   - 当前所有结果 boost 相同（无区分）
   - 需要引入差异化：只 boost focus_types 匹配的结果，其他不 boost

4. **提交未提交的代码**
   - Step 6 (mixed 评测集) + Step 5.0-5.2 (output_suffix + 区段预算 + Router 策略) 均未提交 git
   - 7 个文件 modified + 4 个新脚本

---

## 6. 代码状态

### Git 提交历史 (7 commits ahead of origin/dev)

```
6664a88  Step 4.5: Context budget control - post-merge total token fuse
5c2bdbb  Step 4: Type-Aware Weighting - soft boost and rerank by focus types
1d1a630  Step 3: Adaptive Parameter Control - dynamic top_k and token budgets
a4d2d29  Step 2: Query Router - LLM-based query classification
491398c  feat: add adaptive mode shell (degenerates to hyper) (Step 1)
d7c44c5  fix: entity line Entity CSV truncated by max_token_for_entity_context (Step 0)
90d2ffa  feat: chunk=1000 baseline rebuild + fair evaluation pipeline
```

### 未提交的修改 (Step 5.0-5.2 + Step 6)

**Modified files (6):**
- `evaluate/evaluate_by_scoring.py` — 新增 --output-suffix 参数
- `hyperrag/base.py` — QueryParam 新增 router_policy, forced_complexity
- `hyperrag/context_budget.py` — 重写为区段感知截断 (Sources 55%/Rels 30%/Ents 15%)
- `hyperrag/query_modes.py` — adaptive_query 支持 router_policy; hyper_query 增加日志
- `reproduce/Step_2_extract_question.py` — 新增 --stage/--max-cnt/--seed/--output-prefix
- `reproduce/Step_3_response_question.py` — 新增 --router-policy/--forced-complexity/--output-suffix; oracle 模式逐题注入

**New files (4):**
- `scripts/build_mixed_questions.py` — 合并多 stage 问题集
- `scripts/analyze_mixed_results.py` — 按 stage 分层分析
- `scripts/evaluate_router_on_mixed.py` — Router 准确率评估
- `scripts/test_step5_context_budget.py` — 区段预算单测 (5 tests passed)

### 关键文件路径

| 文件 | 作用 |
|------|------|
| `hyperrag/query_router.py` | LLM Router 分类逻辑 |
| `hyperrag/adaptive_params.py` | 三档参数映射 (simple/medium/complex) |
| `hyperrag/type_aware_weighting.py` | 类型感知软加权 |
| `hyperrag/context_budget.py` | 区段感知预算截断 |
| `hyperrag/query_modes.py` | adaptive_query/hyper_query/naive_query 入口 |
| `hyperrag/base.py` | QueryParam 数据类定义 |
| `hyperrag/query_context.py` | 实体/关系/原文检索与合并 |
| `reproduce/Step_3_response_question.py` | 批量生成回答 |
| `evaluate/evaluate_by_scoring.py` | 五维打分评测 |

---

## 7. 数据库统计

- **数据库名**: neurology_chunk1000
- **超图节点**: 20198 vertices
- **超边**: 21109 hyperedges
- **文本块**: 864 chunks (chunk_size=2400)
- **chunk_size=1000 重分后**: 2317 chunks
- **embedding**: 4096 维 (qwen-8b-embed)
- **LLM**: qwen-27b-int4 (max-model-len=24576, 禁用 thinking)

---

## 8. 评分维度说明

五维评分由 LLM (qwen-27b) 生成，每题每维 0-100 分：

| 维度 | 含义 |
|------|------|
| Comprehensiveness | 回答是否全面覆盖问题所问的所有要点 |
| Diversity | 回答是否提供了多样化的信息视角 |
| Empowerment | 回答是否让读者获得可操作的理解 |
| Logical | 回答的逻辑是否自洽合理 |
| Readability | 回答的可读性和表达清晰度 |

评分文件格式: JSON 数组，每元素为 JSON 字符串（需二次 parse），包含每个维度的 Score + Explanation + Level。
