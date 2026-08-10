# Hyper-RAG 项目任务交接 Prompt

> 把这个文档完整交给下一个 AI，它会理解当前状态并继续执行。

---

## 一、项目概览

**项目名**：Query-Adaptive Domain-Aware Hyper-RAG  
**基础代码**：原版 Hyper-RAG 论文实验代码  
**领域**：医学（neurology，神经病学）  
**核心创新思想**：让系统先理解 Query 类型，再动态决定检索范围、扩散深度和超边权重

**当前阶段**：Phase 1 Baseline 复现——先跑通原版 naive/hyper 的公平评测基准，再进入 Phase 2 创新模块开发。

**当前状态**：chunk_size=2400 的 baseline 已跑完（评测结果：naive=85.6 > hyper=81.3，不公平）。发现 chunk=2400 对 hyper 不公平（hyper 预算太小装不下足够上下文），正在用 chunk_size=1000 + gleaning=1 重建知识库。Smoke test 已通过，下一步是 full rebuild。

---

## 二、技术环境（关键！）

### 模型服务
| 组件 | 地址 | 模型 | 关键约束 |
|------|------|------|---------|
| LLM | `http://10.65.1.110:8002/v1` | `qwen-27b-int4` | vLLM 部署，**max-model-len=24576** |
| Embedding | `http://10.65.1.110:8001/v1` | `qwen-8b-embed` | dim=**4096**，**不支持 matryoshka**，**max=12788** |

### LLM 调用的必须约束
```python
# 必须禁用 thinking，否则推理链会污染输出且极慢
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
# ❌ 不能用 extra_body={"enable_thinking": False}，这是 vLLM 的特殊要求
```

### Python 环境
```bash
# 建库（Step_1）必须在 conda hyperrag 环境运行
# 直接调用 conda Python，不要用 WorkBuddy 自带的隔离 Python
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_1.py ...

# 查询（Step_3）和评测（evaluate/）也要用 conda hyperrag 环境
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_3_response_question.py ...
```

### 配置文件
所有配置在 `D:/projectes/lyw-rag/my_config.py`。

---

## 三、文件结构关键信息

```
D:/projectes/lyw-rag/
├── hyperrag/              # 核心库（已从 operate.py 拆分为 7 个模块）
│   ├── prompt.py          # 15 类医学实体 + 关系 + 高阶关系类型定义
│   ├── llm.py             # LLM 调用封装
│   ├── indexing.py        # 实体抽取、chunk 切分
│   ├── query_modes.py     # naive/hyper/hyper-lite 三种查询模式
│   ├── query_context.py   # 检索上下文构建、combine_contexts 合并去重
│   ├── base.py            # HyperRAG 主类
│   └── utils.py           # truncate_list_by_token_size 等工具
├── reproduce/             # 复现流水线脚本
│   ├── Step_1.py          # 建库（文本切 chunk → 实体/关系抽取 → 向量化）
│   ├── Step_2_extract_question.py  # 生成评测问题
│   └── Step_3_response_question.py # 生成答案（naive/hyper）
├── evaluate/              # 评测脚本
│   ├── evaluate_by_scoring.py    # 五维打分（Comprehensiveness/Diversity/Empowerment/Logical/Readability）
│   └── evaluate_by_selection.py  # pairwise 对比
├── caches/
│   ├── neurology/         # chunk=2400 的知识库（保留不动！已备份到 backups/）
│   ├── neurology_chunk1000_smoke/  # Smoke test 产物（测试通过）
│   └── neurology_chunk1000/       # 目标：新 chunk=1000 全量知识库（待建）
├── backups/
│   └── chunk2400_baseline_20260706/  # 917MB 完整备份
├── scripts/
│   └── analyze_token_budget.py  # Token 预算分析工具
├── docs/
│   ├── experiment_plan.md        # 8 组消融实验计划（E0-E7）
│   ├── chunk1000_rebuild_plan.md # chunk=1000 重建计划
│   └── handoff_prompt.md        # 本文件
└── my_config.py           # API 配置
```

### 重要代码逻辑
- **Step_1 建库**：用 `f.read()` 整体读 JSON 作为单 doc（不是 `json.load`），chunk_id = md5(content)
- **Hyper 双线检索**：entity 线和 relation 线各独立检索 text_units，`combine_contexts` 合并后**去重但不截断**——这是原版架构问题，合并后可能超 context
- **truncate_list_by_token_size**：已修复边界 bug（第一个 chunk 就超预算时返回空列表），改用 `max(1,i)`

---

## 四、当前进度与状态

### ✅ 已完成
1. **Phase 0**：7 个 bug 修复（EMB_DIM、enable_thinking、asyncio 竞态等）
2. **chunk=2400 baseline**：建库成功（20198 vertices, 21109 hyperedges, 864 chunks, 25min）
3. **评测**：43 条问题对齐，五维打分结果：

| 维度 | Naive | Hyper | 差距 |
|------|------:|------:|-----:|
| Comprehensiveness | 88.2 | 75.4 | +12.8 |
| Diversity | 78.8 | 83.3 | -4.5 |
| Empowerment | 76.2 | 64.2 | +12.0 |
| Logical | 92.5 | 90.4 | +2.1 |
| Readability | 92.6 | 93.3 | -0.7 |
| **Overall** | **85.6** | **81.3** | **+4.3** |

4. **完整备份**：`backups/chunk2400_baseline_20260706/`（917MB，全部 17 个文件字节级验证通过）
5. **Step_1.py 参数化**：加了 `--source-data-name`、`--chunk-token-size`、`--chunk-overlap-token-size`、`--gleaning` 四个 CLI 参数
6. **Step_3.py 参数化**：加了 `--max-token-for-text-unit`，hyper 默认预算 4000→8000
7. **Smoke test 通过**：

```
Data: 50 contexts → 9 chunks
Token 控制：8/9 ≤1000，1 个 1001（可忽略）
Entity: 567 entities (590 vertices)
Relationship: 439 hyperedges
Gleaning=1: 18 次 LLM 调用全部成功，无 context 溢出！
耗时: 12.5 分钟
```

### 🔲 待执行（按顺序）

**Step A — Full Rebuild（下一步！）**
```bash
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_1.py \
  --data-name neurology_chunk1000 \
  --source-data-name neurology \
  --chunk-token-size 1000 \
  --chunk-overlap-token-size 150 \
  --gleaning 1
```
预期：~2237 chunks，~4474 次 LLM 调用，2-3 小时。产物落在 `caches/neurology_chunk1000/`。

**Step B — 复制评测问题**
```bash
cp caches/neurology/questions/2_stage.json caches/neurology_chunk1000/questions/
cp caches/neurology/questions/2_stage_ref.json caches/neurology_chunk1000/questions/
```

**Step C — 生成 baseline 回答**
```bash
# Naive (budget=12000, 单线检索)
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode naive

# Hyper (budget=8000, 双线检索，合并后约 12000 tokens 安全)
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode hyper
```

**Step D — 五维打分评测**
```bash
D:/Tools/Conda/envs/hyperrag/python.exe evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode naive --question-stage 2
D:/Tools/Conda/envs/hyperrag/python.exe evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode hyper --question-stage 2
```

**Step E — Pairwise Selection 评测**
```bash
# 双向评测消除位置偏见
D:/Tools/Conda/envs/hyperrag/python.exe evaluate/evaluate_by_selection.py --data-name neurology_chunk1000 --mode-a hyper --mode-b naive --question-stage 2
D:/Tools/Conda/envs/hyperrag/python.exe evaluate/evaluate_by_selection.py --data-name neurology_chunk1000 --mode-a naive --mode-b hyper --question-stage 2
```

---

## 五、Token Budget 精确数据（24576 context window）

### Step_1 实体抽取
- 模板 overhead：2,093 tokens（只用 Example 4 医学示例）
- chunk=1000：总输入 3,093，输出空间 21,483 → 非常充裕
- gleaning=1：history 累计 ~9,173，剩余 15,403 → **完全安全**
- gleaning=2：history 累计 ~12,223，剩余 12,353 → 也安全

### Step_3 Naive 查询
- 模板 overhead：330 tokens（naive_rag_response 130 + query 200）
- budget=12000：总输入 12,330，剩余 8,246 → **可扩展到 16000**

### Step_3 Hyper 查询
- 模板 overhead：418 tokens（rag_response 132 + rag_define 86 + query 200）
- 固定开销：entities CSV(300) + relations CSV(1600) = 1,900
- 预算 X=8000：合并 ~11,200 + 固定 1,900 + overhead 418 ≈ 13,518，剩余 7,058 → **安全**
- 预算 X=10000：合并 ~14,000 + ... ≈ 16,318，剩余 4,258 → 也可行但偏紧

### Embedding 模型 (max=12788)
- chunk=1000 仅用 7.8%，远非瓶颈

---

## 六、关键 Gotchas（已经踩过的坑，不要再踩）

1. **不要改 `caches/neurology/`**——那是 chunk=2400 的 baseline，已备份到 `backups/chunk2400_baseline_20260706/`
2. **建库用 conda hyperrag 环境**，不要用 WorkBuddy 自带的隔离 Python
3. **LLM 调用必须 `enable_thinking=False`**，否则推理链会污染输出
4. **Embedding 不能传 `dimensions` 参数**——qwen-8b-embed 不支持 matryoshka 降维
5. **Hyper 双线检索合并后只去重不截断**——这是原版架构固有问题。所以 hyper 的 `max_token_for_text_unit` 不能设太大（8000 安全，10000 偏紧）
6. **评测脚本输出文件名含 mode 后缀**——避免 naive 和 hyper 互相覆盖
7. **chunk=1000 的 gleaning=1 可行**——chunk=2400 时 gleaning 会因 context 溢出全部失败，chunk=1000 后安全
8. **建库缓存无法复用**——chunk 内容全变了，embedding 和 LLM cache 都得重新算。只复用源文件 `caches/neurology/contexts/neurology_unique_contexts.json`

---

## 七、预期结果

chunk=1000 重建后，naive vs hyper 的对比预期会公平很多：

- naive：12000 tokens ≈ 12 chunks，信息充分
- hyper：8000/线 ≈ 8 chunks/线，双线合并去重后 ≈ 11,200 tokens ≈ 11 chunks，与 naive 接近
- 预期：Comprehensiveness/Empowerment 差距从 +12 缩小到 +3 以内，Diversity 优势保持或扩大

---

## 八、下一步行动摘要

**现在就跑这个命令**：
```bash
cd D:/projectes/lyw-rag
D:/Tools/Conda/envs/hyperrag/python.exe reproduce/Step_1.py \
  --data-name neurology_chunk1000 \
  --source-data-name neurology \
  --chunk-token-size 1000 \
  --chunk-overlap-token-size 150 \
  --gleaning 1
```

等建库完成后，依次执行 Step B → C → D → E（见上面 §四）。
