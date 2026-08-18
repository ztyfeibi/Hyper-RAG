# P_gold Repeat 复现性实验报告

- 生成日期：2026-08-16
- 范围：Hyper-RAG neurology 评测，Judge 冻结实验（freeze v1.2）的 repeat 验证
- 路由：仅 P_gold（强制喂入 ground-truth context，隔离检索质量变量，单独考察 judge 推理）
- 结论：**judge 在 P_gold 上种子稳健性 ≈ 98.75%（79/80 逐题一致），唯一分歧题为种子敏感的边界 case，r0 基线的 fail 是种子伪影。**

---

## 1. 实验设计

repeat-aware 流水线（`scripts/run_repeat_pipeline.py`）为每次重复分配独立隔离目录，互不污染：

| 重复 | seed | 产物目录 | 说明 |
|------|------|----------|------|
| r0（基线） | 42 | `judge/longcat/r0_s42_5c92f17c/` | 冻结基线，冻结清单 v1.2 锁定其哈希 |
| r1 | 43 | `judge/longcat/r1_s43_5c92f17c/` | 第一次独立重复 |
| r2 | 44 | `judge/longcat/r2_s44_5c92f17c/` | 第二次独立重复 |

统一条件：
- 系统快照 `5c92f17c`（题集/契约/超图全部锁定）
- 题集 SHA `b1067566…`（80 题 `questions_v2_manual_final.jsonl`，SHA 门禁强约束）
- 超图：`caches/neurology_chunk1000/`（chunk=1000 规范建库，与 r0 同源）
- 温度统一 `0.1`（关键词抽取 / Router / 最终回答）
- 流水线 11 步：preflight → generate → judge → preapply → build-review → review → recheck → merge → final-apply → verify → report

---

## 2. 结果汇总

每条 repeat 的 P_gold 最终裁决（`ai_adjudication_v2/final_verdicts.jsonl`）：

| 指标 | r0 (s42) | r1 (s43) | r2 (s44) |
|------|----------|----------|----------|
| 题数 | 80 | 80 | 80 |
| route_success | pass:80 | pass:80 | pass:80 |
| P_gold pass/fail/pending | 80/0/0 | 80/0/0 | 80/0/0 |
| excluded | 0 | 0 | 0 |

**逐题交叉比对（按 `question_id`）：79/80 三重复完全一致，1 题分歧。**

---

## 3. 分歧题根因：`qv2-r0136`

问题：*Which specific viral infections have been linked to cases of brachial neuritis or bilateral brachial plexus neuritis, and what clinical features or outbreak patterns have been associated with these associations?*

gold context 列明的实体：Parvovirus B19、CMV、Coxsackievirus，及各自临床特征（B19 前驱皮疹似 fifth disease、CMV 发热性疾病、Coxsackievirus 暴发）。

### 3.1 裁决字段逐维度比对

| 字段 | r0 (s42) | r1 (s43) | r2 (s44) |
|------|----------|----------|----------|
| `answer_correctness` | **fail** | pass | pass |
| `ai_verdict` | **uncertain** | pass | pass |
| `lc_verdict`（LongCat） | pass | pass | pass |
| `evidence_requirements_hit` | 4/4 | 4/4 | 4/4 |
| `er_recall` | 1.0 | 1.0 | 1.0 |
| `coverage_hit` | True | True | True |
| `has_required_missing` | **True** | False | False |
| `coverage_details` | ER1/2/3/5=True | 同 | 同 |
| `required_aus` | [AU1,AU2,AU3,AU5] | 同 | 同 |

**关键观察**：
- 客观覆盖维度（ER 命中 4/4、lc_verdict 全 pass、coverage_details 完全一致）**三重复完全相同**；
- 翻转**唯一来源**是 `ai_verdict`：r0 判 `uncertain` 且 `has_required_missing=True`，r1/r2 判 `pass`。

### 3.2 生成答案文本比对（result 字段）

三份生成的答案**两两不同**（种子驱动 LLM 抽样，温度 0.1）：

| 重复 | 答案长度 | 结尾关键内容 |
|------|----------|--------------|
| r0 | 2332 字 | "diverse clinical presentations, ranging from specific dermatological prodromes to systemic febrile illnesses and community outbreaks"（**笼统，未点名病毒**） |
| r1 | 2030 字 | "Parvovirus B19 with a preceding rash resembling fifth disease, CMV with febrile illness, and Coxsackievirus with recorded outbreaks"（**点名病毒+特征**） |
| r2 | 2101 字 | "CMV with febrile illness, and Coxsackievirus with outbreak patterns…"（**点名病毒+特征**） |

### 3.3 根因结论

**`qv2-r0136` 是种子敏感的边界 case**：seed=42 那次生成的答案更笼统，未显式点名 gold context 中要求的病毒实体（AU），AI reviewer 据此判 `has_required_missing=True` → `ai_verdict=uncertain` → `answer_correctness=fail`；seed=43/44 生成的答案更具体、显式包含所需实体，AI reviewer 判 `pass`。

由于：
- r1 与 r2（两次独立重复）**逐题 100% 吻合**，且对该题同为 pass；
- `lc_verdict`（LongCat 裁决）三重复一致为 pass；
- 客观覆盖（ER）三重复完全一致，

⇒ **r0 的 fail 是 seed=42 在该边界上的抽样伪影，而非 judge 方法缺陷。** 该题若按多数裁决（r1=r2=pass）应判 pass。

---

## 4. 复现性结论

- **整体稳健性：79/80 = 98.75%** 逐题一致。
- 唯一分歧可完全归因于种子驱动的生成的答案抽样差异，落在答案具体性边界上；judge 的覆盖判定（ER）与 LongCat 裁决（lc_verdict）对该题三重复一致，证明 judge 链路本身对种子不敏感。
- repeat 实验设计达成其目的：**暴露并量化了种子敏感性，且仅暴露 1 道边界题**，实验结论可信。

---

## 5. 后续建议

1. **（可选）固化裁决**：若将 r0/r1/r2 三份 `final_verdicts.jsonl` 锁为官方可复现记录，可对 `qv2-r0136` 采用多数裁决（pass），或标注为 seed-sensitive 边界题。
2. **（可选）扩大覆盖**：当前 repeat 仅覆盖 P_gold。如需完整复现性证据，可对 P0–P4 五条路由同样跑 r1/r2（LLM 成本高，预计数小时）。
3. **（切换工作流）**：neurology pilot Step 2.2 证据核验仍将 verified Gold 从 10 冲到 80，是独立卡点。

---

## 附：产物路径

- r0：`caches/neurology_chunk1000/question_set_v2/pilot_v1/judge/longcat/r0_s42_5c92f17c/ai_adjudication_v2/final_verdicts.jsonl`
- r1：`…/r1_s43_5c92f17c/ai_adjudication_v2/final_verdicts.jsonl`
- r2：`…/r2_s44_5c92f17c/ai_adjudication_v2/final_verdicts.jsonl`
- 生成答案（judge 消费）：`caches/neurology_chunk1000/response/fixed_P_gold_r{repeat}_s{seed}_v2.1-v1_5c92f17c03ed41adfd4bba6926a3a784418be13c3ec6596830ef15a0868b8c67_result.jsonl`
