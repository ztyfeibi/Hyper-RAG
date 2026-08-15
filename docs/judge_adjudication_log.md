# Judge 裁决阶段实验日志（repeat 0）

> 生成时间: 2026-08-16
> 范围: LongCat judge r0_s42_5c92f17c → 补充审核 → ai_adjudication_v1/v2 → repeat 0 准入与 Judge 冻结
> 本文档是审计记录，供论文实验章节与后续 repeat 1/2 引用。

---

## 1. 裁决产物版本

| 版本 | 内容 | 状态 |
|------|------|------|
| `ai_adjudication_v1` | 163 条 AI 标注（blind_id 对齐模式），167 条 pending | **已冻结**（哈希见 §2） |
| `verdicts_ai_all_v1.jsonl` | 318 条合并标注 = 151 original + 12 recheck_E + 155 supplementary_D | 已冻结（manifest 在 blind_review_supplementary/） |
| `ai_adjudication_v2` | 318 条标注 + coverage 硬失败，480 = pass 180 / fail 291 / pending 9 | 当前有效版本 |

## 2. v1 覆盖事故与恢复（2026-08-16 02:30）

- **事故**：旧模式测试 fixture 在真实 `ai_adjudication_v1` 目录上执行 `cmd_apply`，导致 v1 于 02:30 被重写。
- **同分布确认**：重写版与原版逐条对比，480 条 route_success/answer_correctness/source/coverage_hit **零差异**；
  仅新增 `review_source` 字段与 manifest/summary 时间戳。即"v1 已被同分布重建"。
- **处置**：已从 `backups/judge_r0_s42_5c92f17c_pre_supplementary_20260815/ai_adjudication_v1` 恢复原版四件套；
  重写版留存于 `backups/ai_adjudication_v1_regen_20260816_0230/`。
- **原版（当前生效）SHA-256 前 16 位**：
  - `final_verdicts.jsonl`: `1ec1afcfe317c77f`
  - `final_summary.json`: `642302b30bc2f782`
  - `pending_supplementary.jsonl`: `6fc8fa81ccdeda4a`
  - `manifest.json`: `44f3056c7f475ac4`
- **根因修复**（commit `6b21221`）：覆盖守卫（目录存在且无 `--overwrite` 即报错）扩展到所有模式；
  旧模式测试改写 `ai_adjudication_v1_testregen` 专用目录；新增守卫回归测试。

## 3. 已知实验局限（论文 Limitations 素材）

1. **原 163 条 AI 标注来源 unknown**：`verdicts_ai_annotated.jsonl` 的审核模型元数据不可证明
   （review_metadata 缺失），标注质量依赖 set_E 复核间接验证（12 条复核中 3 条 verdict 翻转、4 条 fatality 翻转）。
2. **补充审核模型与回答模型同源**：set_D 155 条与 set_E 12 条补充审核使用本地
   qwen-27b-int4（local-vllm），与生成候选回答的是同一模型家族，存在自我偏好（self-preference）风险。
   LongCat judge 层（外部模型）部分对冲，但 AI adjudication 层独立性有限。
3. **rule v3 悬置语义**：unresolved claims（unverifiable）不强转 pass/fail，导致 9 条 pending
   （4 条 uncertain + 5 条 verdict=pass 但含 unverifiable claims，全部来自 set_E）。
   这是规则设计选择而非数据缺失。

## 4. repeat 0 最终分布（ai_adjudication_v2）

- 全局: pass 180 / fail 291 / pending 9（v1 时代 167 条 pending 全部关闭）
- P_gold: 79 pass / 1 fail / 0 pending → 79 题进入 repeat 1/2 候选，1 题排除
- Coverage 冻结: P1=48 / P2=6 / P3=24 / P4=34 / P_gold=80（复算与冻结统计一致）
- pending 9 条全部进入 `excluded_records.jsonl`，exclusion_reason 均为 `unresolved_claims_pending`

## 5. 后续（repeat 1/2 前置）

1. `repeat0_question_eligibility.jsonl`：P_gold fail 1 题排除；9 条 pending 涉及题目标 router 标签不可确定
2. `judge_freeze_manifest.json`：锁定 judge 模型/Prompt/rule v3/coverage v3.1/题集标注哈希
3. 先跑 P_gold repeat 1/2，再 P0-P4 repeat 1/2
