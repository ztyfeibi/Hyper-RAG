# Step 2.1：构建候选证据池（evidence-first）

> 状态：v2 修正完成，待最终验收（2026-08-01）。本阶段**只建证据池，不生成问题、不写 Gold answer**。

## 目标

从 `neurology_chunk1000` 中确定性抽取 80 个 evidence-first 候选，严格配额：

| 结构 | 数量 |
|---|---|
| single_fact（单 chunk/单实体事实） | 20 |
| single_high_arity（单个高元超边） | 15 |
| multi_edge_chain（两个及以上连续超边） | 20 |
| multi_branch（多分支聚合或比较） | 15 |
| similar_subgraph_disambiguation（相似子图消歧） | 10 |

每条候选另生成 3 倍配额备选（共 240），供 Step 2.2 选题。

## 固定输入

- `data_name = neurology_chunk1000`
- `system_snapshot_id = 3518fe4a630e4459372409b55b20f0d6d753ac6313637b4617d3855f3e0dadfa`（Step 1 定稿 `d6f63f6`）
- `seed = 42`
- `candidate_count = 80`
- 加载：`kv_store_text_chunks.json` 与 `hypergraph_chunk_entity_relation.hgdb`

## 可执行的 motif 判定（仅写入 `intended_structure`，不提前写 `verified_structure`）

- **single_fact**：候选资格与排序只依赖合格 chunk（每个合格 chunk 即一个候选单元），实体锚点可选（仅出自该 chunk 的单源实体，有则附上、无也不排除）。Step 2.2 提取原子事实时再补充实体。
- **single_high_arity**：仅需一条超边，`arity ≥ 3`，描述非空，实体集合完整，全部 `source_id` 可解析且指向合格 chunk。
- **multi_edge_chain**：至少两条超边通过共享实体连续相连（恰好共享 1 个实体，形成 A→X→B 二跳路径），且两条边合计覆盖 ≥ 2 个不同有效证据来源。
- **multi_branch**：同一核心实体连接 ≥ `MIN_BRANCHES` 条超边，分支之间除核心实体外两两不相交，且分支来自 ≥ `MIN_BRANCHES` 个不同证据来源（聚合/比较）。
- **similar_subgraph_disambiguation**：两条同 `edge_type` 超边共享 ≥ 2 个实体、各自有独有实体，Jaccard ∈ [`JACCARD_LOW`, `JACCARD_HIGH`)，且来自不同证据来源——结构相似，须靠区分性证据定位目标。

## 每条候选保存的字段

`candidate_id` / `entry_type` / `intended_structure` / `seed_chunk_ids` / `seed_entity_ids` / `hyperedges` / `source_chunk_ids` / `source_text_hashes` / `topology_metrics` / `sampling_seed` / `system_snapshot_id` / `status` / `rejection_reasons`（外加 `evidence_signature`、`pool_role`）。

## 硬过滤

- 所有 `source_id` 必须能在 chunk KV 中解析；chunk 内容不能为空。
- 剔除明显乱码、严重截断、重复内容。
- 超边必须有描述、实体集合和来源。
- 同一 evidence cluster 只保留一个主候选（贪心选择 + `used_chunks`/`used_clusters` 拦截 `evidence_cluster_collision`）。
- 所有集合**先排序再生成 ID**，保证重复运行稳定。

## 确定性约束

- 排序/选择键用 `sha256(f"{seed}|{signature}")`，**不**用内置 `hash()`（受 `PYTHONHASHSEED` 影响）或 `random`（随版本变化）。
- 验证：`PYTHONHASHSEED=0 / 999 / 默认` 三遍重跑 → `evidence_candidates.jsonl` 哈希均为 `f2fa1cb6d2058db0`（v2 确定性成立）。
- ⚠️ 旧记录 `29d44d62251f6f84`（v0）、`26b7bd478e43bfa9`（v1）均为陈旧值（脚本当时未提交、无 git 历史），不作为基准。v2 已改 `ec-v2-*` 前缀，旧 ID 不复用。

## 验收结果（全部 PASS）

- 配额精确 20/15/20/15/10；主候选 80 条 ID 唯一、格式 `ec-v2-NNNN`、与备选无重叠。
- 排序不变量 0 违例：`seed_entity_ids`、每条超边 `entity_ids`、超边列表（`edge_key`）、`source_chunk_ids` 均有序。
- 无任何 `question/answer/gold/difficulty/verified_structure/options` 泄漏；`status` 全为 `pending_evidence_verification`。
- 源 chunk 全部可解析且哈希匹配；超边全部按 `edge_key` 可回查。
- `sampling_report.json` 记录 raw/filtered 计数、各结构数量、cluster 碰撞、拒绝原因分布、语言分布。

## 交付物

- `scripts/build_pilot_evidence_pool.py`（`SCRIPT_VERSION=evidence-pool-v2`）
- `tests/test_evidence_pool.py`（20 测试，纯函数 + 产物验收）
- `caches/neurology_chunk1000/question_set_v2/pilot_v1/`
  - `evidence_candidates.jsonl`（80）、`evidence_candidates_reserve.jsonl`（240）
  - `sampling_manifest.json`、`sampling_report.json`

## 语言分布（客观标注，非难度标签）

en 80 / zh 0 / mixed 0 —— 80 条源证据全部为英文（P2 修正后基于源 chunk 原文统计）。

## 下一步

Step 2.2：基于本证据池生成实际问题并撰写 Gold answer。

## v2 修正记录（2026-08-01，验收反馈后）

针对验收反馈的三项契约问题：

- **[P1] source-first 真正落地**：`build_single_fact` 原从图实体出发、无锚点 chunk 被排除（544/2277 = 23.9%）。现改为**每个合格 chunk 即一个候选单元**，实体锚点可选（有则附、无也不排除），签名仅 `single_fact|{chunk_id}`。验证：single_fact 池规模 = 2277 = 合格 chunk 数。
- **[P1] 产物语义变更须升版**：因候选内容及 ID→证据映射已变，升级 `SCRIPT_VERSION=evidence-pool-v2`、`CANDIDATE_ID_PREFIX=ec-v2`（旧 `ec-v1-*` 不复用），确定性哈希从 `a9e87…` 变为 `f2fa1cb6d2058db0`。
- **[P2] token 单位澄清**：源 token 字段由 `source_token_estimate` 改名为 `source_tiktoken_tokens`，并在 manifest 显式记录 `source_token_unit="tiktoken"`——与 P4 source cap（Qwen token）不是同一单位；Step 2.2 对精确 evidence spans 须改用真实 Qwen tokenizer 计数。

回归测试新增 `test_source_token_unit_is_tiktoken`，并将 `test_single_fact_is_source_first` 改为断言"池规模 == 合格 chunk 数 + 锚点可选"，共 20 测试全过。
