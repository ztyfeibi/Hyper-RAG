# Step 1：实验基础设施

## 目标

让 P_gold、P0–P4 能从冻结契约（`docs/question_set_v2_contract.yaml`, contract v1, 提交 721e66d）中独立、稳定运行，并产生符合 `retrieval-trace-v2` 的完整 Trace。本阶段只跑 fixture 和少量 smoke test，不生成正式问题集。

## 范围（7 个子步骤）

### 1.1 LongCat 与 Tokenizer 预检

新增 `scripts/probe_longcat_api.py`、`scripts/calibrate_tokenizers.py`。

- LongCat 配置走环境变量：`LONGCAT_BASE_URL` / `LONGCAT_API_KEY` / `LONGCAT_MODEL=meituan-longcat/LongCat-2.0`，不提交密钥。
- probe 依次验证：基础调用、temperature=0、严格 JSON 输出、8K/16K 上下文输入、usage/finish_reason/延迟可获得、超时与重试行为。
- Tokenizer 校准：Qwen tokenizer 为正式预算依据，tiktoken 仅作近似；对 prompt、实体表、关系表、source 文本抽样比较；输出 `caches/calibration/tokenizer_calibration.json`。
- 正式运行找不到 Qwen tokenizer 时直接失败，禁止静默回退到 tiktoken。

### 1.2 运行时契约与 Schema Owner

新增 `hyperrag/experiment_contract.py`、`hyperrag/experiment_schema.py`、`tests/test_experiment_contract_runtime.py`。

- 加载契约 YAML，返回类型化 P0–P4 配置，校验契约/Schema 版本。
- 集中提供 question/trace 验证；从 `scripts/validate_question_set_contract.py` 提取共享校验逻辑。
- runner/Judge/Router/分析脚本禁止复制 P0–P4 数值。
- 验收：运行时解析出的所有 top-k 和预算与契约逐项相等。

### 1.3 拆分 QueryParam 检索参数

修改 `hyperrag/base.py`，新增字段：
`chunk_vdb_top_k` / `entity_vdb_top_k` / `relation_vdb_top_k`、
`entity_description_cap` / `relation_description_cap` / `source_text_cap` / `final_context_hard_cap`、
`route_id` / `repeat_id` / `repeat_seed` / `system_snapshot_id` / `contract_version`。

- 新字段为 None 时才回退旧 `top_k` / token 字段；adaptive/hyper 旧实验行为不变。
- 固定 P0–P4 executor 必须使用新字段。
- 同步修改 `query_context.py`、`query_modes.py`、`context_budget.py`。
- 重点验证 P2：`relation_vdb_top_k=0` → 不调用 Relation VDB，但实体邻接扩展仍可生成 Relationships。

### 1.4 统一回答 Prompt 与固定 Executor

新增 `hyperrag/fixed_route_executor.py`、`tests/test_fixed_route_executor.py`。

- `prompt.py` 建立唯一正式回答模板；P0–P4 + P_gold 共享相同 prompt hash。
- 路径语义：P0 空 context / P1 source-only / P2 hyper-lite / P3 standard hyper / P4 expanded hyper / P_gold 仅 Gold evidence spans。
- 接口：`execute_fixed_route(question, route_id, contract, repeat_id, repeat_seed, system_snapshot_id)`。
- P_gold 先用测试 fixture；正式 Gold 在 Step 2 接入。

### 1.5 正式重复运行控制

修改 `reproduce/Step_3_response_question.py`，新增 CLI：
`--fixed-route P0|P1|P2|P3|P4|P_gold`、`--contract`、`--repeat-id`、`--seed`、`--disable-llm-cache`、`--system-snapshot`、`--max-questions`、`--validate-trace`。

- fixed-route 默认：temperature=0.1、enable_llm_cache=false、router disabled、type-aware weighting disabled。
- 每次 repeat 独立输出文件，文件名含 route、repeat、契约版本、snapshot。

### 1.6 完整 Trace 接入

新增 `hyperrag/trace_collector.py`、`tests/test_trace_v2_runtime.py`。

- 保存：VDB 原始/归一化 score 与排名、embedding hash、实体/关系/chunk 候选、seed→entity→hyperedge→source lineage、pre-rerank/post-rerank/post-expansion/post-assembly/final-context IDs、各阶段 section tokens、截断前后 token 与丢失证据 ID、answer/usage/finish_reason/重试/超时/延迟、11 项成本字段。
- 正式运行写盘前调用 `validate_trace_record()`；Trace 不符合 Schema 时该题标记失败，禁止静默保存残缺记录。

### 1.7 初始 System Snapshot

新增 `hyperrag/system_snapshot.py`、`scripts/create_system_snapshot.py`、`tests/test_system_snapshot.py`。

- 快照内容：contract/schema versions、Git commit SHA、P0–P4 配置、prompt hashes、dataset/chunk/entity/relation/hypergraph hashes、回答/Embedding/Judge 模型标识、tokenizer 版本与校准报告 hash、temperature/cache/seed 规则、retriever 与 score normalization 版本。
- 输出 `caches/neurology_chunk1000/snapshots/<snapshot_id>.json`。
- 相同系统状态重复生成必须得到相同 snapshot_id。

## 最终验收

```powershell
python scripts/validate_question_set_contract.py
python -m pytest tests/test_question_set_contract.py tests/test_experiment_contract_runtime.py tests/test_fixed_route_executor.py tests/test_trace_v2_runtime.py tests/test_system_snapshot.py -q
python scripts/probe_longcat_api.py
python scripts/calibrate_tokenizers.py
python scripts/create_system_snapshot.py --data-name neurology_chunk1000
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --question-file mixed_stage_stable_v1 --fixed-route P3 --repeat-id 1 --seed 42 --disable-llm-cache --save-trace --validate-trace --max-questions 1
```

完成标准：五条固定路径参数正确、prompt hash 一致、正式运行不命中缓存、Qwen hard cap 成立、Trace 通过 Schema、snapshot 可复现、现有 Hyper/Adaptive 回归测试不退化。
