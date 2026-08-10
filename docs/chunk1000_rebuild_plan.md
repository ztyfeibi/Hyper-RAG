# Formal Experiment Cache Rebuild Plan: chunk_size=1000

> Date: 2026-07-07  
> Goal: keep the current `caches/neurology` 2400-token cache as the engineering baseline, and build a separate formal experiment cache with smaller chunks for fairer RAG evaluation.

## 1. Background

The current neurology cache was built with:

```text
working_dir = caches/neurology
chunk_token_size = 2400
chunk_overlap_token_size = 120
entity_extract_max_gleaning = 0
embedding_dim = 4096
```

This cache is useful as an engineering baseline because it has already been debugged and can run the Phase 1 baseline pipeline.

However, `chunk_token_size=2400` creates a mismatch with query-time context budgets:

- Default `max_token_for_text_unit=1600` cannot even fit one 2400-token chunk.
- Raising the text budget fixes `naive`, but `hyper` has two retrieval lines, entity and relation, so merged context can exceed the qwen-27b 24K context window.
- Large chunks also reduce retrieval precision because one matched chunk can contain many unrelated medical facts.

Therefore, the formal experiment should use a separate cache with smaller chunks.

## 2. Non-Negotiable Constraints

Do not delete, overwrite, or mutate the current cache:

```text
caches/neurology
```

Build all new artifacts under a new working directory:

```text
caches/neurology_chunk1000
```

The new cache should reuse the original neurology contexts instead of rerunning Step_0:

```text
caches/neurology/contexts/neurology_unique_contexts.json
```

## 3. Target Configuration

Recommended formal experiment cache:

```text
working_dir = caches/neurology_chunk1000
source_data_name = neurology
chunk_token_size = 1000
chunk_overlap_token_size = 120
entity_extract_max_gleaning = 0
llm_model_max_async = 4
embedding_func_max_async = 4
embedding_dim = 4096
```

Rationale:

- `1000` gives much finer retrieval granularity than `2400`.
- It is cheaper than `800`, while still allowing several chunks per retrieval line.
- With `hyper`, each line can use a `6000` to `8000` text budget and still usually stay below the 24K context limit after adding entities and relationships.

## 4. Required Script Changes

Modify `reproduce/Step_1.py` to parameterize chunk rebuilds.

Add CLI arguments:

```text
--source-data-name
--chunk-token-size
--chunk-overlap-token-size
```

Default behavior must remain backward compatible:

```text
--data-name defaults to pipeline_defaults.DATA_NAME
--source-data-name defaults to --data-name
--chunk-token-size defaults to 2400
--chunk-overlap-token-size defaults to 120
```

Expected input path logic:

```text
caches/<source_data_name>/contexts/<source_data_name>_unique_contexts.json
```

Expected output working directory:

```text
caches/<data_name>
```

This lets us run:

```bash
python reproduce/Step_1.py --data-name neurology_chunk1000 --source-data-name neurology --chunk-token-size 1000 --chunk-overlap-token-size 120
```

without touching `caches/neurology`.

## 5. Recommended Metadata

After building the new cache, write a metadata file:

```text
caches/neurology_chunk1000/experiment_metadata.json
```

Suggested content:

```json
{
  "cache_name": "neurology_chunk1000",
  "source_data_name": "neurology",
  "source_context_file": "caches/neurology/contexts/neurology_unique_contexts.json",
  "chunk_token_size": 1000,
  "chunk_overlap_token_size": 120,
  "entity_extract_max_gleaning": 0,
  "llm_model": "qwen-27b-int4",
  "embedding_model": "qwen-8b-embed",
  "embedding_dim": 4096,
  "purpose": "formal experiment cache for fairer retrieval evaluation"
}
```

## 6. Smoke Test First

Before full rebuild, run a small cache build:

```bash
python reproduce/Step_1.py --data-name neurology_chunk1000_smoke --source-data-name neurology --chunk-token-size 1000 --chunk-overlap-token-size 120 --limit 50
```

Smoke test acceptance criteria:

- `caches/neurology_chunk1000_smoke/vdb_chunks.json` exists.
- `caches/neurology_chunk1000_smoke/vdb_entities.json` exists.
- `caches/neurology_chunk1000_smoke/vdb_relationships.json` exists.
- `caches/neurology_chunk1000_smoke/hypergraph_chunk_entity_relation.hgdb` exists.
- Hypergraph has more than 0 vertices and more than 0 hyperedges.
- `HyperRAG.log` contains no repeated context-length or empty-extraction failures.

Only proceed to full rebuild after the smoke test passes.

## 7. Full Rebuild Command

Run:

```bash
python reproduce/Step_1.py --data-name neurology_chunk1000 --source-data-name neurology --chunk-token-size 1000 --chunk-overlap-token-size 120
```

Expected artifacts:

```text
caches/neurology_chunk1000/kv_store_full_docs.json
caches/neurology_chunk1000/kv_store_text_chunks.json
caches/neurology_chunk1000/kv_store_llm_response_cache.json
caches/neurology_chunk1000/vdb_chunks.json
caches/neurology_chunk1000/vdb_entities.json
caches/neurology_chunk1000/vdb_relationships.json
caches/neurology_chunk1000/hypergraph_chunk_entity_relation.hgdb
caches/neurology_chunk1000/HyperRAG.log
caches/neurology_chunk1000/experiment_metadata.json
```

## 8. Query-Time Budget Plan

For `chunk_token_size=1000`, start with:

```text
naive max_token_for_text_unit = 12000
hyper max_token_for_text_unit = 6000
hyper-lite max_token_for_text_unit = 6000
```

Reasoning:

- `naive` has one text retrieval path, so 12000 can include roughly 10 to 12 chunks.
- `hyper` has entity and relation retrieval paths, so each line should be smaller.
- 6000 per line can include roughly 5 to 6 chunks and should leave room for entity and relation tables inside qwen-27b's 24K context.

If `hyper` still hits context overflow, reduce to:

```text
hyper max_token_for_text_unit = 4000
```

If `hyper` underuses source evidence and remains safely below context limits, try:

```text
hyper max_token_for_text_unit = 8000
```

## 9. Required Step_3 Follow-Up

`reproduce/Step_3_response_question.py` currently hardcodes text budgets by mode.

For formal experiments, either:

1. Add CLI arguments:

```text
--max-token-for-text-unit
--max-token-for-entity-context
--max-token-for-relation-context
```

or:

2. Add cache-aware defaults:

```text
if data_name contains "chunk1000":
    naive text budget = 12000
    hyper text budget = 6000
else:
    keep current defaults
```

Option 1 is preferred because it makes experiment conditions explicit.

## 10. Baseline Runs on New Cache

After full rebuild, copy or regenerate the question files for the new cache.

If reusing the cleaned neurology questions, ensure these files exist:

```text
caches/neurology_chunk1000/questions/2_stage.json
caches/neurology_chunk1000/questions/2_stage_ref.json
```

Then run:

```bash
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode naive
python reproduce/Step_3_response_question.py --data-name neurology_chunk1000 --mode hyper
```

Expected outputs:

```text
caches/neurology_chunk1000/response/naive_2_stage_result.json
caches/neurology_chunk1000/response/hyper_2_stage_result.json
caches/neurology_chunk1000/response/naive_2_stage_errors.json
caches/neurology_chunk1000/response/hyper_2_stage_errors.json
```

## 11. Evaluation Plan

Run scoring:

```bash
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode naive --question-stage 2
python evaluate/evaluate_by_scoring.py --data-name neurology_chunk1000 --mode hyper --question-stage 2
```

Run pairwise selection:

```bash
python evaluate/evaluate_by_selection.py --data-name neurology_chunk1000 --mode-a naive --mode-b hyper --question-stage 2
```

Do not compare `caches/neurology` and `caches/neurology_chunk1000` as if they only differ by method. They differ by chunking configuration. Use `neurology_chunk1000` as the formal base for later adaptive-method comparisons.

## 12. Risks and Mitigations

### Risk: More chunks increase Step_1 cost

Mitigation:

- Always run the smoke test first.
- Keep `entity_extract_max_gleaning=0`.
- Keep `llm_model_max_async=4`.

### Risk: More fragmented entities

Mitigation:

- Inspect vertex count, hyperedge count, and `OTHER` type ratio after build.
- Compare with the 2400-token cache as a sanity reference, not as a formal method comparison.

### Risk: Context overflow still occurs in `hyper`

Mitigation:

- Start `hyper` text budget at 6000.
- Reduce to 4000 if needed.
- Track context length failures in `response/*_errors.json`.

### Risk: Evaluation result is confounded by chunking

Mitigation:

- Use the same `neurology_chunk1000` cache for `naive`, `hyper`, and future `adaptive`.
- Report chunk settings in experiment metadata and paper tables.

## 13. Recommended Execution Order

1. Parameterize `reproduce/Step_1.py`.
2. Add metadata writing for rebuilt caches.
3. Run `neurology_chunk1000_smoke`.
4. Inspect smoke output and hypergraph counts.
5. Run full `neurology_chunk1000` rebuild.
6. Copy or regenerate cleaned question files.
7. Parameterize `reproduce/Step_3_response_question.py` budgets.
8. Run `naive` and `hyper` baseline on `neurology_chunk1000`.
9. Run evaluation.
10. Use `neurology_chunk1000` as the formal cache for adaptive-method experiments.

