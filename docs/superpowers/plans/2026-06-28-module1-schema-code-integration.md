# Module 1 Schema Code Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `neurology_schema_v1` into the current refactored Hyper-RAG indexing pipeline so extracted entities, hyperedges, vector records, and query contexts carry stable domain type fields.

**Architecture:** `hyperrag/operate.py` is now a compatibility facade, so implementation must target the refactored modules directly. Prompt changes belong in `hyperrag/prompt.py`; extraction parsing belongs in `hyperrag/extraction.py`; graph merging and persistence belong in `hyperrag/graph_upsert.py`; vector-db payload construction belongs in `hyperrag/indexing.py`; VDB metadata configuration belongs in `hyperrag/hyperrag.py`; query context display belongs in `hyperrag/query_context.py`.

**Tech Stack:** Python async code, HyperRAG local JSON/NanoVectorDB/HypergraphDB storage, prompt templates in `hyperrag/prompt.py`, schema docs in `docs/schema/neurology_schema.yaml`.

---

## Scope

This is still Module 1. It does not implement Query Router, intent classification, type-aware retrieval filtering, adaptive diffusion, or quality-aware reranking.

Target chain:

```text
neurology_schema_v1 -> prompt format -> LLM output parser -> hypergraph storage -> entity/relation VDB metadata and content -> query context visibility
```

---

## Refactored File Map

- `hyperrag/operate.py`: compatibility facade only. Do not add new module-one logic here.
- `hyperrag/prompt.py`: extraction prompt text, default type lists, and examples.
- `hyperrag/extraction.py`: parsing of `Entity`, `Low-order Hyperedge`, and `High-order Hyperedge` records.
- `hyperrag/graph_upsert.py`: merging extracted nodes/edges and writing vertices/hyperedges to HypergraphDB.
- `hyperrag/indexing.py`: indexing orchestration and VDB upsert payload construction.
- `hyperrag/hyperrag.py`: VDB instances and `meta_fields`.
- `hyperrag/query_context.py`: CSV context construction for final LLM prompts.

---

## Acceptance Criteria

- `prompt.py` asks the LLM to emit schema-bound `entity_type` and `edge_type`.
- Low-order hyperedge parser returns `edge_type`.
- High-order hyperedge parser returns `edge_type` and `generalization`.
- Parsers tolerate the old relation format by assigning `edge_type="OTHER"` when the type field is absent.
- HypergraphDB hyperedge data includes `edge_type` and `generalization`.
- Entity VDB records include `entity_type` metadata and type-enhanced embedding content.
- Relationship VDB records include `edge_type` metadata and type-enhanced embedding content.
- `HyperRAG.__post_init__` includes `entity_type` and `edge_type` in vector metadata fields.
- Relationship CSV context includes a `type` column.
- `python -m py_compile` passes for touched modules.
- A small indexing smoke test verifies the new fields exist in generated storage.

---

## Task 1: Prompt Schema Integration

**Files:**

- Modify: `hyperrag/prompt.py`
- Reference: `docs/schema/neurology_schema.yaml`

- [ ] **Step 1: Replace the default entity type list**

Use schema v1 labels:

```python
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "DISEASE",
    "SYMPTOM",
    "SIGN",
    "DRUG",
    "TREATMENT",
    "EXAMINATION",
    "ANATOMICAL_STRUCTURE",
    "PHYSIOLOGICAL_FUNCTION",
    "PATHOLOGICAL_MECHANISM",
    "GENE",
    "PROTEIN",
    "PATHWAY",
    "RISK_FACTOR",
    "DIAGNOSTIC_CRITERION",
    "OTHER",
]
```

- [ ] **Step 2: Add relation type lists**

Add near the entity type list:

```python
PROMPTS["DEFAULT_RELATION_TYPES"] = [
    "CAUSES",
    "ASSOCIATED_WITH",
    "INDICATES",
    "DIAGNOSES",
    "TREATS",
    "PREVENTS",
    "COMPLICATES",
    "LOCATED_IN",
    "AFFECTS",
    "REGULATES",
    "PART_OF",
    "INTERACTS_WITH",
    "MECHANISM_OF",
    "RISK_FACTOR_FOR",
    "DIFFERENTIAL_DIAGNOSIS",
    "CO_OCCURS_WITH",
    "OTHER",
]

PROMPTS["DEFAULT_HIGH_ORDER_RELATION_TYPES"] = [
    "MULTI_FACTOR_MECHANISM",
    "CLINICAL_SYNDROME",
    "DIAGNOSTIC_PATTERN",
    "THERAPEUTIC_STRATEGY",
    "COMORBIDITY_PATTERN",
    "PATHWAY_PROCESS",
    "DIFFERENTIAL_GROUP",
    "OTHER",
]
```

- [ ] **Step 3: Add relation type placeholders to the extraction prompt**

Make `PROMPTS["entity_extraction"]` display:

```text
Relation_types: [{relation_types}]
High_order_relation_types: [{high_order_relation_types}]
```

Also state:

```text
Use only the provided entity and relation type labels. If no label fits, use OTHER. Do not invent new type labels.
```

- [ ] **Step 4: Update hyperedge output formats**

Low-order format:

```text
("Low-order Hyperedge" | entity1 | entity2 | edge_type | description | keywords | strength)
```

High-order format:

```text
("High-order Hyperedge" | e1 | e2 | ... | eN | edge_type | description | generalization | keywords | strength)
```

- [ ] **Step 5: Update examples**

At minimum update `PROMPTS["entity_extraction_examples"][3]`, because `hyperrag/indexing.py` currently selects that example:

```python
example_prompt = PROMPTS["entity_extraction_examples"][3]
```

Preferred: update all examples to the new format so future example selection cannot regress.

- [ ] **Step 6: Verify prompt syntax**

Run:

```powershell
python -m py_compile hyperrag\prompt.py
```

Expected: command exits with code 0.

---

## Task 2: Pass Schema Type Lists Into Prompt Formatting

**Files:**

- Modify: `hyperrag/indexing.py`

- [ ] **Step 1: Extend `context_base`**

In `extract_entities()`, include relation type placeholders:

```python
context_base = dict(
    language=PROMPTS["DEFAULT_LANGUAGE"],
    entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    relation_types=",".join(PROMPTS["DEFAULT_RELATION_TYPES"]),
    high_order_relation_types=",".join(PROMPTS["DEFAULT_HIGH_ORDER_RELATION_TYPES"]),
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    examples=example_str,
)
```

- [ ] **Step 2: Verify indexing syntax**

Run:

```powershell
python -m py_compile hyperrag\indexing.py
```

Expected: command exits with code 0.

---

## Task 3: Parse `edge_type` and High-Order `generalization`

**Files:**

- Modify: `hyperrag/extraction.py`

- [ ] **Step 1: Update low-order parser**

Support both formats:

```text
new: "Low-order Hyperedge" | e1 | e2 | edge_type | description | keywords | strength
old: "Low-order Hyperedge" | e1 | e2 | description | keywords | strength
```

Expected behavior:

```python
if len(record_attributes) >= 7:
    entity_num = len(record_attributes) - 4
    edge_type = clean_str(record_attributes[entity_num].upper())
else:
    entity_num = len(record_attributes) - 3
    edge_type = "OTHER"
```

The returned dict must include:

```python
edge_type=edge_type
```

- [ ] **Step 2: Update high-order parser**

Support both formats:

```text
new: "High-order Hyperedge" | e1 | e2 | ... | eN | edge_type | description | generalization | keywords | strength
old: "High-order Hyperedge" | e1 | e2 | ... | eN | description | generalization | keywords | strength
```

Expected behavior:

```python
if len(record_attributes) >= 8:
    entity_num = len(record_attributes) - 5
    edge_type = clean_str(record_attributes[entity_num].upper())
else:
    entity_num = len(record_attributes) - 4
    edge_type = "OTHER"
```

The returned dict must include:

```python
edge_type=edge_type,
generalization=edge_generalization,
```

- [ ] **Step 3: Verify extraction syntax**

Run:

```powershell
python -m py_compile hyperrag\extraction.py
```

Expected: command exits with code 0.

---

## Task 4: Preserve Type Fields During Hypergraph Upsert

**Files:**

- Modify: `hyperrag/graph_upsert.py`

- [ ] **Step 1: Merge `edge_type` by majority vote**

In `_merge_edges_then_upsert()`, collect existing and incoming edge types:

```python
already_edge_types = []
...
already_edge_types.append(already_edge.get("edge_type", "OTHER"))
...
edge_type = sorted(
    Counter(
        [dp.get("edge_type", "OTHER") for dp in edges_data] + already_edge_types
    ).items(),
    key=lambda x: x[1],
    reverse=True,
)[0][0]
```

- [ ] **Step 2: Merge `generalization` safely**

Collect existing and incoming generalization values:

```python
already_generalization = []
...
already_generalization.append(already_edge.get("generalization", ""))
...
generalization = GRAPH_FIELD_SEP.join(
    sorted(set([dp.get("generalization", "") for dp in edges_data] + already_generalization))
)
```

- [ ] **Step 3: Persist new fields to HypergraphDB**

Update the `upsert_hyperedge()` payload:

```python
dict(
    description=description,
    keywords=filter_keywords,
    source_id=source_id,
    weight=weight,
    edge_type=edge_type,
    generalization=generalization,
)
```

- [ ] **Step 4: Return new fields to indexing**

Update `edge_data`:

```python
edge_data = dict(
    id_set=id_set,
    description=description,
    keywords=filter_keywords,
    edge_type=edge_type,
    generalization=generalization,
)
```

- [ ] **Step 5: Verify graph upsert syntax**

Run:

```powershell
python -m py_compile hyperrag\graph_upsert.py
```

Expected: command exits with code 0.

---

## Task 5: Enrich VDB Payloads

**Files:**

- Modify: `hyperrag/indexing.py`

- [ ] **Step 1: Add `entity_type` metadata and content**

Change entity VDB payload to include:

```python
"content": " | ".join(
    [
        dp.get("entity_type", "OTHER"),
        dp["entity_name"],
        dp.get("description", ""),
    ]
),
"entity_name": dp["entity_name"],
"entity_type": dp.get("entity_type", "OTHER"),
```

- [ ] **Step 2: Add `edge_type` metadata and content**

Change relationship VDB payload to include:

```python
"id_set": dp["id_set"],
"edge_type": dp.get("edge_type", "OTHER"),
"content": " | ".join(
    [
        dp.get("edge_type", "OTHER"),
        str(dp["id_set"]),
        dp.get("keywords", ""),
        dp.get("generalization", ""),
        dp.get("description", ""),
    ]
),
```

- [ ] **Step 3: Verify indexing syntax**

Run:

```powershell
python -m py_compile hyperrag\indexing.py
```

Expected: command exits with code 0.

---

## Task 6: Expose Type Metadata From Vector Stores

**Files:**

- Modify: `hyperrag/hyperrag.py`

- [ ] **Step 1: Add entity type metadata field**

Change:

```python
meta_fields={"entity_name"},
```

to:

```python
meta_fields={"entity_name", "entity_type"},
```

- [ ] **Step 2: Add edge type metadata field**

Change:

```python
meta_fields={"id_set"},
```

to:

```python
meta_fields={"id_set", "edge_type"},
```

- [ ] **Step 3: Verify HyperRAG syntax**

Run:

```powershell
python -m py_compile hyperrag\hyperrag.py
```

Expected: command exits with code 0.

---

## Task 7: Show Relation Types In Query Context

**Files:**

- Modify: `hyperrag/query_context.py`

- [ ] **Step 1: Add `type` column to entity-line relationship CSV**

In `_build_entity_query_context()`, add `type` to the relationship table header and add:

```python
e.get("edge_type", "OTHER")
```

before description in each relationship row.

- [ ] **Step 2: Include `edge_type` in returned hyperedges**

Add:

```python
"edge_type": e.get("edge_type", "OTHER"),
```

to returned hyperedge dicts.

- [ ] **Step 3: Apply the same change to relation-line context**

In `_build_relation_query_context()`, add the same CSV `type` column and returned `edge_type` field.

- [ ] **Step 4: Verify query context syntax**

Run:

```powershell
python -m py_compile hyperrag\query_context.py
```

Expected: command exits with code 0.

---

## Task 8: Full Syntax Verification

**Files:**

- Verify: `hyperrag/prompt.py`
- Verify: `hyperrag/extraction.py`
- Verify: `hyperrag/graph_upsert.py`
- Verify: `hyperrag/indexing.py`
- Verify: `hyperrag/hyperrag.py`
- Verify: `hyperrag/query_context.py`

- [ ] **Step 1: Run compile check**

Run:

```powershell
python -m py_compile hyperrag\prompt.py hyperrag\extraction.py hyperrag\graph_upsert.py hyperrag\indexing.py hyperrag\hyperrag.py hyperrag\query_context.py
```

Expected: command exits with code 0.

- [ ] **Step 2: Confirm `operate.py` remains a facade**

Run:

```powershell
rg -n "_handle_single_relationship_extraction|_merge_edges_then_upsert|extract_entities" hyperrag\operate.py
```

Expected:

```text
Only import/re-export lines should appear. No new implementation should be added to operate.py.
```

---

## Task 9: Small Indexing Smoke Test

**Files:**

- Use existing code paths.
- Use an isolated temporary working directory.

- [ ] **Step 1: Prepare a tiny input**

Use one or two Neurology-like snippets containing:

```text
anti-NMDA receptor antibody
autoimmune encephalitis
ovarian teratoma
resection
spinal fluid analysis
```

- [ ] **Step 2: Build a tiny HyperRAG working directory**

Create `HyperRAG(working_dir=...)` and call `insert()` with the snippets.

Expected: indexing completes without parser exceptions.

- [ ] **Step 3: Inspect generated storage**

Check generated storage for:

```text
vdb_entities.json contains entity_type
vdb_relationships.json contains edge_type
hypergraph hyperedge data contains edge_type and generalization
```

- [ ] **Step 4: Run one query**

Ask:

```text
What is the relationship between anti-NMDA receptor antibody, autoimmune encephalitis, and ovarian teratoma?
```

Expected: the constructed relationship context includes a `type` column.

---

## Task 10: Full Rebuild Decision Point

**Files:**

- Use: `reproduce/Step_1.py`
- Use: `reproduce/Step_3_response_question.py`

- [ ] **Step 1: Do not run full rebuild until smoke passes**

Full rebuild is expensive and changes cache artifacts. Only run it after Task 9 passes.

- [ ] **Step 2: Rebuild the knowledge base**

Back up or isolate the current `caches/neurology` directory before running full `Step_1`.

Run:

```powershell
python reproduce\Step_1.py --data-name neurology
```

Expected: knowledge base rebuild completes.

- [ ] **Step 3: Regenerate answers**

Run:

```powershell
python reproduce\Step_3_response_question.py
```

Expected: responses are generated using the typed index.

---

## Task 11: Module 1 Review Metrics

**Files:**

- Optional create: `docs/schema/module1_integration_review.md`

- [ ] **Step 1: Record typed extraction coverage**

Measure or manually inspect:

```text
entity_type coverage
edge_type coverage
OTHER ratio
missing edge_type count
high-order generalization coverage
```

- [ ] **Step 2: Record whether schema v1 needs adjustment**

If many entities or relations become `OTHER`, update the review note with examples and a recommendation for `neurology_schema_v1.1`.

Expected: Module 1 has enough evidence to decide whether schema v1 is good enough for Module 2.

---

## Non-Goals

- Do not implement query router.
- Do not implement complexity classification.
- Do not implement hard type filtering.
- Do not implement adaptive 2-hop diffusion.
- Do not introduce a new storage backend.
- Do not rewrite the refactored module layout.

---

## Handoff Summary

This plan updates Module 1 for the already-refactored operator layout. The key correction from the older plan is that `operate.py` must remain a facade; actual changes go into `prompt.py`, `extraction.py`, `graph_upsert.py`, `indexing.py`, `hyperrag.py`, and optionally `query_context.py`.
