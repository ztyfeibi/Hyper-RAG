#!/usr/bin/env python
"""Validate the Question Set v2.1 experiment contract.

This script is the machine-readable gate for the frozen contract defined in
``docs/question_set_v2_contract.yaml`` and the two JSON schemas under
``docs/schema/``. It is intentionally dependency-free (stdlib + PyYAML only) so
the acceptance command ``python scripts/validate_question_set_contract.py``
cannot fail on a missing third-party package.

What it checks
--------------
1. The contract YAML loads and carries all six frozen version strings.
2. P0..P4 policy ladder is complete: exactly five stages, and the
   candidate / pre-merge / post-merge arrays are all length 5 and aligned.
3. The post-merge section allocation ratios sum to ~1.0.
4. The forbidden ambiguous label ``minimum_successful_route`` does not appear
   anywhere in the contract.
5. Both JSON schemas parse and contain the expected top-level keys.
6. A built-in EXAMPLE question item validates against the question schema.
7. A built-in EXAMPLE trace record validates against the trace schema.
8. A built-in set of example per-route runs can be used to derive
   ``successful_route_set``, ``minimum_sufficient_route_by_policy`` and
   ``pareto_optimal_routes`` (and the derived values match the expected ones).

Exit code 0 = all checks passed; 1 = at least one check failed.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("FATAL: PyYAML is required (pip install pyyaml)\n")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CONTRACT_YAML = REPO_ROOT / "docs" / "question_set_v2_contract.yaml"
SCHEMA_DIR = REPO_ROOT / "docs" / "schema"
QUESTION_SCHEMA = SCHEMA_DIR / "question_item_v2.schema.json"
TRACE_SCHEMA = SCHEMA_DIR / "retrieval_trace_v2.schema.json"

REQUIRED_VERSIONS = [
    "contract_version",
    "trace_schema_version",
    "success_rule_version",
    "metric_version",
    "cost_schema_version",
]

# --------------------------------------------------------------------------- #
# Shared Schema Owner (hyperrag/experiment_schema.py), loaded by file path so
# importing it never triggers hyperrag/__init__ heavy dependencies.
# Single source of truth since Step 1.2 -- do NOT re-implement these here.
# --------------------------------------------------------------------------- #
import importlib.util


def _load_schema_owner():
    path = REPO_ROOT / "hyperrag" / "experiment_schema.py"
    spec = importlib.util.spec_from_file_location(
        "_hyperrag_experiment_schema", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_S = _load_schema_owner()

# Re-exported names; tests and downstream scripts keep using these symbols.
SchemaValidationError = _S.SchemaValidationError
validate_schema = _S.validate_schema
validate_question_item = _S.validate_question_item
check_question_cross_field = _S.check_question_cross_field
validate_trace_record = _S.validate_trace_record
check_trace_cross_field = _S.check_trace_cross_field
derive_labels = _S.derive_labels
COST_FIELDS = _S.COST_FIELDS
POLICY_STAGES = _S.POLICY_STAGES
_stable_success = _S._stable_success
_is_cost_complete = _S._is_cost_complete
_dominates = _S._dominates
_cost_vector = _S._cost_vector
_median = _S._median
_median_cost_vector = _S._median_cost_vector
_aggregate_routes = _S._aggregate_routes


# --------------------------------------------------------------------------- #
# Built-in examples
# --------------------------------------------------------------------------- #
EXAMPLE_QUESTION = {
    "question_id": "qv2-example-0001",
    "language": "en",
    "question": ("What are the two primary afferent pathways that convey "
                 "nociceptive signals from the periphery to the thalamus?"),
    "gold_answer": ("The spinothalamic tract and the dorsal column-medial "
                    "lemniscus pathway convey nociceptive afferent signals to "
                    "thalamic nuclei."),
    "answer_units": [
        {"unit_id": "AU1", "claim": "The spinothalamic tract is a primary "
         "nociceptive afferent pathway to the thalamus.", "required": True},
        {"unit_id": "AU2", "claim": "The dorsal column-medial lemniscus pathway "
         "also conveys some nociceptive afferent input.", "required": True},
    ],
    "evidence_requirements": [
        {"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
         "alternative_chunk_ids": ["chunk-abc", "chunk-def"]},
        {"requirement_id": "ER2", "answer_unit_ids": ["AU2"],
         "alternative_chunk_ids": ["chunk-ghi"]},
    ],
    "evidence_spans": [
        {"unit_id": "AU1", "evidence_spans": [
            {"chunk_id": "chunk-abc", "start_char": 418, "end_char": 672,
             "quote": "..."}]},
        {"unit_id": "AU2", "evidence_spans": [
            {"chunk_id": "chunk-ghi", "start_char": 100, "end_char": 300,
             "quote": "..."}]},
    ],
    "required_subgraph": {
        "required_vertices": ["ent-nociceptor", "ent-thalamus"],
        "required_hyperedges": [
            {"hyperedge_id": "he-1",
             "entity_set": ["ent-nociceptor", "ent-thalamus"],
             "source_chunk_ids": ["chunk-abc"]}],
        "answer_unit_links": {"AU1": ["he-1"]},
        "topology_metrics": {"edge_count": 1, "max_arity": 2,
                             "path_depth": 1, "branch_width": 1},
    },
    "intended_structure": "two-or-more-connected-hyperedges",
    "verified_structure": "two-or-more-connected-hyperedges",
}

EXAMPLE_TRACE = {
    "question_id": "qv2-example-0001",
    "run_id": "run-001",
    "route_id": "P3",
    "seed": 42,
    "system_snapshot_id": "snap-2026-07-29",
    "answer_text": ("The spinothalamic tract and the dorsal column-medial "
                    "lemniscus pathway convey nociceptive afferent signals."),
    "finish_reason": "stop",
    "retries": 0,
    "timeouts": 0,
    "stage_evidence_loss": {
        "pre_rerank": 0, "post_graph_expansion": 0, "post_assembly": 0,
    },
    "keyword_extraction": {
        "entity_keywords": ["nociception", "thalamus"],
        "relation_keywords": ["afferent pathway"],
        "keyword_result_hash": "sha256:abc",
    },
    "retrievers": [
        {
            "retriever_id": "entity_vdb",
            "retriever_version": "nano-vectordb-1",
            "query_embedding_hash": "sha256:ent",
            "score_semantics": "cosine similarity (higher = more similar)",
            "score_direction": "higher_is_better",
            "score_normalization_method": "min-max",
            "score_normalization_version": "v1",
            "candidates": [
                {"candidate_id": "ent-1", "candidate_type": "entity", "rank": 1,
                 "raw_retrieval_score": 0.91, "normalized_score": 0.91,
                 "reranker_score": None,
                 "source_provenance_ids": ["chunk-abc"]}
            ],
        },
        {
            "retriever_id": "relation_vdb",
            "retriever_version": "nano-vectordb-1",
            "query_embedding_hash": "sha256:rel",
            "score_semantics": "cosine similarity (higher = more similar)",
            "score_direction": "higher_is_better",
            "score_normalization_method": "min-max",
            "score_normalization_version": "v1",
            "candidates": [],
        },
    ],
    "stages": {
        "pre_rerank_ids": ["ent-1"],
        "post_rerank_ids": ["ent-1"],
        "post_graph_expansion_ids": ["he-1"],
        "post_assembly_source_ids": ["chunk-abc"],
        "final_context_source_ids": ["chunk-abc"],
        "entity_ids": ["ent-1"],
        "hyperedge_ids": ["he-1"],
        "tokens_before_truncation": 3200,
        "tokens_after_truncation": 3000,
        "truncated_tokens": 200,
        "evidence_group_coverage": {"ER1": True, "ER2": False},
    },
    "cost": {
        "input_tokens": 3200, "output_tokens": 300,
        "retrieval_latency_ms": 120, "end_to_end_latency_ms": 1500,
        "llm_calls": 2, "embedding_calls": 1, "reranker_calls": 0,
        "graph_expansion_calls": 1, "retrieved_candidate_count": 10,
        "gpu_seconds": 0.5, "api_cost": 0.01,
    },
    "judge": {
        "verdict": "pass", "critical_error": False,
        "evidence_sufficient": True, "raw_json": {},
        "human_review_status": "pending",
    },
}

# Example per-route RUNS used to demonstrate label derivation.
# Each question/route carries REPEAT runs (success bool per repeat) so the
# frozen operational_stable_success_v1 rule can be applied: 3/3 -> success,
# 0/3|1/3 -> fail, 2/3 -> needs 2 more (>=4/5), >=5 runs -> >=4/5 to pass.
# Costs are a per-route template (repeated across repeats); aggregation takes
# the per-component median, and a component ABSENT from every run is kept as
# None (never defaulted to 0).
_COST_TEMPLATES = {
    "P0": {"input_tokens": 0, "output_tokens": 300, "retrieval_latency_ms": 0,
           "end_to_end_latency_ms": 800, "llm_calls": 1, "embedding_calls": 0,
           "reranker_calls": 0, "graph_expansion_calls": 0, "retrieved_candidate_count": 0,
           "gpu_seconds": 0.1, "api_cost": 0.005},
    "P1": {"input_tokens": 4000, "output_tokens": 300, "retrieval_latency_ms": 1200,
           "end_to_end_latency_ms": 1500, "llm_calls": 2, "embedding_calls": 1,
           "reranker_calls": 0, "graph_expansion_calls": 0, "retrieved_candidate_count": 20,
           "gpu_seconds": 0.3, "api_cost": 0.01},
    "P2": {"input_tokens": 7000, "output_tokens": 300, "retrieval_latency_ms": 2200,
           "end_to_end_latency_ms": 2500, "llm_calls": 2, "embedding_calls": 1,
           "reranker_calls": 0, "graph_expansion_calls": 1, "retrieved_candidate_count": 40,
           "gpu_seconds": 0.5, "api_cost": 0.02},
    "P3": {"input_tokens": 5000, "output_tokens": 300, "retrieval_latency_ms": 1500,
           "end_to_end_latency_ms": 1800, "llm_calls": 2, "embedding_calls": 1,
           "reranker_calls": 0, "graph_expansion_calls": 1, "retrieved_candidate_count": 30,
           "gpu_seconds": 0.35, "api_cost": 0.015},
    "P4": {"input_tokens": 4000, "output_tokens": 300, "retrieval_latency_ms": 1100,
           "end_to_end_latency_ms": 1400, "llm_calls": 2, "embedding_calls": 1,
           "reranker_calls": 0, "graph_expansion_calls": 1, "retrieved_candidate_count": 25,
           "gpu_seconds": 0.3, "api_cost": 0.012},
    # Q2-specific variants to create a genuine Pareto tie (token vs latency).
    "P1_q2": {"input_tokens": 2000, "output_tokens": 300, "retrieval_latency_ms": 2800,
               "end_to_end_latency_ms": 3000, "llm_calls": 2, "embedding_calls": 1,
               "reranker_calls": 0, "graph_expansion_calls": 0, "retrieved_candidate_count": 15,
               "gpu_seconds": 0.6, "api_cost": 0.02},
    "P4_q2": {"input_tokens": 6000, "output_tokens": 300, "retrieval_latency_ms": 700,
               "end_to_end_latency_ms": 900, "llm_calls": 2, "embedding_calls": 1,
               "reranker_calls": 0, "graph_expansion_calls": 1, "retrieved_candidate_count": 25,
               "gpu_seconds": 0.2, "api_cost": 0.014},
}


def _mk_runs(qid, route, success_pattern, cost_key):
    """Expand a success pattern (list of bools) into repeat run dicts."""
    cost = dict(_COST_TEMPLATES[cost_key])
    return [{"question_id": qid, "route_id": route, "success": s, "cost": dict(cost)}
            for s in success_pattern]


EXAMPLE_RUNS = []
# Q1: P2/P3/P4 succeed (3/3); P4 cheapest on every component -> Pareto = {P4}.
EXAMPLE_RUNS += _mk_runs("qv2-ex1", "P0", [False, False, False], "P0")
EXAMPLE_RUNS += _mk_runs("qv2-ex1", "P1", [False, False, False], "P1")
EXAMPLE_RUNS += _mk_runs("qv2-ex1", "P2", [True, True, True], "P2")
EXAMPLE_RUNS += _mk_runs("qv2-ex1", "P3", [True, True, True], "P3")
EXAMPLE_RUNS += _mk_runs("qv2-ex1", "P4", [True, True, True], "P4")
# Q2: P1 (low token / high latency) and P4 (high token / low latency) -> tie.
EXAMPLE_RUNS += _mk_runs("qv2-ex2", "P0", [False, False, False], "P0")
EXAMPLE_RUNS += _mk_runs("qv2-ex2", "P1", [True, True, True], "P1_q2")
EXAMPLE_RUNS += _mk_runs("qv2-ex2", "P2", [False, False, False], "P2")
EXAMPLE_RUNS += _mk_runs("qv2-ex2", "P3", [False, False, False], "P3")
EXAMPLE_RUNS += _mk_runs("qv2-ex2", "P4", [True, True, True], "P4_q2")
# Q3: all five succeed; P0 (no retrieval, all-zero cost) dominates everyone.
EXAMPLE_RUNS += _mk_runs("qv2-ex3", "P0", [True, True, True], "P0")
EXAMPLE_RUNS += _mk_runs("qv2-ex3", "P1", [True, True, True], "P1")
EXAMPLE_RUNS += _mk_runs("qv2-ex3", "P2", [True, True, True], "P2")
EXAMPLE_RUNS += _mk_runs("qv2-ex3", "P3", [True, True, True], "P3")
EXAMPLE_RUNS += _mk_runs("qv2-ex3", "P4", [True, True, True], "P4")
# Q4: 5 repeats per route exercising the 2/3 -> 4/5 rule. P3 cheaper -> Pareto={P3}.
EXAMPLE_RUNS += _mk_runs("qv2-ex4", "P2", [True, True, False, True, True], "P2")
EXAMPLE_RUNS += _mk_runs("qv2-ex4", "P3", [True, True, True, True, False], "P3")
# Q5: 2/3 at exactly 3 trials is NOT a stopping point -> route undecided -> excluded.
EXAMPLE_RUNS += _mk_runs("qv2-ex5", "P2", [True, True, False], "P2")
# Q6: 10 locked-test boundary runs; 9/10 >= 8/10 -> success (demonstrates the
# 10-trial stopping point; 7/10 would be fail, covered by unit tests).
EXAMPLE_RUNS += _mk_runs("qv2-ex6", "P3",
                         [True, True, True, True, True,
                          True, True, True, False, True], "P3")

EXPECTED_DERIVED = {
    "qv2-ex1": {
        "successful_route_set": ["P2", "P3", "P4"],
        "minimum_sufficient_route_by_policy": "P2",
        "pareto_optimal_routes": ["P4"],
    },
    "qv2-ex2": {
        "successful_route_set": ["P1", "P4"],
        "minimum_sufficient_route_by_policy": "P1",
        "pareto_optimal_routes": ["P1", "P4"],
    },
    "qv2-ex3": {
        "successful_route_set": ["P0", "P1", "P2", "P3", "P4"],
        "minimum_sufficient_route_by_policy": "P0",
        "pareto_optimal_routes": ["P0"],
    },
    "qv2-ex4": {
        "successful_route_set": ["P2", "P3"],
        "minimum_sufficient_route_by_policy": "P2",
        "pareto_optimal_routes": ["P3"],
    },
    "qv2-ex5": {
        "successful_route_set": [],
        "minimum_sufficient_route_by_policy": None,
        "pareto_optimal_routes": [],
    },
    "qv2-ex6": {
        "successful_route_set": ["P3"],
        "minimum_sufficient_route_by_policy": "P3",
        "pareto_optimal_routes": ["P3"],
    },
}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_versions(contract, errors):
    for key in REQUIRED_VERSIONS:
        if key not in contract:
            errors.append(f"contract missing version key: {key}")
        elif not isinstance(contract[key], str) or not contract[key]:
            errors.append(f"contract version key {key} is not a non-empty string")


def check_policy_ladder(contract, errors):
    ladder = contract.get("policy_ladder")
    if not isinstance(ladder, dict):
        errors.append("contract.policy_ladder missing or not a mapping")
        return
    if ladder.get("stages") != POLICY_STAGES:
        errors.append(f"policy_ladder.stages must be exactly {POLICY_STAGES}")
    n = len(POLICY_STAGES)
    for section in ("candidate_budget", "pre_merge_caps",
                    "post_merge_hard_control"):
        sec = ladder.get(section)
        if not isinstance(sec, dict):
            errors.append(f"policy_ladder.{section} missing")
            continue
        for field, vec in sec.items():
            if field == "section_allocation_ratios":
                continue
            if not isinstance(vec, list) or len(vec) != n:
                errors.append(
                    f"policy_ladder.{section}.{field} must be a length-{n} list")
                continue
            # Every budget value must be a non-negative number.
            for i, val in enumerate(vec):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(
                        f"policy_ladder.{section}.{field}[{i}] must be a number")
                elif val < 0:
                    errors.append(
                        f"policy_ladder.{section}.{field}[{i}] = {val} is negative")
    # section allocation ratios must sum to ~1.0
    ratios = (ladder.get("post_merge_hard_control", {})
              .get("section_allocation_ratios"))
    if isinstance(ratios, dict):
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(
                f"section_allocation_ratios sum={total}, expected 1.0")


def check_forbidden_terms(contract, errors):
    FORBIDDEN = "minimum_successful_route"

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "minimum_successful_route":
                    return True
                if isinstance(v, str) and "minimum_successful_route" in v:
                    return True
                if _walk(v):
                    return True
        elif isinstance(node, list):
            for item in node:
                if _walk(item):
                    return True
        return False

    if _walk(contract):
        errors.append("forbidden ambiguous term 'minimum_successful_route' "
                      "found in contract")


def check_schemas(question_schema, trace_schema, errors):
    for name, schema in (("question_item_v2", question_schema),
                         ("retrieval_trace_v2", trace_schema)):
        if "$schema" not in schema:
            errors.append(f"{name} schema missing $schema")
        if "type" not in schema:
            errors.append(f"{name} schema missing top-level type")
        if "$id" not in schema:
            errors.append(f"{name} schema missing $id")


def check_example_validates(question_schema, trace_schema, errors):
    try:
        validate_question_item(EXAMPLE_QUESTION, question_schema, "$")
    except SchemaValidationError as e:
        errors.append(f"EXAMPLE_QUESTION failed: {e}")
    try:
        validate_trace_record(EXAMPLE_TRACE, trace_schema, "$")
    except SchemaValidationError as e:
        errors.append(f"EXAMPLE_TRACE failed: {e}")


def check_label_derivation(errors):
    derived = derive_labels(EXAMPLE_RUNS)
    for qid, expected in EXPECTED_DERIVED.items():
        got = derived.get(qid)
        if got != expected:
            errors.append(
                f"label derivation mismatch for {qid}: "
                f"expected {expected}, got {got}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# Expected structure quota (frozen 2026-07-29; mirrors STRUCTURE_QUOTA in code).
EXPECTED_STRUCTURE_QUOTA = {
    "single_fact": 20,
    "single_high_arity": 15,
    "multi_edge_chain": 20,
    "multi_branch": 15,
    "similar_subgraph_disambiguation": 10,
}
VALID_STRUCTURES = set(EXPECTED_STRUCTURE_QUOTA)
_QID_RE = re.compile(r"^qv2-[A-Za-z0-9_-]+$")
_AUID_RE = re.compile(r"^AU[0-9]+$")


def check_question_file(question_file: Path, errors):
    """Validate an actual question set file (questions_v2.jsonl) against the
    question_item_v2 schema + aggregate contract rules. Self-contained: does
    not require jsonschema (it is not installed in the hyperrag env)."""
    if not question_file.exists():
        errors.append(f"question file not found: {question_file}")
        return
    raw_lines = question_file.read_text(encoding="utf-8").splitlines()
    items = []
    for ln, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception as e:  # noqa: BLE001
            errors.append(f"line {ln}: invalid JSON ({e})")
            return

    if len(items) != 80:
        errors.append(f"question count = {len(items)} (expected 80)")

    ids = [it.get("question_id") for it in items]
    if len(set(ids)) != len(ids):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        errors.append(f"duplicate question_id: {dup}")
    texts = [it.get("question") for it in items]
    if len(set(texts)) != len(texts):
        errors.append("duplicate question text detected")

    required_top = ["question_id", "language", "question", "gold_answer",
                    "answer_units", "evidence_requirements", "evidence_spans"]
    for it in items:
        qid = str(it.get("question_id", "<missing>"))
        for fld in required_top:
            if fld not in it:
                errors.append(f"{qid}: missing required field '{fld}'")
        if not _QID_RE.match(str(it.get("question_id", ""))):
            errors.append(f"{qid}: question_id does not match ^qv2-...$")
        if it.get("language") != "en":
            errors.append(f"{qid}: language != 'en' ({it.get('language')})")
        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"{qid}: question empty")
        ga = it.get("gold_answer")
        if not isinstance(ga, str) or not ga.strip():
            errors.append(f"{qid}: gold_answer empty")
        aus = it.get("answer_units")
        if not isinstance(aus, list) or not (1 <= len(aus) <= 5):
            errors.append(f"{qid}: answer_units count "
                          f"{len(aus) if isinstance(aus, list) else 'NA'} not in 1..5")
        else:
            for au in aus:
                if not isinstance(au, dict):
                    errors.append(f"{qid}: answer_unit is not an object")
                    continue
                if "unit_id" not in au or "claim" not in au or "required" not in au:
                    errors.append(f"{qid}: answer_unit missing unit_id/claim/required")
                if not _AUID_RE.match(str(au.get("unit_id", ""))):
                    errors.append(f"{qid}: bad unit_id {au.get('unit_id')}")
        es = it.get("evidence_spans")
        if not isinstance(es, list) or len(es) < 1:
            errors.append(f"{qid}: evidence_spans missing/empty")
        else:
            for grp in es:
                if not isinstance(grp, dict):
                    errors.append(f"{qid}: evidence_spans group is not an object")
                    continue
                if not _AUID_RE.match(str(grp.get("unit_id", ""))):
                    errors.append(f"{qid}: evidence_spans bad unit_id {grp.get('unit_id')}")
                spans = grp.get("evidence_spans")
                if not isinstance(spans, list) or len(spans) < 1:
                    errors.append(f"{qid}: {grp.get('unit_id')} has no spans")
                    continue
                for sp in spans:
                    for k in ("chunk_id", "start_char", "end_char", "quote"):
                        if k not in sp:
                            errors.append(f"{qid}: span missing '{k}'")
        if it.get("intended_structure") not in VALID_STRUCTURES:
            errors.append(f"{qid}: intended_structure "
                          f"'{it.get('intended_structure')}' not in allowed set")

    struct_counts = Counter(it.get("intended_structure") for it in items)
    for struct, exp in EXPECTED_STRUCTURE_QUOTA.items():
        got = struct_counts.get(struct, 0)
        if got != exp:
            errors.append(f"structure quota {struct}: got {got}, expected {exp}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(CONTRACT_YAML),
                        help="path to the contract YAML")
    parser.add_argument("--schema-dir", default=str(SCHEMA_DIR),
                        help="directory containing the JSON schemas")
    parser.add_argument("--question-file", default=None,
                        help="path to questions_v2.jsonl to validate against the "
                             "question_item_v2 schema + aggregate contract rules")
    args = parser.parse_args(argv)

    contract_path = Path(args.contract)
    schema_dir = Path(args.schema_dir)
    question_schema_path = schema_dir / "question_item_v2.schema.json"
    trace_schema_path = schema_dir / "retrieval_trace_v2.schema.json"

    errors = []

    if args.question_file:
        check_question_file(Path(args.question_file), errors)

    # Load contract
    if not contract_path.exists():
        errors.append(f"contract file not found: {contract_path}")
    else:
        try:
            contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"contract YAML failed to parse: {e}")
            contract = None

    # Load schemas
    question_schema = trace_schema = None
    if not question_schema_path.exists():
        errors.append(f"question schema not found: {question_schema_path}")
    else:
        try:
            question_schema = json.loads(
                question_schema_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"question schema failed to parse: {e}")
    if not trace_schema_path.exists():
        errors.append(f"trace schema not found: {trace_schema_path}")
    else:
        try:
            trace_schema = json.loads(
                trace_schema_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"trace schema failed to parse: {e}")

    # Run checks
    if contract is not None:
        check_versions(contract, errors)
        check_policy_ladder(contract, errors)
        check_forbidden_terms(contract, errors)
    if question_schema is not None and trace_schema is not None:
        check_schemas(question_schema, trace_schema, errors)
        check_example_validates(question_schema, trace_schema, errors)
    check_label_derivation(errors)

    # Report
    print("=" * 68)
    print("Question Set v2.1 Experiment Contract — Validation")
    print("=" * 68)
    if errors:
        print(f"\nFAILED with {len(errors)} error(s):\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print("\nRESULT: FAIL")
        return 1

    print("\nAll checks passed:\n")
    print("  [1] contract version strings        OK")
    print("  [2] P0..P4 policy ladder (len-5)    OK")
    print("  [3] section allocation ratios sum   OK")
    print("  [4] forbidden term absent           OK")
    print("  [5] both JSON schemas parse         OK")
    print("  [6] EXAMPLE question/trace validate OK")
    print("  [7] label derivation matches        OK")
    if args.question_file:
        print(f"  [8] question file {Path(args.question_file).name:<24} OK")
    if contract is not None:
        print(f"\ncontract_version = {contract.get('contract_version')}")
        print(f"trace_schema_version = {contract.get('trace_schema_version')}")
    derived = derive_labels(EXAMPLE_RUNS)
    print("\nExample derived labels:")
    for qid, labels in derived.items():
        print(f"  {qid}: min_sufficient={labels['minimum_sufficient_route_by_policy']}, "
              f"pareto={labels['pareto_optimal_routes']}")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
