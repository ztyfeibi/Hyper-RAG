#!/usr/bin/env python
"""Tests for the Question Set v2.1 experiment contract.

Run with:  python -m pytest tests/test_question_set_contract.py -q
Fallback (no pytest):  python tests/test_question_set_contract.py

The fallback runner works WITHOUT pytest: a small local ``raises`` helper
replaces ``pytest.raises`` so schema-violation tests still run.
"""

import json
import os
import sys
from pathlib import Path

import yaml

# Make the validator module importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_question_set_contract as v  # noqa: E402

CONTRACT_YAML = REPO_ROOT / "docs" / "question_set_v2_contract.yaml"
QUESTION_SCHEMA = REPO_ROOT / "docs" / "schema" / "question_item_v2.schema.json"
TRACE_SCHEMA = REPO_ROOT / "docs" / "schema" / "retrieval_trace_v2.schema.json"


# --------------------------------------------------------------------------- #
# pytest-free assertion helper
# --------------------------------------------------------------------------- #
try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, etype, exc, tb):
        if etype is None:
            raise AssertionError(f"expected {self.exc.__name__} not raised")
        return issubclass(etype, self.exc)


def raises(exc):
    """pytest.raises when pytest is available, else a local equivalent."""
    if pytest is not None:
        return pytest.raises(exc)
    return _Raises(exc)


def _load_contract():
    return yaml.safe_load(CONTRACT_YAML.read_text(encoding="utf-8"))


def _load_question_schema():
    return json.loads(QUESTION_SCHEMA.read_text(encoding="utf-8"))


def _load_trace_schema():
    return json.loads(TRACE_SCHEMA.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Contract-level tests
# --------------------------------------------------------------------------- #
def test_contract_version_strings_present():
    contract = _load_contract()
    for key in v.REQUIRED_VERSIONS:
        assert key in contract, f"missing version key {key}"
        assert isinstance(contract[key], str) and contract[key]


def test_contract_references_schemas():
    contract = _load_contract()
    owner = contract.get("schema_owner", {})
    assert "question_item" in owner
    assert "retrieval_trace" in owner
    assert QUESTION_SCHEMA.exists()
    assert TRACE_SCHEMA.exists()


def test_policy_ladder_complete_and_length5():
    contract = _load_contract()
    ladder = contract["policy_ladder"]
    assert ladder["stages"] == v.POLICY_STAGES
    n = len(v.POLICY_STAGES)
    for section in ("candidate_budget", "pre_merge_caps", "post_merge_hard_control"):
        sec = ladder[section]
        for field, vec in sec.items():
            if field == "section_allocation_ratios":
                continue
            assert isinstance(vec, list) and len(vec) == n, (
                f"{section}.{field} must be length-{n}")


def test_section_allocation_ratios_sum_to_one():
    contract = _load_contract()
    ratios = contract["policy_ladder"]["post_merge_hard_control"][
        "section_allocation_ratios"]
    total = sum(ratios.values())
    assert abs(total - 1.0) < 1e-6, f"ratios sum={total}"


def test_policy_ladder_rejects_negative_budget():
    import copy
    contract = _load_contract()
    bad = copy.deepcopy(contract)
    bad["policy_ladder"]["pre_merge_caps"]["entity_description_cap"][2] = -5
    errors = []
    v.check_policy_ladder(bad, errors)
    assert any("negative" in e for e in errors), errors


def test_forbidden_term_absent_from_contract():
    contract = _load_contract()
    errors = []
    v.check_forbidden_terms(contract, errors)
    assert errors == [], f"forbidden term check failed: {errors}"


def test_label_semantics_required_outputs():
    contract = _load_contract()
    outs = set(contract["label_semantics"]["required_outputs"])
    expected = {
        "successful_route_set",
        "minimum_sufficient_route_by_policy",
        "route_cost_vectors",
        "pareto_optimal_routes",
        "minimum_cost_route_by_profile",
    }
    assert expected.issubset(outs)


def test_cost_profiles_defined():
    contract = _load_contract()
    profiles = set(contract["cost_vector"]["cost_profiles"])
    assert {"token_first", "latency_first", "local_gpu", "api_billing"}.issubset(profiles)


def test_cost_fields_cover_contract_raw_fields():
    contract = _load_contract()
    raw = set(contract["cost_vector"]["raw_fields"])
    covered = set(v.COST_FIELDS)
    assert raw == covered, (
        f"validator COST_FIELDS must equal contract raw_fields; "
        f"missing {raw - covered}, extra {covered - raw}")


# --------------------------------------------------------------------------- #
# Schema-level tests
# --------------------------------------------------------------------------- #
def test_schemas_parse_and_have_required_keys():
    q = _load_question_schema()
    t = _load_trace_schema()
    for name, schema in (("question", q), ("trace", t)):
        assert "$schema" in schema
        assert "$id" in schema
        assert "type" in schema


def test_example_question_validates():
    q = _load_question_schema()
    v.validate_question_item(v.EXAMPLE_QUESTION, q, "$")


def test_example_trace_validates():
    t = _load_trace_schema()
    v.validate_trace_record(v.EXAMPLE_TRACE, t, "$")


def test_question_rejects_bad_language():
    q = _load_question_schema()
    bad = dict(v.EXAMPLE_QUESTION)
    bad["language"] = "zh"
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_empty_text():
    q = _load_question_schema()
    bad = dict(v.EXAMPLE_QUESTION)
    bad["question"] = ""
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_missing_evidence_spans():
    q = _load_question_schema()
    bad = {k: val for k, val in v.EXAMPLE_QUESTION.items() if k != "evidence_spans"}
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_dangling_answer_unit():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    # Reference a non-existent answer unit AU999 in an evidence requirement.
    bad["evidence_requirements"][0]["answer_unit_ids"].append("AU999")
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_dangling_evidence_span_unit():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    bad["evidence_spans"].append({
        "unit_id": "AU999",
        "evidence_spans": [{"chunk_id": "chunk-x", "start_char": 0,
                            "end_char": 10, "quote": "..."}]})
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_start_gt_end_char():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    bad["evidence_spans"][0]["evidence_spans"][0]["start_char"] = 999
    bad["evidence_spans"][0]["evidence_spans"][0]["end_char"] = 10
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_start_eq_end_char():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    # zero-length span: start_char == end_char must be rejected (strict <).
    bad["evidence_spans"][0]["evidence_spans"][0]["start_char"] = 100
    bad["evidence_spans"][0]["evidence_spans"][0]["end_char"] = 100
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_required_au_without_span():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    # Drop the evidence span for the REQUIRED answer unit AU2.
    bad["evidence_spans"] = [sp for sp in bad["evidence_spans"]
                             if sp["unit_id"] != "AU2"]
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_required_au_not_referenced():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    # Remove AU2 from its evidence requirement (still has a span, but no ER).
    for er in bad["evidence_requirements"]:
        if "AU2" in er["answer_unit_ids"]:
            er["answer_unit_ids"] = [a for a in er["answer_unit_ids"] if a != "AU2"]
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_duplicate_answer_unit_id():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    bad["answer_units"].append({"unit_id": "AU1", "claim": "dup", "required": False})
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_question_rejects_duplicate_evidence_requirement_id():
    q = _load_question_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_QUESTION))
    bad["evidence_requirements"].append({
        "requirement_id": "ER1", "answer_unit_ids": ["AU1"],
        "alternative_chunk_ids": ["chunk-abc"]})
    with raises(v.SchemaValidationError):
        v.validate_question_item(bad, q, "$")


def test_trace_rejects_empty_stages():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    bad["stages"] = {}
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


def test_trace_rejects_empty_judge():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    bad["judge"] = {}
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


def test_trace_rejects_incomplete_cost():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    # Keep only 4 cost fields; the contract requires all 11.
    bad["cost"] = {k: bad["cost"][k] for k in
                   ("input_tokens", "output_tokens", "llm_calls", "embedding_calls")}
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


def test_trace_rejects_reranker_score_zero_when_no_reranker():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    # reranker_calls is 0, but a candidate reports reranker_score=0 (faked).
    bad["cost"]["reranker_calls"] = 0
    bad["retrievers"][0]["candidates"][0]["reranker_score"] = 0
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


def test_trace_rejects_missing_retriever_score_fields():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    # Drop the newly-required retriever-level reproducibility fields.
    for key in ("score_semantics", "score_normalization_method",
                "score_normalization_version"):
        del bad["retrievers"][0][key]
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


def test_trace_rejects_missing_candidate_score_fields():
    t = _load_trace_schema()
    bad = json.loads(json.dumps(v.EXAMPLE_TRACE))
    # Drop the newly-required candidate-level fields.
    cand = bad["retrievers"][0]["candidates"][0]
    for key in ("normalized_score", "reranker_score", "source_provenance_ids"):
        del cand[key]
    with raises(v.SchemaValidationError):
        v.validate_trace_record(bad, t, "$")


# --------------------------------------------------------------------------- #
# Label derivation / stability-rule tests
# --------------------------------------------------------------------------- #
def test_stable_success_rule():
    # 3/3 -> success; 0/3,1/3 -> fail; 2/3 -> undecided (None)
    assert v._stable_success(3, 3) is True
    assert v._stable_success(0, 3) is False
    assert v._stable_success(1, 3) is False
    assert v._stable_success(2, 3) is None
    # 5 trials -> 4/5 or 5/5 success; 0..3/5 fail
    assert v._stable_success(4, 5) is True
    assert v._stable_success(5, 5) is True
    assert v._stable_success(3, 5) is False
    assert v._stable_success(0, 5) is False
    # 10 trials (locked-test boundary) -> 8..10/10 success; 0..7/10 fail
    assert v._stable_success(8, 10) is True
    assert v._stable_success(9, 10) is True
    assert v._stable_success(10, 10) is True
    assert v._stable_success(7, 10) is False   # boundary: 7/10 must NOT pass
    assert v._stable_success(4, 10) is False   # would wrongly pass under old >=4 rule
    assert v._stable_success(0, 10) is False
    # any other trial count -> undecided (None)
    assert v._stable_success(1, 1) is None
    assert v._stable_success(4, 4) is None
    assert v._stable_success(6, 6) is None


def test_label_derivation_matches_expected():
    derived = v.derive_labels(v.EXAMPLE_RUNS)
    for qid, expected in v.EXPECTED_DERIVED.items():
        assert derived[qid] == expected, f"mismatch for {qid}"


def test_derive_labels_aggregates_repeats():
    # 5 repeats per route; 2/3 then 4/5 scenario -> P2 and P3 both successful.
    runs = (v._mk_runs("q", "P2", [True, True, False, True, True], "P2")
            + v._mk_runs("q", "P3", [True, True, True, True, False], "P3"))
    derived = v.derive_labels(runs)["q"]
    assert set(derived["successful_route_set"]) == {"P2", "P3"}
    assert derived["minimum_sufficient_route_by_policy"] == "P2"
    assert derived["pareto_optimal_routes"] == ["P3"]


def test_undecided_route_excluded_from_successful():
    # Exactly 2/3 at 3 trials is NOT a stopping point -> excluded.
    runs = v._mk_runs("q", "P2", [True, True, False], "P2")
    derived = v.derive_labels(runs)["q"]
    assert derived["successful_route_set"] == []
    assert derived["minimum_sufficient_route_by_policy"] is None


def test_missing_cost_not_treated_as_zero():
    # A route with NO cost recorded must NOT be falsely reported cheapest.
    none_cost = {k: None for k in v.COST_FIELDS}
    summary_missing = {"question_id": "q", "route_id": "P2", "n_runs": 3,
                       "pass_count": 3, "stable_success": True, "cost": dict(none_cost)}
    summary_real = {"question_id": "q", "route_id": "P3", "n_runs": 3,
                    "pass_count": 3, "stable_success": True,
                    "cost": {"input_tokens": 5000, "output_tokens": 300,
                             "retrieval_latency_ms": 1500, "end_to_end_latency_ms": 1800,
                             "llm_calls": 2, "embedding_calls": 1, "reranker_calls": 0,
                             "graph_expansion_calls": 1, "retrieved_candidate_count": 30,
                             "gpu_seconds": 0.35, "api_cost": 0.015}}
    # Neither dominates the other: missing components are skipped, not zeroed.
    assert v._dominates(summary_real, summary_missing) is False
    assert v._dominates(summary_missing, summary_real) is False


def test_incomplete_cost_excluded_from_pareto():
    # P2 records ONLY input_tokens=1 (all other components absent -> None);
    # P3 has a complete cost vector. Even though P2 is "lower" on input_tokens,
    # it must NOT enter Pareto (and must NOT dominate P3): an incomplete cost
    # vector can never be declared cheaper than a fully-measured one.
    incomplete = {"input_tokens": 1}  # every other field absent -> None
    complete = {"input_tokens": 5000, "output_tokens": 300,
                "retrieval_latency_ms": 1500, "end_to_end_latency_ms": 1800,
                "llm_calls": 2, "embedding_calls": 1, "reranker_calls": 0,
                "graph_expansion_calls": 1, "retrieved_candidate_count": 30,
                "gpu_seconds": 0.35, "api_cost": 0.015}
    runs = (
        [{"question_id": "q", "route_id": "P2", "success": True, "cost": dict(incomplete)}
         for _ in range(3)]
        + [{"question_id": "q", "route_id": "P3", "success": True, "cost": dict(complete)}
           for _ in range(3)]
    )
    derived = v.derive_labels(runs)["q"]
    # P2 is still "successful" (3/3 stopping point met) but excluded from Pareto.
    assert set(derived["successful_route_set"]) == {"P2", "P3"}
    assert derived["pareto_optimal_routes"] == ["P3"]
    assert "P2" not in derived["pareto_optimal_routes"]
    # sanity: _is_cost_complete reflects the gate
    by_route = {s["route_id"]: s for s in v._aggregate_routes(runs)}
    assert v._is_cost_complete(by_route["P2"]) is False
    assert v._is_cost_complete(by_route["P3"]) is True
    # a real-vs-incomplete pair is never mutually dominant
    assert v._dominates(by_route["P3"], by_route["P2"]) is False
    assert v._dominates(by_route["P2"], by_route["P3"]) is False


# --------------------------------------------------------------------------- #
# End-to-end validator gate
# --------------------------------------------------------------------------- #
def test_validator_script_exits_zero():
    assert v.main([]) == 0


# --------------------------------------------------------------------------- #
# Fallback runner (no pytest)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_functions = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
