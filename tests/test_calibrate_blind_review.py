#!/usr/bin/env python3
"""test_calibrate_blind_review.py — 校准脚本测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_blind_review as cal  # noqa: E402


# ---------------------------------------------------------------------------
# 工具：构建测试记录
# ---------------------------------------------------------------------------
def _mk_rec(bid, route, qid, ai_verdict, ai_au, lc_verdict, lc_au,
            ai_fatality="none", ai_claims=None, cov_hit=True, au_shape="single",
            lc_unsup_count=0, has_contradicted=False, lc_derived=None):
    all_aus = set(ai_au.keys()) | set(lc_au.keys())
    return {
        "blind_id": bid, "set": cal._set_of(bid), "route": route, "qid": qid,
        "au_shape": au_shape, "cov_hit": cov_hit,
        "n_unsupported_lc": lc_unsup_count, "has_contradicted": has_contradicted,
        "lc_verdict": lc_verdict, "lc_derived": lc_derived or lc_verdict,
        "lc_human_review": False,
        "lc_au_status": dict(lc_au),
        "lc_unsup_count": lc_unsup_count,
        "ai_verdict": ai_verdict, "ai_fatality": ai_fatality,
        "ai_au_status": dict(ai_au),
        "ai_unsup_claims": ai_claims or [],
        "ai_unsup_count": len(ai_claims) if ai_claims else 0,
        "required_aus": all_aus, "all_aus": all_aus,
        "n_aus": len(all_aus), "is_multi_au": len(all_aus) > 1,
    }


# ---------------------------------------------------------------------------
# _derive_claim_fatality
# ---------------------------------------------------------------------------
def test_derive_claim_fatality():
    assert cal._derive_claim_fatality("contradicted") == "fatal"
    assert cal._derive_claim_fatality("unsupported_noncritical") == "harmless"
    assert cal._derive_claim_fatality("unverifiable") == "harmless"
    assert cal._derive_claim_fatality("supported_by_source") == "none"
    assert cal._derive_claim_fatality("unknown") == "harmless"


# ---------------------------------------------------------------------------
# au_binary_prf
# ---------------------------------------------------------------------------
def test_au_binary_all_match():
    r = _mk_rec("BL-001", "P0", "q1", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"})
    m = cal.au_binary_prf([r])
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_au_binary_fp():
    # LongCat says supported, AI says missing
    r = _mk_rec("BL-002", "P0", "q2", "fail", {"AU1": "missing"},
                "pass", {"AU1": "supported"})
    m = cal.au_binary_prf([r])
    assert m["fp"] == 1 and m["tp"] == 0
    assert m["precision"] == 0.0


def test_au_binary_fn():
    # AI says supported, LongCat says missing
    r = _mk_rec("BL-003", "P0", "q3", "pass", {"AU1": "supported"},
                "fail", {"AU1": "missing"})
    m = cal.au_binary_prf([r])
    assert m["fn"] == 1 and m["tp"] == 0
    assert m["recall"] == 0.0


def test_au_binary_multi_au():
    r = _mk_rec("BL-004", "P0", "q4", "pass",
                {"AU1": "supported", "AU2": "missing", "AU3": "supported"},
                "pass", {"AU1": "supported", "AU2": "missing", "AU3": "contradicted"})
    m = cal.au_binary_prf([r])
    # AU1: both supported -> TP
    # AU2: both not -> TN
    # AU3: LC contradicted (not_supported), AI supported -> FN
    assert m["tp"] == 1 and m["tn"] == 1 and m["fn"] == 1


# ---------------------------------------------------------------------------
# au_confusion_matrix
# ---------------------------------------------------------------------------
def test_au_confusion_matrix():
    r = _mk_rec("BL-005", "P0", "q5", "pass",
                {"AU1": "supported", "AU2": "missing"},
                "pass", {"AU1": "supported", "AU2": "contradicted"})
    cm = cal.au_confusion_matrix([r])
    labels = cm["labels"]
    assert "supported" in labels and "missing" in labels and "contradicted" in labels
    assert cm["matrix"]["supported"]["supported"] == 1
    assert cm["matrix"]["missing"]["contradicted"] == 1


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------
def test_kappa_perfect():
    labels = ["pass", "fail", "pass", "fail"]
    k = cal.cohen_kappa(labels, labels, ["pass", "fail"])
    assert k == 1.0


def test_kappa_random():
    import random
    rng = random.Random(42)
    l1 = [rng.choice(["pass", "fail"]) for _ in range(100)]
    l2 = [rng.choice(["pass", "fail"]) for _ in range(100)]
    k = cal.cohen_kappa(l1, l2, ["pass", "fail"])
    assert -0.3 < k < 0.3  # 随机应接近 0


def test_kappa_empty():
    assert cal.cohen_kappa([], [], ["pass"]) is None


# ---------------------------------------------------------------------------
# verdict_agreement
# ---------------------------------------------------------------------------
def test_verdict_agreement_perfect():
    r = _mk_rec("BL-010", "P0", "q10", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"})
    m = cal.verdict_agreement([r])
    assert m["three_class"]["accuracy"] == 1.0
    assert m["three_class"]["cohen_kappa"] == 1.0
    assert m["false_pass"] == 0 and m["false_fail"] == 0


def test_verdict_false_pass():
    # LongCat=pass, AI=fail
    r = _mk_rec("BL-011", "P0", "q11", "fail", {"AU1": "contradicted"},
                "pass", {"AU1": "supported"})
    m = cal.verdict_agreement([r])
    assert m["false_pass"] == 1
    assert m["false_fail"] == 0


def test_verdict_false_fail():
    # LongCat=fail, AI=pass
    r = _mk_rec("BL-012", "P0", "q12", "pass", {"AU1": "supported"},
                "fail", {"AU1": "missing"})
    m = cal.verdict_agreement([r])
    assert m["false_fail"] == 1
    assert m["false_pass"] == 0


def test_verdict_binary_drop_uncertain():
    r1 = _mk_rec("BL-013", "P0", "q13", "pass", {"AU1": "supported"},
                 "pass", {"AU1": "supported"})
    r2 = _mk_rec("BL-014", "P0", "q14", "uncertain", {"AU1": "missing"},
                 "uncertain", {"AU1": "missing"})
    m = cal.verdict_agreement([r1, r2])
    # 只有 r1 进入二分类（r2 双方 uncertain 被排除）
    assert m["binary_drop_uncertain"]["n"] == 1
    assert m["binary_drop_uncertain"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# unsupported_metrics
# ---------------------------------------------------------------------------
def test_unsupported_both_detected():
    r = _mk_rec("BL-020", "P0", "q20", "fail", {"AU1": "supported"},
                "fail", {"AU1": "supported"},
                ai_claims=[{"claim": "x", "status": "unverifiable"}],
                lc_unsup_count=2)
    m = cal.unsupported_metrics([r])
    assert m["record_level"]["both_detected"] == 1
    assert m["record_level"]["longcat_only"] == 0
    assert m["record_level"]["ai_only"] == 0


def test_unsupported_ai_only():
    # LongCat 未检出, AI 检出
    r = _mk_rec("BL-021", "P0", "q21", "fail", {"AU1": "supported"},
                "pass", {"AU1": "supported"},
                ai_claims=[{"claim": "x", "status": "unsupported_noncritical"}],
                lc_unsup_count=0)
    m = cal.unsupported_metrics([r])
    assert m["record_level"]["ai_only"] == 1
    assert m["record_level"]["longcat_only"] == 0


def test_unsupported_set_c_false_negative():
    # set_C: LongCat 说 0, AI 找到
    r = _mk_rec("BL-150", "P_gold", "q50", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"},
                ai_claims=[{"claim": "x", "status": "unverifiable"}],
                lc_unsup_count=0)
    # BL-150 -> set C (n >= 144)
    assert cal._set_of("BL-150") == "C"
    m = cal.unsupported_metrics([r])
    assert m["set_c_negative_control"]["ai_found_unsupported"] == 1
    assert m["set_c_negative_control"]["longcat_false_negative_rate"] == 1.0


def test_unsupported_set_c_correct():
    # set_C: both say no unsupported
    r = _mk_rec("BL-151", "P_gold", "q51", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"}, ai_claims=[], lc_unsup_count=0)
    m = cal.unsupported_metrics([r])
    assert m["set_c_negative_control"]["ai_found_unsupported"] == 0
    assert m["set_c_negative_control"]["longcat_false_negative_rate"] == 0.0


# ---------------------------------------------------------------------------
# bootstrap_metric
# ---------------------------------------------------------------------------
def test_bootstrap_returns_tuple():
    r = _mk_rec("BL-030", "P0", "q30", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"})
    result = cal.bootstrap_metric([r], lambda rs: cal.au_binary_prf(rs)["precision"])
    assert len(result) == 3
    assert result[0] == 1.0  # point estimate


def test_bootstrap_empty():
    result = cal.bootstrap_metric([], lambda rs: 0.5)
    assert result == (None, None, None)


# ---------------------------------------------------------------------------
# apply_rule_v3
# ---------------------------------------------------------------------------
def test_rule_v3_all_supported_pass():
    lc = {"question_id": "q1", "route": "P0", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"}, "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=True, has_evidence_gate=False)
    assert v["answer_correctness"] == "pass"
    assert v["route_success"] == "pass"


def test_rule_v3_contradicted_fail():
    lc = {"question_id": "q2", "route": "P1", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "fail", "au_status": {"AU1": "contradicted"}, "unsupported_fatality": "fatal", "unsupported_claims": []}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=True, has_evidence_gate=True)
    assert v["answer_correctness"] == "fail"
    assert v["route_success"] == "fail"


def test_rule_v3_coverage_fail_hard_fail():
    # coverage=False must be hard fail, not overridden
    lc = {"question_id": "q3", "route": "P2", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"}, "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=False, has_evidence_gate=True)
    assert v["route_success"] == "fail"
    assert v["answer_correctness"] == "pass"  # answer is pass, but coverage fails


def test_rule_v3_p0_no_gate():
    lc = {"question_id": "q4", "route": "P0", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"}, "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=False, has_evidence_gate=False)
    assert v["route_success"] == "pass"  # P0 ignores coverage


def test_rule_v3_fatal_claim_fail():
    lc = {"question_id": "q5", "route": "P1", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "fail", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "fatal",
          "unsupported_claims": [{"claim": "wrong data", "status": "contradicted"}]}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=True, has_evidence_gate=True)
    assert v["answer_correctness"] == "fail"
    assert v["has_fatal_claim"] is True


def test_rule_v3_unverifiable_pending():
    lc = {"question_id": "q6", "route": "P1", "answer_units": [{"unit_id": "AU1", "status": "supported"}]}
    ai = {"verdict": "uncertain", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "harmless",
          "unsupported_claims": [{"claim": "unknown claim", "status": "unverifiable"}]}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=True, has_evidence_gate=True)
    assert v["answer_correctness"] == "pending"


def test_rule_v3_no_ai_longcat_only():
    lc = {"question_id": "q7", "route": "P1",
          "answer_units": [{"unit_id": "AU1", "status": "missing"}],
          "verdict": "fail", "derived_verdict": "uncertain"}
    v = cal.apply_rule_v3(lc, None, coverage_hit=True, has_evidence_gate=True)
    assert v["source"] == "longcat_only"
    assert v["answer_correctness"] == "uncertain"  # missing but no AI -> derived


def test_rule_v3_missing_with_ai_pending():
    lc = {"question_id": "q8", "route": "P1", "answer_units": [{"unit_id": "AU1", "status": "missing"}]}
    ai = {"verdict": "uncertain", "au_status": {"AU1": "missing"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(lc, ai, coverage_hit=True, has_evidence_gate=True)
    assert v["answer_correctness"] == "pending"


# ---------------------------------------------------------------------------
# _set_of
# ---------------------------------------------------------------------------
def test_set_of():
    assert cal._set_of("BL-001") == "A"
    assert cal._set_of("BL-100") == "A"
    assert cal._set_of("BL-101") == "B"
    assert cal._set_of("BL-143") == "B"
    assert cal._set_of("BL-144") == "C"
    assert cal._set_of("BL-163") == "C"


# ---------------------------------------------------------------------------
# validate_annotations (mock-free, uses real data)
# ---------------------------------------------------------------------------
def test_validate_real_data():
    """用真实数据跑 validate（文件应存在）。"""
    result = cal.validate_annotations()
    assert result["passed"] is True
    assert len(result["errors"]) == 0
    assert result["stats"]["total"] == 163
    assert result["stats"]["set_A"] == 100
    assert result["stats"]["set_B"] == 43
    assert result["stats"]["set_C"] == 20
