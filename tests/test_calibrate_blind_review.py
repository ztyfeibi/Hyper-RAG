#!/usr/bin/env python3
"""test_calibrate_blind_review.py — 校准脚本测试（v2 修复版）。

覆盖九大阻塞问题的回归测试 + 端到端场景（真实 coverage 流、精确统计、
480 条约束、裁决规则、kappa 退化、manifest 哈希、原始文件未改等）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_blind_review as cal  # noqa: E402

# 端到端运行 CLI 等价的 args（审核元数据不可证明 -> 全部 unknown）
# 输出到测试专用目录，禁止触碰真实 ai_adjudication_v1（历史产物不可覆盖）
_TEST_VERSION = "ai_adjudication_v1_testregen"
ARGS_UNKNOWN = argparse.Namespace(
    review_model=None, review_provider=None, review_temperature=None,
    review_prompt_path=None, annotation_type=None,
    output_version=_TEST_VERSION, overwrite=True)

# 原始文件哈希（模块导入时采集，早于任何 fixture / apply 运行）
_ORIG_FILES = (
    [cal.BLIND_DIR / "verdicts_ai_annotated.jsonl", cal.BLIND_DIR / "sample_manifest.json"]
    + [cal._JUDGE_DIR / f"{r}_verdict.jsonl" for r in cal.ROUTES]
    + [cal._JUDGE_DIR / "summary.json"]
)
_ORIG_HASHES = {p.name: cal.sha256_file(p) for p in _ORIG_FILES}


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
# fixture：v2 派生 + 全量 apply（模块级，各跑一次）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v2_ready():
    stats = cal.annotate_v2(cal.BLIND_DIR / cal.ANN_V1, cal.BLIND_DIR / cal.ANN_V2)
    return stats


@pytest.fixture(scope="module")
def applied(v2_ready):
    rc = cal.cmd_apply(ARGS_UNKNOWN)
    assert rc == 0, "cmd_apply 应成功（fail-closed 未触发）"
    return cal._JUDGE_DIR / _TEST_VERSION


# ---------------------------------------------------------------------------
# 问题四：_derive_claim_fatality / derive_record_fatality（v2 映射）
# ---------------------------------------------------------------------------
def test_derive_claim_fatality():
    assert cal._derive_claim_fatality("contradicted") == "fatal"
    assert cal._derive_claim_fatality("unsupported_noncritical") == "harmless"
    assert cal._derive_claim_fatality("unverifiable") == "unresolved"  # v2 改动
    assert cal._derive_claim_fatality("supported_by_source") == "none"
    assert cal._derive_claim_fatality("unknown") == "unresolved"  # 未知 fail-safe 悬置


def test_record_fatality_priority():
    """record-level 由 claim-level 重新派生：fatal > unresolved > harmless > none。"""
    c = lambda f: {"claim": "x", "status": "s", "fatality": f}
    assert cal.derive_record_fatality([]) == "none"
    assert cal.derive_record_fatality([c("harmless"), c("none")]) == "harmless"
    assert cal.derive_record_fatality([c("harmless"), c("unresolved")]) == "unresolved"
    assert cal.derive_record_fatality([c("unresolved"), c("fatal")]) == "fatal"
    assert cal.derive_record_fatality([c("none")]) == "none"


def test_annotate_v2_adds_fatality_and_preserves_original(tmp_path, v2_ready):
    """v2 生成：每条 claim 加显式 fatality，record 级重新派生，原文件不动。"""
    src = cal.BLIND_DIR / cal.ANN_V1
    dst = tmp_path / "v2.jsonl"
    orig_sha = cal.sha256_file(src)
    stats = cal.annotate_v2(src, dst)
    assert stats["records"] == 163
    assert cal.sha256_file(src) == orig_sha  # 原文件未改
    rows = [json.loads(l) for l in open(dst, encoding="utf-8")]
    assert len(rows) == 163
    for r in rows:
        for c in r["unsupported_claims"]:
            assert c.get("fatality") in cal.VALID_CLAIM_FATALITY
            assert c["fatality"] == cal.CLAIM_FATALITY_MAP[c["status"]]
        assert r["unsupported_fatality"] == cal.derive_record_fatality(r["unsupported_claims"])
        assert "unsupported_fatality_original" in r


def test_validate_require_fatality_is_error():
    """require_fatality=True 时 claim 缺 fatality 必须是 ERROR（不再 warning 通过）。"""
    rows = [{"blind_id": "BL-001", "verdict": "pass", "au_status": {"AU1": "supported"},
             "unsupported_fatality": "none",
             "unsupported_claims": [{"claim": "x", "status": "unverifiable"}]}]
    result = cal.validate_annotations(rows, require_fatality=True)
    assert result["passed"] is False
    assert any("缺独立 fatality" in e for e in result["errors"])

    rows2 = [dict(rows[0])]
    rows2[0] = dict(rows2[0])
    rows2[0]["unsupported_claims"] = [{"claim": "x", "status": "unverifiable", "fatality": "unresolved"}]
    # record fatality 与派生不一致 -> ERROR
    result2 = cal.validate_annotations(rows2, require_fatality=True)
    assert any("claim 派生" in e for e in result2["errors"])


# ---------------------------------------------------------------------------
# AU / kappa / verdict 指标（含问题六 kappa 退化）
# ---------------------------------------------------------------------------
def test_au_binary_all_match():
    r = _mk_rec("BL-001", "P0", "q1", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"})
    m = cal.au_binary_prf([r])
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_au_confusion_matrix():
    r = _mk_rec("BL-005", "P0", "q5", "pass",
                {"AU1": "supported", "AU2": "missing"},
                "pass", {"AU1": "supported", "AU2": "contradicted"})
    cm = cal.au_confusion_matrix([r])
    assert cm["matrix"]["supported"]["supported"] == 1
    assert cm["matrix"]["missing"]["contradicted"] == 1


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
    assert -0.3 < k < 0.3


def test_kappa_empty():
    assert cal.cohen_kappa([], [], ["pass"]) is None


def test_kappa_degenerate_returns_none():
    """问题六：类别无变化（双方全 pass）时 pe=1，必须返回 None 而非 1.0。"""
    labels = ["pass"] * 10
    assert cal.cohen_kappa(labels, labels, ["pass", "fail", "uncertain"]) is None
    # 一方单一类别、另一方同类别为主 -> pe 接近 1 的另一形态
    assert cal.cohen_kappa(["fail"] * 5, ["fail"] * 5, ["pass", "fail"]) is None


def test_bootstrap_degenerate_reports_valid_count():
    """问题六：bootstrap 退化轮忽略并报告 n_valid；全退化 CI=N/A。"""
    r = _mk_rec("BL-030", "P0", "q30", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"})
    # metric 恒 None（kappa 退化）-> (None, None, None, 0)
    result = cal.bootstrap_metric([r], lambda rs: None)
    assert result == (None, None, None, 0)
    # 正常 metric -> 4 元组且 n_valid>0
    result2 = cal.bootstrap_metric([r], lambda rs: cal.au_binary_prf(rs)["precision"], n_boot=50)
    assert len(result2) == 4
    assert result2[0] == 1.0 and result2[3] > 0


def test_bootstrap_empty():
    assert cal.bootstrap_metric([], lambda rs: 0.5) == (None, None, None, 0)


def test_verdict_false_pass_fail():
    r = _mk_rec("BL-011", "P0", "q11", "fail", {"AU1": "contradicted"},
                "pass", {"AU1": "supported"})
    m = cal.verdict_agreement([r])
    assert m["false_pass"] == 1 and m["false_fail"] == 0


def test_unsupported_set_c_false_negative():
    r = _mk_rec("BL-150", "P_gold", "q50", "pass", {"AU1": "supported"},
                "pass", {"AU1": "supported"},
                ai_claims=[{"claim": "x", "status": "unverifiable", "fatality": "unresolved"}],
                lc_unsup_count=0)
    m = cal.unsupported_metrics([r])
    assert m["set_c_negative_control"]["ai_found_unsupported"] == 1
    assert m["claim_level"]["fatality_dist"].get("unresolved") == 1


# ---------------------------------------------------------------------------
# 问题二/三：apply_rule_v3 裁决规则
# ---------------------------------------------------------------------------
def _lc(qid="q1", route="P1", units=None):
    return {"question_id": qid, "route": route,
            "answer_units": units or [{"unit_id": "AU1", "status": "supported"}]}


def test_rule_v3_all_supported_pass():
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(route="P0"), ai, coverage_hit=None,
                          has_evidence_gate=False, required_aus={"AU1"})
    assert v["answer_correctness"] == "pass"
    assert v["route_success"] == "pass"


def test_rule_v3_required_contradicted_fail():
    ai = {"verdict": "fail", "au_status": {"AU1": "contradicted"},
          "unsupported_fatality": "fatal", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(), ai, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "fail"
    assert v["route_success"] == "fail"


def test_rule_v3_required_missing_fail():
    """问题三：required AU missing（unsupported）必须 fail，不得 pending。"""
    ai = {"verdict": "uncertain", "au_status": {"AU1": "missing"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(), ai, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "fail"
    assert v["has_required_missing"] is True


def test_rule_v3_required_missing_not_in_au_status_fail():
    """AI 标注漏填的 required AU（缺键）按 missing 处理 -> fail。"""
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(units=[{"unit_id": "AU1", "status": "supported"},
                                     {"unit_id": "AU2", "status": "supported"}]),
                          ai, coverage_hit=True, has_evidence_gate=True,
                          required_aus={"AU1", "AU2"})
    assert v["answer_correctness"] == "fail"


def test_rule_v3_optional_au_missing_not_fail():
    """optional AU missing 不触发 fail。"""
    ai = {"verdict": "pass", "au_status": {"AU1": "supported", "AU2": "missing"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(), ai, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "pass"


def test_rule_v3_coverage_fail_hard_fail():
    """coverage=False 是 route_success 硬 fail；answer_correctness 保持原值。"""
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(route="P2"), ai, coverage_hit=False,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["route_success"] == "fail"
    assert v["answer_correctness"] == "pass"


def test_rule_v3_p0_no_gate():
    ai = {"verdict": "pass", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "none", "unsupported_claims": []}
    v = cal.apply_rule_v3(_lc(route="P0"), ai, coverage_hit=False,
                          has_evidence_gate=False, required_aus={"AU1"})
    assert v["route_success"] == "pass"


def test_rule_v3_fatal_claim_fail():
    ai = {"verdict": "fail", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "fatal",
          "unsupported_claims": [{"claim": "wrong data", "status": "contradicted",
                                  "fatality": "fatal"}]}
    v = cal.apply_rule_v3(_lc(), ai, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "fail"
    assert v["has_fatal_claim"] is True


def test_rule_v3_unverifiable_pending():
    ai = {"verdict": "uncertain", "au_status": {"AU1": "supported"},
          "unsupported_fatality": "unresolved",
          "unsupported_claims": [{"claim": "unknown claim", "status": "unverifiable",
                                  "fatality": "unresolved"}]}
    v = cal.apply_rule_v3(_lc(), ai, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "pending"
    assert v["has_unresolved_claim"] is True


def test_rule_v3_no_ai_never_auto_pass():
    """问题二：无 AI 终审不得自动 pass——LongCat AU 全 supported 也必须 pending。"""
    lc = _lc(units=[{"unit_id": "AU1", "status": "supported"}])
    lc["verdict"] = "pass"
    lc["derived_verdict"] = "pass"
    v = cal.apply_rule_v3(lc, None, coverage_hit=True,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["source"] == "longcat_only"
    assert v["answer_correctness"] == "pending"
    assert v["route_success"] == "pending"


def test_rule_v3_no_ai_p0_pending():
    """问题二：P0 未审核一律 pending（进补充审核队列）。"""
    lc = _lc(route="P0", units=[{"unit_id": "AU1", "status": "supported"}])
    lc["verdict"] = "pass"
    v = cal.apply_rule_v3(lc, None, coverage_hit=None,
                          has_evidence_gate=False, required_aus={"AU1"})
    assert v["answer_correctness"] == "pending"
    assert v["route_success"] == "pending"


def test_rule_v3_no_ai_coverage_fail():
    """问题二：未审核 + coverage=false -> route fail 但 answer_correctness 保持 pending。"""
    lc = _lc(units=[{"unit_id": "AU1", "status": "supported"}])
    v = cal.apply_rule_v3(lc, None, coverage_hit=False,
                          has_evidence_gate=True, required_aus={"AU1"})
    assert v["answer_correctness"] == "pending"
    assert v["route_success"] == "fail"


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
# 问题七：align fail-closed
# ---------------------------------------------------------------------------
def test_align_fail_closed_missing_mapping(v2_ready, monkeypatch):
    """blind_id 无 manifest 映射必须抛 AlignError，不得静默跳过。"""
    manifest = cal.load_sample_manifest()
    # 删掉一个 blind_id 的映射
    for key in ("set_A_main_100", "set_B_pgold_pending_43", "set_C_neg_control_20"):
        if manifest.get(key):
            manifest[key] = manifest[key][1:]  # 去掉第一条
            break
    monkeypatch.setattr(cal, "load_sample_manifest", lambda: manifest)
    with pytest.raises(cal.AlignError, match="无映射"):
        cal.align_records()


def test_align_fail_closed_parse_not_ok(v2_ready, monkeypatch):
    """对齐记录 parse_ok 非 true 必须抛 AlignError。"""
    orig = cal.load_longcat_verdicts
    # 取 set_A 第一条映射的 (route, qid)，确保在 163 条对齐范围内
    entry = cal.load_sample_manifest()["set_A_main_100"][0]

    def patched():
        data = orig()
        rec = data[entry["route"]][entry["qid"]]
        data[entry["route"]][entry["qid"]] = dict(rec, parse_ok=False)
        return data

    monkeypatch.setattr(cal, "load_longcat_verdicts", patched)
    with pytest.raises(cal.AlignError, match="parse_ok"):
        cal.align_records()


def test_align_ok_returns_163(v2_ready):
    records = cal.align_records()
    assert len(records) == 163


# ---------------------------------------------------------------------------
# 问题一：真实 coverage（端到端，真实数据）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cov_map():
    return cal.compute_coverage_map()


def test_coverage_exact_frozen_stats(cov_map):
    """全量 coverage 统计必须与冻结结果一致：P1=48 P2=6 P3=24 P4=34 P_gold=80。"""
    errors = cal.verify_coverage_against_frozen(cov_map)
    assert errors == []
    stats = cal.coverage_route_stats(cov_map)
    assert stats["P1"]["pass"] == 48
    assert stats["P2"]["pass"] == 6
    assert stats["P3"]["pass"] == 24
    assert stats["P4"]["pass"] == 34
    assert stats["P_gold"]["pass"] == 80
    assert stats["P_gold"]["n"] == 80 and stats["P_gold"]["no_context"] == 0


def test_coverage_record_fields(cov_map):
    """每条 gated 记录保存 coverage_hit/规则版本/ER 计数/详情/context_hash。"""
    n_checked = 0
    for (route, qid), v in cov_map.items():
        if route == "P0":
            assert v["coverage_hit"] is None and v["coverage_rule_version"] is None
            continue
        n_checked += 1
        assert v["coverage_rule_version"] == "v3.1"
        assert isinstance(v["coverage_hit"], bool)
        assert isinstance(v["evidence_requirements_total"], int)
        assert isinstance(v["evidence_requirements_hit"], int)
        assert v["evidence_requirements_hit"] <= v["evidence_requirements_total"]
        assert isinstance(v["coverage_details"], dict)
        assert len(v["coverage_details"]) == v["evidence_requirements_total"]
        assert v["context_hash"] and len(v["context_hash"]) == 64
    assert n_checked == 400  # 五条 gated 路径 × 80


def test_coverage_context_hash_matches_result_file(cov_map):
    """context_hash 必须来自 result 文件的真实 context（抽查 P2 一条）。"""
    import judge_longcat as jl
    rf = cal._ROOT / jl.result_file("P2", cal.REPEAT, cal.SEED, cal.SNAPSHOT)
    contexts = jl.load_result_contexts(rf)
    (route, qid), v = next(((k, v) for k, v in cov_map.items() if k[0] == "P2"))
    assert v["context_hash"] == cal.sha256_str(contexts[qid])


def test_coverage_frozen_mismatch_detected():
    """篡改冻结常量时 verify 必须检出不一致（防回归：空 map 统计为 0 != 冻结 48）。"""
    saved = dict(cal.FROZEN_COVERAGE_PASS)
    try:
        cal.FROZEN_COVERAGE_PASS["P1"] = 47  # 模拟与复算不一致的冻结值
        assert cal.verify_coverage_against_frozen({}) != []
    finally:
        cal.FROZEN_COVERAGE_PASS.clear()
        cal.FROZEN_COVERAGE_PASS.update(saved)


# ---------------------------------------------------------------------------
# 问题八：apply 端到端输出四件套
# ---------------------------------------------------------------------------
def test_apply_final_verdicts_480(applied):
    fv = [json.loads(l) for l in open(applied / "final_verdicts.jsonl", encoding="utf-8")]
    assert len(fv) == 480
    keys = [(r["route"], r["question_id"]) for r in fv]
    assert len(set(keys)) == 480  # (route, qid) 唯一
    for route in cal.ROUTES:
        assert sum(1 for r in fv if r["route"] == route) == 80
    for r in fv:
        assert r["answer_correctness"] in ("pass", "fail", "pending")
        assert r["route_success"] in ("pass", "fail", "pending")
        assert r["adjudication_rule_version"] == "v3"
        if r["route"] == "P0":
            assert r["coverage_hit"] is None
        else:
            assert isinstance(r["coverage_hit"], bool)
            assert r["coverage_rule_version"] == "v3.1"
            assert r["context_hash"]


def test_apply_no_auto_pass_for_unreviewed(applied):
    """未审核（source=longcat_only）记录不得出现 answer_correctness=pass。"""
    fv = [json.loads(l) for l in open(applied / "final_verdicts.jsonl", encoding="utf-8")]
    unreviewed = [r for r in fv if r["source"] == "longcat_only"]
    assert len(unreviewed) == 480 - 163
    assert all(r["answer_correctness"] == "pending" for r in unreviewed)


def test_apply_coverage_fail_hard_fail_distribution(applied):
    """gated 路径 coverage=false 的记录 route_success 必为 fail（无论是否审核）。"""
    fv = [json.loads(l) for l in open(applied / "final_verdicts.jsonl", encoding="utf-8")]
    cov_fail = [r for r in fv if r["route"] != "P0" and r["coverage_hit"] is False]
    assert len(cov_fail) == (32 + 74 + 56 + 46)  # P1..P4 冻结 coverage fail 数
    assert all(r["route_success"] == "fail" for r in cov_fail)


def test_apply_pending_supplementary_content(applied):
    """补充审核队列：不含已审核 163 条、隐藏 route/blind_id、含盲审材料。"""
    rows = [json.loads(l) for l in
            open(applied / "pending_supplementary.jsonl", encoding="utf-8")]
    assert len(rows) > 0
    fv = [json.loads(l) for l in open(applied / "final_verdicts.jsonl", encoding="utf-8")]
    reviewed = {r["question_id"] for r in fv if r["blind_id"] is not None}
    by_qid = {}
    for r in fv:
        by_qid.setdefault(r["question_id"], []).append(r)
    for r in rows:
        assert "route" not in r and "blind_id" not in r  # 隐藏路径标识
        assert r["reason"] == "no_ai_review"
        assert r["question"]
        assert r["candidate_answer"]
        assert isinstance(r["answer_units"], list) and r["answer_units"]
        assert isinstance(r["evidence_spans"], dict)
        # 该 qid 必须至少存在一条未审核且 pending 的记录（队列成员资格）
        assert any(x["source"] == "longcat_only" and x["route_success"] == "pending"
                   for x in by_qid[r["question_id"]]), \
            f"{r['question_id']} 不应在补充审核队列中"


def test_pending_qid_not_reviewed_on_any_route(applied):
    rows = [json.loads(l) for l in
            open(applied / "pending_supplementary.jsonl", encoding="utf-8")]
    fv = [json.loads(l) for l in open(applied / "final_verdicts.jsonl", encoding="utf-8")]
    reviewed = {r["question_id"] for r in fv if r["blind_id"] is not None}
    pending_qids = {r["question_id"] for r in rows}
    # 一个 qid 可能在 P0 未审核（进队列）而 P1 已审核；队列只按 (route,qid) 排除，
    # 但材料隐藏 route——校验：队列 qid 至少存在一条 pending 的未审核记录
    by_qid = {}
    for r in fv:
        by_qid.setdefault(r["question_id"], []).append(r)
    for qid in pending_qids:
        assert any(r["source"] == "longcat_only" and r["route_success"] == "pending"
                   for r in by_qid[qid]), f"{qid} 不应有任何未审核 pending 记录"


def test_apply_final_summary(applied):
    s = json.loads((applied / "final_summary.json").read_text(encoding="utf-8"))
    assert s["total_records"] == 480
    assert s["adjudication_rule_version"] == "v3"
    assert s["coverage_rule_version"] == "v3.1"
    assert s["ai_annotated_records"] == 163
    assert s["coverage_frozen_check"]["match"] is True
    assert s["coverage_frozen_check"]["expected_pass"] == cal.FROZEN_COVERAGE_PASS
    for route in cal.ROUTES:
        assert s["by_route"][route]["n"] == 80


def test_apply_manifest_hashes(applied):
    """manifest 的全部 SHA-256 必须与实际文件一致。"""
    m = json.loads((applied / "manifest.json").read_text(encoding="utf-8"))
    assert m["adjudication_rule_version"] == "v3"
    assert m["coverage_rule_version"] == "v3.1"
    assert m["counts"]["total_records"] == 480
    assert m["counts"]["per_route"] == {r: 80 for r in cal.ROUTES}
    assert m["provenance"]["review_model"] == "unknown"
    assert m["provenance"]["provenance_complete"] is False
    for fname, h in m["output_sha256"].items():
        assert cal.sha256_file(applied / fname) == h
    assert cal.sha256_file(cal.BLIND_DIR / cal.ANN_V1) == m["input_sha256"]["ai_annotated_v1"]
    assert cal.sha256_file(cal.BLIND_DIR / cal.ANN_V2) == m["input_sha256"]["ai_annotated_v2"]
    for route in cal.ROUTES:
        assert cal.sha256_file(cal._JUDGE_DIR / f"{route}_verdict.jsonl") == \
            m["input_sha256"]["longcat_verdicts"][route]
    assert m["coverage_recomputation"]["frozen_match"] is True
    assert len(m["script_sha256"]) == 64
    assert len(m["judge_script_sha256"]) == 64


def test_original_files_unchanged(applied):
    """apply 全流程后原始 LongCat 判定 / AI 标注 / summary 未被改动。"""
    for p in _ORIG_FILES:
        assert cal.sha256_file(p) == _ORIG_HASHES[p.name], f"{p.name} 被改动"


def test_v2_not_overwriting_v1(applied):
    """v2 是独立新文件，v1 内容保持原样。"""
    v1 = [json.loads(l) for l in open(cal.BLIND_DIR / cal.ANN_V1, encoding="utf-8")]
    for r in v1:
        for c in r.get("unsupported_claims", []):
            assert "fatality" not in c  # 原文件无派生字段
        assert "unsupported_fatality_original" not in r


# ---------------------------------------------------------------------------
# validate 端到端（真实数据）
# ---------------------------------------------------------------------------
def test_validate_real_data(v2_ready):
    """用真实数据跑 validate 结构校验（v2 应全部通过）。"""
    v2 = cal.load_annotated_v2()
    result = cal.validate_annotations(v2, require_fatality=True)
    assert result["passed"] is True
    assert len(result["errors"]) == 0
    assert result["stats"]["total"] == 163
    assert result["stats"]["set_A"] == 100
    assert result["stats"]["set_B"] == 43
    assert result["stats"]["set_C"] == 20


def test_apply_refuses_to_overwrite_real_v1(v2_ready):
    """旧模式默认 output_version=ai_adjudication_v1 且目录已存在时必须拒绝（防 2026-08-16 重写事故复发）。"""
    args = argparse.Namespace(
        review_model=None, review_provider=None, review_temperature=None,
        review_prompt_path=None, annotation_type=None)  # 无 output_version -> 默认 v1
    rc = cal.cmd_apply(args)
    assert rc == 1, "ai_adjudication_v1 已存在且未加 --overwrite，必须返回 1"


def test_apply_refuses_to_overwrite_real_v1(v2_ready):
    """旧模式默认 output_version=ai_adjudication_v1 且目录已存在时必须拒绝（防 2026-08-16 重写事故复发）。"""
    args = argparse.Namespace(
        review_model=None, review_provider=None, review_temperature=None,
        review_prompt_path=None, annotation_type=None)  # 无 output_version -> 默认 v1
    rc = cal.cmd_apply(args)
    assert rc == 1, "ai_adjudication_v1 已存在且未加 --overwrite，必须返回 1"
