#!/usr/bin/env python3
"""test_calibrate_apply_v2.py — 合并标注 apply（ai_adjudication_v2）端到端测试。

覆盖：
- 318 条合并标注按 (route, question_id) 对齐，480 条最终裁决
- 未审核 162 条全部为 coverage 硬失败（route_success=fail）
- E 组 uncertain 不被强转 pass/fail；unresolved 保持 pending 进 excluded_records
- 输出目录守卫（已存在默认报错 / 禁止写入 ai_adjudication_v1）
- ai_adjudication_v1 产物哈希不变
- review_metadata.json 离线重建（D=155 / E=12 / SHA-256 / schema 分布）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_blind_review as cal  # noqa: E402

SUP_DIR = cal._JUDGE_DIR / "blind_review_supplementary"
ANN_ALL = SUP_DIR / "verdicts_ai_all_v1.jsonl"

# 测试专用输出目录（--overwrite 保证可重复运行）；真实产物目录 ai_adjudication_v2
# 由正式命令生成，存在时由 test_real_v2_dir_matches 校验
E2E_VERSION = "ai_adjudication_v2_e2e"
V1_DIR = cal._JUDGE_DIR / "ai_adjudication_v1"

_V1_FILES = ("final_verdicts.jsonl", "final_summary.json",
             "pending_supplementary.jsonl", "manifest.json")
# v1 基线在 applied_v2 fixture 内、合并模式 apply 之前采集（同一 pytest 会话中
# 旧模式测试会合法地重生成 v1，文件级哈希只有相对基线可比）
_V1_SNAPSHOT: dict[str, str] = {}

ARGS_APPLY = argparse.Namespace(
    review_model=None, review_provider=None, review_temperature=None,
    review_prompt_path=None, annotation_type=None,
    annotation_file=str(ANN_ALL), output_version=E2E_VERSION, overwrite=True)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# fixture：跑一次合并模式 apply（测试专用目录）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def applied_v2():
    for f in _V1_FILES:
        _V1_SNAPSHOT[f] = hashlib.sha256((V1_DIR / f).read_bytes()).hexdigest()
    rc = cal.cmd_apply(ARGS_APPLY)
    assert rc == 0, "合并模式 cmd_apply 应成功"
    return cal._JUDGE_DIR / E2E_VERSION


# ---------------------------------------------------------------------------
# 1. 基本约束：318 / 162 / 480 / 每路径 80 / coverage 冻结对账
# ---------------------------------------------------------------------------
def test_annotation_count_318():
    rows = _load_jsonl(ANN_ALL)
    assert len(rows) == 318
    keys = [(r["route"], r["question_id"]) for r in rows]
    assert len(set(keys)) == 318  # (route, question_id) 唯一


def test_final_480_unique_per_route_80(applied_v2):
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    assert len(fv) == 480
    keys = [(r["route"], r["question_id"]) for r in fv]
    assert len(set(keys)) == 480
    for route in cal.ROUTES:
        assert sum(1 for r in fv if r["route"] == route) == 80
    assert (applied_v2 / "excluded_records.jsonl").exists()
    assert not (applied_v2 / "pending_supplementary.jsonl").exists()  # v2 不再排队


def test_summary_counts(applied_v2):
    s = json.loads((applied_v2 / "final_summary.json").read_text(encoding="utf-8"))
    assert s["annotation_count"] == 318
    assert s["unreviewed_count"] == 162
    assert s["total_records"] == 480
    assert s["apply_mode"] != "legacy"
    assert s["P_gold_route_success"]["pass"] + s["P_gold_route_success"]["fail"] \
        + s["P_gold_route_success"]["pending"] == 80


def test_coverage_frozen(applied_v2):
    """coverage 逐条复算与冻结统计一致：P1:48 / P2:6 / P3:24 / P4:34 / P_gold:80。"""
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    from collections import Counter
    cov_pass = Counter(r["route"] for r in fv if r["coverage_hit"] is True)
    assert cov_pass == {"P1": 48, "P2": 6, "P3": 24, "P4": 34, "P_gold": 80}
    s = json.loads((applied_v2 / "final_summary.json").read_text(encoding="utf-8"))
    assert s["coverage_frozen_check"]["match"] is True


# ---------------------------------------------------------------------------
# 2. 未审核 162 条 = coverage 硬失败
# ---------------------------------------------------------------------------
def test_unreviewed_162_all_coverage_hard_fail(applied_v2):
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    unreviewed = [r for r in fv if r["source"] == "longcat_only"]
    assert len(unreviewed) == 162
    for r in unreviewed:
        assert r["coverage_hit"] is False, f"{r['route']}/{r['question_id']} 非 coverage 硬失败"
        assert r["route_success"] == "fail"
        assert r["answer_correctness"] == "pending"  # 未审核不判答案对错


def test_ai_reviewed_318(applied_v2):
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    assert sum(1 for r in fv if r["source"] == "ai_review") == 318
    assert sum(1 for r in fv if r["review_source"] == "supplementary_D") == 155
    assert sum(1 for r in fv if r["review_source"] == "recheck_E") == 12
    assert sum(1 for r in fv if r["review_source"] == "original") == 151


# ---------------------------------------------------------------------------
# 3. pending：uncertain / unresolved 不强转，全部进 excluded_records
# ---------------------------------------------------------------------------
def test_no_167_pending_only_unresolved_left(applied_v2):
    """route_success 不再有 167 条 pending；剩余 pending 只来自 unresolved 悬置。"""
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    pending = [r for r in fv if r["route_success"] == "pending"]
    assert len(pending) < 167
    # 所有 pending 均为已审核且含 unresolved claim（或未审核 + coverage 通过，
    # 实际数据中不存在——162 条未审核全部 coverage 硬失败）
    for r in pending:
        assert r["source"] == "ai_review"
        assert r["has_unresolved_claim"] is True


def test_e_uncertain_not_forced(applied_v2):
    """E 组 4 条 uncertain 不被强行转成 pass/fail。"""
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    fmap = {(r["route"], r["question_id"]): r for r in fv}
    ann = _load_jsonl(ANN_ALL)
    e_uncertain = [r for r in ann
                   if r.get("review_source") == "recheck_E" and r["verdict"] == "uncertain"]
    assert len(e_uncertain) == 4
    for a in e_uncertain:
        fr = fmap[(a["route"], a["question_id"])]
        assert fr["route_success"] == "pending", \
            f"E uncertain {a['question_id']} 被强转为 {fr['route_success']}"


def test_all_pending_in_excluded_with_reason(applied_v2):
    """所有 route_success=pending 记录都进入 excluded_records.jsonl 且带 reason。"""
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    ex = _load_jsonl(applied_v2 / "excluded_records.jsonl")
    pending_keys = {(r["route"], r["question_id"])
                    for r in fv if r["route_success"] == "pending"}
    ex_keys = {(r["route"], r["question_id"]) for r in ex}
    assert pending_keys == ex_keys
    for r in ex:
        assert r["exclusion_reason"] in ("unresolved_claims_pending",
                                         "no_ai_review_pending")
    s = json.loads((applied_v2 / "final_summary.json").read_text(encoding="utf-8"))
    assert s["excluded_count"] == len(ex)


def test_uncertain_verdict_never_pass(applied_v2):
    """全部 uncertain 标注（含 D 组）不会被裁成 pass。"""
    fv = _load_jsonl(applied_v2 / "final_verdicts.jsonl")
    fmap = {(r["route"], r["question_id"]): r for r in fv}
    ann = _load_jsonl(ANN_ALL)
    for a in ann:
        if a["verdict"] == "uncertain":
            assert fmap[(a["route"], a["question_id"])]["route_success"] != "pass"


# ---------------------------------------------------------------------------
# 4. 输出目录守卫与 v1 不可变
# ---------------------------------------------------------------------------
def test_output_dir_exists_guard(applied_v2):
    """--output-version 目录已存在且未加 --overwrite 时必须报错（禁止静默覆盖）。"""
    assert applied_v2.exists()
    args = argparse.Namespace(
        review_model=None, review_provider=None, review_temperature=None,
        review_prompt_path=None, annotation_type=None,
        annotation_file=str(ANN_ALL), output_version=E2E_VERSION, overwrite=False)
    rc = cal.cmd_apply(args)
    assert rc == 1


def test_annotation_file_forbidden_for_v1():
    """--annotation-file 模式禁止写入 ai_adjudication_v1。"""
    args = argparse.Namespace(
        review_model=None, review_provider=None, review_temperature=None,
        review_prompt_path=None, annotation_type=None,
        annotation_file=str(ANN_ALL), output_version="ai_adjudication_v1",
        overwrite=False)
    rc = cal.cmd_apply(args)
    assert rc == 1


def test_annotation_file_missing():
    args = argparse.Namespace(
        review_model=None, review_provider=None, review_temperature=None,
        review_prompt_path=None, annotation_type=None,
        annotation_file="caches/nonexistent.jsonl", output_version="x_v2",
        overwrite=False)
    rc = cal.cmd_apply(args)
    assert rc == 1


def test_v1_hashes_unchanged(applied_v2):
    """合并模式 apply 之后 ai_adjudication_v1 四件套哈希不变（相对 apply 前基线）。"""
    for f, h in _V1_SNAPSHOT.items():
        assert hashlib.sha256((V1_DIR / f).read_bytes()).hexdigest() == h, \
            f"ai_adjudication_v1/{f} 在合并模式 apply 中被改动"


# ---------------------------------------------------------------------------
# 5. manifest：318 条标注 + 全量输入输出哈希
# ---------------------------------------------------------------------------
def test_manifest(applied_v2):
    m = json.loads((applied_v2 / "manifest.json").read_text(encoding="utf-8"))
    assert m["apply_mode"] == "merged_annotation"
    assert m["counts"]["total_records"] == 480
    assert m["counts"]["per_route"] == {r: 80 for r in cal.ROUTES}
    assert m["counts"]["ai_annotated"] == 318
    assert m["counts"]["unreviewed"] == 162
    assert m["counts"]["excluded"] == m["counts"]["excluded"]  # 存在即可
    # 标注文件哈希
    assert m["input_sha256"]["annotation_file"] == cal.sha256_file(ANN_ALL)
    # 输出哈希与实际文件一致
    for fname, h in m["output_sha256"].items():
        assert cal.sha256_file(applied_v2 / fname) == h
    # 补充审核溯源（review_metadata.json）
    ap = m["annotation_provenance"]
    assert ap is not None
    assert ap["review_provider"] in ("local-vllm", "SiliconFlow")
    assert len(ap["sha256"]) == 64


# ---------------------------------------------------------------------------
# 6. load_merged_annotations 校验（fail-closed）
# ---------------------------------------------------------------------------
def _ann_row(**kw):
    base = {"route": "P0", "question_id": "qv2-0001", "blind_id": None,
            "review_id": "SR-D-001", "review_source": "supplementary_D",
            "au_status": {"AU1": "supported"}, "verdict": "pass",
            "unsupported_fatality": "none", "unsupported_claims": []}
    base.update(kw)
    return base


def test_load_merged_annotations_rejects_bad(tmp_path):
    p = tmp_path / "ann.jsonl"
    p.write_text(json.dumps(_ann_row(verdict="bogus"), ensure_ascii=False) + "\n",
                 encoding="utf-8")
    with pytest.raises(cal.AlignError, match="verdict"):
        cal.load_merged_annotations(p)


def test_load_merged_annotations_rejects_dup(tmp_path):
    p = tmp_path / "ann.jsonl"
    p.write_text(json.dumps(_ann_row()) + "\n" + json.dumps(_ann_row()) + "\n",
                 encoding="utf-8")
    with pytest.raises(cal.AlignError, match="重复"):
        cal.load_merged_annotations(p)


def test_load_merged_annotations_ok(tmp_path):
    p = tmp_path / "ann.jsonl"
    p.write_text(json.dumps(_ann_row()) + "\n", encoding="utf-8")
    rows = cal.load_merged_annotations(p)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 7. 真实 ai_adjudication_v2 目录（正式 apply 后存在时校验）
# ---------------------------------------------------------------------------
def test_real_v2_dir_matches():
    real = cal._JUDGE_DIR / "ai_adjudication_v2"
    if not real.exists():
        pytest.skip("ai_adjudication_v2 尚未由正式命令生成")
    s = json.loads((real / "final_summary.json").read_text(encoding="utf-8"))
    assert s["annotation_count"] == 318
    assert s["unreviewed_count"] == 162
    assert s["total_records"] == 480
    fv = _load_jsonl(real / "final_verdicts.jsonl")
    assert len(fv) == 480
