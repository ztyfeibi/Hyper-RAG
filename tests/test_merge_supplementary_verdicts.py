# -*- coding: utf-8 -*-
"""merge_supplementary_verdicts.py 单元测试（合成 fixture，不依赖 caches 真实数据）。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import merge_supplementary_verdicts as msv
import run_supplementary_review as rsr


def _ann(bid, verdict="pass", fatality="harmless"):
    return {"blind_id": bid, "au_status": {"AU1": "supported"}, "verdict": verdict,
            "unsupported_fatality": fatality,
            "unsupported_claims": [{"claim": "x is y.", "status": "unverifiable"}],
            "notes": ""}


def _sup(rid, verdict="fail", fatality="fatal", schema="claim_id_v2"):
    return {"review_id": rid, "au_status": {"AU1": "supported"}, "verdict": verdict,
            "unsupported_fatality": fatality,
            "unsupported_claims": [{"claim": "x is y.", "status": "unverifiable",
                                    "claim_id": "C1", "prefilled": True}],
            "notes": "", "response_schema": schema}


def _fv_row(route, qid, bid=None, status="pass"):
    return {"route": route, "question_id": qid, "blind_id": bid,
            "route_success": status, "source": "ai_review" if bid else "longcat_only"}


@pytest.fixture
def fixtures(tmp_path, monkeypatch):
    """2 条原标注（BL-001 原, BL-002 被 E 复核）+ 1 条 D 新审。"""
    sup = tmp_path / "sup"
    sup.mkdir()
    ann = [_ann("BL-001"), _ann("BL-002", verdict="uncertain", fatality="unresolved")]
    fv = [
        _fv_row("P0", "qv2-0001", "BL-001"),
        _fv_row("P1", "qv2-0002", "BL-002"),
        _fv_row("P2", "qv2-0003", None, status="pending"),
        _fv_row("P3", "qv2-0004", None, status="fail"),  # 非 pending 的无 blind_id 行
    ]
    sm = {
        "set_D_unreviewed_155": [
            {"review_id": "SR-D-001", "route": "P2", "question_id": "qv2-0003",
             "blind_id": None}],
        "set_E_recheck_12": [
            {"review_id": "SR-E-001", "route": "P1", "question_id": "qv2-0002",
             "blind_id": "BL-002", "ai_verdict": "uncertain"}],
    }
    rec_d = [_sup("SR-D-001")]
    rec_e = [_sup("SR-E-001", verdict="pass", fatality="harmless")]

    (tmp_path / "ann.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ann), encoding="utf-8")
    (tmp_path / "fv.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fv), encoding="utf-8")
    (sup / "sample_manifest.json").write_text(json.dumps(sm), encoding="utf-8")
    (sup / "D.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rec_d), encoding="utf-8")
    (sup / "E.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rec_e), encoding="utf-8")

    monkeypatch.setattr(msv, "ANN", tmp_path / "ann.jsonl")
    monkeypatch.setattr(msv, "FINAL", tmp_path / "fv.jsonl")
    monkeypatch.setattr(msv, "SM", sup / "sample_manifest.json")
    monkeypatch.setattr(msv, "OUT", sup / "out.jsonl")
    monkeypatch.setattr(msv, "OUT_MANIFEST", sup / "out_manifest.json")
    monkeypatch.setattr(rsr, "SET_OUTPUT", {"D": sup / "D.jsonl", "E": sup / "E.jsonl"})
    return {"tmp_path": tmp_path, "sup": sup}


def test_merge_happy_path(fixtures):
    manifest = msv.merge()
    assert manifest["stats"]["total"] == 3
    assert manifest["stats"]["by_source"] == {"original": 1, "recheck_E": 1,
                                              "supplementary_D": 1}

    rows = [json.loads(l) for l in (fixtures["sup"] / "out.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    by_key = {(r["route"], r["question_id"]): r for r in rows}

    # original
    o = by_key[("P0", "qv2-0001")]
    assert o["review_source"] == "original" and o["blind_id"] == "BL-001"
    assert o["verdict"] == "pass" and o["superseded"] is None

    # recheck_E 覆盖 + 溯源
    e = by_key[("P1", "qv2-0002")]
    assert e["review_source"] == "recheck_E" and e["review_id"] == "SR-E-001"
    assert e["verdict"] == "pass"  # 新值
    assert e["superseded"]["original_verdict"] == "uncertain"  # 旧值保留
    assert e["superseded"]["original_fatality"] == "unresolved"

    # supplementary_D 新增
    d = by_key[("P2", "qv2-0003")]
    assert d["review_source"] == "supplementary_D" and d["blind_id"] is None
    assert d["response_schema"] == "claim_id_v2"

    # 翻转统计
    assert len(manifest["stats"]["e_recheck_flips"]["verdict"]) == 1
    assert len(manifest["stats"]["e_recheck_flips"]["fatality"]) == 1
    # 未审核剩余 = fv 总数 4 - 3
    assert manifest["stats"]["unreviewed_remaining"] == 1
    # manifest 输入哈希齐全
    assert set(manifest["inputs"]) == {"verdicts_ai_annotated", "final_verdicts",
                                       "verdicts_ai_supplementary_D",
                                       "verdicts_ai_recheck_E", "sample_manifest"}


def test_dry_run_not_write(fixtures):
    msv.merge(write=False)
    assert not (fixtures["sup"] / "out.jsonl").exists()
    assert not (fixtures["sup"] / "out_manifest.json").exists()


def test_e_blind_id_not_in_annotated(fixtures):
    sm = json.loads((fixtures["sup"] / "sample_manifest.json").read_text(encoding="utf-8"))
    sm["set_E_recheck_12"][0]["blind_id"] = "BL-999"
    (fixtures["sup"] / "sample_manifest.json").write_text(json.dumps(sm), encoding="utf-8")
    with pytest.raises(SystemExit, match="不在原 annotated"):
        msv.merge()


def test_set_d_keys_mismatch_pending(fixtures):
    sm = json.loads((fixtures["sup"] / "sample_manifest.json").read_text(encoding="utf-8"))
    sm["set_D_unreviewed_155"][0]["question_id"] = "qv2-9999"
    (fixtures["sup"] / "sample_manifest.json").write_text(json.dumps(sm), encoding="utf-8")
    with pytest.raises(SystemExit, match="set_D keys"):
        msv.merge()


def test_bad_enum_rejected(fixtures):
    d = [_sup("SR-D-001", verdict="excellent")]  # 非法 verdict
    (fixtures["sup"] / "D.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in d), encoding="utf-8")
    with pytest.raises(SystemExit, match="verdict 非法"):
        msv.merge()


def test_missing_d_record(fixtures):
    (fixtures["sup"] / "D.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="set_D 记录缺失"):
        msv.merge()


def test_duplicate_route_qid_rejected(fixtures, monkeypatch):
    # 让 D 的 (route,qid) 与某条原标注相同 → 重复
    sm = json.loads((fixtures["sup"] / "sample_manifest.json").read_text(encoding="utf-8"))
    fv = [json.loads(l) for l in (fixtures["tmp_path"] / "fv.jsonl")
          .read_text(encoding="utf-8").splitlines()]
    sm["set_D_unreviewed_155"][0].update(route="P0", question_id="qv2-0001")
    # 保持一致性校验通过：把 fv 的 pending-nobid 行改成同 key
    for r in fv:
        if r["route"] == "P2":
            r.update(route="P0", question_id="qv2-0001")
    (fixtures["sup"] / "sample_manifest.json").write_text(json.dumps(sm), encoding="utf-8")
    (fixtures["tmp_path"] / "fv.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fv), encoding="utf-8")
    with pytest.raises(SystemExit, match="重复"):
        msv.merge()
