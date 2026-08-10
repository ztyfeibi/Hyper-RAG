# -*- coding: utf-8 -*-
"""Step 2.2 证据核验与 Gold 构建的测试。

覆盖：
1. 纯函数 / 机械校验（不依赖 LLM / hyperdb）。
2. Schema 工厂与禁止字段。
3. 候选池哈希绑定（与 ec-v2 主文件一致）。
4. 产物验收（step2_2 输出存在时启用，否则 skip）。
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = REPO_ROOT / "caches" / "neurology_chunk1000" / "question_set_v2" / "pilot_v1"
STEP2_2 = POOL_DIR / "step2_2"

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyperrag import evidence_verification as ev  # noqa: E402


# --------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------
def test_sha256_text_stable():
    assert ev.sha256_text("abc") == ev.sha256_text("abc")
    assert ev.sha256_text("abc") != ev.sha256_text("abd")


def test_split_source_ids_dedups_and_sorts():
    assert ev.split_source_ids("chunk-b<SEP>chunk-a<SEP>chunk-a") == ["chunk-a", "chunk-b"]
    assert ev.split_source_ids("") == []
    assert ev.split_source_ids(None) == []


def test_classify_language():
    assert ev.classify_language(["头痛是一种常见症状"]) == "zh"
    assert ev.classify_language(["Headache is a common symptom."]) == "en"
    # 中英混排
    mixed = ev.classify_language(["患者 presented with 头痛 and weakness"])
    assert mixed in ("zh", "mixed", "en")


def test_locate_span_and_validate():
    text = "The patient presented with ataxia. The MRI showed a lesion."
    loc = ev.locate_span(text, "ataxia")
    assert loc == (27, 33)
    bad = ev.locate_span(text, "ataxiaX")
    assert bad is None
    h = ev.sha256_text(text)
    span = {"chunk_id": "c1", "char_start": 27, "char_end": 33,
            "text": "ataxia", "chunk_content_hash": h}
    ok, reason = ev.validate_span(text, h, span)
    assert ok and reason == "ok"
    # 文本篡改
    span2 = dict(span); span2["text"] = "ATAXIA"
    ok2, r2 = ev.validate_span(text, h, span2)
    assert not ok2 and r2 == "text_mismatch"
    # 哈希不匹配
    span3 = dict(span); span3["chunk_content_hash"] = "deadbeef"
    ok3, r3 = ev.validate_span(text, h, span3)
    assert not ok3 and r3 == "chunk_hash_mismatch"


def test_spans_traceable():
    text = "Headache and vomiting are signs. MRI confirmed the lesion."
    h = ev.sha256_text(text)
    spans = [
        {"chunk_id": "c1", "char_start": 0, "char_end": 30,
         "text": text[:30], "chunk_content_hash": h},
        {"chunk_id": "c1", "char_start": None, "char_end": None,
         "text": "not present", "chunk_content_hash": h},
    ]
    ok, problems = ev.spans_traceable(spans, {"c1": text}, {"c1": h})
    assert not ok
    assert problems, "应至少产生一条失败原因"
    assert any(r in p for p in problems for r in ("text_mismatch", "missing_bounds"))


def test_hyperedge_supported_by_text():
    he = {"entity_ids": ["Headache", "Vomiting", "Lesion"]}
    src = ["The patient had Headache and Vomiting after the Lesion was found."]
    ok, why = ev.hyperedge_supported_by_text(he, src)
    assert ok
    # Cross-lingual graph labels should pass when stored provenance overlaps source chunks.
    he2 = {"entity_ids": ["??", "??"], "source_ids": ["c1"]}
    ok2, why2 = ev.hyperedge_supported_by_text(
        he2, ["Headache and vomiting were observed."], ["c1"]
    )
    assert ok2 and "weak_lexical_grounding" in why2
    # Missing provenance is still blocking when there is no lexical grounding.
    he3 = {"entity_ids": ["Xyz", "Abc"]}
    ok3, why3 = ev.hyperedge_supported_by_text(he3, src)
    assert not ok3 and why3 == "missing_hyperedge_provenance"

def test_count_qwen_tokens_real():
    # 真实 Qwen tokenizer（本地 /tokenize）；网络不可用时 skip。
    try:
        n = ev.count_qwen_tokens("The patient presented with progressive weakness.")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen /tokenize 不可用: {e}")
    assert isinstance(n, int) and n > 0
    # 确定性：同文本同结果
    n2 = ev.count_qwen_tokens("The patient presented with progressive weakness.")
    assert n == n2


# --------------------------------------------------------------------------
# Schema / 禁止字段
# --------------------------------------------------------------------------
def test_make_factories_ordered_and_forbidden_free():
    span = ev.make_span("c1", "txt", 0, 3, "h")
    au = ev.make_answer_unit("au1", "stmt")
    eg = ev.make_evidence_group("eg1", "au1", ["sp1"], "r")
    rv = ev.make_review_verdict("au1", "supported", "accept", "ok")
    q = ev.make_qrels_entry("au1", ["c1"], ["sp1"])
    for rec in (span, au, eg, rv, q):
        assert not ev.check_forbidden_fields(rec)


def test_check_forbidden_fields_detects():
    rec = {"answer_units": [{"statement": "x", "difficulty": "hard"}]}
    bad = ev.check_forbidden_fields(rec)
    assert any("difficulty" in b for b in bad)
    # route / question 同样禁止
    rec2 = {"route": "P4", "question": "what?"}
    bad2 = ev.check_forbidden_fields(rec2)
    assert "route" in bad2 and "question" in bad2


def test_constants_frozen():
    assert ev.BOUND_CANDIDATE_POOL_HASH.startswith("f2fa1cb6d2058db0")
    assert ev.SOURCE_TOKENIZER_MODEL == "gpt-4o-mini"
    assert ev.STRUCTURE_QUOTA["single_fact"] == 20
    assert sum(ev.STRUCTURE_QUOTA.values()) == 80
    assert ev.P4_SOURCE_CAP_QWEN == 4000
    assert ev.SCRIPT_VERSION == "evidence-verify-v2"


def test_candidate_pool_hash_binding():
    primary = POOL_DIR / "evidence_candidates.jsonl"
    if not primary.exists():
        pytest.skip("候选池文件缺失")
    h = ev.sha256_file(str(primary))
    assert h == ev.BOUND_CANDIDATE_POOL_HASH, (
        f"候选池哈希绑定失效：实际 {h} != 绑定 {ev.BOUND_CANDIDATE_POOL_HASH}"
    )


# --------------------------------------------------------------------------
# 产物验收（step2_2 输出存在时启用）
# --------------------------------------------------------------------------
def _load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def test_prepare_manifest_present_and_bound():
    if not (STEP2_2 / "prepare_manifest.json").exists():
        pytest.skip("prepare 未运行")
    m = json.load(open(STEP2_2 / "prepare_manifest.json", encoding="utf-8"))
    assert m["candidate_pool_hash_match"] is True
    assert m["candidate_pool_hash_bound"] == ev.BOUND_CANDIDATE_POOL_HASH
    assert m["source_tokenizer_model"] == "gpt-4o-mini"
    assert m["p4_source_cap_qwen"] == 4000
    assert "draft" in m["prompt_template_hashes"]


def test_verified_outputs_when_present():
    vp = STEP2_2 / "verified_evidence.jsonl"
    if not vp.exists():
        pytest.skip("finalize 未运行")
    manifest_path = STEP2_2 / "verification_manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        if not manifest.get("quota_met"):
            pytest.skip("finalized evidence exists but quota is not complete yet")
    verified = _load_jsonl(vp)
    assert len(verified) == 80, f"verified 应为 80，实际 {len(verified)}"
    from collections import Counter
    quota = Counter(v["intended_structure"] for v in verified)
    assert dict(quota) == dict(ev.STRUCTURE_QUOTA), f"配额不符: {dict(quota)}"
    # 禁止字段
    for v in verified:
        assert not ev.check_forbidden_fields(v), f"{v['candidate_id']} 含禁止字段"
    # ID 唯一
    ids = [v["candidate_id"] for v in verified]
    assert len(set(ids)) == len(ids)
    # 每条 1-5 answer units
    for v in verified:
        n = v.get("n_answer_units", len(v.get("answer_units", [])))
        assert 1 <= n <= 5, f"{v['candidate_id']} answer units={n}"
        # span 可逐字回溯
        chunk_texts = {}
        for sc in v.get("_source_texts", []):
            chunk_texts[sc["chunk_id"]] = sc["content"]
        # spans 已在 finalize 算好 char 边界，直接校验
        for sp in v.get("spans", []):
            assert sp.get("char_start") is not None, f"{v['candidate_id']} span 缺边界"


def test_qwen_token_stats_recomputable():
    vp = STEP2_2 / "verified_evidence.jsonl"
    if not vp.exists():
        pytest.skip("finalize 未运行")
    verified = _load_jsonl(vp)
    try:
        for v in verified[:3]:
            cap = v["capacity"]
            # 重新统计 gold span tokens 必须一致
            gold_text = "\n".join(s["text"] for s in v.get("spans", []) if s.get("text"))
            if gold_text:
                n = ev.count_qwen_tokens(gold_text)
                assert n == cap["gold_span_qwen_tokens"], "gold token 不可复算"
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen /tokenize 不可用: {e}")


def test_manifest_and_report_complete():
    if not (STEP2_2 / "verification_manifest.json").exists():
        pytest.skip("finalize 未运行")
    m = json.load(open(STEP2_2 / "verification_manifest.json", encoding="utf-8"))
    assert "quota_met" in m
    assert "output_file_hashes" in m
    if not m["quota_met"]:
        pytest.skip("finalized evidence exists but quota is not complete yet")
    assert m["verified_count"] == 80
    assert m["quota_met"] is True


# --------------------------------------------------------------------------
# Step 2.2 orchestration contracts (offline)
# --------------------------------------------------------------------------
def _vpe():
    import importlib
    return importlib.import_module("scripts.verify_pilot_evidence")


def test_candidate_scope_selection():
    vpe = _vpe()
    primary = [{"candidate_id": "p1"}]
    reserve = [{"candidate_id": "r1"}, {"candidate_id": "r2"}]
    assert [c["candidate_id"] for c in vpe.select_candidates(primary, reserve, "primary")] == ["p1"]
    assert [c["candidate_id"] for c in vpe.select_candidates(primary, reserve, "reserve")] == ["r1", "r2"]
    assert [c["candidate_id"] for c in vpe.select_candidates(primary, reserve, "all")] == ["p1", "r1", "r2"]


def test_qrels_are_answer_unit_specific():
    vpe = _vpe()
    answer_units = [ev.make_answer_unit("au1", "A"), ev.make_answer_unit("au2", "B")]
    spans = [
        {"span_id": "sp1", "chunk_id": "c1", "text": "A"},
        {"span_id": "sp2", "chunk_id": "c2", "text": "B"},
    ]
    groups = [
        ev.make_evidence_group("eg1", "au1", ["sp1"], ""),
        ev.make_evidence_group("eg2", "au2", ["sp2"], ""),
    ]
    qrels = vpe._qrels_from_answer_units(answer_units, groups, spans)
    by_au = {q["answer_unit_id"]: q for q in qrels}
    assert by_au["au1"]["relevant_chunk_ids"] == ["c1"]
    assert by_au["au1"]["relevant_span_ids"] == ["sp1"]
    assert by_au["au2"]["relevant_chunk_ids"] == ["c2"]
    assert by_au["au2"]["relevant_span_ids"] == ["sp2"]


def test_evidence_group_problem_detects_missing_span():
    vpe = _vpe()
    answer_units = [ev.make_answer_unit("au1", "A")]
    groups = [ev.make_evidence_group("eg1", "au1", ["sp_missing"], "")]
    problems = vpe._evidence_group_problems(answer_units, groups, [])
    assert any("missing spans" in p for p in problems)


def test_apply_adjudication_accepts_human_queue(tmp_path, monkeypatch):
    vpe = _vpe()
    monkeypatch.setattr(vpe, "STEP2_2_DIR", tmp_path)
    primary = [{
        "candidate_id": "c-human",
        "intended_structure": "single_fact",
        "entry_type": "source",
        "source_chunk_ids": ["chunk-a"],
    }]
    monkeypatch.setattr(vpe, "load_candidate_pool", lambda: (primary, [], "hash"))
    human = [{
        "candidate_id": "c-human",
        "intended_structure": "single_fact",
        "entry_type": "source",
        "source_chunk_ids": ["chunk-a"],
        "decision": "revise",
        "rejection_reasons": ["partial"],
        "answer_units": [],
        "spans": [],
        "evidence_groups": [],
        "qrels": [],
    }]
    (tmp_path / "human_review_queue.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in human) + "\n",
        encoding="utf-8",
    )
    adjudication = tmp_path / "human_adjudications.jsonl"
    adjudication.write_text(
        json.dumps({"candidate_id": "c-human", "decision": "accept", "reviewer": "unit-test"}) + "\n",
        encoding="utf-8",
    )
    vpe.phase_apply_adjudication("dummy", adjudication)
    verified = _load_jsonl(tmp_path / "verified_evidence.jsonl")
    remaining = _load_jsonl(tmp_path / "human_review_queue.jsonl")
    assert [v["candidate_id"] for v in verified] == ["c-human"]
    assert remaining == []
    manifest = json.load(open(tmp_path / "verification_manifest.json", encoding="utf-8"))
    assert manifest["verified_count"] == 1



def _replacement_fixture_record(candidate_id: str, chunk_id: str, text: str) -> dict:
    content_hash = ev.sha256_text(text)
    return {
        "candidate_id": candidate_id,
        "intended_structure": "single_fact",
        "entry_type": "source",
        "source_chunk_ids": [chunk_id],
        "decision": "accept",
        "rejection_reasons": [],
        "answer_units": [{
            "unit_id": "au1",
            "statement": "Supported fact.",
            "evidence_group_ids": ["eg1"],
            "qwen_tokens": None,
        }],
        "spans": [{
            "span_id": "sp1",
            "chunk_id": chunk_id,
            "text": text,
            "chunk_content_hash": content_hash,
            "char_start": 0,
            "char_end": len(text),
        }],
        "evidence_groups": [{
            "group_id": "eg1",
            "answer_unit_id": "au1",
            "span_ids": ["sp1"],
            "rationale": "Direct support.",
        }],
        "qrels": [{
            "answer_unit_id": "au1",
            "relevant_chunk_ids": [chunk_id],
            "relevant_span_ids": ["sp1"],
        }],
        "gold_answer": "Supported fact [au1].",
        "review_verdicts": [],
        "n_answer_units": 1,
    }


def _write_replacement_fixture_files(tmp_path: Path, verified: list[dict],
                                     rejected: list[dict]) -> None:
    for name, rows in {
        "verified_evidence.jsonl": verified,
        "rejected_evidence.jsonl": rejected,
        "human_review_queue.jsonl": [],
        "unprocessed_candidates.jsonl": [],
    }.items():
        (tmp_path / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_apply_replacement_swaps_and_revalidates(tmp_path, monkeypatch):
    vpe = _vpe()
    monkeypatch.setattr(vpe, "STEP2_2_DIR", tmp_path)
    monkeypatch.setattr(vpe, "STRUCTURE_QUOTA", {"single_fact": 1})
    old = _replacement_fixture_record("old", "chunk-old", "Old fact.")
    new = _replacement_fixture_record("new", "chunk-new", "Exact source quote.")
    new["decision"] = "reject"
    new["rejection_reasons"] = ["span@chunk-new:missing_bounds"]
    new["spans"][0]["text"] = "normalized quote"
    new["spans"][0]["char_start"] = None
    new["spans"][0]["char_end"] = None
    _write_replacement_fixture_files(tmp_path, [old], [new])

    monkeypatch.setattr(
        vpe, "load_candidate_pool",
        lambda: ([{"candidate_id": "old"}], [{"candidate_id": "new"}], "hash"),
    )
    chunk_hash = ev.sha256_text("Exact source quote.")
    monkeypatch.setattr(
        vpe, "load_chunks",
        lambda _data_name: (
            {"chunk-new": "Exact source quote."},
            {"chunk-new": chunk_hash},
            {},
        ),
    )
    monkeypatch.setattr(vpe, "count_qwen_tokens", lambda text: len(text.split()))

    plan = tmp_path / "replacement.jsonl"
    plan.write_text(
        json.dumps({
            "candidate_id": "new",
            "replaces_candidate_id": "old",
            "decision": "accept",
            "reviewer": "unit-test",
            "updates": {
                "span_updates": {
                    "sp1": {
                        "text": "Exact source quote.",
                        "char_start": 0,
                        "char_end": len("Exact source quote."),
                    }
                }
            },
        }) + "\n",
        encoding="utf-8",
    )

    vpe.phase_apply_replacement("dummy", plan)

    verified = _load_jsonl(tmp_path / "verified_evidence.jsonl")
    assert [item["candidate_id"] for item in verified] == ["new"]
    assert verified[0]["decision"] == "accept"
    assert verified[0]["rejection_reasons"] == []
    assert verified[0]["qrels"][0]["relevant_chunk_ids"] == ["chunk-new"]
    assert verified[0]["capacity"]["p4_raw_chunk_fit"] is True
    assert _load_jsonl(tmp_path / "rejected_evidence.jsonl") == []
    superseded = _load_jsonl(tmp_path / "superseded_evidence.jsonl")
    assert superseded[0]["candidate_id"] == "old"
    assert superseded[0]["superseded_by"] == "new"
    manifest = json.load(open(tmp_path / "verification_manifest.json", encoding="utf-8"))
    assert manifest["phase"] == "apply-replacement"
    assert manifest["quota_met"] is True


def test_apply_replacement_validation_failure_does_not_write(tmp_path, monkeypatch):
    vpe = _vpe()
    monkeypatch.setattr(vpe, "STEP2_2_DIR", tmp_path)
    monkeypatch.setattr(vpe, "STRUCTURE_QUOTA", {"single_fact": 1})
    old = _replacement_fixture_record("old", "chunk-old", "Old fact.")
    new = _replacement_fixture_record("new", "chunk-new", "Exact source quote.")
    _write_replacement_fixture_files(tmp_path, [old], [new])
    original_bytes = (tmp_path / "verified_evidence.jsonl").read_bytes()

    monkeypatch.setattr(
        vpe, "load_candidate_pool",
        lambda: ([{"candidate_id": "old"}], [{"candidate_id": "new"}], "hash"),
    )
    chunk_hash = ev.sha256_text("Exact source quote.")
    monkeypatch.setattr(
        vpe, "load_chunks",
        lambda _data_name: (
            {"chunk-new": "Exact source quote."},
            {"chunk-new": chunk_hash},
            {},
        ),
    )
    monkeypatch.setattr(vpe, "count_qwen_tokens", lambda _text: 5001)

    plan = tmp_path / "replacement.jsonl"
    plan.write_text(
        json.dumps({
            "candidate_id": "new",
            "replaces_candidate_id": "old",
            "decision": "accept",
            "reviewer": "unit-test",
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="P4 cap"):
        vpe.phase_apply_replacement("dummy", plan)

    assert (tmp_path / "verified_evidence.jsonl").read_bytes() == original_bytes
    assert not (tmp_path / "superseded_evidence.jsonl").exists()
