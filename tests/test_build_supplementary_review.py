"""补充盲审集构建测试（Step 1 验收）。"""

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

bsr = importlib.import_module("build_supplementary_review")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("supplementary")
    info = bsr.run_build(out_dir=out)
    return out, info


def _read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


# ---------- 数量与唯一性 ----------

def test_set_counts(built):
    out, info = built
    assert info["set_d"] == 155
    assert info["set_e"] == 12


def test_review_ids_unique_167(built):
    out, _ = built
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    ids = [e["review_id"] for e in manifest["set_D_unreviewed_155"]] + \
          [e["review_id"] for e in manifest["set_E_recheck_12"]]
    assert len(ids) == 167
    assert len(set(ids)) == 167
    assert all(r.startswith("SR-D-") for r in ids[:155])
    assert all(r.startswith("SR-E-") for r in ids[155:])


def test_route_qid_unique(built):
    out, _ = built
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    keys = [(e["route"], e["question_id"])
            for e in manifest["set_D_unreviewed_155"] + manifest["set_E_recheck_12"]]
    assert len(set(keys)) == 167


def test_manifest_members_match_final_verdicts(built):
    """manifest 成员必须精确等于 final_verdicts 中两类 pending 记录。"""
    out, _ = built
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    fv = _read_jsonl(bsr.OUT_DIR / "ai_adjudication_v1" / "final_verdicts.jsonl")
    d_expect = {(r["route"], r["question_id"]) for r in fv
                if r.get("source") == "longcat_only" and r.get("route_success") == "pending"}
    e_expect = {(r["route"], r["question_id"]) for r in fv
                if r.get("source") == "ai_review" and r.get("route_success") == "pending"}
    d_got = {(e["route"], e["question_id"]) for e in manifest["set_D_unreviewed_155"]}
    e_got = {(e["route"], e["question_id"]) for e in manifest["set_E_recheck_12"]}
    assert d_got == d_expect and len(d_expect) == 155
    assert e_got == e_expect and len(e_expect) == 12
    # set_E 全部带原 blind_id（复核覆盖目标）
    assert all(e["blind_id"] for e in manifest["set_E_recheck_12"])
    assert all(e["blind_id"] is None for e in manifest["set_D_unreviewed_155"])


# ---------- 泄漏检查 ----------

def test_md_no_leak(built):
    out, _ = built
    pats = [re.compile(r"qv2-\d"), re.compile(r"\bP_gold\b"),
            re.compile(r"\bP[0-4]\b"),
            re.compile(r"blind_review|\.jsonl|caches[/\\]|fixed_")]
    for name in ("set_D_unreviewed_155.md", "set_E_recheck_12.md"):
        text = (out / name).read_text(encoding="utf-8")
        for p in pats:
            m = p.search(text)
            assert m is None, f"{name} 泄漏: {p.pattern!r} -> {m and m.group(0)!r}"


def test_templates_no_leak(built):
    out, _ = built
    pats = [re.compile(r"qv2-\d"), re.compile(r"\bP_gold\b"), re.compile(r"\bP[0-4]\b")]
    for name in ("verdicts_template_D.jsonl", "verdicts_template_E.jsonl"):
        text = (out / name).read_text(encoding="utf-8")
        for p in pats:
            assert p.search(text) is None, f"{name} 泄漏: {p.pattern}"


def test_review_guide_mentions_rules(built):
    out, _ = built
    guide = (out / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
    for kw in ("supported_by_source", "unsupported_noncritical", "contradicted",
               "unverifiable", "harmless", "fatal", "set_D", "set_E",
               "不得把 uncertain 强行改成 fail"):
        assert kw in guide, f"REVIEW_GUIDE 缺少 {kw}"


# ---------- 材料完整性 ----------

def test_md_entry_count_and_sections(built):
    out, _ = built
    d_text = (out / "set_D_unreviewed_155.md").read_text(encoding="utf-8")
    e_text = (out / "set_E_recheck_12.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^### SR-D-\d{3}$", d_text, re.M)) == 155
    assert len(re.findall(r"^### SR-E-\d{3}$", e_text, re.M)) == 12
    for kw in ("**Question**:", "**Answer units**", "**Evidence spans**",
               "**Candidate answer**:", "**人工判定栏**"):
        assert d_text.count(kw) == 155, f"set_D 缺 {kw}"
        assert e_text.count(kw) == 12, f"set_E 缺 {kw}"
    # set_E 必须附原审核判定（复核语义）
    assert e_text.count("**原审核判定**") == 12
    # set_D 不附原审核判定
    assert "**原审核判定**" not in d_text


def test_md_required_aus_marked(built):
    """required AU（final_verdicts.required_aus）在材料中带 * 标记。"""
    out, _ = built
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    d_text = (out / "set_D_unreviewed_155.md").read_text(encoding="utf-8")
    # 抽前 3 条验证：required_aus 中每个 unit_id 在对应条目里出现
    for e in manifest["set_D_unreviewed_155"][:3]:
        for au in e["required_aus"] or []:
            assert re.search(rf"\({re.escape(au)} \(required\)\*|\*.*{au}", d_text) or \
                   f"{au} (required)" in d_text


def test_template_d_structure_and_claims_prefilled(built):
    out, _ = built
    rows = _read_jsonl(out / "verdicts_template_D.jsonl")
    assert len(rows) == 155
    fv = _read_jsonl(bsr.OUT_DIR / "ai_adjudication_v1" / "final_verdicts.jsonl")
    lc_claims = {}
    for route in bsr.jl.ROUTES:
        for r in _read_jsonl(bsr.OUT_DIR / f"{route}_verdict.jsonl"):
            lc_claims[(route, r["question_id"])] = r.get("unsupported_claims") or []
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    n_prefilled = 0
    for row, e in zip(rows, manifest["set_D_unreviewed_155"]):
        assert row["review_id"] == e["review_id"]
        assert set(row.keys()) >= {"review_id", "au_status", "verdict",
                                   "unsupported_fatality", "unsupported_claims", "notes"}
        assert row["au_status"] == {} and row["verdict"] == ""
        expect = lc_claims[(e["route"], e["question_id"])]
        assert [c["claim"] for c in row["unsupported_claims"]] == expect
        assert all(c["status"] == "" for c in row["unsupported_claims"])
        n_prefilled += len(expect)
    assert n_prefilled > 0


def test_template_e_claims_from_original_annotation(built):
    out, _ = built
    rows = _read_jsonl(out / "verdicts_template_E.jsonl")
    assert len(rows) == 12
    manifest = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    ai_ann = {r["blind_id"]: r for r in
              _read_jsonl(bsr.OUT_DIR / "blind_review" / "verdicts_ai_annotated.jsonl")}
    for row, e in zip(rows, manifest["set_E_recheck_12"]):
        orig = ai_ann[e["blind_id"]]
        expect = [c["claim"] for c in orig.get("unsupported_claims") or []]
        assert [c["claim"] for c in row["unsupported_claims"]] == expect
        assert all(c["status"] == "" for c in row["unsupported_claims"])


def test_build_deterministic(tmp_path):
    """两次构建产物逐字节一致（无随机性、无时间戳）。"""
    a, b = tmp_path / "a", tmp_path / "b"
    bsr.run_build(out_dir=a)
    bsr.run_build(out_dir=b)
    for fa in sorted(a.iterdir()):
        fb = b / fa.name
        assert fb.exists()
        assert fa.read_bytes() == fb.read_bytes(), f"非确定性: {fa.name}"


def test_real_dir_outputs_exist():
    """默认目录（真实产物）六件套齐全且与代码约定一致。"""
    sup = bsr.SUP_DIR
    for name in ("set_D_unreviewed_155.md", "set_E_recheck_12.md",
                 "verdicts_template_D.jsonl", "verdicts_template_E.jsonl",
                 "REVIEW_GUIDE.md", "sample_manifest.json"):
        assert (sup / name).exists(), f"缺 {name}"
