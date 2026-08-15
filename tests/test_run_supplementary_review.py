# -*- coding: utf-8 -*-
"""run_supplementary_review.py 离线测试（mock LLM，真实 Step 1 材料）。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_supplementary_review as rsr  # noqa: E402


@pytest.fixture(scope="module")
def materials():
    d_md = rsr.SUP_DIR / "set_D_unreviewed_155.md"
    e_md = rsr.SUP_DIR / "set_E_recheck_12.md"
    return {
        "d_sections": rsr.split_md_sections(d_md.read_text(encoding="utf-8")),
        "e_sections": rsr.split_md_sections(e_md.read_text(encoding="utf-8")),
        "d_tpl": rsr.load_templates("D"),
        "e_tpl": rsr.load_templates("E"),
    }


# ---------------------------------------------------------------------------
# 材料解析
# ---------------------------------------------------------------------------

class TestMaterialParsing:
    def test_sections_match_templates(self, materials):
        assert len(materials["d_sections"]) == 155
        assert len(materials["e_sections"]) == 12
        assert set(materials["d_sections"]) == {t["review_id"] for t in materials["d_tpl"]}
        assert set(materials["e_sections"]) == {t["review_id"] for t in materials["e_tpl"]}

    def test_sections_contain_own_review_id_header(self, materials):
        """回归锁定：分节保留 '### SR-xxx' 头行（模型需要据此返回 review_id）。"""
        for secs in (materials["d_sections"], materials["e_sections"]):
            for rid, text in secs.items():
                assert text.startswith(f"### {rid}\n"), rid

    def test_extract_au_ids(self, materials):
        for tpl in materials["d_tpl"][:20] + materials["e_tpl"]:
            au = rsr.extract_au_ids(materials["d_sections" if tpl["review_id"].startswith("SR-D") else "e_sections"][tpl["review_id"]])
            assert au and au == sorted(au, key=lambda x: int(x[2:]))
            assert len(au) >= 1

    def test_sections_no_leak(self, materials):
        import re
        pats = [r"qv2-\d", r"\bP_gold\b", r"\bP[0-4]\b", r"caches[/\\]"]
        for secs in (materials["d_sections"], materials["e_sections"]):
            for rid, text in secs.items():
                for p in pats:
                    assert not re.search(p, text), f"{rid} 泄漏 {p}"


# ---------------------------------------------------------------------------
# 记录校验
# ---------------------------------------------------------------------------

def _valid_record(tpl, au_ids):
    prefilled = tpl["unsupported_claims"]
    return {
        "review_id": tpl["review_id"],
        "au_status": {a: "supported" for a in au_ids},
        "verdict": "pass",
        "unsupported_fatality": "none",
        "claim_status_by_id": {f"C{i}": "unverifiable"
                               for i in range(1, len(prefilled) + 1)},
        "additional_claims": [],
        "notes": "",
    }


def _cids(tpl):
    return [f"C{i}" for i in range(1, len(tpl["unsupported_claims"]) + 1)]


class TestNormalizeRecord:
    def test_normalize_maps_ids_to_original_text(self, materials):
        tpl = next(t for t in materials["d_tpl"] if t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        raw = _valid_record(tpl, au)
        raw["claim_status_by_id"]["C1"] = "contradicted"
        raw["additional_claims"] = [{"claim": "extra stmt", "status": "unverifiable"}]
        out = rsr.normalize_record(raw, tpl)
        claims = out["unsupported_claims"]
        assert claims[0]["claim"] == tpl["unsupported_claims"][0]["claim"]
        assert claims[0]["status"] == "contradicted"
        assert claims[0]["claim_id"] == "C1" and claims[0]["prefilled"] is True
        assert claims[-1] == {"claim": "extra stmt", "status": "unverifiable",
                              "claim_id": None, "prefilled": False}
        assert out["response_schema"] == "claim_id_v2"
        assert out["review_id"] == tpl["review_id"]

    def test_normalize_zero_claims(self, materials):
        tpl = next(t for t in materials["d_tpl"] if not t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        out = rsr.normalize_record(_valid_record(tpl, au), tpl)
        assert out["unsupported_claims"] == []


def _cids(tpl):
    return [f"C{i}" for i in range(1, len(tpl["unsupported_claims"]) + 1)]


class TestNormalizeRecord:
    def test_normalize_maps_ids_to_original_text(self, materials):
        tpl = next(t for t in materials["d_tpl"] if t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        raw = _valid_record(tpl, au)
        raw["claim_status_by_id"]["C1"] = "contradicted"
        raw["additional_claims"] = [{"claim": "extra stmt", "status": "unverifiable"}]
        out = rsr.normalize_record(raw, tpl)
        claims = out["unsupported_claims"]
        assert claims[0]["claim"] == tpl["unsupported_claims"][0]["claim"]
        assert claims[0]["status"] == "contradicted"
        assert claims[0]["claim_id"] == "C1" and claims[0]["prefilled"] is True
        assert claims[-1] == {"claim": "extra stmt", "status": "unverifiable",
                              "claim_id": None, "prefilled": False}
        assert out["response_schema"] == "claim_id_v2"
        assert out["review_id"] == tpl["review_id"]

    def test_normalize_zero_claims(self, materials):
        tpl = next(t for t in materials["d_tpl"] if not t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        out = rsr.normalize_record(_valid_record(tpl, au), tpl)
        assert out["unsupported_claims"] == []


class TestValidateRecord:
    def test_valid_passes(self, materials):
        tpl = materials["d_tpl"][0]
        sec = materials["d_sections"][tpl["review_id"]]
        au = rsr.extract_au_ids(sec)
        assert rsr.validate_record(_valid_record(tpl, au), tpl, au, _cids(tpl)) == []

    def test_review_id_mismatch(self, materials):
        tpl = materials["d_tpl"][0]
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["review_id"] = "SR-D-999"
        errs = rsr.validate_record(rec, tpl, au, _cids(tpl))
        assert any("review_id" in e for e in errs)

    def test_missing_au_rejected(self, materials):
        tpl = materials["d_tpl"][0]
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["au_status"].pop(au[0])
        errs = rsr.validate_record(rec, tpl, au, _cids(tpl))
        assert any("缺 AU" in e for e in errs)

    def test_bad_verdict_rejected(self, materials):
        tpl = materials["d_tpl"][0]
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["verdict"] = "maybe"
        assert rsr.validate_record(rec, tpl, au, _cids(tpl))

    def test_missing_prefilled_claim_id_rejected(self, materials):
        tpl = next(t for t in materials["d_tpl"] if t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["claim_status_by_id"].pop("C1")
        errs = rsr.validate_record(rec, tpl, au, _cids(tpl))
        assert any("claim_status_by_id 缺编号" in e for e in errs)

    def test_bad_claim_status_rejected(self, materials):
        tpl = next(t for t in materials["d_tpl"] if len(t["unsupported_claims"]) >= 2)
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["claim_status_by_id"]["C2"] = "supported"
        errs = rsr.validate_record(rec, tpl, au, _cids(tpl))
        assert any("claim status 非法" in e for e in errs)

    def test_no_echo_of_claim_text_required(self, materials):
        """回归锁定：claim_id schema 下模型无需抄写 claim 原文也合法。"""
        tpl = next(t for t in materials["d_tpl"] if t["unsupported_claims"])
        au = rsr.extract_au_ids(materials["d_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec.pop("unsupported_claims", None)  # 原始响应里根本没有该字段
        assert rsr.validate_record(rec, tpl, au, _cids(tpl)) == []

    def test_unresolved_fatality_accepted(self, materials):
        """契约允许 unresolved fatality（Step 2 规格四值）。"""
        tpl = materials["e_tpl"][0]
        au = rsr.extract_au_ids(materials["e_sections"][tpl["review_id"]])
        rec = _valid_record(tpl, au)
        rec["unsupported_fatality"] = "unresolved"
        assert rsr.validate_record(rec, tpl, au, _cids(tpl)) == []


# ---------------------------------------------------------------------------
# run_set：mock LLM 端到端（limit + 断点续跑 + dry-run）
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolate_output(tmp_path, monkeypatch):
    """把两个 set 的输出重定向到临时文件，保留 md/模板读真实材料。"""
    outs = {"D": tmp_path / "out_D.jsonl", "E": tmp_path / "out_E.jsonl"}
    monkeypatch.setattr(rsr, "SET_OUTPUT", outs)
    return outs


def _fake_llm_ok(materials):
    def fake(prompt, system_prompt):
        rid = next(m.group(1) for m in [__import__("re").search(r"### (SR-[DE]-\d{3})", prompt)] if m)
        pool = materials["d_tpl"] if rid.startswith("SR-D") else materials["e_tpl"]
        tpl = next(t for t in pool if t["review_id"] == rid)
        au = rsr.extract_au_ids(next(s for r, s in
                                     (list(materials["d_sections"].items()) if rid.startswith("SR-D")
                                      else list(materials["e_sections"].items()))
                                     if r == rid))
        return json.dumps(_valid_record(tpl, au), ensure_ascii=False)
    return fake


class TestRunSet:
    def test_dry_run_no_api(self, isolate_output, monkeypatch, materials):
        def boom(*a, **k):
            raise AssertionError("dry-run 不应调 LLM")
        monkeypatch.setattr(rsr, "llm_call", boom)
        stats = rsr.run_set("D", limit=3, dry_run=True)
        assert stats["todo"] > 0 and stats["done"] == 0
        assert not isolate_output["D"].exists()

    def test_run_limit_and_resume(self, isolate_output, monkeypatch, materials):
        monkeypatch.setattr(rsr, "llm_call", _fake_llm_ok(materials))
        s1 = rsr.run_set("D", limit=3)
        assert s1["done"] == 3
        rows = [json.loads(l) for l in open(isolate_output["D"], encoding="utf-8")]
        assert len(rows) == 3
        # 续跑：已完成 3 条跳过
        s2 = rsr.run_set("D", limit=2)
        assert s2["skipped"] == 3 and s2["done"] == 2
        rows = [json.loads(l) for l in open(isolate_output["D"], encoding="utf-8")]
        assert len(rows) == 5
        assert len({r["review_id"] for r in rows}) == 5

    def test_parse_retry_then_success(self, isolate_output, monkeypatch, materials):
        calls = {"n": 0}

        def flaky(prompt, system_prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not-json"
            return _fake_llm_ok(materials)(prompt, system_prompt)

        monkeypatch.setattr(rsr, "llm_call", flaky)
        stats = rsr.run_set("E", limit=1, parse_retry=3)
        assert stats["done"] == 1 and stats["parse_retries"] == 1

    def test_exhausted_retry_raises(self, isolate_output, monkeypatch, materials):
        monkeypatch.setattr(rsr, "llm_call", lambda *a: "bad")
        with pytest.raises(SystemExit):
            rsr.run_set("E", limit=1, parse_retry=2)

    def test_full_set_E_end_to_end(self, isolate_output, monkeypatch, materials):
        monkeypatch.setattr(rsr, "llm_call", _fake_llm_ok(materials))
        stats = rsr.run_set("E")
        assert stats["done"] == 12
        rows = [json.loads(l) for l in open(isolate_output["E"], encoding="utf-8")]
        assert len(rows) == 12
        assert {r["review_id"] for r in rows} == {t["review_id"] for t in materials["e_tpl"]}


class TestMetadata:
    def test_write_metadata_fields(self, tmp_path, monkeypatch, materials):
        class A:
            set = "both"
            limit = None
            parse_retry = 3
        meta_path = rsr.write_metadata({"D": {"done": 3}, "E": {"done": 1}}, A())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["review_model"]  # 当前配置 qwen-27b-int4（本地）
        assert meta["review_temperature"] == 0.0
        assert meta["review_provider"] in ("SiliconFlow", "local-vllm")
        assert "siliconflow" in rsr.LLM_BASE_URL_SILICONFLOW or \
            meta["review_provider"] == "local-vllm"
        assert len(meta["prompt_guide_sha256"]) == 64
        assert set(meta["material_md_sha256"]) == {"D", "E"}
        assert set(meta["template_sha256"]) == {"D", "E"}
        assert len(meta["script_sha256"]) == 64
        assert meta["stats"] == {"D": {"done": 3}, "E": {"done": 1}}
