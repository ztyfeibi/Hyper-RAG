# -*- coding: utf-8 -*-
"""Step 5 准入清单 + Step 6 冻结清单的验收测试（真实产物）。"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

import build_repeat0_eligibility as elig
import freeze_judge as fz


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def records():
    return _load_jsonl(elig.JUDGE_DIR / "repeat0_question_eligibility.jsonl")


@pytest.fixture(scope="module")
def summary():
    return json.loads((elig.JUDGE_DIR / "repeat0_eligibility_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def freeze():
    return json.loads((fz.JUDGE_DIR / "judge_freeze_manifest.json").read_text(encoding="utf-8"))


class TestEligibility:
    def test_80_questions_and_dist(self, records, summary):
        assert len(records) == 80
        assert summary["eligibility_dist"] == {
            "eligible": 70, "router_label_indeterminate": 9, "excluded": 1}
        assert summary["repeat12_candidate_count"] == 79

    def test_excluded_is_p_gold_fail(self, records):
        ex = [r for r in records if r["eligibility"] == "excluded"]
        assert len(ex) == 1
        assert ex[0]["route_success"]["P_gold"] == "fail"
        assert ex[0]["reason"] == "p_gold_fail"

    def test_indeterminate_have_pending_routes(self, records):
        ind = [r for r in records if r["eligibility"] == "router_label_indeterminate"]
        assert len(ind) == 9
        for r in ind:
            assert r["pending_routes"], f"{r['question_id']} 无 pending 路径却被标记 indeterminate"
            assert r["reason"].startswith("unresolved_pending_routes:")

    def test_eligible_have_no_pending_and_gold_pass(self, records):
        for r in records:
            if r["eligibility"] == "eligible":
                assert not r["pending_routes"]
                assert r["route_success"]["P_gold"] == "pass"

    def test_route_success_covers_six_routes(self, records):
        for r in records:
            assert set(r["route_success"]) == set(elig.ROUTES)

    def test_pending_count_consistent_with_v2(self, records):
        fv = _load_jsonl(elig.V2_DIR / "final_verdicts.jsonl")
        n_pending = sum(1 for r in fv if r["route_success"] == "pending")
        assert n_pending == sum(len(r["pending_routes"]) for r in records)


class TestFreeze:
    def test_judge_identity(self, freeze):
        assert freeze["judge"]["judge_model"] == "meituan-longcat/LongCat-2.0"
        assert freeze["judge"]["adjudication_rule_version"] == "v2"
        assert freeze["judge"]["coverage_rule_version"] == "v3.1"
        assert freeze["judge"]["judge_prompt_hash"]

    def test_adjudication_identity(self, freeze):
        assert freeze["adjudication"]["rule_version"] == "v3"
        assert freeze["adjudication"]["frozen_coverage_pass"] == {
            "P1": 48, "P2": 6, "P3": 24, "P4": 34, "P_gold": 80}
        assert freeze["adjudication"]["effective_version"] == "ai_adjudication_v2"

    def test_v1_hashes_match_log(self, freeze):
        log_prefixes = {
            "final_verdicts.jsonl": "1ec1afcfe317c77f",
            "final_summary.json": "642302b30bc2f782",
            "pending_supplementary.jsonl": "6fc8fa81ccdeda4a",
            "manifest.json": "44f3056c7f475ac4",
        }
        for f, prefix in log_prefixes.items():
            assert freeze["input_sha256"]["v1_outputs"][f].startswith(prefix)

    def test_supplementary_review_provenance(self, freeze):
        sr = freeze["supplementary_review"]
        assert sr["provider"] == "local-vllm"
        assert sr["model"]
        assert sr["disable_thinking"] is True

    def test_input_hashes_live(self, freeze):
        # 标注文件与准入清单哈希必须与当前盘上一致
        assert freeze["input_sha256"]["annotation_all_v1"] == fz.sha256_file(fz.ANN_ALL)
        assert freeze["input_sha256"]["eligibility"] == fz.sha256_file(fz.ELIG)

    def test_freeze_version(self, freeze):
        assert freeze["freeze_version"] == fz.FREEZE_VERSION  # v1.2（repeat-aware 流水线冻结）

    def test_guide_sha256_not_null(self, freeze):
        g = freeze["supplementary_review"]["guide_sha256"]
        assert g is not None
        assert isinstance(g, str) and len(g) == 64

    def test_guide_sha256_matches_file(self, freeze):
        # 与 run_supplementary_review 的 prompt_guide_sha256 同口径（read_text LF 归一）
        assert freeze["supplementary_review"]["guide_sha256"] == fz.sha256_guide(fz.REVIEW_GUIDE)

    def test_refuses_overwrite(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["freeze_judge.py"])
        rc = fz.main()
        assert rc == 1, "已存在的 freeze manifest 禁止静默覆盖"


class TestFreezeVerify:
    def test_verify_passes_unchanged(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["freeze_judge.py", "--verify"])
        assert fz.main() == 0
        assert "Judge freeze verification: PASS" in capsys.readouterr().out

    def test_verify_detects_tampered_input(self, monkeypatch, tmp_path, capsys):
        fake = tmp_path / "questions_tampered.jsonl"
        fake.write_text('{"tampered": true}\n', encoding="utf-8")
        monkeypatch.setattr(fz, "QUESTIONS_FILE", fake)
        monkeypatch.setattr(sys, "argv", ["freeze_judge.py", "--verify"])
        assert fz.main() == 1
        captured = capsys.readouterr()
        assert "questions_file" in (captured.out + captured.err)

    def test_verify_does_not_rewrite_manifest(self, monkeypatch):
        before = fz.sha256_file(fz.OUT)
        monkeypatch.setattr(sys, "argv", ["freeze_judge.py", "--verify"])
        assert fz.main() == 0
        assert fz.sha256_file(fz.OUT) == before, "--verify 必须只读，不得改写 manifest"

    def test_verify_mutually_exclusive_with_overwrite(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["freeze_judge.py", "--verify", "--overwrite"])
        with pytest.raises(SystemExit) as e:
            fz.main()
        assert e.value.code == 2
