# -*- coding: utf-8 -*-
"""test_repeat_pipeline.py — repeat-aware 总控流水线测试（合成 fixture，零外部依赖）。

覆盖：
  1.  目录隔离：r1 路径与 r0 冻结目录完全分离
  2.  review_id 编号：r0=SR-D-001 / r1=R1-D-001
  3.  repeat=0 拒绝执行
  4.  --dry-run 零执行零写盘
  5.  preflight freeze 失败 → 流水线中止（后续步骤不执行）
  6.  preflight 失败零写盘
  7.  P_gold 禁止 --max-questions + 强制 gold context/cache/trace 参数
  8.  P_gold 80 题 gold context 门禁
  9.  题集 SHA 漂移门禁
  10. 11 步固定顺序
  11. --phase 单阶段过滤
  12. --resume 跳过已完成步骤
  13. r1 merge 不读 r0 原 163 标注（目录隔离 + base_annotation none）
  14. r1 merge E 覆盖 D 不重复计数
  15. verify 验收：条数/路由/P_gold 检查
  16. calibrate --no-annotations 与 --annotation-file 互斥
"""
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_repeat_pipeline as rrp  # noqa: E402
from repeat_context import RepeatContext  # noqa: E402

SNAP = "5c92f17c03ed41adfd4bba6926a3a784418be13c3ec6596830ef15a0868b8c67"
QUESTIONS_SHA = "b1067566df3303197e991d56873b9406ad860e15f4c25ec67852a3939a8a7ed2"


class FakeProc:
    def __init__(self, returncode=0):
        self.returncode = returncode


def _rc(repeat=1, seed=43):
    return RepeatContext(repeat=repeat, seed=seed, snapshot=SNAP)


# ---------------------------------------------------------------------------
# 1. 目录隔离
# ---------------------------------------------------------------------------
def test_dir_isolation_r1_vs_r0():
    rc1 = _rc()
    rc0 = RepeatContext(repeat=0, seed=42, snapshot=SNAP)
    assert rc1.judge_dir != rc0.judge_dir
    assert rc1.judge_dir.name == f"r1_s43_{SNAP[:8]}"
    assert rc0.judge_dir.name == f"r0_s42_{SNAP[:8]}"
    assert rc1.supplementary_dir != rc0.supplementary_dir
    assert rc1.judge_dir not in rc0.judge_dir.parents
    # 结果文件名含 repeat/seed，不会覆盖 r0
    assert "r1_s43" in rc1.result_file("P_gold").name
    assert "r0_s42" in rc0.result_file("P_gold").name


# ---------------------------------------------------------------------------
# 2. review_id 编号
# ---------------------------------------------------------------------------
def test_review_id_format():
    assert _rc(0, 42).review_id("D", 1) == "SR-D-001"
    assert _rc(1, 43).review_id("D", 1) == "R1-D-001"
    assert _rc(2, 44).review_id("E", 12) == "R2-E-012"


# ---------------------------------------------------------------------------
# 3. repeat=0 拒绝
# ---------------------------------------------------------------------------
def test_refuses_repeat0(tmp_path):
    rc = _rc(0, 42)
    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    problems = rrp.preflight_problems(rc, ["P_gold"], QUESTIONS_SHA)
    assert any("repeat=0" in p or "repeat=0 不允许" in p for p in problems)
    rc2 = _rc(-1, 43)
    problems2 = rrp.preflight_problems(rc2, ["P_gold"], QUESTIONS_SHA)
    assert problems2, "repeat<1 必须拒绝"


# ---------------------------------------------------------------------------
# 4. --dry-run 零执行
# ---------------------------------------------------------------------------
def test_dry_run_no_execution(tmp_path):
    rc = _rc()
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return FakeProc(0)

    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rc_code = rrp.run_pipeline(steps, rc, ["P_gold"], phase="generate",
                               dry_run=True, runner=fake_run)
    assert rc_code == 0
    assert calls == [], "dry-run 不得执行任何子进程"


def test_dry_run_no_write(tmp_path, monkeypatch):
    """dry-run + preflight：不写任何文件（题集缺失也会先被门禁拦住）。"""
    rc = _rc()
    # questions_file 指向 tmp（不存在）→ 门禁失败，返回 1，且零写盘
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: tmp_path / "no_such.jsonl"))
    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rc_code = rrp.run_pipeline(steps, rc, ["P_gold"], phase="all", dry_run=True)
    assert rc_code == 1
    assert list(tmp_path.rglob("*")) == [], "preflight 失败必须零写盘"


# ---------------------------------------------------------------------------
# 5. preflight freeze 失败 → 中止
# ---------------------------------------------------------------------------
def test_preflight_freeze_fail_stops_pipeline(tmp_path, monkeypatch):
    rc = _rc()
    # 题集/gold 门禁先伪造通过
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: qfile))
    monkeypatch.setattr(rrp, "sha256_file", lambda p: QUESTIONS_SHA)
    gfile = tmp_path / "gold.jsonl"
    gfile.write_text("\n".join(["{}"] * 80) + "\n", encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "gold_context_file",
                        property(lambda self: gfile))

    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd[1] if len(cmd) > 1 else str(cmd))
        # freeze verify 失败
        if "freeze_judge.py" in str(cmd):
            return FakeProc(1)
        return FakeProc(0)

    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rc_code = rrp.run_pipeline(steps, rc, ["P_gold"], phase="all", runner=fake_run)
    assert rc_code == 1
    # freeze 失败后不得继续任何步骤
    assert len(executed) == 1, f"fail-fast 失败：实际执行了 {executed}"


# ---------------------------------------------------------------------------
# 6. preflight 失败零写盘（题集哈希漂移）
# ---------------------------------------------------------------------------
def test_preflight_sha_mismatch_zero_write(tmp_path, monkeypatch):
    rc = _rc()
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: qfile))
    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rc_code = rrp.run_pipeline(steps, rc, ["P_gold"], phase="all", dry_run=False)
    assert rc_code == 1
    assert list(tmp_path.parent.rglob("r1_s43*")) == [], "门禁失败不得创建 repeat 目录"


# ---------------------------------------------------------------------------
# 7. P_gold 强约束
# ---------------------------------------------------------------------------
def test_pgold_command_constraints():
    rc = _rc()
    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    gen = [s for s in steps if s.name == "generate[P_gold]"][0]
    joined = " ".join(gen.cmd)
    assert "--max-questions" not in joined, "P_gold 禁止 --max-questions"
    assert "--gold-context-file" in joined
    assert "--disable-llm-cache" in joined
    assert "--save-trace" in joined
    assert "--validate-trace" in joined
    assert "--expected-question-sha256" in joined
    assert "--question-file-path" in joined, "必须显式锁定题集文件，否则 Step_3 回退到 2_stage.json"
    assert str(rrp.jl.QUESTIONS_FILE) in joined, "题集文件路径必须是冻结的 questions_v2_manual_final.jsonl"
    assert "--data-name" in joined, "必须显式锁定 data_name，否则 Step_3 回退默认 neurology（chunk=2400 旧超图）"
    assert rc.data_name in joined, f"data_name 必须是规范 chunk=1000 建库 {rc.data_name}"
    assert "--debug-relax-constraints" not in joined
    assert f"--repeat-id 1" in joined and "--seed 43" in joined
    assert SNAP in joined


def test_non_pgold_no_gold_context_arg():
    rc = _rc()
    steps = rrp.build_steps(rc, ["P0", "P_gold"], QUESTIONS_SHA)
    gen_p0 = [s for s in steps if s.name == "generate[P0]"][0]
    assert "--gold-context-file" not in " ".join(gen_p0.cmd)
    # 非 P_gold 路由也必须显式锁定题集文件（共用 80 题冻结题集）
    assert "--question-file-path" in " ".join(gen_p0.cmd)
    # 非 P_gold 路由同样必须锁定 data_name（否则用错 chunk=2400 超图）
    assert "--data-name" in " ".join(gen_p0.cmd)
    assert rc.data_name in " ".join(gen_p0.cmd)


# ---------------------------------------------------------------------------
# 8. P_gold 80 题门禁
# ---------------------------------------------------------------------------
def test_pgold_80_questions_gate(tmp_path, monkeypatch):
    rc = _rc()
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: qfile))
    monkeypatch.setattr(rrp, "sha256_file", lambda p: QUESTIONS_SHA)
    for n, expect_fail in [(79, True), (80, False), (81, True)]:
        gfile = tmp_path / "gold.jsonl"
        gfile.write_text("\n".join(["{}"] * n) + "\n", encoding="utf-8")
        monkeypatch.setattr(RepeatContext, "gold_context_file",
                            property(lambda self, g=gfile: g))
        problems = rrp.preflight_problems(rc, ["P_gold"], QUESTIONS_SHA)
        if expect_fail:
            assert any("gold context 条数" in p for p in problems), f"n={n} 应拒绝"
        else:
            assert not any("gold context" in p for p in problems), f"n={n} 应通过"


def test_pgold_missing_gold_file(tmp_path, monkeypatch):
    rc = _rc()
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: tmp_path / "nope.jsonl"))
    monkeypatch.setattr(RepeatContext, "gold_context_file",
                        property(lambda self: tmp_path / "no_gold.jsonl"))
    problems = rrp.preflight_problems(rc, ["P_gold"], QUESTIONS_SHA)
    assert any("gold context 文件" in p or "题集文件缺失" in p for p in problems)


# ---------------------------------------------------------------------------
# 9. 题集 SHA 漂移门禁
# ---------------------------------------------------------------------------
def test_questions_sha_mismatch_gate(tmp_path, monkeypatch):
    rc = _rc()
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("changed content\n", encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "questions_file",
                        property(lambda self: qfile))
    problems = rrp.preflight_problems(rc, ["P_gold"], QUESTIONS_SHA)
    assert any("SHA-256 漂移" in p for p in problems)


# ---------------------------------------------------------------------------
# 10. 11 步固定顺序
# ---------------------------------------------------------------------------
def test_step_order():
    rc = _rc()
    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    names = [s.name.split("[")[0] for s in steps]
    assert names == rrp.STEP_ORDER
    assert len(names) == 11


# ---------------------------------------------------------------------------
# 11. --phase 过滤
# ---------------------------------------------------------------------------
def test_phase_filtering():
    rc = _rc()
    steps = rrp.build_steps(rc, ["P0", "P_gold"], QUESTIONS_SHA)
    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd)
        return FakeProc(0)

    rc_code = rrp.run_pipeline(steps, rc, ["P0", "P_gold"], phase="judge",
                               runner=fake_run)
    assert rc_code == 0
    assert len(executed) == 1
    assert "judge_longcat.py" in str(executed[0])
    assert "--routes" in executed[0] and "P0" in executed[0] and "P_gold" in executed[0]


def test_phase_generate_multi_route():
    rc = _rc()
    steps = rrp.build_steps(rc, ["P0", "P_gold"], QUESTIONS_SHA)
    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd)
        return FakeProc(0)

    rrp.run_pipeline(steps, rc, ["P0", "P_gold"], phase="generate",
                     dry_run=True, runner=fake_run)
    assert executed == []


# ---------------------------------------------------------------------------
# 12. --resume 跳过
# ---------------------------------------------------------------------------
def test_resume_skips_completed(tmp_path, monkeypatch):
    rc = _rc()
    monkeypatch.setattr(RepeatContext, "result_file",
                        lambda self, route: tmp_path / f"fixed_{route}_result.jsonl")
    (tmp_path / "fixed_P_gold_result.jsonl").write_text("done\n", encoding="utf-8")
    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd)
        return FakeProc(0)

    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rc_code = rrp.run_pipeline(steps, rc, ["P_gold"], phase="generate",
                               resume=True, runner=fake_run)
    assert rc_code == 0
    assert executed == [], "产物已存在的 generate 步骤应被 resume 跳过"


def test_resume_without_marker_runs(tmp_path, monkeypatch):
    rc = _rc()
    monkeypatch.setattr(RepeatContext, "result_file",
                        lambda self, route: tmp_path / f"fixed_{route}_result.jsonl")
    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd)
        return FakeProc(0)

    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rrp.run_pipeline(steps, rc, ["P_gold"], phase="generate",
                     resume=True, runner=fake_run)
    assert len(executed) == 1


# ---------------------------------------------------------------------------
# 13/14. r1 merge 不读 r0 标注 + E 覆盖 D
# ---------------------------------------------------------------------------
def _r1_sup_fixture(tmp_path):
    """r1 supplementary 目录合成 fixture：3 条 D + 1 条 E（E 复核 D 第 2 条）。"""
    sup = tmp_path / "sup"
    sup.mkdir()
    sm = {
        "set_D_unreviewed_3": [
            {"review_id": f"R1-D-00{i}", "route": "P_gold",
             "question_id": f"qv2-000{i}", "blind_id": None}
            for i in (1, 2, 3)],
        "set_E_recheck_1": [
            {"review_id": "R1-E-001", "route": "P_gold",
             "question_id": "qv2-0002", "blind_id": None}],
    }
    (sup / "sample_manifest.json").write_text(json.dumps(sm), encoding="utf-8")

    def rec(rid, verdict="fail", fatality="fatal"):
        return {"review_id": rid, "au_status": {"AU1": "supported"},
                "verdict": verdict, "unsupported_fatality": fatality,
                "unsupported_claims": [{"claim": "x.", "status": "unverifiable"}],
                "notes": "", "response_schema": "claim_id_v2"}

    (sup / "D.jsonl").write_text(
        "".join(json.dumps(rec(f"R1-D-00{i}"), ensure_ascii=False) + "\n"
                for i in (1, 2, 3)), encoding="utf-8")
    (sup / "E.jsonl").write_text(
        json.dumps(rec("R1-E-001", verdict="pass", fatality="harmless"),
                   ensure_ascii=False) + "\n", encoding="utf-8")
    return sup


def test_merge_r1_no_r0_annotation_read(tmp_path, monkeypatch):
    """r1 merge（base none）不得触碰 r0 blind_review 标注目录。"""
    import merge_supplementary_verdicts as msv
    import run_supplementary_review as rsr
    from repeat_context import DEFAULT_RC

    sup = _r1_sup_fixture(tmp_path)
    fv = [{"route": "P_gold", "question_id": f"qv2-000{i}", "blind_id": None,
           "route_success": "pending", "source": "longcat_only"} for i in (1, 2, 3)]
    (tmp_path / "fv.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fv), encoding="utf-8")

    rc1 = _rc()
    msv.set_context(rc1)
    monkeypatch.setattr(msv, "FINAL", tmp_path / "fv.jsonl")
    monkeypatch.setattr(msv, "SM", sup / "sample_manifest.json")
    monkeypatch.setattr(rsr, "SET_OUTPUT", {"D": sup / "D.jsonl", "E": sup / "E.jsonl"})
    monkeypatch.setattr(msv, "OUT", sup / "verdicts_ai_all_r1.jsonl")

    opened = []
    real_open = Path.read_text

    def spy_open(self, *a, **k):
        opened.append(str(self))
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy_open)
    manifest = msv.merge(write=False)
    monkeypatch.setattr(Path, "read_text", real_open)

    # 读取过的文件不得包含 r0 blind_review 原标注
    r0_ann = str(DEFAULT_RC.blind_dir / "verdicts_ai_annotated.jsonl")
    assert not any("blind_review/verdicts_ai_annotated" in p for p in opened), \
        f"r1 merge 禁止读取 r0 原 163 标注: {[p for p in opened if 'annotated' in p]}"
    assert manifest["base_annotation"] == "none"
    assert manifest["stats"]["total"] == 3
    try:
        msv.set_context(DEFAULT_RC)
    finally:
        pass


def test_merge_r1_e_overrides_d_no_double_count(tmp_path, monkeypatch):
    import merge_supplementary_verdicts as msv
    import run_supplementary_review as rsr

    sup = _r1_sup_fixture(tmp_path)
    fv = [{"route": "P_gold", "question_id": f"qv2-000{i}", "blind_id": None,
           "route_success": "pending", "source": "longcat_only"} for i in (1, 2, 3)]
    (tmp_path / "fv.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fv), encoding="utf-8")

    msv.set_context(_rc())
    monkeypatch.setattr(msv, "FINAL", tmp_path / "fv.jsonl")
    monkeypatch.setattr(msv, "SM", sup / "sample_manifest.json")
    monkeypatch.setattr(rsr, "SET_OUTPUT", {"D": sup / "D.jsonl", "E": sup / "E.jsonl"})
    out = sup / "verdicts_ai_all_r1.jsonl"
    monkeypatch.setattr(msv, "OUT", out)

    manifest = msv.merge(write=True)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3, "E 覆盖 D，不得重复计数"
    assert manifest["stats"]["by_source"] == {"supplementary_D": 2, "recheck_E": 1}
    by_qid = {r["question_id"]: r for r in rows}
    e = by_qid["qv2-0002"]
    assert e["review_source"] == "recheck_E" and e["review_id"] == "R1-E-001"
    assert e["verdict"] == "pass"
    assert e["superseded"]["original_verdict"] == "fail"
    assert e["superseded"]["original_review"] == "verdicts_ai_supplementary_D.jsonl"


def test_merge_r1_rejects_original163(tmp_path, monkeypatch):
    import merge_supplementary_verdicts as msv
    msv.set_context(_rc())
    with pytest.raises(SystemExit, match="仅限 r0"):
        msv.merge(base_annotation="original163", write=False)


# ---------------------------------------------------------------------------
# 15. verify 验收
# ---------------------------------------------------------------------------
def test_verify_problems_detects_shortage(tmp_path, monkeypatch):
    rc = _rc()
    v2 = tmp_path / "ai_adjudication_v2"
    v2.mkdir()
    rows = [{"route": "P_gold", "question_id": f"qv2-000{i}"}
            for i in range(1, 40)]  # 只有 39 条
    (v2 / "final_verdicts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "judge_dir",
                        property(lambda self: tmp_path))
    problems = rrp.verify_problems(rc, ["P_gold"])
    assert any("39" in p for p in problems)
    assert any("final_summary 缺失" in p for p in problems)


def test_verify_problems_pass(tmp_path, monkeypatch):
    rc = _rc()
    v2 = tmp_path / "ai_adjudication_v2"
    v2.mkdir()
    rows = [{"route": "P_gold", "question_id": f"qv2-000{i}"}
            for i in range(1, 81)]
    (v2 / "final_verdicts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (v2 / "final_summary.json").write_text(json.dumps({
        "total_records": 80, "repeat": 1,
        "P_gold_route_success": {"pass": 30, "fail": 40, "pending": 10},
    }), encoding="utf-8")
    monkeypatch.setattr(RepeatContext, "judge_dir",
                        property(lambda self: tmp_path))
    assert rrp.verify_problems(rc, ["P_gold"]) == []


# ---------------------------------------------------------------------------
# 16. --no-annotations 与 --annotation-file 互斥
# ---------------------------------------------------------------------------
def test_no_annotations_mutex(tmp_path, monkeypatch, capsys):
    import calibrate_blind_review as cal

    ann = tmp_path / "ann.jsonl"
    ann.write_text("{}", encoding="utf-8")
    args = type("A", (), {})()
    args.annotation_file = str(ann)
    args.no_annotations = True
    rc_code = cal.cmd_apply(args)
    assert rc_code == 1
    assert "互斥" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 补充：recheck 无材料跳过
# ---------------------------------------------------------------------------
def test_recheck_skipped_without_e_material(tmp_path, monkeypatch):
    rc = _rc()
    monkeypatch.setattr(RepeatContext, "supplementary_dir",
                        property(lambda self: tmp_path / "sup_none"))
    executed = []

    def fake_run(cmd, cwd=None):
        executed.append(cmd)
        return FakeProc(0)

    steps = rrp.build_steps(rc, ["P_gold"], QUESTIONS_SHA)
    rrp.run_pipeline(steps, rc, ["P_gold"], phase="recheck", runner=fake_run)
    assert executed == [], "无 E 材料时 recheck 应跳过"
