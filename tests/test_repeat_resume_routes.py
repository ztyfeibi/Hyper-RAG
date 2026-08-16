"""回归测试：--resume 路线覆盖度感知（P_gold-only → 六路径扩展）。

背景（用户发现）：r1/r2 已存在 P_gold 共享产物后，用 `--resume` 从 P_gold-only 扩展到
P0-P4+P_gold 六路径时，原逻辑以“单文件存在”判 skip，导致 preapply/final-apply 被错误跳过，
最终只产出 80 条而非 6×80=480。

修复：合并裁决步骤（preapply/final-apply/review/merge）与 judge 的 resume 判据改为
“产物覆盖全部请求路线”，而非单文件存在。本文件锁定该行为：
  - 合并步骤仅覆盖部分路线时不跳过；
  - 覆盖全部路线时（含二次 resume）跳过；
  - generate[route] 每路径独立，P_gold 产物保留不重算；
  - judge 增量（per-route verdict 完整才跳过）；
  - 扩展后最终 480 条、P_gold 记录原样保留、无重复。
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import judge_longcat as jl
import scripts.run_repeat_pipeline as rrp

_RET = types.SimpleNamespace(returncode=0)
SNAPSHOT = "5c92f17c03ed41adfd4bba6926a3a784418be13c3ec6596830ef15a0868b8c67"
SIX = ["P0", "P1", "P2", "P3", "P4", "P_gold"]


class _FakeRC:
    """隔离用 RepeatContext 替身：所有路径落在 tmp，不触碰真实实验产物。"""

    def __init__(self, tmp: Path, repeat: int = 1, seed: int = 43,
                 snapshot: str = SNAPSHOT):
        self._tmp = Path(tmp)
        self.repeat = repeat
        self.seed = seed
        self.snapshot = snapshot
        self.data_name = "neurology_chunk1000"

    @property
    def judge_dir(self) -> Path:
        return self._tmp

    @property
    def supplementary_dir(self) -> Path:
        return self._tmp / "blind_review_supplementary"

    def result_file(self, route: str) -> Path:
        return self._tmp / f"fixed_{route}_r{self.repeat}_s{self.seed}_result.jsonl"

    def verdict_file(self, route: str) -> Path:
        return self._tmp / f"{route}_verdict.jsonl"

    @property
    def questions_file(self) -> Path:
        return Path(jl.QUESTIONS_FILE)

    @property
    def gold_context_file(self) -> Path:
        return Path(jl.GOLD_CTX_FILE)


# ---------------------------------------------------------------------------
# 单元：_should_skip 路线覆盖度
# ---------------------------------------------------------------------------
def test_should_skip_preapply_incomplete_when_only_pgold(tmp_path):
    rc = _FakeRC(tmp_path)
    pre = rrp.Step("preapply", [], marker=rc.judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl")
    pre.marker.parent.mkdir(parents=True, exist_ok=True)
    with open(pre.marker, "w", encoding="utf-8") as f:
        for i in range(rrp.P_GOLD_N_QUESTIONS):
            f.write(json.dumps({"route": "P_gold", "question_id": f"q{i:03d}"}) + "\n")
    assert rrp._should_skip(pre, rc, SIX) is False, "仅 P_gold 时 preapply 不应跳过"


def test_should_skip_preapply_complete_when_six(tmp_path):
    rc = _FakeRC(tmp_path)
    pre = rrp.Step("preapply", [], marker=rc.judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl")
    pre.marker.parent.mkdir(parents=True, exist_ok=True)
    with open(pre.marker, "w", encoding="utf-8") as f:
        for r in SIX:
            for i in range(rrp.P_GOLD_N_QUESTIONS):
                f.write(json.dumps({"route": r, "question_id": f"q{i:03d}"}) + "\n")
    assert rrp._should_skip(pre, rc, SIX) is True, "六路径完整时 preapply 应跳过"


def test_should_skip_final_apply_route_aware(tmp_path):
    rc = _FakeRC(tmp_path)
    fin = rrp.Step("final-apply", [], marker=rc.judge_dir / "ai_adjudication_v2" / "final_verdicts.jsonl")
    fin.marker.parent.mkdir(parents=True, exist_ok=True)
    with open(fin.marker, "w", encoding="utf-8") as f:
        for i in range(rrp.P_GOLD_N_QUESTIONS):
            f.write(json.dumps({"route": "P_gold", "question_id": f"q{i:03d}"}) + "\n")
    assert rrp._should_skip(fin, rc, SIX) is False, "仅 P_gold 时 final-apply 不应跳过"
    with open(fin.marker, "a", encoding="utf-8") as f:
        for r in ["P0", "P1", "P2", "P3", "P4"]:
            for i in range(rrp.P_GOLD_N_QUESTIONS):
                f.write(json.dumps({"route": r, "question_id": f"q{i:03d}"}) + "\n")
    assert rrp._should_skip(fin, rc, SIX) is True, "补齐六路径后 final-apply 应跳过"


def test_should_skip_judge_route_aware(tmp_path):
    rc = _FakeRC(tmp_path)
    jd = rrp.Step("judge", [], marker=rc.verdict_file("P0"))
    with open(rc.verdict_file("P_gold"), "w", encoding="utf-8") as f:
        for i in range(rrp.P_GOLD_N_QUESTIONS):
            f.write(json.dumps({"question_id": f"q{i:03d}", "parse_ok": True}) + "\n")
    assert rrp._should_skip(jd, rc, SIX) is False, "仅 P_gold verdict 时 judge 不应跳过"
    for r in ["P0", "P1", "P2", "P3", "P4"]:
        with open(rc.verdict_file(r), "w", encoding="utf-8") as f:
            for i in range(rrp.P_GOLD_N_QUESTIONS):
                f.write(json.dumps({"question_id": f"q{i:03d}", "parse_ok": True}) + "\n")
    assert rrp._should_skip(jd, rc, SIX) is True, "六路径 verdict 完整时 judge 应跳过"


def test_should_skip_generate_per_route_preserved(tmp_path):
    rc = _FakeRC(tmp_path)
    gen_pg = rrp.Step("generate[P_gold]", [], marker=rc.result_file("P_gold"))
    gen_pg.marker.parent.mkdir(parents=True, exist_ok=True)
    gen_pg.marker.write_text(json.dumps({"question_id": "q000"}) + "\n")
    assert rrp._should_skip(gen_pg, rc, SIX) is True, "generate[P_gold] 产物存在应跳过（保留）"
    gen_p0 = rrp.Step("generate[P0]", [], marker=rc.result_file("P0"))
    assert rrp._should_skip(gen_p0, rc, SIX) is False, "generate[P0] 产物缺失应运行"


# ---------------------------------------------------------------------------
# 集成：P_gold-only → 六路径扩展 → 二次 resume 全跳过
# ---------------------------------------------------------------------------
def _simulate_complete(step: rrp.Step, rc: _FakeRC, routes: list[str]) -> None:
    """模拟某步骤成功执行：按其性质写出产物（judge 增量，仅补缺失路线）。"""
    if step.name == "judge":
        for r in routes:
            vf = rc.verdict_file(r)
            if not vf.exists():
                vf.parent.mkdir(parents=True, exist_ok=True)
                with open(vf, "w", encoding="utf-8") as f:
                    for i in range(rrp.P_GOLD_N_QUESTIONS):
                        f.write(json.dumps({"question_id": f"q{i:03d}", "parse_ok": True}) + "\n")
        return
    if step.name and step.name.startswith("generate"):
        cmd = step.cmd
        ri = cmd.index("--fixed-route")
        r = cmd[ri + 1]
        pf = rc.result_file(r)
        pf.parent.mkdir(parents=True, exist_ok=True)
        with open(pf, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "q000", "result": f"ans-{r}"}) + "\n")
        return
    m = step.marker
    if m is not None:
        m.parent.mkdir(parents=True, exist_ok=True)
        with open(m, "w", encoding="utf-8") as f:
            for r in routes:
                for i in range(rrp.P_GOLD_N_QUESTIONS):
                    f.write(json.dumps({"route": r, "question_id": f"q{i:03d}",
                                        "verdict": "pass"}) + "\n")
        # 写 final_summary.json 供 verify 校验（仅 final-apply 的 v2 目录被读取）
        if step.name == "final-apply":
            summary = {
                "total_records": rrp.P_GOLD_N_QUESTIONS * len(routes),
                "repeat": rc.repeat,
                "route_success_dist": {"pass": rrp.P_GOLD_N_QUESTIONS * len(routes)},
                "P_gold_route_success": ({"pass": rrp.P_GOLD_N_QUESTIONS}
                                          if "P_gold" in routes else {}),
                "excluded_count": 0,
                "exclusion_reason_dist": {},
            }
            (m.parent / "final_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False), encoding="utf-8")


def _make_runner(steps, rc, routes, calls):
    cmd_to_step = {tuple(s.cmd): s for s in steps}
    def runner(cmd, cwd=None):
        calls.append(tuple(cmd))
        s = cmd_to_step.get(tuple(cmd))
        if s is not None:
            _simulate_complete(s, rc, routes)
        return _RET
    return runner


def test_resume_extension_pgold_to_six(tmp_path):
    rc = _FakeRC(tmp_path)

    # Phase A: P_gold-only 全量运行（无 resume），产生 P_gold 共享产物
    steps_a = rrp.build_steps(rc, ["P_gold"], rrp.QUESTIONS_SHA256)
    calls_a: list = []
    rrp.run_pipeline(steps_a, rc, ["P_gold"], phase="all",
                     dry_run=False, resume=False,
                     runner=_make_runner(steps_a, rc, ["P_gold"], calls_a))
    assert calls_a, "Phase A 应实际执行步骤"

    # Phase B: 六路径 + --resume（扩展）
    steps_b = rrp.build_steps(rc, SIX, rrp.QUESTIONS_SHA256)
    calls_b: list = []
    rrp.run_pipeline(steps_b, rc, SIX, phase="all",
                     dry_run=False, resume=True,
                     runner=_make_runner(steps_b, rc, SIX, calls_b))
    b_cmds = set(calls_b)

    # P_gold 产物保留：generate[P_gold] 必须跳过（不重算）
    gen_pg = [s for s in steps_b if s.name == "generate[P_gold]"][0]
    assert tuple(gen_pg.cmd) not in b_cmds, "generate[P_gold] 应被跳过（保留已有回答）"

    # P0-P4 必须新生成
    for r in ["P0", "P1", "P2", "P3", "P4"]:
        gen = [s for s in steps_b if s.name == f"generate[{r}]"][0]
        assert tuple(gen.cmd) in b_cmds, f"generate[{r}] 应运行"

    # 合并裁决必须重跑以覆盖六路径
    pre = [s for s in steps_b if s.name == "preapply"][0]
    fin = [s for s in steps_b if s.name == "final-apply"][0]
    jd = [s for s in steps_b if s.name == "judge"][0]
    assert tuple(pre.cmd) in b_cmds, "preapply 应重跑（覆盖六路径）"
    assert tuple(fin.cmd) in b_cmds, "final-apply 应重跑（覆盖六路径）"
    assert tuple(jd.cmd) in b_cmds, "judge 应运行（补齐 P0-P4）"

    # 保留校验：P_gold 回答文件未被覆盖，P_gold verdict 仍在
    assert rc.result_file("P_gold").read_text(encoding="utf-8").count("ans-P_gold") == 1
    assert rc.verdict_file("P_gold").exists()

    # Phase C: 六路径 + --resume 再次运行 → 工作步骤全部跳过（preflight 门禁每次都跑）
    calls_c: list = []
    rrp.run_pipeline(steps_b, rc, SIX, phase="all",
                     dry_run=False, resume=True,
                     runner=_make_runner(steps_b, rc, SIX, calls_c))
    work_cmds = {tuple(s.cmd) for s in steps_b if s.name != "preflight"}
    ran_work = [c for c in calls_c if c in work_cmds]
    assert not ran_work, f"二次 resume 应全跳过工作步骤，却执行了 {len(ran_work)} 步: {[c[:3] for c in ran_work]}"


# ---------------------------------------------------------------------------
# 数据级：扩展后 480 条、P_gold 原样保留、无重复
# ---------------------------------------------------------------------------
def test_pgold_records_preserved_no_duplication(tmp_path):
    rc = _FakeRC(tmp_path)
    pre = rrp.Step("preapply", [], marker=rc.judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl")
    pre.marker.parent.mkdir(parents=True, exist_ok=True)
    pgold_records = [{"route": "P_gold", "question_id": f"q{i:03d}",
                      "verdict": "pass", "note": "pgold-unique"}
                     for i in range(rrp.P_GOLD_N_QUESTIONS)]
    with open(pre.marker, "w", encoding="utf-8") as f:
        for r in pgold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 模拟扩展：六路径最终产物复用 P_gold 记录原样 + 新增 P0-P4
    six_file = rc.judge_dir / "ai_adjudication_v2" / "final_verdicts.jsonl"
    six_file.parent.mkdir(parents=True, exist_ok=True)
    with open(six_file, "w", encoding="utf-8") as f:
        for r in pgold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in SIX:
            if r == "P_gold":
                continue
            for i in range(rrp.P_GOLD_N_QUESTIONS):
                f.write(json.dumps({"route": r, "question_id": f"q{i:03d}",
                                    "verdict": "pass"}, ensure_ascii=False) + "\n")

    assert rrp._covers_routes(rc, SIX, six_file), "six 文件应覆盖全部路线"
    rows = [json.loads(l) for l in six_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = [(r["route"], r["question_id"]) for r in rows]
    assert len(rows) == 80 * 6, f"应为 480 条，实际 {len(rows)}"
    assert len(keys) == len(set(keys)), "存在重复 (route, question_id)"
    pgold_in_six = [r for r in rows if r["route"] == "P_gold"]
    assert all(r.get("note") == "pgold-unique" for r in pgold_in_six), \
        "P_gold 记录应原样保留（无覆盖/无重复）"
