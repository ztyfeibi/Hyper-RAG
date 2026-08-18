"""盲审链陈旧产物检测 + resume 判据回归测试。

背景（2026-08-18 r2 六路径事故）：
preapply 从 P_gold-only(80) 用 --overwrite 重生成为六路径(480) 后，
`build-review` 的 skip 判据只看 `sample_manifest.json` 是否存在 → 被误跳过 →
`verdicts_template_D.jsonl` 仍是陈旧 80 条 → review 空转（待跑 0）→
merge 一致性校验才爆（且报错被 [:3] 截断，显示成"只差 3 题"）。

更危险的是 review_id 为序号编码（R{n}-D-001…），随 pending 集合漂移。
若静默重建模板，run_supplementary_review 按 review_id 断点续跑会把旧裁决
复用到**不同 cell** 上 → 静默数据污染。故必须 fail-closed。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_repeat_pipeline as rrp  # noqa: E402


class _FakeRC:
    """只提供 judge_dir 的最小上下文（被测函数仅用到它）。"""

    def __init__(self, judge_dir: Path):
        self.judge_dir = judge_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _mk_preapply(judge_dir: Path, no_bid: list[tuple], with_bid: list[tuple],
                 resolved: int = 0) -> None:
    """构造 preapply final_verdicts.jsonl。"""
    rows = []
    for route, qid in no_bid:
        rows.append({"route": route, "question_id": qid,
                     "route_success": "pending", "blind_id": None})
    for route, qid in with_bid:
        rows.append({"route": route, "question_id": qid,
                     "route_success": "pending", "blind_id": f"B-{route}-{qid}"})
    for i in range(resolved):
        rows.append({"route": "P0", "question_id": f"qv2-x{i:03d}",
                     "route_success": "pass", "blind_id": None})
    _write_jsonl(judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl", rows)


def _mk_manifest(judge_dir: Path, d_cells: list[tuple],
                 e_cells: list[tuple]) -> None:
    sup = judge_dir / "blind_review_supplementary"
    sup.mkdir(parents=True, exist_ok=True)
    obj = {
        "source_final_verdicts": "ai_adjudication_pre/final_verdicts.jsonl",
        "adjudication_rule_version": "v3",
        f"set_D_unreviewed_{len(d_cells)}": [
            {"review_id": f"R2-D-{i + 1:03d}", "route": rt, "question_id": qid,
             "blind_id": None}
            for i, (rt, qid) in enumerate(d_cells)
        ],
        f"set_E_recheck_{len(e_cells)}": [
            {"review_id": f"R2-E-{i + 1:03d}", "route": rt, "question_id": qid,
             "blind_id": f"B-{rt}-{qid}"}
            for i, (rt, qid) in enumerate(e_cells)
        ],
    }
    (sup / "sample_manifest.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# _pending_cells / _manifest_cells
# --------------------------------------------------------------------------

def test_pending_cells_splits_by_blind_id(tmp_path):
    _mk_preapply(tmp_path, no_bid=[("P0", "q1"), ("P1", "q2")],
                 with_bid=[("P2", "q3")], resolved=5)
    no_bid, with_bid = rrp._pending_cells(_FakeRC(tmp_path))
    assert no_bid == {("P0", "q1"), ("P1", "q2")}
    assert with_bid == {("P2", "q3")}


def test_pending_cells_missing_file_returns_empty(tmp_path):
    assert rrp._pending_cells(_FakeRC(tmp_path)) == (set(), set())


def test_manifest_cells_reads_dynamic_keys(tmp_path):
    _mk_manifest(tmp_path, d_cells=[("P0", "q1"), ("P1", "q2")],
                 e_cells=[("P2", "q3")])
    d, e = rrp._manifest_cells(_FakeRC(tmp_path))
    assert d == {("P0", "q1"), ("P1", "q2")}
    assert e == {("P2", "q3")}


# --------------------------------------------------------------------------
# _build_review_covers —— 核心事故判据
# --------------------------------------------------------------------------

def test_build_review_covers_true_when_consistent(tmp_path):
    cells = [("P0", "q1"), ("P1", "q2")]
    _mk_preapply(tmp_path, no_bid=cells, with_bid=[])
    _mk_manifest(tmp_path, d_cells=cells, e_cells=[])
    assert rrp._build_review_covers(_FakeRC(tmp_path)) is True


def test_build_review_covers_false_when_manifest_stale(tmp_path):
    """事故复现：manifest 只有 P_gold 80，preapply 已是六路径 268。"""
    pgold = [("P_gold", f"qv2-{i:04d}") for i in range(1, 81)]
    six = pgold + [("P0", f"qv2-{i:04d}") for i in range(1, 81)]
    _mk_preapply(tmp_path, no_bid=six, with_bid=[])
    _mk_manifest(tmp_path, d_cells=pgold, e_cells=[])   # 陈旧
    assert rrp._build_review_covers(_FakeRC(tmp_path)) is False


def test_build_review_covers_false_when_manifest_missing(tmp_path):
    _mk_preapply(tmp_path, no_bid=[("P0", "q1")], with_bid=[])
    assert rrp._build_review_covers(_FakeRC(tmp_path)) is False


def test_build_review_covers_distinguishes_d_and_e(tmp_path):
    """cell 总数相同但 D/E 归属不同，也必须判为不一致。"""
    _mk_preapply(tmp_path, no_bid=[("P0", "q1")], with_bid=[("P1", "q2")])
    _mk_manifest(tmp_path, d_cells=[("P0", "q1"), ("P1", "q2")], e_cells=[])
    assert rrp._build_review_covers(_FakeRC(tmp_path)) is False


# --------------------------------------------------------------------------
# _review_output_covers —— 盲审输出无 route 字段
# --------------------------------------------------------------------------

def test_review_output_covers_true_when_all_ids_done(tmp_path):
    sup = tmp_path / "blind_review_supplementary"
    _write_jsonl(sup / "verdicts_template_D.jsonl",
                 [{"review_id": "R2-D-001"}, {"review_id": "R2-D-002"}])
    _write_jsonl(sup / "verdicts_ai_supplementary_D.jsonl",
                 [{"review_id": "R2-D-001"}, {"review_id": "R2-D-002"}])
    assert rrp._review_output_covers(_FakeRC(tmp_path), "D") is True


def test_review_output_covers_false_when_partial(tmp_path):
    sup = tmp_path / "blind_review_supplementary"
    _write_jsonl(sup / "verdicts_template_D.jsonl",
                 [{"review_id": f"R2-D-{i:03d}"} for i in range(1, 6)])
    _write_jsonl(sup / "verdicts_ai_supplementary_D.jsonl",
                 [{"review_id": "R2-D-001"}])
    assert rrp._review_output_covers(_FakeRC(tmp_path), "D") is False


def test_review_output_covers_empty_template_e_is_covered(tmp_path):
    """set_E 为空（无 uncertain）时空输出即视为已覆盖 —— 保持既有行为。"""
    sup = tmp_path / "blind_review_supplementary"
    (sup).mkdir(parents=True, exist_ok=True)
    (sup / "verdicts_template_E.jsonl").write_text("", encoding="utf-8")
    (sup / "verdicts_ai_recheck_E.jsonl").write_text("", encoding="utf-8")
    assert rrp._review_output_covers(_FakeRC(tmp_path), "E") is True


def test_review_output_covers_route_field_absent_does_not_break(tmp_path):
    """回归：旧实现用 _covers_routes 对无 route 字段的输出恒 False。"""
    sup = tmp_path / "blind_review_supplementary"
    rows = [{"review_id": "R2-D-001", "verdict": "pass"}]
    _write_jsonl(sup / "verdicts_template_D.jsonl", rows)
    _write_jsonl(sup / "verdicts_ai_supplementary_D.jsonl", rows)
    assert rrp._covers_routes(_FakeRC(tmp_path), ["P0"],
                              sup / "verdicts_ai_supplementary_D.jsonl") is False
    assert rrp._review_output_covers(_FakeRC(tmp_path), "D") is True


# --------------------------------------------------------------------------
# stale_blind_review_problems —— fail-closed 守卫
# --------------------------------------------------------------------------

def test_stale_no_manifest_is_clean(tmp_path):
    _mk_preapply(tmp_path, no_bid=[("P0", "q1")], with_bid=[])
    assert rrp.stale_blind_review_problems(_FakeRC(tmp_path)) == []


def test_stale_consistent_is_clean(tmp_path):
    cells = [("P0", "q1")]
    _mk_preapply(tmp_path, no_bid=cells, with_bid=[])
    _mk_manifest(tmp_path, d_cells=cells, e_cells=[])
    _write_jsonl(tmp_path / "blind_review_supplementary"
                 / "verdicts_ai_supplementary_D.jsonl",
                 [{"review_id": "R2-D-001"}])
    assert rrp.stale_blind_review_problems(_FakeRC(tmp_path)) == []


def test_stale_mismatch_without_verdicts_allows_rebuild(tmp_path):
    """仅 manifest 陈旧、无 AI 裁决可污染 → 允许 build-review 重建。"""
    _mk_preapply(tmp_path, no_bid=[("P0", "q1"), ("P1", "q2")], with_bid=[])
    _mk_manifest(tmp_path, d_cells=[("P0", "q1")], e_cells=[])
    assert rrp.stale_blind_review_problems(_FakeRC(tmp_path)) == []


def test_stale_mismatch_with_verdicts_is_fail_closed(tmp_path):
    """事故场景：陈旧 manifest + 已有裁决 → 必须报问题，禁止静默重建。"""
    _mk_preapply(tmp_path, no_bid=[("P0", "q1"), ("P1", "q2")], with_bid=[])
    _mk_manifest(tmp_path, d_cells=[("P0", "q1")], e_cells=[])
    _write_jsonl(tmp_path / "blind_review_supplementary"
                 / "verdicts_ai_supplementary_D.jsonl",
                 [{"review_id": "R2-D-001", "verdict": "pass"}])
    problems = rrp.stale_blind_review_problems(_FakeRC(tmp_path))
    assert problems, "陈旧 manifest + 已有裁决必须 fail-closed"
    joined = " ".join(problems)
    assert "verdicts_ai_supplementary_D.jsonl" in joined
    assert "删除" in joined


def test_stale_empty_verdict_file_allows_rebuild(tmp_path):
    """0 字节裁决文件不算污染源（如 set_E 空文件）。"""
    _mk_preapply(tmp_path, no_bid=[("P0", "q1"), ("P1", "q2")], with_bid=[])
    _mk_manifest(tmp_path, d_cells=[("P0", "q1")], e_cells=[])
    sup = tmp_path / "blind_review_supplementary"
    (sup / "verdicts_ai_supplementary_D.jsonl").write_text("", encoding="utf-8")
    assert rrp.stale_blind_review_problems(_FakeRC(tmp_path)) == []


def test_stale_no_pending_defers_to_earlier_steps(tmp_path):
    """preapply 无 pending（或未产出）时不在此处报错。"""
    _mk_preapply(tmp_path, no_bid=[], with_bid=[], resolved=3)
    _mk_manifest(tmp_path, d_cells=[("P0", "q1")], e_cells=[])
    assert rrp.stale_blind_review_problems(_FakeRC(tmp_path)) == []


# --------------------------------------------------------------------------
# _should_skip 分派
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["build-review", "review", "recheck"])
def test_should_skip_blind_steps_not_route_aware(name, tmp_path):
    """回归：这三步不得走 _covers_routes（其产物无 route 字段/非 JSONL）。"""
    assert name not in rrp._ROUTE_AWARE_COMBINED


def test_should_skip_build_review_uses_pending_consistency(tmp_path):
    sup = tmp_path / "blind_review_supplementary"
    marker = sup / "sample_manifest.json"
    step = rrp.Step("build-review", [], marker=marker)
    pgold = [("P_gold", f"qv2-{i:04d}") for i in range(1, 81)]
    _mk_preapply(tmp_path, no_bid=pgold + [("P0", "q1")], with_bid=[])
    _mk_manifest(tmp_path, d_cells=pgold, e_cells=[])   # 陈旧
    rc = _FakeRC(tmp_path)
    assert marker.exists(), "marker 存在但仍不得跳过"
    assert rrp._should_skip(step, rc, ["P0", "P_gold"]) is False


def test_should_skip_review_uses_review_ids(tmp_path):
    sup = tmp_path / "blind_review_supplementary"
    out = sup / "verdicts_ai_supplementary_D.jsonl"
    step = rrp.Step("review", [], marker=out)
    _write_jsonl(sup / "verdicts_template_D.jsonl",
                 [{"review_id": "R2-D-001"}, {"review_id": "R2-D-002"}])
    _write_jsonl(out, [{"review_id": "R2-D-001"}])
    rc = _FakeRC(tmp_path)
    assert rrp._should_skip(step, rc, ["P0"]) is False       # 模板未跑完
    _write_jsonl(out, [{"review_id": "R2-D-001"}, {"review_id": "R2-D-002"}])
    assert rrp._should_skip(step, rc, ["P0"]) is True
