"""方向 A 回归测试：parse_ok=false 作为一等公民 judge_error，且 resume 不再累积重复行。

核心修复：
- read_done_any：parse_failed cell 视为已完成（不再反复重试）。
- write_record：同 qid 替换写盘（修复 resume 追加导致 80->81 行的 bloat）。
- _judge_covers：用 read_done_any，parse_failed 路线仍判为已覆盖。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import judge_longcat as jl  # noqa: E402
import run_repeat_pipeline as rrp  # noqa: E402


def _write(out: Path, recs):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                   encoding="utf-8")


def test_read_done_any_includes_parse_failed(tmp_path):
    out = tmp_path / "P2_verdict.jsonl"
    _write(out, [
        {"question_id": "qv2-r0001", "parse_ok": True, "verdict": "pass"},
        {"question_id": "qv2-r0136", "parse_ok": False, "response": "truncated"},
    ])
    assert jl.read_done_any(out) == {"qv2-r0001", "qv2-r0136"}
    # 旧语义仍只认 parse_ok（仅用于并发写入幂等守卫）
    assert jl.read_done_ok(out) == {"qv2-r0001"}


def test_write_record_replaces_same_qid(tmp_path):
    out = tmp_path / "P2_verdict.jsonl"
    # 首次写入 parse_failed
    assert jl.write_record(out, "qv2-r0136",
                           {"question_id": "qv2-r0136", "parse_ok": False}) is True
    # resume 重跑同一题（确定性失败）：替换而非追加 -> 仍为 1 行
    assert jl.write_record(out, "qv2-r0136",
                           {"question_id": "qv2-r0136", "parse_ok": False}) is True
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1, f"resume 不应累积重复行，实际 {len(rows)} 行"
    assert rows[0]["question_id"] == "qv2-r0136"


def test_write_record_concurrent_guard_on_parse_ok(tmp_path):
    out = tmp_path / "P0_verdict.jsonl"
    assert jl.write_record(out, "qv2-r0001",
                           {"question_id": "qv2-r0001", "parse_ok": True}) is True
    # 已存在 parse_ok 记录：第二次写入被守卫拦截
    assert jl.write_record(out, "qv2-r0001",
                           {"question_id": "qv2-r0001", "parse_ok": True}) is False
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1


def test_write_record_appends_new_qid(tmp_path):
    out = tmp_path / "P0_verdict.jsonl"
    jl.write_record(out, "qv2-r0001", {"question_id": "qv2-r0001", "parse_ok": True})
    jl.write_record(out, "qv2-r0002", {"question_id": "qv2-r0002", "parse_ok": True})
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["question_id"] for r in rows} == {"qv2-r0001", "qv2-r0002"}


class _FakeRC:
    def __init__(self, mapping):
        self._m = mapping

    def verdict_file(self, route):
        # 缺失路线返回不存在的路径：_judge_covers 应据此返回 False
        return self._m.get(route, Path(f"/nonexistent/{route}_verdict.jsonl"))


def test_judge_covers_true_when_parse_failed_present(tmp_path):
    """P0 含 1 个 parse_failed，但 80 个 distinct cell -> 视为已覆盖（不再重试）。"""
    p0 = tmp_path / "P0_verdict.jsonl"
    recs = [{"question_id": f"qv2-r{i:04d}", "parse_ok": True} for i in range(1, 81)]
    recs[79] = {"question_id": "qv2-r0080", "parse_ok": False}  # 末位 cell 确定性 judge 失败
    _write(p0, recs)
    p1 = tmp_path / "P1_verdict.jsonl"
    _write(p1, [{"question_id": f"qv2-r{i:04d}", "parse_ok": True} for i in range(1, 81)])
    rc = _FakeRC({"P0": p0, "P1": p1})
    assert rrp._judge_covers(rc, ["P0", "P1"]) is True


def test_judge_covers_false_when_route_missing(tmp_path):
    p0 = tmp_path / "P0_verdict.jsonl"
    _write(p0, [{"question_id": f"qv2-r{i:04d}", "parse_ok": True} for i in range(1, 81)])
    rc = _FakeRC({"P0": p0})  # P1 文件不存在
    assert rrp._judge_covers(rc, ["P0", "P1"]) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
