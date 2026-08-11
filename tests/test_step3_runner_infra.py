# -*- coding: utf-8 -*-
"""Step_3 实验运行基础设施回归测试（2026-08-11 修复项）。

覆盖七项修复中的可自动化部分：

1. **run_identity 断点续跑** —— 同 qid 但身份不同（换 snapshot/seed/题集）
   必须判定为 stale 并清理重跑，不得误跳；同身份才算真续跑。
2. **res["trace_data"] 来源** —— 测试只验证 build 链路使用的输入语义（规范化
   trace_out），运行期行为由 test_fixed_route_executor 覆盖。
3. **result/trace 原子性** —— _write_record_pair 成对写入；resume 阶段
   _rebuild_trace_from_results 补缺行、删孤儿行，保证两文件一一对应。
4. **manifest 状态** —— _compute_run_status 的 complete/incomplete 边界；
   truncated / question_file_total 由 main 内联构造（此处验证状态函数）。
5. **verify 全量加载** —— scripts.verify_trace_completeness._load_route 返回
   全部记录（非第一条），并检测 result/trace 行数一致性。
6. **extract_queries_v2** —— JSONL / JSON dict 数组 / JSON str 数组 / 空题报错。
"""

import json

import pytest

from reproduce.Step_3_response_question import (
    extract_queries_v2,
    _run_identity,
    _record_identity,
    _load_result_records,
    _classify_existing_records,
    _rewrite_jsonl_keep,
    _rebuild_trace_from_results,
    _write_record_pair,
    _compute_run_status,
    _check_locked_question_hash,
)
from scripts.verify_trace_completeness import (
    _load_route,
    _check_route_identity,
    _check_manifest,
)


# --------------------------------------------------------------------- #
# 通用构造
# --------------------------------------------------------------------- #
def _mk_result(qid, route="P1", repeat=0, seed=42, cv="question-set-v2.1-contract-v1",
               snap="SNAP", qhash="QHASH", answer="ANSWER"):
    """构造一条符合 runner 输出结构的 result 记录（含内嵌 trace）。"""
    return {
        "question_id": qid,
        "query": f"q-{qid}",
        "result": answer,
        "route_id": route,
        "mode": "naive",
        "context": f"ctx-{qid}",
        "prompt_template": "formal_answer_response",
        "prompt_hash": "ph",
        "contract_version": cv,
        "repeat_id": repeat,
        "seed": seed,
        "seed_supported": True,
        "system_snapshot_id": snap,
        "question_file_hash": qhash,
        "gold_context_file_hash": None,
        "trace": {
            "question_id": qid,
            "run_id": f"{route}_r{repeat}_s{seed}_{qid}",
            "route_id": route,
            "seed": seed,
            "system_snapshot_id": snap,
            "answer_text": answer,
            "finish_reason": "stop",
            "retrievers": [],
            "stages": {"tokens_after_truncation": 10},
            "cost": {},
        },
    }


# --------------------------------------------------------------------- #
# extract_queries_v2
# --------------------------------------------------------------------- #
def test_extract_queries_v2_jsonl(tmp_path):
    p = tmp_path / "qs.jsonl"
    p.write_text(
        json.dumps({"question_id": "qv2-0011", "question": "q1"}) + "\n"
        + json.dumps({"question_id": "qv2-0015", "question": "q2"}) + "\n"
        + json.dumps({"question_id": "qv2-0021", "question": "q3"}) + "\n",
        encoding="utf-8",
    )
    queries, qids = extract_queries_v2(str(p))
    assert queries == ["q1", "q2", "q3"]
    assert qids == ["qv2-0011", "qv2-0015", "qv2-0021"]


def test_extract_queries_v2_json_dict_array(tmp_path):
    p = tmp_path / "qs.json"
    p.write_text(json.dumps([
        {"question_id": "a", "question": "qa"},
        {"question_id": "b", "question": "qb"},
    ]), encoding="utf-8")
    queries, qids = extract_queries_v2(str(p))
    assert queries == ["qa", "qb"]
    assert qids == ["a", "b"]


def test_extract_queries_v2_json_str_array(tmp_path):
    """旧格式（纯字符串数组）退化 qid 为整数索引。"""
    p = tmp_path / "qs.json"
    p.write_text(json.dumps(["qa", "qb"]), encoding="utf-8")
    queries, qids = extract_queries_v2(str(p))
    assert queries == ["qa", "qb"]
    assert qids == ["0", "1"]


def test_extract_queries_v2_empty_question_raises(tmp_path):
    p = tmp_path / "qs.jsonl"
    p.write_text(json.dumps({"question_id": "x", "question": ""}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="empty question"):
        extract_queries_v2(str(p))


# --------------------------------------------------------------------- #
# run_identity：六要素
# --------------------------------------------------------------------- #
def test_run_identity_distinguishes_all_six_factors():
    base = ("P1", 0, 42, "cv", "snap", "hash")
    identity = _run_identity(*base)
    for i in range(6):
        variant = list(base)
        variant[i] = f"CHANGED-{i}"
        assert _run_identity(*variant) != identity, f"factor {i} 应改变身份"


def test_record_identity_roundtrip():
    rec = _mk_result("qv2-0011")
    assert _record_identity(rec) == _run_identity(
        "P1", 0, 42, rec["contract_version"], "SNAP", "QHASH")


def test_record_identity_none_when_fields_missing():
    rec = _mk_result("q1")
    rec.pop("seed")
    assert _record_identity(rec) is None


# --------------------------------------------------------------------- #
# 断点续跑分类：同身份跳过 / 异身份清理
# --------------------------------------------------------------------- #
def test_classify_same_identity_is_processed():
    rec = _mk_result("qv2-0011")
    identity = _run_identity("P1", 0, 42, rec["contract_version"], "SNAP", "QHASH")
    processed, stale = _classify_existing_records([rec], identity)
    assert processed == {"qv2-0011"}
    assert stale == set()


def test_classify_different_snapshot_is_stale():
    """同一 qid 换 snapshot -> 判定为异身份（stale），不得跳过。"""
    rec = _mk_result("qv2-0011", snap="OLD_SNAP")
    identity = _run_identity("P1", 0, 42, rec["contract_version"], "NEW_SNAP", "QHASH")
    processed, stale = _classify_existing_records([rec], identity)
    assert processed == set()
    assert stale == {"qv2-0011"}


def test_classify_different_seed_is_stale():
    rec = _mk_result("qv2-0011", seed=1)
    identity = _run_identity("P1", 0, 42, rec["contract_version"], "SNAP", "QHASH")
    processed, stale = _classify_existing_records([rec], identity)
    assert stale == {"qv2-0011"}


def test_classify_mixed_records():
    same = _mk_result("qv2-0011")
    stale = _mk_result("qv2-0015", seed=99)
    identity = _run_identity("P1", 0, 42, same["contract_version"], "SNAP", "QHASH")
    processed, stale_ids = _classify_existing_records([same, stale], identity)
    assert processed == {"qv2-0011"}
    assert stale_ids == {"qv2-0015"}


# --------------------------------------------------------------------- #
# 清理：_rewrite_jsonl_keep
# --------------------------------------------------------------------- #
def test_rewrite_jsonl_keeps_only_selected(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(
        json.dumps(_mk_result(qid)) for qid in ("a", "b", "c")) + "\n",
        encoding="utf-8")
    _rewrite_jsonl_keep(str(p), {"a", "c"})
    qids = [r["question_id"] for r in _load_result_records(str(p))]
    assert qids == ["a", "c"]


def test_rewrite_jsonl_skips_missing_file(tmp_path):
    _rewrite_jsonl_keep(str(tmp_path / "nope.jsonl"), {"a"})  # 不抛错


# --------------------------------------------------------------------- #
# trace 对齐：_rebuild_trace_from_results（result 内嵌 trace 为唯一事实源）
# --------------------------------------------------------------------- #
def test_rebuild_trace_fills_missing_and_drops_orphans(tmp_path):
    result_file = tmp_path / "r.jsonl"
    trace_file = tmp_path / "t.jsonl"
    recs = [_mk_result("a"), _mk_result("b"), _mk_result("c")]
    result_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n",
                           encoding="utf-8")
    # trace 缺 b 的行，且带一个孤儿行 x
    trace_file.write_text(
        json.dumps(recs[0]["trace"]) + "\n"
        + json.dumps(recs[2]["trace"]) + "\n"
        + json.dumps({"question_id": "x", "run_id": "orphan"}) + "\n",
        encoding="utf-8")

    _rebuild_trace_from_results(str(result_file), str(trace_file))

    qids = [json.loads(l)["question_id"] for l in trace_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert qids == ["a", "b", "c"], "应补上缺失的 b 并删掉孤儿 x"


def test_rebuild_trace_always_from_result_embedded(tmp_path):
    """重建必须**始终**以 result 内嵌 trace 为准：已有 trace 行的内容被篡改
    （含顺序错乱、字段漂移）时，重建后应恢复为 result 内嵌版本，而非保留旧行。"""
    result_file = tmp_path / "r.jsonl"
    trace_file = tmp_path / "t.jsonl"
    recs = [_mk_result("a", answer="ANS_A"), _mk_result("b", answer="ANS_B")]
    result_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n",
                           encoding="utf-8")
    # 已有 trace：a 行内容被篡改（answer_text 漂移），b 行被删，还带孤儿 x
    tampered = dict(recs[0]["trace"])
    tampered["answer_text"] = "TAMPERED"
    trace_file.write_text(
        json.dumps(tampered) + "\n"
        + json.dumps({"question_id": "x", "run_id": "orphan"}) + "\n",
        encoding="utf-8")

    _rebuild_trace_from_results(str(result_file), str(trace_file))

    lines = [json.loads(l) for l in trace_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [t["question_id"] for t in lines] == ["a", "b"]
    assert lines[0]["answer_text"] == "ANS_A", "必须恢复为 result 内嵌 trace，而非保留篡改行"
    assert lines[1]["answer_text"] == "ANS_B"


def test_rebuild_trace_from_scratch_when_trace_missing(tmp_path):
    result_file = tmp_path / "r.jsonl"
    trace_file = tmp_path / "t.jsonl"
    result_file.write_text(json.dumps(_mk_result("a")) + "\n", encoding="utf-8")
    _rebuild_trace_from_results(str(result_file), str(trace_file))
    qids = [json.loads(l)["question_id"] for l in trace_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert qids == ["a"]


# --------------------------------------------------------------------- #
# 原子写入：_write_record_pair
# --------------------------------------------------------------------- #
def test_write_record_pair_writes_both_files(tmp_path):
    rf = open(tmp_path / "r.jsonl", "w", encoding="utf-8")
    tf = open(tmp_path / "t.jsonl", "w", encoding="utf-8")
    try:
        rec = _mk_result("qv2-0011")
        _write_record_pair(rf, tf, rec, rec["trace"])
    finally:
        tf.close()
        rf.close()
    r_lines = [l for l in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t_lines = [l for l in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(r_lines) == 1 and len(t_lines) == 1
    assert json.loads(r_lines[0])["question_id"] == "qv2-0011"
    assert json.loads(t_lines[0])["question_id"] == "qv2-0011"


def test_write_record_pair_trace_none_skips_trace(tmp_path):
    rf = open(tmp_path / "r.jsonl", "w", encoding="utf-8")
    tf = open(tmp_path / "t.jsonl", "w", encoding="utf-8")
    try:
        _write_record_pair(rf, tf, {"question_id": "q"}, None)
    finally:
        tf.close()
        rf.close()
    assert (tmp_path / "t.jsonl").read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------- #
# manifest 状态
# --------------------------------------------------------------------- #
def test_run_status_complete_when_covered_and_no_error():
    assert _compute_run_status(3, 3, 0) == "complete"


def test_run_status_incomplete_when_not_covered():
    assert _compute_run_status(2, 3, 0) == "incomplete"


def test_run_status_incomplete_when_errors():
    assert _compute_run_status(3, 3, 1) == "incomplete"


def test_run_status_complete_when_empty_request():
    assert _compute_run_status(0, 0, 0) == "complete"


# --------------------------------------------------------------------- #
# verify 全量加载：_load_route
# --------------------------------------------------------------------- #
def _write_route_files(base, route="P1", n=3, with_trace=True):
    stem = f"fixed_{route}_r0_s42_v2.1-v1_SNAP"
    lines = [json.dumps(_mk_result(f"qv2-00{i:02d}")) for i in range(1, n + 1)]
    (base / f"{stem}_result.jsonl").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    if with_trace:
        tlines = [json.dumps(_mk_result(f"qv2-00{i:02d}")["trace"])
                  for i in range(1, n + 1)]
        (base / f"{stem}_trace.jsonl").write_text("\n".join(tlines) + "\n",
                                                  encoding="utf-8")


def test_load_route_returns_all_records(tmp_path):
    _write_route_files(tmp_path, n=3)
    results, traces, err = _load_route(tmp_path, "P1", "SNAP", 0, 42, "v2.1-v1")
    assert err is None
    assert len(results) == 3 and len(traces) == 3
    assert [r["question_id"] for r in results] == ["qv2-0001", "qv2-0002", "qv2-0003"]


def test_load_route_reports_result_trace_mismatch(tmp_path):
    _write_route_files(tmp_path, n=3, with_trace=False)  # 只有 result，无 trace
    results, traces, err = _load_route(tmp_path, "P1", "SNAP", 0, 42, "v2.1-v1")
    assert err is not None and "trace" in err
    assert len(results) == 3 and traces == []


def test_load_route_falls_back_to_json_array(tmp_path):
    stem = f"fixed_P1_r0_s42_v2.1-v1_SNAP"
    (tmp_path / f"{stem}_result.json").write_text(
        json.dumps([_mk_result("qv2-001"), _mk_result("qv2-002")]),
        encoding="utf-8")
    (tmp_path / f"{stem}_trace.jsonl").write_text(
        json.dumps(_mk_result("qv2-001")["trace"]) + "\n"
        + json.dumps(_mk_result("qv2-002")["trace"]) + "\n",
        encoding="utf-8")
    results, traces, err = _load_route(tmp_path, "P1", "SNAP", 0, 42, "v2.1-v1")
    assert err is None
    assert len(results) == 2 and len(traces) == 2


def test_load_route_missing_files(tmp_path):
    results, traces, err = _load_route(tmp_path, "P4", "SNAP", 0, 42, "v2.1-v1")
    assert err is not None and "缺少结果文件" in err


# --------------------------------------------------------------------- #
# 题集哈希锁定：_check_locked_question_hash（fail-fast 判定）
# --------------------------------------------------------------------- #
def test_locked_hash_match_returns_true(capsys):
    assert _check_locked_question_hash("abc", "abc") is True
    out = capsys.readouterr().out
    assert "verified" in out


def test_locked_hash_mismatch_returns_false(capsys):
    assert _check_locked_question_hash("expected", "actual") is False
    out = capsys.readouterr().out
    assert "FATAL" in out and "mismatch" in out


# --------------------------------------------------------------------- #
# verify 增强：ID 一致性 / 重复 / manifest 状态
# --------------------------------------------------------------------- #
def test_route_identity_consistent():
    recs = [_mk_result("qv2-0011"), _mk_result("qv2-0015")]
    traces = [r["trace"] for r in recs]
    assert _check_route_identity(recs, traces) == []


def test_route_identity_mismatch_order():
    recs = [_mk_result("qv2-0011"), _mk_result("qv2-0015")]
    traces = [recs[1]["trace"], recs[0]["trace"]]  # 顺序颠倒
    problems = _check_route_identity(recs, traces)
    assert problems, "result/trace ID 顺序不一致必须报错"


def test_route_identity_duplicate_id():
    recs = [_mk_result("qv2-0011"), _mk_result("qv2-0011")]  # 重复 qid
    traces = [r["trace"] for r in recs]
    problems = _check_route_identity(recs, traces)
    assert any("重复" in p for p in problems)


def test_route_identity_missing_id():
    recs = [_mk_result("qv2-0011"), {"query": "no-id"}]
    traces = [r["trace"] for r in recs if "trace" in r]
    problems = _check_route_identity(recs, traces)
    assert any("缺 question_id" in p for p in problems)


def test_manifest_complete_ok(tmp_path):
    base = tmp_path
    (base / "fixed_P1_r0_s42_v2.1-v1_SNAP_manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8")
    assert _check_manifest(base, "fixed_P1_r0_s42_v2.1-v1_SNAP", "P1") == []


def test_manifest_incomplete_fails(tmp_path):
    base = tmp_path
    (base / "fixed_P1_r0_s42_v2.1-v1_SNAP_manifest.json").write_text(
        json.dumps({"status": "incomplete"}), encoding="utf-8")
    problems = _check_manifest(base, "fixed_P1_r0_s42_v2.1-v1_SNAP", "P1")
    assert any("incomplete" in p for p in problems)


def test_manifest_missing_fails(tmp_path):
    problems = _check_manifest(tmp_path, "fixed_P1_r0_s42_v2.1-v1_SNAP", "P1")
    assert problems and "缺失" in problems[0]
