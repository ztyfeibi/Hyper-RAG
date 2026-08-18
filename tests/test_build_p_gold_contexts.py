# -*- coding: utf-8 -*-
"""P_gold contexts 构建脚本回归测试。

覆盖契约的核心约束：

1. ID 映射按后缀（qv2-XXXX -> ec-v2-XXXX），**不依赖行号**；乱序输入仍按题集顺序输出；
2. 缺失 verified candidate / decision != accept -> fail；
3. required AU 无 span / 空 evidence group / 不存在的 group -> fail；
4. span 与 chunk 原文切片不一致 / chunk 缺失 -> fail；
5. 重复 span 按 (chunk_id, start, end) 去重，[Evidence N] 编号连续；
6. 多行证据被规范成单行；
7. **修改 gold_answer 不改变 context**（上下文只由 span.text 组成）；
8. question_id 重复 -> fail；
9. 真实数据集成：恰好 80 条且 ID 顺序 == 锁定题集顺序。
"""

import json
from pathlib import Path

import pytest

from scripts.build_p_gold_contexts import (
    PGoldBuildError,
    build_p_gold_contexts,
    load_questions,
    load_verified,
    _map_candidate_to_question,
)

REPO = Path(__file__).resolve().parents[1]
REAL_QUESTION = (
    REPO / "caches" / "neurology_chunk1000" / "question_set_v2" / "pilot_v1"
    / "question_generation" / "questions_v2_manual_final.jsonl")
REAL_VERIFIED = (
    REPO / "caches" / "neurology_chunk1000" / "question_set_v2" / "pilot_v1"
    / "step2_2" / "verified_evidence.jsonl")
REAL_CHUNKS = REPO / "caches" / "neurology_chunk1000" / "kv_store_text_chunks.json"


def _count_fake(text: str) -> int:
    """确定性伪 token 计数：1 token ≈ 4 chars（仅测试用）。"""
    return max(1, (len(text) + 3) // 4)


# --------------------------------------------------------------------- #
# fixture 构造
# --------------------------------------------------------------------- #
def _mk_question(qid: str, n_au: int = 2):
    return {
        "question_id": qid,
        "question": f"Question for {qid}?",
        "gold_answer": f"gold answer for {qid}",
        "answer_units": [
            {"unit_id": f"AU{i + 1}", "claim": f"claim {i + 1} of {qid}",
             "required": True}
            for i in range(n_au)
        ],
    }


def _mk_verified(cid: str, n_au: int = 2):
    """构造 verified record：每个 AU 一个 group、一个 span（chunk-x）。

    span 文本 = "claim {i} of {qid}"，偏移按单 chunk 拼接计算（' | ' 分隔），
    由 _rebuild_chunks 重建为一致的内容。
    """
    qid = "qv2-" + cid[len("ec-v2-"):]
    return {
        "candidate_id": cid,
        "decision": "accept",
        "answer_units": [
            {"unit_id": f"au{i + 1}", "statement": f"claim {i + 1} of {qid}",
             "evidence_group_ids": [f"eg{i + 1}"]}
            for i in range(n_au)
        ],
        "evidence_groups": [
            {"group_id": f"eg{i + 1}", "answer_unit_id": f"au{i + 1}",
             "span_ids": [f"sp{i + 1}"]}
            for i in range(n_au)
        ],
        "spans": [
            {"span_id": f"sp{i + 1}", "chunk_id": "chunk-x",
             "char_start": 0, "char_end": 0,
             "text": f"source text {i + 1} of {qid}"}
            for i in range(n_au)
        ],
    }


def _rebuild_chunks(verifieds):
    """按全局 span 顺序重建 chunk-x 内容与 spans 偏移（' | ' 分隔）。

    返回 (chunks, verifieds_new)——verifieds_new 为深拷贝（偏移已重写）。
    """
    verifieds = [json.loads(json.dumps(v)) for v in verifieds]
    all_spans = []
    for v in verifieds:
        all_spans.extend(v["spans"])
    content = " | ".join(s["text"] for s in all_spans)
    cur = 0
    for s in all_spans:
        s["char_start"] = cur
        cur += len(s["text"])
        s["char_end"] = cur
        cur += 3  # " | "
    chunks = {"chunk-x": {"content": content}}
    return chunks, verifieds


def _write(tmp_path, questions, verifieds, chunks):
    qf = tmp_path / "questions.jsonl"
    vf = tmp_path / "verified.jsonl"
    cf = tmp_path / "chunks.json"
    qf.write_text("\n".join(json.dumps(q) for q in questions) + "\n",
                  encoding="utf-8")
    vf.write_text("\n".join(json.dumps(v) for v in verifieds) + "\n",
                  encoding="utf-8")
    cf.write_text(json.dumps(chunks), encoding="utf-8")
    return qf, vf, cf


def _build(tmp_path, questions, verifieds, cap=10000):
    chunks, verifieds = _rebuild_chunks(verifieds)
    qf, vf, cf = _write(tmp_path, questions, verifieds, chunks)
    return build_p_gold_contexts(qf, vf, cf, _count_fake, context_hard_cap=cap)


# --------------------------------------------------------------------- #
# 1. ID 映射（按后缀，不依赖行号）
# --------------------------------------------------------------------- #
def test_id_mapping_by_suffix_not_line_number(tmp_path):
    # 题集顺序 qv2-0011, qv2-0015；verified 顺序颠倒 -> 仍按题集顺序输出
    qs = [_mk_question("qv2-0011", 1), _mk_question("qv2-0015", 1)]
    vs = [_mk_verified("ec-v2-0015", 1), _mk_verified("ec-v2-0011", 1)]
    res = _build(tmp_path, qs, vs)
    assert res.errors == [], res.errors
    assert [r.question_id for r in res.records] == ["qv2-0011", "qv2-0015"]
    assert "source text 1 of qv2-0011" in res.records[0].context
    assert "source text 1 of qv2-0015" in res.records[1].context


def test_missing_verified_candidate_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 1), _mk_question("qv2-0099", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    res = _build(tmp_path, qs, vs)
    assert any("qv2-0099" in e and "找不到" in e for e in res.errors), res.errors


def test_duplicate_question_id_fails(tmp_path):
    qf = tmp_path / "q.jsonl"
    qf.write_text(
        json.dumps(_mk_question("qv2-0011")) + "\n"
        + json.dumps(_mk_question("qv2-0011")) + "\n", encoding="utf-8")
    with pytest.raises(PGoldBuildError, match="question_id 重复"):
        load_questions(qf)


def test_candidate_mapping_suffix_missing(tmp_path):
    # candidate ec-v2-7777 的后缀在题集中不存在
    qf, vf, _ = _write(tmp_path, [_mk_question("qv2-0011", 1)],
                       [_mk_verified("ec-v2-7777", 1)], {})
    with pytest.raises(PGoldBuildError, match="无法映射"):
        _map_candidate_to_question(load_questions(qf), load_verified(vf))


# --------------------------------------------------------------------- #
# 2. AU / group / span 完整性
# --------------------------------------------------------------------- #
def test_required_au_without_span_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 2)]
    vs = [_mk_verified("ec-v2-0011", 2)]
    vs[0]["evidence_groups"][1]["span_ids"] = []  # AU2 的 group 无 span
    res = _build(tmp_path, qs, vs)
    assert any("au2" in e or "AU2" in e for e in res.errors), res.errors


def test_empty_evidence_group_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 2)]
    vs = [_mk_verified("ec-v2-0011", 2)]
    vs[0]["answer_units"][1]["evidence_group_ids"] = ["eg2", "egX"]  # egX 不存在
    res = _build(tmp_path, qs, vs)
    assert any("egX" in e and "不存在" in e for e in res.errors), res.errors


def test_span_chunk_mismatch_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    chunks, verifieds = _rebuild_chunks(vs)
    sp = verifieds[0]["spans"][0]
    chunks["chunk-x"]["content"] = "X" * len(sp["text"])  # 同长不同内容
    qf, vf, cf = _write(tmp_path, qs, verifieds, chunks)
    res = build_p_gold_contexts(qf, vf, cf, _count_fake, 10000)
    assert any("与原文切片不一致" in e for e in res.errors), res.errors


def test_missing_chunk_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    qf, vf, cf = _write(tmp_path, qs, vs, {"chunk-other": {"content": "x"}})
    res = build_p_gold_contexts(qf, vf, cf, _count_fake, 10000)
    assert any("不存在于 chunks" in e for e in res.errors), res.errors


def test_decision_not_accept_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    vs[0]["decision"] = "reject"
    res = _build(tmp_path, qs, vs)
    assert any("decision" in e and "accept" in e for e in res.errors), res.errors


# --------------------------------------------------------------------- #
# 3. 组装格式
# --------------------------------------------------------------------- #
def test_duplicate_spans_deduped_and_numbered(tmp_path):
    # AU1 与 AU2 的 group 引用同一个 span
    qs = [_mk_question("qv2-0011", 2)]
    vs = [_mk_verified("ec-v2-0011", 2)]
    vs[0]["evidence_groups"][1]["span_ids"] = ["sp1"]  # AU2 复用 sp1
    res = _build(tmp_path, qs, vs)
    assert res.errors == [], res.errors
    ctx = res.records[0].context
    assert ctx.count("[Evidence") == 1, "重复 span 必须只出现一次"
    assert "[Evidence 1]" in ctx and "[Evidence 2]" not in ctx
    assert len(res.records[0].spans) == 1


def test_multiline_span_normalized_to_single_line(tmp_path):
    qs = [_mk_question("qv2-0011", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    text = "first line\n\nsecond\t\tcolumn  \nthird"
    sp = vs[0]["spans"][0]
    sp["text"] = text
    res = _build(tmp_path, qs, vs)
    assert res.errors == [], res.errors
    ctx = res.records[0].context
    assert "\n" not in ctx and "\t" not in ctx
    assert "  " not in ctx, "连续空白必须压缩为单空格"
    assert ctx == "[Evidence 1] first line second column third"


def test_gold_answer_does_not_affect_context(tmp_path):
    qs_a = [_mk_question("qv2-0011", 2)]
    qs_b = [_mk_question("qv2-0011", 2)]
    qs_b[0]["gold_answer"] = "COMPLETELY DIFFERENT gold answer"
    vs = [_mk_verified("ec-v2-0011", 2)]
    res_a = _build(tmp_path, qs_a, vs)
    res_b = _build(tmp_path, qs_b, vs)
    assert res_a.errors == [] and res_b.errors == []
    assert res_a.records[0].context == res_b.records[0].context
    assert "COMPLETELY DIFFERENT" not in res_b.records[0].context
    # AU claim 也不得进入 context（claim 只是元数据，非原文证据）
    assert "claim 1 of qv2-0011" not in res_a.records[0].context


def test_context_format_evidence_blocks_in_au_order(tmp_path):
    qs = [_mk_question("qv2-0011", 3)]
    vs = [_mk_verified("ec-v2-0011", 3)]
    res = _build(tmp_path, qs, vs)
    assert res.errors == [], res.errors
    ctx = res.records[0].context
    assert ctx.startswith("[Evidence 1] ")
    assert "[Evidence 2] " in ctx and "[Evidence 3] " in ctx
    i1 = ctx.index("source text 1 of qv2-0011")
    i2 = ctx.index("source text 2 of qv2-0011")
    i3 = ctx.index("source text 3 of qv2-0011")
    assert i1 < i2 < i3, "Evidence 块顺序必须与 AU 顺序一致"


# --------------------------------------------------------------------- #
# 4. token 上限
# --------------------------------------------------------------------- #
def test_over_cap_fails(tmp_path):
    qs = [_mk_question("qv2-0011", 1)]
    vs = [_mk_verified("ec-v2-0011", 1)]
    res = _build(tmp_path, qs, vs, cap=1)
    assert any("超过输入安全上限" in e for e in res.errors), res.errors


# --------------------------------------------------------------------- #
# 5. 真实数据集成（文件存在时）
# --------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (REAL_QUESTION.exists() and REAL_VERIFIED.exists() and REAL_CHUNKS.exists()),
    reason="真实实验数据不存在")
def test_real_data_80_lines_and_question_order():
    # 用确定性伪计数（离线），仅验证映射/组装/校验逻辑
    res = build_p_gold_contexts(REAL_QUESTION, REAL_VERIFIED, REAL_CHUNKS,
                                _count_fake, context_hard_cap=100000)
    assert res.errors == [], f"真实数据构建失败: {res.errors[:5]}"
    assert res.stats["question_count"] == 80
    assert res.stats["context_line_count"] == 80
    assert res.stats["missing_AU"] == 0
    assert res.stats["missing_group"] == 0
    assert res.stats["span_mismatch"] == 0
    assert res.stats["duplicate_question_id"] == 0
    assert res.stats["empty_context"] == 0
    assert res.stats["over_cap"] == 0
    # ID 顺序 == 题集顺序
    qids = [q["question_id"] for q in load_questions(REAL_QUESTION)]
    assert [r.question_id for r in res.records] == qids
    # 全部 context 非空、单行、[Evidence N] 块数与 span 数一致
    for r in res.records:
        assert r.context and "\n" not in r.context
        assert r.context.count("[Evidence") == len(r.spans)
    # 每个 [Evidence N] 块内容 == 对应 span.text（strip 后）
    for r in res.records:
        for n, sp in enumerate(r.spans, start=1):
            assert f"[Evidence {n}] {sp.text.strip()}" in r.context
