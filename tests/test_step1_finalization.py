# -*- coding: utf-8 -*-
"""Step 1 收尾验收测试：预合并 Qwen 计数 + Trace 完整性。

对应三项收尾问题中可自动化的两项：

问题 2 —— 预合并 cap 仍按 tiktoken 执行
    契约 ``units.token_unit = qwen_tokenizer_token``，pre_merge_caps
    (entity/relation/source) 都是 Qwen token；热路径 ``truncate_list_by_token_size``
    按 tiktoken 计数，必须先经校准 **保守** 转换，否则实际保留量偏大、
    P2-P4 的执行参数偏离冻结契约。

问题 3 —— Trace 不完整
    3.1 ``finish_reason`` 全链路采集（llm -> executor -> Step_3）；
    3.2 P_gold 的 context token 用真实 Qwen 计数（此前恒为 0）；
    3.3 候选级 ``source_provenance_ids`` 真实填充（此前硬编码 null）；
    3.4 ``stages.provenance`` / ``stages.graph_lineage`` 真实构造。
"""

import asyncio

import pytest

from hyperrag.base import QueryParam
from hyperrag.tokenizer_calibration import (
    get_cached_calibration,
    qwen_cap_to_tiktoken_budget,
    qwen_to_tiktoken_budget,
    reset_calibration_cache,
)
from hyperrag.trace_collector import build_trace_record, validate_and_finalize
from hyperrag.experiment_schema import SchemaValidationError


# --------------------------------------------------------------------- #
# 问题 2：预合并 cap 的单位口径
# --------------------------------------------------------------------- #
def test_contract_caps_converted_to_tiktoken_budget():
    """拆分字段（契约 Qwen 单位）必须经保守转换后才喂给 tiktoken 截断。"""
    calib = get_cached_calibration()
    assert calib is not None, "缺少 tokenizer 校准报告，无法验收（应 fail-fast）"
    max_ratio = float(calib["ratio_stats"]["max"])
    assert max_ratio > 1.0, "Qwen 分词更细，max_ratio 应 > 1"

    p = QueryParam(
        entity_description_cap=250,
        relation_description_cap=1200,
        source_text_cap=3000,
    )
    assert p.caps_in_qwen_unit() is True
    # effective_* 仍返回契约原值（Qwen 单位，用于上报）
    assert p.effective_entity_cap() == 250
    assert p.effective_relation_cap() == 1200
    assert p.effective_source_cap() == 3000
    # tiktoken_* 是实际执行预算，必须严格更小（保守）
    assert p.tiktoken_entity_cap() == qwen_to_tiktoken_budget(250, calib) < 250
    assert p.tiktoken_relation_cap() == qwen_to_tiktoken_budget(1200, calib) < 1200
    assert p.tiktoken_source_cap() == qwen_to_tiktoken_budget(3000, calib) < 3000
    # 保守性保证：tiktoken 预算 * max_ratio 不超过 Qwen 上限
    assert p.tiktoken_entity_cap() * max_ratio <= 250
    assert p.tiktoken_relation_cap() * max_ratio <= 1200
    assert p.tiktoken_source_cap() * max_ratio <= 3000


def test_legacy_caps_not_converted():
    """未设拆分字段的旧路径（hyper/naive/adaptive）保持 tiktoken 原值，行为不变。"""
    p = QueryParam(
        max_token_for_entity_context=300,
        max_token_for_relation_context=1600,
        max_token_for_text_unit=1600,
    )
    assert p.caps_in_qwen_unit() is False
    assert p.tiktoken_entity_cap() == 300
    assert p.tiktoken_relation_cap() == 1600
    assert p.tiktoken_source_cap() == 1600


def test_zero_cap_stays_zero():
    """cap=0（如 P1 关闭实体/关系线）转换后仍为 0，不得被抬成 1。"""
    p = QueryParam(entity_description_cap=0, relation_description_cap=0)
    assert p.tiktoken_entity_cap() == 0
    assert p.tiktoken_relation_cap() == 0


def test_missing_calibration_fails_fast():
    """无校准报告时禁止静默按 tiktoken 口径执行 —— 必须抛错。"""
    with pytest.raises(ValueError):
        qwen_to_tiktoken_budget(1000, None)


def test_budget_cache_is_consistent():
    reset_calibration_cache()
    a = qwen_cap_to_tiktoken_budget(1200)
    b = qwen_cap_to_tiktoken_budget(1200)
    assert a == b and a is not None


def test_truncate_uses_calibrated_encoder():
    """截断函数默认编码器需与校准脚本口径一致（gpt-4o-mini）。"""
    import inspect
    from hyperrag.utils import truncate_list_by_token_size
    from scripts.calibrate_tokenizers import TIKTOKEN_MODEL_NAME

    sig = inspect.signature(truncate_list_by_token_size)
    assert sig.parameters["model_name"].default == TIKTOKEN_MODEL_NAME


# --------------------------------------------------------------------- #
# 问题 3.3 / 3.4：候选 provenance + stages.provenance / graph_lineage
# --------------------------------------------------------------------- #
_TRACE = {
    "entity_vdb_ids": ["E1", "E2"],
    "entity_vdb_scores": [0.9, 0.8],
    "entity_vdb_source_ids": [["chunk-a", "chunk-b"], ["chunk-c"]],
    "relation_vdb_ids": ["R1"],
    "relation_vdb_scores": [0.7],
    "relation_vdb_source_ids": [["chunk-b"]],
    "entity_post_graph_ids": ["E1", "E2"],
    "entity_post_truncate_ids": ["E1"],
    "entity_line_relation_ids": ["ER1"],
    "entity_line_chunk_ids": ["chunk-a"],
    "relation_post_filter_ids": ["R1"],
    "relation_post_truncate_ids": ["R1"],
    "relation_line_entity_ids": ["RE1"],
    "relation_line_chunk_ids": ["chunk-b"],
    "merged_chunk_ids": ["chunk-a", "chunk-b"],
    "graph_expansion_calls": 12,
    "keyword_result_hash": "kh",
    "context_hash": "ch",
    "context_tokens": 900,
    "context_tokens_before_truncate": 1000,
}


def _rec(trace=None, **kw):
    base = dict(
        trace_data=trace if trace is not None else _TRACE,
        question_id="q1", run_id="r1", route_id="P3", seed=42,
        system_snapshot_id="snap", answer_text="A",
    )
    base.update(kw)
    return build_trace_record(**base)


def test_source_provenance_ids_populated():
    rec = _rec()
    ent = next(r for r in rec["retrievers"] if r["retriever_id"] == "entity")
    rel = next(r for r in rec["retrievers"] if r["retriever_id"] == "relation")
    assert ent["candidates"][0]["source_provenance_ids"] == ["chunk-a", "chunk-b"]
    assert ent["candidates"][1]["source_provenance_ids"] == ["chunk-c"]
    assert rel["candidates"][0]["source_provenance_ids"] == ["chunk-b"]
    validate_and_finalize(rec)


def test_source_provenance_null_when_not_collected():
    """未采集 provenance 时如实记 null，绝不用空列表伪装成"采集到但为空"。"""
    trace = dict(_TRACE)
    trace.pop("entity_vdb_source_ids")
    trace.pop("relation_vdb_source_ids")
    rec = _rec(trace)
    for r in rec["retrievers"]:
        for c in r["candidates"]:
            assert c["source_provenance_ids"] is None
    validate_and_finalize(rec)


def test_provenance_misaligned_length_is_not_silently_used():
    """provenance 与候选数不匹配时降级为 null，不得错位对齐。"""
    trace = dict(_TRACE, entity_vdb_source_ids=[["chunk-a"]])  # 1 != 2
    rec = _rec(trace)
    ent = next(r for r in rec["retrievers"] if r["retriever_id"] == "entity")
    assert all(c["source_provenance_ids"] is None for c in ent["candidates"])


def test_stages_provenance_and_graph_lineage_present():
    rec = _rec()
    prov = rec["stages"]["provenance"]
    assert prov["entity_line"]["enabled"] is True
    assert prov["entity_line"]["vdb_candidates"] == 2
    assert prov["relation_line"]["enabled"] is True
    assert prov["chunk_line"]["enabled"] is False   # hyper 路径不查 chunk VDB
    assert prov["keyword_result_hash"] == "kh"

    lin = rec["stages"]["graph_lineage"]
    assert lin["entity_line"]["seed_entities"] == ["E1", "E2"]
    assert lin["entity_line"]["expanded_hyperedges"] == ["ER1"]
    assert lin["relation_line"]["context_hyperedges"] == ["R1"]
    assert lin["relation_line"]["expanded_entities"] == ["RE1"]
    assert lin["merged_source_chunks"] == ["chunk-a", "chunk-b"]
    assert lin["graph_expansion_calls"] == 12
    validate_and_finalize(rec)


def test_finish_reason_threaded_into_record():
    rec = _rec(finish_reason="stop")
    assert rec["finish_reason"] == "stop"
    validate_and_finalize(rec)
    # 未采集时仍允许 null（Schema 允许），但不得伪造 "stop"
    assert _rec()["finish_reason"] is None


# --------------------------------------------------------------------- #
# 问题 3.1 / 3.2：finish_reason 采集 + P_gold 真实 Qwen 计数
# --------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, text, finish_reason):
        choice = type("C", (), {
            "message": type("M", (), {"content": text})(),
            "finish_reason": finish_reason,
        })()
        self.choices = [choice]
        self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 7})()


def test_llm_records_finish_reason(monkeypatch):
    """openai_complete_if_cache 必须把真实 finish_reason 写入旁路 dict。"""
    from hyperrag import llm as llm_mod

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = type("Chat", (), {"completions": self})()

        async def create(self, **kwargs):
            assert "_record_finish_reason_into" not in kwargs, \
                "旁路参数必须在调 API 前 pop 掉"
            return _FakeResp("ANSWER", "length")

    monkeypatch.setattr(llm_mod, "AsyncOpenAI", lambda *a, **kw: _FakeClient())
    box = {}
    out = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        llm_mod.openai_complete_if_cache(
            "m", "p", base_url="http://x/v1", _record_finish_reason_into=box)
    )
    assert out == "ANSWER"
    assert box["finish_reason"] == "length"
    assert box["from_cache"] is False


def test_gold_route_reports_real_qwen_tokens(monkeypatch):
    """P_gold 的 gold_context 是真实进 prompt 的证据，token 数不得记 0。"""
    from hyperrag import fixed_route_executor as fre
    from hyperrag.experiment_contract import load_contract

    monkeypatch.setattr(fre, "_count_context_qwen_tokens", lambda ctx: 300 if ctx else 0)

    class _Rag:
        async def llm_model_func(self, prompt, system_prompt=None,
                                 history_messages=None, **kw):
            kw.pop("_record_usage_into", None)
            fin = kw.pop("_record_finish_reason_into", None)
            if fin is not None:
                fin["finish_reason"] = "stop"
            return "GOLD ANSWER"

    res = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        fre.execute_fixed_route(
            "Q", "P_gold", load_contract(), _Rag(),
            repeat_id=0, repeat_seed=42, system_snapshot_id="snap",
            gold_context="G" * 50, save_trace=True, trace_data={},
        )
    )
    assert res["finish_reason"] == "stop"
    td = res["trace_data"]
    assert td["context_tokens"] == 300, "P_gold context token 必须是真实 Qwen 计数"
    assert td["context_tokens_before_truncate"] == 300
    assert td["qwen_count_source"] == "vllm_tokenize"

    rec = build_trace_record(
        td, question_id="q", run_id="r", route_id="P_gold", seed=42,
        system_snapshot_id="snap", answer_text=res["answer"],
        finish_reason=res["finish_reason"], cost=res["cost"],
    )
    assert rec["route_id"] == "gold"
    assert rec["finish_reason"] == "stop"
    assert rec["stages"]["tokens_after_truncation"] == 300
    validate_and_finalize(rec)
