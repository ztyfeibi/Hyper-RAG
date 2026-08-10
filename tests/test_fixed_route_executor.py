"""Step 1.4 tests: 统一回答 Prompt 与固定 Executor。

Coverage:
1. route_to_mode 映射正确。
2. build_query_param 从契约构造参数：拆分字段 == 契约值、legacy 字段同步、
   mode 正确、系统边界与 bookkeeping 注入。
3. formal_answer_prompt_hash 对同一模板稳定（prompt hash 一致）。
4. execute_fixed_route：
   - P0 不检索（context 为空），统一模板被调用。
   - 检索路径（P1/P3）经 naive/hyper 取 context，再走统一模板。
   - P_gold 使用传入 gold_context。
   - 所有路径 prompt 模板与 hash 相同（公平性核心）。
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperrag.experiment_contract import load_contract
from hyperrag.fixed_route_executor import (
    build_query_param,
    execute_fixed_route,
    formal_answer_prompt,
    formal_answer_prompt_hash,
    route_to_mode,
)


CONTRACT = load_contract()


@dataclass
class FakeRag:
    """Minimal HyperRAG stand-in: a real dataclass so asdict() works."""
    llm_model_func: Any = None
    chunks_vdb: Any = None
    text_chunks: Any = None
    entities_vdb: Any = None
    relationships_vdb: Any = None
    chunk_entity_relation_hypergraph: Any = None


def _make_rag():
    return FakeRag(
        llm_model_func=AsyncMock(return_value="ANSWER"),
        chunks_vdb=MagicMock(),
        text_chunks=MagicMock(),
        entities_vdb=MagicMock(),
        relationships_vdb=MagicMock(),
        chunk_entity_relation_hypergraph=MagicMock(),
    )


# --------------------------------------------------------------------------
# 1. route_to_mode
# --------------------------------------------------------------------------
def test_route_to_mode():
    assert route_to_mode("P1") == "naive"
    assert route_to_mode("P2") == "hyper"
    assert route_to_mode("P3") == "hyper"
    assert route_to_mode("P4") == "hyper"


def test_route_to_mode_rejects_unknown():
    import hyperrag.fixed_route_executor as fx
    from hyperrag.experiment_contract import ContractError
    with pytest.raises(ContractError):
        fx.route_to_mode("P9")


# --------------------------------------------------------------------------
# 2. build_query_param
# --------------------------------------------------------------------------
@pytest.mark.parametrize("route_id", ["P0", "P1", "P2", "P3", "P4"])
def test_build_query_param_matches_contract(route_id):
    rc = CONTRACT.route(route_id)
    qp = build_query_param(rc, CONTRACT, route_id, repeat_id=1, repeat_seed=42,
                           system_snapshot_id="snap-x")

    # 拆分字段 == 契约值
    assert qp.chunk_vdb_top_k == rc.chunk_vdb_top_k
    assert qp.entity_vdb_top_k == rc.entity_vdb_top_k
    assert qp.relation_vdb_top_k == rc.relation_vdb_top_k
    assert qp.entity_description_cap == rc.entity_description_cap
    assert qp.relation_description_cap == rc.relation_description_cap
    assert qp.source_text_cap == rc.source_text_cap
    assert qp.final_context_hard_cap == rc.final_context_hard_cap

    # legacy 字段与拆分字段同步（无脏值）
    assert qp.max_token_for_entity_context == rc.entity_description_cap
    assert qp.max_token_for_relation_context == rc.relation_description_cap
    assert qp.max_token_for_text_unit == rc.source_text_cap
    assert qp.max_total_context_tokens == rc.final_context_hard_cap

    # 系统边界
    assert qp.llm_temperature == CONTRACT.system_boundary.temperature
    assert qp.max_response_tokens == CONTRACT.system_boundary.max_response_tokens

    # 固定路径开关
    assert qp.router_policy == "fixed"
    assert qp.enable_type_aware_weighting is False

    # bookkeeping
    assert qp.route_id == route_id
    assert qp.repeat_id == 1
    assert qp.repeat_seed == 42
    assert qp.system_snapshot_id == "snap-x"
    assert qp.contract_version == CONTRACT.contract_version

    # mode
    expected_mode = "llm" if rc.is_llm_only else route_to_mode(route_id)
    assert qp.mode == expected_mode


def test_p0_is_llm_only_mode():
    rc = CONTRACT.route("P0")
    assert rc.is_llm_only is True
    qp = build_query_param(rc, CONTRACT, "P0")
    assert qp.mode == "llm"


# --------------------------------------------------------------------------
# 3. prompt hash stability
# --------------------------------------------------------------------------
def test_prompt_hash_stable():
    h1 = formal_answer_prompt_hash()
    h2 = formal_answer_prompt_hash()
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_prompt_has_no_stray_braces():
    # ensure .format works with only the two declared placeholders
    p = formal_answer_prompt()
    out = p.format(context_data="CTX", response_type="Single paragraph")
    assert "CTX" in out
    assert "Single paragraph" in out


# --------------------------------------------------------------------------
# 4. execute_fixed_route
# --------------------------------------------------------------------------
def test_execute_p0_no_retrieval():
    rag = _make_rag()
    with patch.object(__import__("hyperrag.fixed_route_executor", fromlist=["x"]),
                      "hyper_query") as _h, \
         patch.object(__import__("hyperrag.fixed_route_executor", fromlist=["x"]),
                      "naive_query") as _n:
        result = __import__("asyncio").run(
            execute_fixed_route("q", "P0", CONTRACT, rag, repeat_id=1,
                                repeat_seed=42, system_snapshot_id="s")
        )
    # P0 must not call any retrieval
    _h.assert_not_called()
    _n.assert_not_called()
    assert result["mode"] == "llm"
    assert result["context"] == ""
    assert result["answer"] == "ANSWER"
    # unified prompt invoked with empty data tables
    _, kwargs = rag.llm_model_func.call_args
    assert "Data tables" in kwargs["system_prompt"]
    assert "---Goal---" in kwargs["system_prompt"]


def test_execute_p1_uses_naive_context():
    rag = _make_rag()
    with patch("hyperrag.fixed_route_executor.naive_query",
               AsyncMock(return_value="NAIVE_CTX")) as mock_naive, \
         patch("hyperrag.fixed_route_executor.hyper_query") as mock_hyper:
        result = __import__("asyncio").run(
            execute_fixed_route("q", "P1", CONTRACT, rag)
        )
    mock_naive.assert_called_once()
    mock_hyper.assert_not_called()
    assert result["context"] == "NAIVE_CTX"
    assert result["mode"] == "naive"
    # unified prompt received the retrieved context
    _, kwargs = rag.llm_model_func.call_args
    assert "NAIVE_CTX" in kwargs["system_prompt"]


def test_execute_p1_accepts_dict_return_contract():
    """回归：naive_query 在 only_need_context=True + save_trace 时返回
    {"context": str, "trace_data": dict}（真实返回契约）。
    executor 必须提取 context，绝不能丢弃成空串（曾导致 P1 退化为 P0）。"""
    rag = _make_rag()
    real_shape = {"context": "REAL_NAIVE_CTX", "trace_data": {"chunk_vdb_ids": ["c1"]}}
    with patch("hyperrag.fixed_route_executor.naive_query",
               AsyncMock(return_value=real_shape)) as mock_naive, \
         patch("hyperrag.fixed_route_executor.hyper_query") as mock_hyper:
        result = __import__("asyncio").run(
            execute_fixed_route("q", "P1", CONTRACT, rag, save_trace=True,
                                trace_data={})
        )
    mock_naive.assert_called_once()
    mock_hyper.assert_not_called()
    assert result["context"] == "REAL_NAIVE_CTX"
    assert result["context"] != ""  # P1 绝不能退化为 P0
    _, kwargs = rag.llm_model_func.call_args
    assert "REAL_NAIVE_CTX" in kwargs["system_prompt"]


def test_execute_p3_uses_hyper_context():
    rag = _make_rag()
    with patch("hyperrag.fixed_route_executor.hyper_query",
               AsyncMock(return_value={"context": "HYPER_CTX"})) as mock_hyper, \
         patch("hyperrag.fixed_route_executor.naive_query") as mock_naive:
        result = __import__("asyncio").run(
            execute_fixed_route("q", "P3", CONTRACT, rag)
        )
    mock_hyper.assert_called_once()
    mock_naive.assert_not_called()
    assert result["context"] == "HYPER_CTX"
    assert result["mode"] == "hyper"


def test_execute_p_gold_uses_gold_context():
    rag = _make_rag()
    with patch("hyperrag.fixed_route_executor.hyper_query") as mock_hyper, \
         patch("hyperrag.fixed_route_executor.naive_query") as mock_naive:
        result = __import__("asyncio").run(
            execute_fixed_route("q", "P_gold", CONTRACT, rag,
                                gold_context="GOLD_EVIDENCE")
        )
    mock_hyper.assert_not_called()
    mock_naive.assert_not_called()
    assert result["mode"] == "gold"
    assert result["context"] == "GOLD_EVIDENCE"
    _, kwargs = rag.llm_model_func.call_args
    assert "GOLD_EVIDENCE" in kwargs["system_prompt"]


def test_all_routes_share_prompt_hash():
    """公平性核心：所有固定路径的 prompt 模板与 hash 完全相同。"""
    rag = _make_rag()
    hashes = {}
    with patch("hyperrag.fixed_route_executor.hyper_query",
               AsyncMock(return_value={"context": "X"})), \
         patch("hyperrag.fixed_route_executor.naive_query",
               AsyncMock(return_value="X")):
        for rid in ["P0", "P1", "P2", "P3", "P4", "P_gold"]:
            gc = "G" if rid == "P_gold" else None
            res = __import__("asyncio").run(
                execute_fixed_route("q", rid, CONTRACT, rag, gold_context=gc)
            )
            hashes[rid] = res["prompt_hash"]
    assert len(set(hashes.values())) == 1
    assert hashes["P0"] == formal_answer_prompt_hash()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
