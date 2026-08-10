"""Step 1.6 tests: 完整 Trace 接入 (retrieval-trace-v2)。

Coverage:
1. build_trace_record 从 trace_data 组装出包含所有顶层必填字段的记录。
2. 记录通过 validate_and_finalize（符合 retrieval-trace-v2 Schema）。
3. retrievers / stages / cost / judge 子结构字段齐全且类型正确。
4. route_id 映射：P_gold -> gold（schema 枚举）。
5. 残缺记录（缺 answer_text）被 validate_trace_record 拒绝。
6. 不引入 schema 禁止的额外字段（additionalProperties=false）。
"""

from hyperrag.experiment_schema import (
    load_trace_schema,
    validate_trace_record,
    SchemaValidationError,
)
from hyperrag.trace_collector import build_trace_record, validate_and_finalize


SCHEMA = load_trace_schema()

FAKE_TRACE = {
    "entity_vdb_ids": ["E1", "E2"],
    "entity_vdb_scores": [0.91, 0.88],   # 真实 cosine 分数（契约禁止 0 占位）
    "relation_vdb_ids": ["R1"],
    "relation_vdb_scores": [0.77],
    "merged_chunk_ids": ["chunk-a", "chunk-b", "chunk-c"],
    "entity_query_embedding_hash": "h-ent",
    "relation_query_embedding_hash": "h-rel",
    "entity_post_graph_ids": ["E1", "E2"],
    "relation_post_filter_ids": ["R1"],
    "entity_post_truncate_ids": ["E1"],
    "relation_post_truncate_ids": ["R1"],
    "entity_line_chunk_ids": ["E1", "E2"],
    "entity_line_relation_ids": ["ER1"],
    "relation_line_entity_ids": ["RE1"],
    "context_tokens_before_truncate": 1500,
    "context_tokens": 1200,
    "context_truncated": True,
    "context_hash": "abc",
    "entity_keywords": "a, b, c",
    "relation_keywords": ["x", "y"],
    "keyword_result_hash": "kh",
}


def _build(**overrides):
    kw = dict(
        trace_data=FAKE_TRACE,
        question_id="q1",
        run_id="P3-r1-s42",
        route_id="P3",
        seed=42,
        system_snapshot_id="snap-1",
        answer_text="ANSWER",
    )
    kw.update(overrides)
    return build_trace_record(**kw)


def test_build_has_all_top_level_keys():
    rec = _build()
    required = {
        "question_id", "run_id", "route_id", "seed", "system_snapshot_id",
        "answer_text", "finish_reason", "retries", "timeouts",
        "stage_evidence_loss", "keyword_extraction", "retrievers", "stages",
        "cost", "judge",
    }
    assert required.issubset(rec.keys())
    # no stray top-level keys
    assert set(rec.keys()) == required


def test_record_validates():
    rec = _build()
    # must not raise
    validated = validate_and_finalize(rec, schema=SCHEMA)
    assert validated is rec


def test_retrievers_structure():
    rec = _build()
    retr = {r["retriever_id"]: r for r in rec["retrievers"]}
    # hyper 路径只有 entity/relation 两条 VDB 直查线；
    # chunk 来自图扩展、无直接 VDB 分数，契约禁止伪造 chunk retriever。
    assert set(retr.keys()) == {"entity", "relation"}
    for r in rec["retrievers"]:
        assert r["score_direction"] in ("higher_is_better", "lower_is_better")
        assert isinstance(r["candidates"], list) and r["candidates"]
        for c in r["candidates"]:
            assert set(c.keys()) == {
                "candidate_id", "candidate_type", "rank",
                "raw_retrieval_score", "normalized_score",
                "reranker_score", "source_provenance_ids",
            }
            assert isinstance(c["raw_retrieval_score"], (int, float))
            assert c["rank"] >= 1
    # 真实分数按序进入候选
    ent = retr["entity"]
    assert [c["raw_retrieval_score"] for c in ent["candidates"]] == [0.91, 0.88]


def test_chunk_retriever_only_with_direct_vdb():
    """chunk retriever 仅在 P1/naive 直查 chunk VDB 时出现（带真实分数）。"""
    td = dict(FAKE_TRACE)
    td["chunk_vdb_ids"] = ["chunk-a", "chunk-b"]
    td["chunk_vdb_scores"] = [0.66, 0.61]
    rec = _build(trace_data=td)
    retr = {r["retriever_id"]: r for r in rec["retrievers"]}
    assert set(retr.keys()) == {"entity", "relation", "chunk"}
    assert [c["raw_retrieval_score"] for c in retr["chunk"]["candidates"]] == [0.66, 0.61]


def test_missing_scores_rejected():
    """上游未提供真实分数时必须拒绝组装，禁止以 0 占位伪造。"""
    td = dict(FAKE_TRACE)
    td.pop("entity_vdb_scores")
    try:
        _build(trace_data=td)
        raised = False
    except SchemaValidationError:
        raised = True
    assert raised, "缺真实分数应抛 SchemaValidationError，而不是补 0"


def test_stages_and_cost_completeness():
    rec = _build()
    assert rec["stages"]["tokens_before_truncation"] == 1500
    assert rec["stages"]["tokens_after_truncation"] == 1200
    assert rec["stages"]["truncated_tokens"] == 300
    cost_keys = {
        "input_tokens", "output_tokens", "retrieval_latency_ms",
        "end_to_end_latency_ms", "llm_calls", "embedding_calls",
        "reranker_calls", "graph_expansion_calls", "retrieved_candidate_count",
        "gpu_seconds", "api_cost",
    }
    assert cost_keys.issubset(rec["cost"].keys())
    assert set(rec["cost"].keys()) == cost_keys
    judge_keys = {"verdict", "critical_error", "evidence_sufficient", "human_review_status"}
    assert judge_keys.issubset(rec["judge"].keys())


def test_gold_route_id_mapping():
    rec = _build(route_id="P_gold")
    assert rec["route_id"] == "gold"
    validate_and_finalize(rec, schema=SCHEMA)  # must still validate


def test_keyword_list_normalization():
    rec = _build()
    assert rec["keyword_extraction"]["entity_keywords"] == ["a", "b", "c"]
    assert rec["keyword_extraction"]["relation_keywords"] == ["x", "y"]


def test_malformed_record_rejected():
    rec = _build()
    del rec["answer_text"]  # remove a required field
    try:
        validate_trace_record(rec, SCHEMA)
        raised = False
    except SchemaValidationError:
        raised = True
    assert raised, "missing answer_text should be rejected by schema"


def test_cost_override_merges():
    rec = _build(cost={"input_tokens": 123, "llm_calls": 4})
    assert rec["cost"]["input_tokens"] == 123
    assert rec["cost"]["llm_calls"] == 4
    # untouched defaults preserved
    assert rec["cost"]["output_tokens"] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_hyperedge_ids_include_entity_line_expansion():
    """回归：P2 只开实体线时，Relationships 区段全部来自实体邻接扩展
    (entity_line_relation_ids)。hyperedge_ids 必须合并两条线并稳定去重，
    不能只取 relation_post_truncate_ids（曾导致 P2 超边 lineage 为空）。"""
    td = dict(FAKE_TRACE)
    td["relation_post_truncate_ids"] = ["HE-rel-1"]
    td["entity_line_relation_ids"] = ["HE-ent-1", "HE-rel-1", "HE-ent-2"]
    rec = _build(trace_data=td)
    st = rec["stages"]
    # 合并 + 首现顺序去重
    assert st["hyperedge_ids"] == ["HE-rel-1", "HE-ent-1", "HE-ent-2"]
    # post_graph_expansion 也要包含实体线扩展产物
    assert "HE-ent-1" in st["post_graph_expansion_ids"]


def test_hyperedge_ids_p2_entity_line_only():
    """P2 形态：关系线关闭（relation_post_truncate_ids 为空），
    超边只来自实体线，hyperedge_ids 不得为空。"""
    td = dict(FAKE_TRACE)
    td["relation_post_truncate_ids"] = []
    td["entity_line_relation_ids"] = ["HE-a", "HE-b"]
    rec = _build(trace_data=td)
    assert rec["stages"]["hyperedge_ids"] == ["HE-a", "HE-b"]
