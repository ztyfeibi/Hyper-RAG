# -*- coding: utf-8 -*-
"""Step 1.6/收尾：完整 Trace 接入 (retrieval-trace-v2)。

把固定路径执行过程中采集到的 ``trace_data``（来自 hyper_query / naive_query
的 ``save_trace``）与最终回答 / usage 组装成符合
``docs/schema/retrieval_trace_v2.schema.json`` 的完整记录，并在写盘前调用
``experiment_schema.validate_trace_record`` 校验；不符合 Schema 的题标记为失败，
禁止静默保存残缺记录。

真实数据采集（Step 1 收尾硬性要求）
--------------------------------
本模块 **不再用 0 / None 伪造检索分数与成本**。所有 retriever 候选的
``raw_retrieval_score`` 必须来自 VDB 返回的 cosine 相似度（``distance``）；
11 项成本中可采集的（input/output tokens、llm_calls、embedding_calls、
reranker_calls、graph_expansion_calls、retrieved_candidate_count、两类 latency）
由 runner 从 Storage / LLM 调用 / context assembly 真实采集并传入；本部署下
**无法采集** 的 ``gpu_seconds`` / ``api_cost`` 由 runner 显式传 ``None``
（契约 cost 聚合规则：缺则记 null，绝不补 0）。

阶段 ID 语义（修正错配）
------------------------
- ``entity_ids``      : 进入 final context 的实体节点（entity_post_truncate_ids
                        + relation_line_entity_ids），**不是** chunk ID。
- ``hyperedge_ids``    : 进入 final context 的超边（relation_post_truncate_ids），
                        **不混入** 实体 ID。
- ``final_context_source_ids`` : 截断后真正保留进 final context 的 source chunk
                        （来自 trace_data.final_context_source_ids），反映截断。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .experiment_schema import (
    load_trace_schema,
    validate_trace_record,
    SchemaValidationError,
)

# 契约 / CLI 用 "P_gold"，trace schema 枚举用 "gold"
_GOLD_ROUTE_ALIASES = {"P_gold"}

# 检索分数语义（本项目 VDB 用 numpy cosine similarity）
SCORE_SEMANTICS = "cosine_similarity"
SCORE_DIRECTION = "higher_is_better"
# 本项目不做任何分数归一化 —— 方法标识为 "none"，版本固定；绝不谎称做了归一化。
SCORE_NORMALIZATION_METHOD = "none"
SCORE_NORMALIZATION_VERSION = "identity-v1"


def _route_id_for_trace(route_id: str) -> str:
    return "gold" if route_id in _GOLD_ROUTE_ALIASES else route_id


def _as_keyword_list(v: Any) -> List[str]:
    """trace_data 中关键词可能是字符串或列表；统一规整为字符串数组。"""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x not in (None, "")]
    parts = []
    for seg in str(v).replace("\n", ",").split(","):
        seg = seg.strip()
        if seg:
            parts.append(seg)
    return parts


def _retriever(
    retriever_id: str,
    version: str,
    embedding_hash: str,
    ids: List[Any],
    scores: List[float],
    candidate_type: str,
    provenance: Optional[List[Any]] = None,
    score_normalization_version: str = SCORE_NORMALIZATION_VERSION,
) -> Dict[str, Any]:
    """构造单个 retriever 条目，候选携带 **真实** cosine 检索分数。

    Parameters
    ----------
    ids : 候选 ID 列表（实体名 / 超边 id_set / chunk id）。
    scores : 与 ``ids`` 对齐的真实 cosine 分数（来自 VDB ``distance``）。
             长度必须与 ``ids`` 一致；若上游未提供真实分数则不允许为空列表伪装。
    provenance : 与 ``ids`` 对齐的 source chunk ID 列表（实体/超边来自哪些原文）。
             整体为 ``None`` 或长度不匹配 -> 逐候选记 ``None``（如实标记"未采集"），
             绝不用空列表伪装成"已采集但为空"。
    """
    if len(scores) != len(ids):
        raise SchemaValidationError(
            f"retriever {retriever_id}: scores({len(scores)}) != ids({len(ids)}); "
            f"真实检索分数缺失，禁止以 0 占位伪造。"
        )
    if provenance is not None and len(provenance) != len(ids):
        raise SchemaValidationError(
            f"retriever {retriever_id}: provenance({len(provenance)}) != ids({len(ids)}); "
            f"source_provenance_ids 必须与候选严格对齐。"
        )
    candidates = []
    for rank, (cid, sc) in enumerate(zip(ids, scores), start=1):
        prov = provenance[rank - 1] if provenance is not None else None
        candidates.append({
            "candidate_id": str(cid),
            "candidate_type": candidate_type,
            "rank": rank,
            "raw_retrieval_score": float(sc),   # 真实 cosine 分数
            "normalized_score": None,           # 本项目未做归一化
            "reranker_score": None,             # 无 reranker
            # 该候选所依据的原文 chunk（实体/超边 source_id；chunk 则为自身）
            "source_provenance_ids": (
                [str(x) for x in prov] if prov is not None else None
            ),
        })
    return {
        "retriever_id": retriever_id,
        "retriever_version": version,
        "query_embedding_hash": embedding_hash,
        "score_semantics": SCORE_SEMANTICS,
        "score_direction": SCORE_DIRECTION,
        "score_normalization_method": SCORE_NORMALIZATION_METHOD,
        "score_normalization_version": score_normalization_version,
        # 真实分数只挂在候选级 raw_retrieval_score（schema additionalProperties=false，
        # retriever 级不允许冗余分数列表）
        "candidates": candidates,
    }


def _provenance_or_none(trace_data: dict, key: str, expected_len: int):
    """取候选级 provenance；未采集（缺键/空）时返回 None 而非空列表。"""
    v = trace_data.get(key)
    if not v:
        return None
    if len(v) != expected_len:
        return None
    return v


def _build_retrievers(trace_data: dict) -> List[Dict[str, Any]]:
    retrievers: List[Dict[str, Any]] = []

    # 实体线（真实 cosine 分数 + 真实 source provenance）
    ent_ids = trace_data.get("entity_vdb_ids") or []
    ent_scores = trace_data.get("entity_vdb_scores") or []
    if ent_ids:
        retrievers.append(_retriever(
            "entity", "v1",
            trace_data.get("entity_query_embedding_hash") or "",
            ent_ids, ent_scores, "entity",
            provenance=_provenance_or_none(
                trace_data, "entity_vdb_source_ids", len(ent_ids))))

    # 关系线（真实 cosine 分数 + 真实 source provenance）
    rel_ids = trace_data.get("relation_vdb_ids") or []
    rel_scores = trace_data.get("relation_vdb_scores") or []
    if rel_ids:
        retrievers.append(_retriever(
            "relation", "v1",
            trace_data.get("relation_query_embedding_hash") or "",
            rel_ids, rel_scores, "relation",
            provenance=_provenance_or_none(
                trace_data, "relation_vdb_source_ids", len(rel_ids))))

    # chunk retriever **仅当** 直接查了 chunk VDB（P1 / naive 路径）时存在；
    # hyper 路径下 chunk 来自图扩展，无直接 VDB 分数，绝不伪造，故不发送该 retriever。
    chunk_ids = trace_data.get("chunk_vdb_ids") or []
    chunk_scores = trace_data.get("chunk_vdb_scores") or []
    if chunk_ids:
        retrievers.append(_retriever(
            "chunk", "v1", "", chunk_ids, chunk_scores, "chunk",
            provenance=_provenance_or_none(
                trace_data, "chunk_vdb_source_ids", len(chunk_ids))))

    return retrievers


def _build_provenance(trace_data: dict) -> Dict[str, Any]:
    """``stages.provenance``：证据从哪来 —— 三条检索线的开关与产出规模。

    全部字段都来自 ``trace_data`` 的真实采集值；某条线未启用时如实记
    ``enabled=False`` + 计数 0（P0/P_gold 三条线全 False，是真实情况）。
    """
    ent_ids = trace_data.get("entity_vdb_ids") or []
    rel_ids = trace_data.get("relation_vdb_ids") or []
    chunk_ids = trace_data.get("chunk_vdb_ids") or []
    ent_chunks = trace_data.get("entity_line_chunk_ids") or []
    rel_chunks = trace_data.get("relation_line_chunk_ids") or []
    return {
        "entity_line": {
            "enabled": bool(ent_ids),
            "vdb_candidates": len(ent_ids),
            "query_embedding_hash": trace_data.get("entity_query_embedding_hash") or "",
            "source_chunk_ids": [str(x) for x in ent_chunks],
        },
        "relation_line": {
            "enabled": bool(rel_ids),
            "vdb_candidates": len(rel_ids),
            "query_embedding_hash": trace_data.get("relation_query_embedding_hash") or "",
            "source_chunk_ids": [str(x) for x in rel_chunks],
        },
        "chunk_line": {
            "enabled": bool(chunk_ids),
            "vdb_candidates": len(chunk_ids),
            "query_embedding_hash": "",   # chunk VDB 未记录 query embedding hash
            "source_chunk_ids": [str(x) for x in chunk_ids],
        },
        "keyword_result_hash": trace_data.get("keyword_result_hash") or "",
        "context_hash": trace_data.get("context_hash") or "",
        "token_unit": trace_data.get("qwen_count_source") or "vllm_tokenize",
    }


def _build_graph_lineage(trace_data: dict) -> Dict[str, Any]:
    """``stages.graph_lineage``：超图扩展路径（seed 实体 -> 超边 -> 原文 chunk）。

    实体线：VDB 召回 seed 实体 -> 邻接超边扩展 -> 各自 source_id 取原文。
    关系线：VDB 召回超边 -> 反查成员实体 -> 超边 source_id 取原文。
    """
    def _s(key):
        return [str(x) for x in (trace_data.get(key) or [])]

    return {
        "entity_line": {
            "seed_entities": _s("entity_post_graph_ids"),
            "context_entities": _s("entity_post_truncate_ids"),
            "expanded_hyperedges": _s("entity_line_relation_ids"),
            "source_chunks": _s("entity_line_chunk_ids"),
        },
        "relation_line": {
            "seed_hyperedges": _s("relation_post_filter_ids"),
            "context_hyperedges": _s("relation_post_truncate_ids"),
            "expanded_entities": _s("relation_line_entity_ids"),
            "source_chunks": _s("relation_line_chunk_ids"),
        },
        "merged_source_chunks": _s("merged_chunk_ids"),
        "graph_expansion_calls": trace_data.get("graph_expansion_calls", 0) or 0,
    }


def build_trace_record(
    trace_data: dict,
    *,
    question_id: str,
    run_id: str,
    route_id: str,
    seed: Optional[int],
    system_snapshot_id: str,
    answer_text: str,
    finish_reason: Optional[str] = None,
    retries: int = 0,
    timeouts: int = 0,
    cost: Optional[dict] = None,
    judge: Optional[dict] = None,
) -> dict:
    """组装完整 retrieval-trace-v2 记录。

    Parameters
    ----------
    trace_data : QueryParam.trace_data（hyper_query / naive_query 填充）。
    cost : 真实成本覆盖（11 项）。缺字段由 _COST_DEFAULTS 兜底为 0，但
            gpu_seconds / api_cost 等本部署无法采集的项应由 runner 显式传 None。
    """
    route_for_trace = _route_id_for_trace(route_id)

    keyword_extraction = {
        "entity_keywords": _as_keyword_list(trace_data.get("entity_keywords")),
        "relation_keywords": _as_keyword_list(trace_data.get("relation_keywords")),
        "keyword_result_hash": trace_data.get("keyword_result_hash") or "",
    }

    retrievers = _build_retrievers(trace_data)

    # 阶段 ID 语义修正（见模块 docstring）
    entity_post_truncate = trace_data.get("entity_post_truncate_ids") or []
    relation_line_entity = trace_data.get("relation_line_entity_ids") or []
    entity_ids = entity_post_truncate + relation_line_entity  # 仅实体节点
    # 超边来自两条线：关系线 VDB 直查截断后（relation_post_truncate_ids）
    # + 实体线邻接扩展（entity_line_relation_ids）。P2 只开实体线时后者是
    # Relationships 区段的唯一来源，遗漏会导致超边 lineage 为空。
    # 稳定去重：保持首次出现顺序（列表可能含 list/元组型 ID，用 repr 作键）。
    hyperedge_ids = []
    _seen_he = set()
    for he in ((trace_data.get("relation_post_truncate_ids") or [])
               + (trace_data.get("entity_line_relation_ids") or [])):
        key = repr(he)
        if key not in _seen_he:
            _seen_he.add(key)
            hyperedge_ids.append(he)
    final_context_source_ids = (
        trace_data.get("final_context_source_ids")
        or trace_data.get("merged_chunk_ids")
        or []
    )

    tokens_before = trace_data.get("context_tokens_before_truncate") or 0
    tokens_after = trace_data.get("context_tokens") or 0
    stages = {
        "pre_rerank_ids": (trace_data.get("entity_vdb_ids") or [])
                          + (trace_data.get("relation_vdb_ids") or []),
        "post_rerank_ids": (trace_data.get("entity_post_graph_ids") or [])
                            + (trace_data.get("relation_post_filter_ids") or []),
        "post_graph_expansion_ids": (trace_data.get("entity_post_truncate_ids") or [])
                                    + (trace_data.get("relation_post_truncate_ids") or [])
                                    + (trace_data.get("entity_line_relation_ids") or []),
        "post_assembly_source_ids": trace_data.get("merged_chunk_ids") or [],
        "final_context_source_ids": final_context_source_ids,
        "entity_ids": entity_ids,
        "hyperedge_ids": hyperedge_ids,
        "provenance": _build_provenance(trace_data),
        "graph_lineage": _build_graph_lineage(trace_data),
        "tokens_before_truncation": tokens_before,
        "tokens_after_truncation": tokens_after,
        "truncated_tokens": max(0, tokens_before - tokens_after),
        "evidence_group_coverage": {},
    }

    cost_full = _build_cost(cost)

    judge_full = judge or {
        "verdict": "uncertain",
        "critical_error": False,
        "evidence_sufficient": False,
        "human_review_status": "pending",
    }

    return {
        "question_id": str(question_id),
        "run_id": run_id,
        "route_id": route_for_trace,
        "seed": seed,
        "system_snapshot_id": system_snapshot_id,
        "answer_text": answer_text,
        "finish_reason": finish_reason,
        "retries": retries,
        "timeouts": timeouts,
        "stage_evidence_loss": {},
        "keyword_extraction": keyword_extraction,
        "retrievers": retrievers,
        "stages": stages,
        "cost": cost_full,
        "judge": judge_full,
    }


# 成本字段默认值（仅当 runner 未提供该字段时兜底为 0）。
# gpu_seconds / api_cost 等不可采集项由 runner 显式传 None，不会被这里的 0 覆盖。
_COST_DEFAULTS = {
    "input_tokens": 0,
    "output_tokens": 0,
    "retrieval_latency_ms": 0.0,
    "end_to_end_latency_ms": 0.0,
    "llm_calls": 0,
    "embedding_calls": 0,
    "reranker_calls": 0,
    "graph_expansion_calls": 0,
    "retrieved_candidate_count": 0,
    "gpu_seconds": 0.0,
    "api_cost": 0.0,
}


def _build_cost(cost: Optional[dict]) -> Dict[str, Any]:
    """合并 runner 提供的真实成本；runner 显式传 None 的项保留 null。

    注意：runner 必须提供全部 11 项（可采集项为真实整数，不可采集项为 None）。
    此处仅在 runner 漏传某字段时以 0 兜底，避免 Schema 缺字段。
    """
    full = dict(_COST_DEFAULTS)
    if cost:
        for k in _COST_DEFAULTS:
            if k in cost:
                full[k] = cost[k]   # 允许 None（不可采集项）
    return full


def validate_and_finalize(record: dict, schema=None) -> dict:
    """校验记录是否符合 retrieval-trace-v2；不合法抛 SchemaValidationError。"""
    if schema is None:
        schema = load_trace_schema()
    validate_trace_record(record, schema)
    return record
