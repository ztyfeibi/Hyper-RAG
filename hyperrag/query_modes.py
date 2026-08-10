"""Non-streaming query modes for HyperRAG."""

import asyncio
import hashlib
import json
import time

from .base import BaseHypergraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam, TextChunkSchema
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .query_router import route_query, QueryRoute
from .adaptive_params import apply_adaptive_params, get_adaptive_param_dict
from .context_budget import apply_qwen_context_budget
from .query_context import (
    _build_entity_query_context,
    _build_relation_query_context,
    combine_contexts,
)
from .query_keywords import parse_low_level_keywords, parse_query_keywords
from .utils import (
    compute_mdhash_id,
    deduplicate_by_key,
    list_of_list_to_csv,
    logger,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)


async def hyper_query(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """完整 Hyper-RAG 查询。

    同时走两条线：
    - low_level_keywords -> 实体向量库 -> 超图 vertex 扩展；
    - high_level_keywords -> 关系向量库 -> 超图 hyperedge 扩展。
    两条线合并后，把 Entities / Relationships / Sources 交给 LLM 回答。
    """
    entity_context = None
    relation_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    # 第一步：先让 LLM 从用户问题里抽两类关键词。
    # low_level_keywords 偏实体；high_level_keywords 偏关系/主题。
    result = await use_model_func(
        kw_prompt, temperature=query_param.llm_temperature
    )

    try:
        entity_keywords, relation_keywords = parse_query_keywords(result, kw_prompt)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return PROMPTS["fail_response"]
    """
        Perform different actions based on keywords:
            ll_keywords: Find information based on low-level keywords.
            hl_keywords: Define topic information based on high-level keywords.
    """
    if entity_keywords:
        """
        low_level_context: Retrieves vertices and their first-order neighbor hyperedges.
        high_level_context: Retrieves hyperedges and their first-order neighbor vertices.
        """
        # 实体线输入：低阶关键词字符串。
        # 实体线输出：相关实体、这些实体连接的超边、以及 source_id 对应的原文 chunk。
        entity_context = await _build_entity_query_context(
            entity_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if relation_keywords:
        # 关系线输入：高阶关键词字符串。
        # 关系线输出：相关超边、超边连接的实体、以及 source_id 对应的原文 chunk。
        relation_context = await _build_relation_query_context(
            relation_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    """
        combine the information from the local_query and global_query,
        so that we can have the final retrieval information.
    """
    # P2（关系线禁用）/ P4（实体线禁用）等固定路径下，对应 context 为 None
    context = combine_contexts(
        relation_context.get("context") if relation_context else None,
        entity_context.get("context") if entity_context else None,
    )

    # Step 4.5: Apply total-context budget (section-aware fuse).
    # 契约 final_context_hard_cap 是 **Qwen token**；经校准保守转换为 tiktoken
    # 预算再截断，保证最坏情况不突破 Qwen 硬上限（问题 2）。
    context, context_budget = apply_qwen_context_budget(
        context,
        qwen_hard_cap=query_param.effective_final_cap(),
        tiktoken_model_name=global_config.get("tiktoken_model_name", "gpt-4o-mini"),
        section_allocation_ratios=query_param.section_allocation_ratios,
    )

    # Log section-aware budget info for debugging
    logger.info(
        f"Context budget (Qwen-aware): truncated={context_budget.get('context_truncated', False)}, "
        f"qwen_before={context_budget.get('context_tokens_qwen_before', 0)}, "
        f"qwen_after={context_budget.get('context_tokens_qwen_after', 0)}, "
        f"tiktoken_budget={context_budget.get('tiktoken_budget')}, "
        f"entity={context_budget.get('entity_tokens', 0)}, "
        f"relation={context_budget.get('relation_tokens', 0)}, "
        f"source={context_budget.get('source_tokens', 0)}"
    )

    # 保留结构化结果，供 Web-UI 展示检索证据和超图关系。
    _ec = entity_context or {}
    _rc = relation_context or {}
    contextJson = {
        "entities": deduplicate_by_key(_ec.get("entities", []) + _rc.get("entities", []), "entity_name"),
        "hyperedges": deduplicate_by_key(_ec.get("hyperedges", []) + _rc.get("hyperedges", []), "entity_set"),
        "text_units": deduplicate_by_key(_ec.get("text_units", []) + _rc.get("text_units", []), "content")
    }

    # Step 5.4: Collect trace data for reproducibility diagnostics.
    # Populates query_param.trace_data (shared dict via dataclasses.replace).
    if query_param.save_trace:
        # Entity-line: 3-stage IDs + chunk IDs
        entity_vdb_ids = (
            entity_context.get("entity_vdb_ids", []) if entity_context else []
        )
        entity_query_embedding_hash = (
            entity_context.get("entity_query_embedding_hash", "") if entity_context else ""
        )
        entity_post_graph_ids = (
            entity_context.get("entity_post_graph_ids", []) if entity_context else []
        )
        entity_post_truncate_ids = (
            entity_context.get("entity_post_truncate_ids", []) if entity_context else []
        )
        entity_line_relation_ids = (
            entity_context.get("entity_line_relation_ids", []) if entity_context else []
        )
        entity_line_chunk_ids = (
            entity_context.get("text_unit_ids", []) if entity_context else []
        )

        # Relation-line: 3-stage IDs + chunk IDs
        relation_vdb_ids = (
            relation_context.get("relation_vdb_ids", []) if relation_context else []
        )
        relation_query_embedding_hash = (
            relation_context.get("relation_query_embedding_hash", "") if relation_context else ""
        )
        relation_post_filter_ids = (
            relation_context.get("relation_post_filter_ids", []) if relation_context else []
        )
        relation_post_truncate_ids = (
            relation_context.get("relation_post_truncate_ids", []) if relation_context else []
        )
        relation_line_entity_ids = (
            relation_context.get("relation_line_entity_ids", []) if relation_context else []
        )
        relation_line_chunk_ids = (
            relation_context.get("text_unit_ids", []) if relation_context else []
        )

        # Merged chunk IDs after dedup
        merged_chunk_ids = [
            compute_mdhash_id(t["content"], prefix="chunk-")
            for t in contextJson.get("text_units", [])
        ]
        # 截断后真正保留进 final context 的 source chunk（区段感知尾截断：
        # 幸存的是前缀，按 content 是否在最终 context 中判定，绝不臆造）。
        final_context_source_ids = [
            cid for cid, tu in zip(merged_chunk_ids, contextJson.get("text_units", []))
            if tu.get("content") and tu["content"] in context
        ]

        # Context budget metrics
        # 契约要求上报 **实际 Qwen token 计数**（非 tiktoken 近似值）。
        context_tokens = context_budget.get("context_tokens_qwen_after", 0)
        context_tokens_before = context_budget.get("context_tokens_qwen_before", 0)
        context_truncated = context_budget.get("context_truncated", False)
        # 真实检索计数（来自两条检索线的热路径采集；绝不补零）
        embedding_calls = (
            (entity_context.get("embedding_calls", 0) if entity_context else 0)
            + (relation_context.get("embedding_calls", 0) if relation_context else 0)
        )
        graph_expansion_calls = (
            (entity_context.get("graph_expansion_calls", 0) if entity_context else 0)
            + (relation_context.get("graph_expansion_calls", 0) if relation_context else 0)
        )
        retrieval_latency_ms = (
            (entity_context.get("retrieval_latency_ms", 0.0) if entity_context else 0.0)
            + (relation_context.get("retrieval_latency_ms", 0.0) if relation_context else 0.0)
        )
        retrieved_candidate_count = (
            (entity_context.get("retrieved_candidate_count", 0) if entity_context else 0)
            + (relation_context.get("retrieved_candidate_count", 0) if relation_context else 0)
        )

        # Root input to both VDB lines: the raw keyword extraction response.
        # Recompute a stable hash so trace stays small (no full raw text).
        keyword_result_hash = hashlib.sha256(
            result.encode("utf-8", errors="replace")
        ).hexdigest()
        query_param.trace_data.update({
            # Keyword extraction (root input to both VDB lines)
            "entity_keywords": entity_keywords if entity_keywords else "",
            "relation_keywords": relation_keywords if relation_keywords else "",
            "keyword_result_hash": keyword_result_hash,
            # Query embedding hashes (diagnose embedding API stability)
            "entity_query_embedding_hash": entity_query_embedding_hash,
            "relation_query_embedding_hash": relation_query_embedding_hash,
            # Entity-line（含真实 cosine 检索分数，绝不补零）
            "entity_vdb_ids": entity_vdb_ids,
            "entity_vdb_scores": (
                entity_context.get("entity_vdb_scores", []) if entity_context else []
            ),
            # 问题 3：候选级 source provenance（与 entity_vdb_ids 对齐）
            "entity_vdb_source_ids": (
                entity_context.get("entity_vdb_source_ids", []) if entity_context else []
            ),
            "entity_post_graph_ids": entity_post_graph_ids,
            "entity_post_truncate_ids": entity_post_truncate_ids,
            "entity_line_relation_ids": entity_line_relation_ids,
            "entity_line_chunk_ids": entity_line_chunk_ids,
            # Relation-line（含真实 cosine 检索分数，绝不补零）
            "relation_vdb_ids": relation_vdb_ids,
            "relation_vdb_scores": (
                relation_context.get("relation_vdb_scores", []) if relation_context else []
            ),
            # 问题 3：候选级 source provenance（与 relation_vdb_ids 对齐）
            "relation_vdb_source_ids": (
                relation_context.get("relation_vdb_source_ids", []) if relation_context else []
            ),
            "relation_post_filter_ids": relation_post_filter_ids,
            "relation_post_truncate_ids": relation_post_truncate_ids,
            "relation_line_entity_ids": relation_line_entity_ids,
            "relation_line_chunk_ids": relation_line_chunk_ids,
            # Merged
            "merged_chunk_ids": merged_chunk_ids,
            "final_context_source_ids": final_context_source_ids,
            # Context budget（上报 Qwen token 真实估算值）
            "context_tokens": context_tokens,
            "context_tokens_before_truncate": context_tokens_before,
            "context_truncated": context_truncated,
            # 真实检索成本计数（绝不以 0 占位掩盖缺失；缺失字段由 trace_collector 标记）
            "embedding_calls": embedding_calls,
            "graph_expansion_calls": graph_expansion_calls,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieved_candidate_count": retrieved_candidate_count,
            # Context hash
            "context_hash": hashlib.md5(context.encode()).hexdigest(),
            # Query params (legacy fields retained for backward compatibility)
            "effective_params": {
                "mode": query_param.mode,
                "top_k": query_param.top_k,
                "max_token_for_text_unit": query_param.max_token_for_text_unit,
                "max_token_for_entity_context": query_param.max_token_for_entity_context,
                "max_token_for_relation_context": query_param.max_token_for_relation_context,
                "max_total_context_tokens": query_param.max_total_context_tokens,
                "max_response_tokens": query_param.max_response_tokens,
                "llm_temperature": query_param.llm_temperature,
                # --- Step 1.3 split retrieval params (raw) ---
                "chunk_vdb_top_k": query_param.chunk_vdb_top_k,
                "entity_vdb_top_k": query_param.entity_vdb_top_k,
                "relation_vdb_top_k": query_param.relation_vdb_top_k,
                "entity_description_cap": query_param.entity_description_cap,
                "relation_description_cap": query_param.relation_description_cap,
                "source_text_cap": query_param.source_text_cap,
                "final_context_hard_cap": query_param.final_context_hard_cap,
                "route_id": query_param.route_id,
                "repeat_id": query_param.repeat_id,
                "repeat_seed": query_param.repeat_seed,
                "system_snapshot_id": query_param.system_snapshot_id,
                "contract_version": query_param.contract_version,
                # --- Step 1.3 split retrieval params (effective resolved) ---
                "effective_chunk_top_k": query_param.effective_chunk_top_k(),
                "effective_entity_top_k": query_param.effective_entity_top_k(),
                "effective_relation_top_k": query_param.effective_relation_top_k(),
                "effective_entity_cap": query_param.effective_entity_cap(),
                "effective_relation_cap": query_param.effective_relation_cap(),
                "effective_source_cap": query_param.effective_source_cap(),
                "effective_final_cap": query_param.effective_final_cap(),
                # --- 问题 2：契约 cap 是 Qwen token，热路径截断按 tiktoken 计数。
                # 记录实际生效的 tiktoken 预算，便于核对"是否按冻结契约执行"。
                "caps_unit": ("qwen_tokenizer_token"
                              if query_param.caps_in_qwen_unit() else "tiktoken"),
                "tiktoken_entity_cap": query_param.tiktoken_entity_cap(),
                "tiktoken_relation_cap": query_param.tiktoken_relation_cap(),
                "tiktoken_source_cap": query_param.tiktoken_source_cap(),
            },
        })

    if query_param.only_need_context:
        if query_param.effective_final_cap() is not None:
            return {"context": context, "context_budget": context_budget}
        return context
    if context is None:
        return PROMPTS["fail_response"]
    define_str = ""
    if entity_keywords or relation_keywords:
        """
        High-level keywords serve as qualifiers to the topic information
        """
        entity_keywords = entity_keywords if entity_keywords else ""
        relation_keywords = relation_keywords if relation_keywords else ""
        define_str = PROMPTS["rag_define"]
        define_str = define_str.format(ll_keywords=entity_keywords,hl_keywords=relation_keywords)
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query + define_str,
        system_prompt=sys_prompt,
        max_tokens=query_param.max_response_tokens,
        temperature=query_param.llm_temperature,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    if query_param.return_type == "json":
        contextJson["response"] = response
        response = contextJson
    return response

async def adaptive_query(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Adaptive v3: Router classifies query, parameters adapted, type-aware weighting enabled.

    Step 5.2: supports three router_policy modes:
    - ``llm``    : call route_query (default, original behavior)
    - ``fixed``  : skip router, use query_param.forced_complexity
    - ``oracle`` : skip router, use query_param.forced_complexity (set per-question by Step_3)

    Step 5.1: context budget is now section-aware, preserving Sources.
    Route + adaptive_params + context_budget are attached to the context-only return.
    """
    # --- Router policy: determine route ---
    router_policy = getattr(query_param, "router_policy", "llm")
    forced_complexity = getattr(query_param, "forced_complexity", None)

    if router_policy in ("fixed", "oracle") and forced_complexity:
        # Skip LLM router; construct route directly from forced_complexity
        route = QueryRoute(
            query_type="unknown",
            complexity=forced_complexity,
            focus_types=["entity", "relation", "text"],
            reason=f"router_policy={router_policy}, forced_complexity={forced_complexity}",
        )
        logger.info(
            f"Adaptive route (policy={router_policy}): complexity={route.complexity} "
            f"(forced, no LLM call)"
        )
    else:
        # Default: call LLM router
        route = await route_query(query, query_param, global_config)
        logger.info(
            f"Adaptive route (policy=llm): type={route.query_type}, complexity={route.complexity}, "
            f"focus={route.focus_types}"
        )

    adaptive_param = apply_adaptive_params(query_param, route)

    # Step 4: pass focus_types and enable weighting for type-aware re-ranking
    from dataclasses import replace
    adaptive_param = replace(
        adaptive_param,
        route_focus_types=route.focus_types,
        enable_type_aware_weighting=True,
    )

    result = await hyper_query(
        query,
        knowledge_hypergraph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        adaptive_param,
        global_config,
    )

    # Step 5.4: Add route info to trace data (trace_data is shared dict
    # between query_param and adaptive_param via dataclasses.replace).
    if query_param.save_trace:
        query_param.trace_data["route"] = route.to_dict()
        query_param.trace_data["router_policy"] = router_policy

    # For context-only mode, attach route + adaptive_params + context_budget
    if query_param.only_need_context:
        if isinstance(result, dict):
            payload = result
        else:
            payload = {"context": result}
        payload["route"] = route.to_dict()
        payload["adaptive_params"] = get_adaptive_param_dict(route)
        payload["router_policy"] = router_policy
        return payload

    return result


async def hyper_query_lite(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """轻量 Hyper-RAG 查询。

    主要走实体线：low_level_keywords -> entities_vdb -> vertex/邻接超边/source chunks。
    相比 hyper_query，它不主动走 high_level_keywords 的关系向量检索。
    """

    entity_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(
        kw_prompt, temperature=query_param.llm_temperature
    )

    try:
        entity_keywords = parse_low_level_keywords(result, kw_prompt)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return PROMPTS["fail_response"]
    """
        Perform different actions based on keywords:
            ll_keywords: Find information based on low-level keywords.
    """
    if entity_keywords:
        """
        low_level_context: Retrieves vertices and their first-order neighbor hyperedges.
        high_level_context: Retrieves hyperedges and their first-order neighbor vertices.
        """
        entity_context = await _build_entity_query_context(
            entity_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    """
        combine the information from the local_query and global_query,
        so that we can have the final retrieval information.
    """
    context = entity_context.get("context")

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    define_str = ""
    if entity_keywords:
        """
        High-level keywords serve as qualifiers to the topic information
        """
        entity_keywords = entity_keywords if entity_keywords else ""
        define_str = PROMPTS["rag_define"]
        define_str = define_str.format(ll_keywords=entity_keywords, hl_keywords="")
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query + define_str,
        system_prompt=sys_prompt,
        max_tokens=query_param.max_response_tokens,
        temperature=query_param.llm_temperature,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    if query_param.return_type == "json":
        entity_context["response"] = response
        response = entity_context
    return response

async def graph_query(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # Graph-RAG 对照查询：只保留二元关系，模拟传统图 RAG 的 pairwise 边。
    """
    检索和返回 hypergraph db 中的成对关系
    """
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(
        kw_prompt, temperature=query_param.llm_temperature
    )
    try:
        entity_keywords, relation_keywords = parse_query_keywords(result, kw_prompt)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return PROMPTS["fail_response"]

    # 只处理二元关系
    def filter_pairwise_edges(edges):
        return [e for e in edges if isinstance(e.get("id_set"), (list, tuple)) and len(e["id_set"]) == 2]

    # 获取所有相关的二元关系
    relation_context = None
    if relation_keywords:
        results = await relationships_vdb.query(relation_keywords, top_k=query_param.effective_relation_top_k())
        if not len(results):
            return PROMPTS["fail_response"]
        edge_datas = await asyncio.gather(
            *[knowledge_hypergraph_inst.get_hyperedge(r['id_set']) for r in results]
        )
        edge_degree = await asyncio.gather(
            *[knowledge_hypergraph_inst.hyperedge_degree(e['id_set']) for e in results]
        )
        edge_datas = [
            {"id_set": k["id_set"], "rank": d, **v}
            for k, v, d in zip(results, edge_datas, edge_degree)
            if v is not None
        ]
        # 只保留二元关系
        edge_datas = filter_pairwise_edges(edge_datas)
        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        edge_datas = truncate_list_by_token_size(
            edge_datas,
            key=lambda x: x["description"],
            max_token_size=query_param.tiktoken_relation_cap(),
        )
        # 相关实体
        entity_names = set()
        for e in edge_datas:
            for f in e["id_set"]:
                if await knowledge_hypergraph_inst.has_vertex(f):
                    entity_names.add(f)
        node_datas = await asyncio.gather(
            *[knowledge_hypergraph_inst.get_vertex(entity_name) for entity_name in entity_names]
        )
        node_degrees = await asyncio.gather(
            *[knowledge_hypergraph_inst.vertex_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
            if n is not None
        ]
        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size=query_param.tiktoken_entity_cap(),
        )
        # 相关文本
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in edge_datas
        ]
        all_text_units_lookup = {}
        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await text_chunks_db.get_by_id(c_id),
                        "order": index,
                    }
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None and v["data"] is not None
        ]
        all_text_units = sorted(all_text_units, key=lambda x: x["order"])
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"]["content"],
            max_token_size=query_param.tiktoken_source_cap(),
        )
        all_text_units = [t["data"] for t in all_text_units]
        # 格式化 context
        relations_section_list = [
            ["id", "entity set", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(edge_datas):
            relations_section_list.append(
                [
                    i,
                    e["id_set"],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_of_list_to_csv(relations_section_list)
        entites_section_list = [["id", "entity", "type", "description", "additional properties", "rank"]]
        for i, n in enumerate(node_datas):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n.get("additional_properties", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_of_list_to_csv(entites_section_list)
        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(all_text_units):
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_of_list_to_csv(text_units_section_list)
        context_string = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
        contextJson = {
            "context": context_string,
            "entities": [
                {
                    "id": i,
                    "entity_name": n["entity_name"],
                    "entity_type": n.get("entity_type", "UNKNOWN"),
                    "description": n.get("description", "UNKNOWN"),
                    "additional_properties": n.get("additional_properties", "UNKNOWN"),
                    "rank": n["rank"]
                }
                for i, n in enumerate(node_datas)
            ],
            "hyperedges": [
                {
                    "id": i,
                    "entity_set": e["id_set"],
                    "description": e["description"],
                    "keywords": e["keywords"],
                    "weight": e["weight"],
                    "rank": e["rank"]
                }
                for i, e in enumerate(edge_datas)
            ],
            "text_units": [
                {
                    "id": i,
                    "content": t["content"]
                }
                for i, t in enumerate(all_text_units)
            ]
        }
        if query_param.only_need_context:
            return context_string
        if context_string is None:
            return PROMPTS["fail_response"]
        define_str = ""
        if entity_keywords or relation_keywords:
            entity_keywords = entity_keywords if entity_keywords else ""
            relation_keywords = relation_keywords if relation_keywords else ""
            define_str = PROMPTS["rag_define"]
            define_str = define_str.format(ll_keywords=entity_keywords,hl_keywords=relation_keywords)
        sys_prompt_temp = PROMPTS["rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context_string, response_type=query_param.response_type
        )
        response = await use_model_func(
            query + define_str,
            system_prompt=sys_prompt,
            max_tokens=query_param.max_response_tokens,
            temperature=query_param.llm_temperature,
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        if query_param.return_type == "json":
            contextJson["response"] = response
            response = contextJson
        return response
    else:
        return PROMPTS["fail_response"]

async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """普通 RAG 查询：query -> chunks_vdb -> text_chunks_db -> LLM。"""
    use_model_func = global_config["llm_model_func"]
    # 真实检索耗时 + 真实 cosine 分数（绝不补零）
    _t0 = time.perf_counter()
    results = await chunks_vdb.query(query, top_k=query_param.effective_chunk_top_k())
    retrieval_latency_ms = (time.perf_counter() - _t0) * 1000.0
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunk_scores = [float(r.get("distance", 0.0)) for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.tiktoken_source_cap(),
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    # 真实检索成本计数（绝不补零）
    if query_param.save_trace:
        # 契约 units.token_unit=qwen_tokenizer_token：上报 **真实** Qwen 计数
        # （vLLM /tokenize），不再用校准比例估算；端点不可用则 fail-fast。
        from .qwen_tokenizer import get_qwen_token_counter
        _qc = get_qwen_token_counter()
        section_before = "--New Chunk--\n".join([c["content"] for c in chunks])
        qwen_before = _qc(section_before)
        qwen_after = _qc(section)
        # 真实计数强制 final hard cap：tiktoken 截断只是近似，实测仍超限
        # 则从尾部继续丢 chunk 直至合规（禁止上报超限 context）。
        _cap = query_param.effective_final_cap()
        while _cap is not None and qwen_after > _cap and maybe_trun_chunks:
            maybe_trun_chunks = maybe_trun_chunks[:-1]
            section = "--New Chunk--\n".join(
                [c["content"] for c in maybe_trun_chunks])
            qwen_after = _qc(section)
        final_ids = [
            compute_mdhash_id(c["content"], prefix="chunk-")
            for c in maybe_trun_chunks
        ]
        query_param.trace_data.update({
            "chunk_vdb_ids": chunks_ids,
            "chunk_vdb_scores": chunk_scores,
            # 问题 3：chunk 候选的 provenance 就是它自身（原文即证据来源）
            "chunk_vdb_source_ids": [[cid] for cid in chunks_ids],
            "merged_chunk_ids": chunks_ids,
            "final_context_source_ids": final_ids,
            "context_tokens": qwen_after,
            "context_tokens_before_truncate": qwen_before,
            "context_truncated": len(maybe_trun_chunks) < len(chunks),
            "context_hash": hashlib.md5(section.encode()).hexdigest(),
            "qwen_count_source": "vllm_tokenize",
            "embedding_calls": 1,
            "graph_expansion_calls": 0,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieved_candidate_count": len(results),
        })
    if query_param.only_need_context:
        return {"context": section, "trace_data": dict(query_param.trace_data)}
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        max_tokens=query_param.max_response_tokens,
        temperature=query_param.llm_temperature,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    if query_param.return_type == "json":
        response = {
            "response": response,
        }
    return response

async def llm_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    # 纯 LLM 查询：不检索任何本地数据，作为无 RAG 基线。
    """
    只调用 LLM，不进行任何数据查询。
    """
    use_model_func = global_config["llm_model_func"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data="", response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        max_tokens=query_param.max_response_tokens,
        temperature=query_param.llm_temperature,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    if query_param.return_type == "json":
        response = {
            "response": response,
        }
    return response
