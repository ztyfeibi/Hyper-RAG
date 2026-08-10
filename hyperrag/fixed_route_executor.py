# -*- coding: utf-8 -*-
"""Step 1.4: 固定路径执行器 —— 统一回答 Prompt + 固定 P0-P4 检索配置。

设计目标
--------
1. **检索参数唯一来源**：P0-P4 的全部 top-k / 预算 100% 来自冻结契约
   （``experiment_contract.RouteConfig``）。runner / 分析脚本禁止复制这些数值。
2. **统一回答模板**：P0-P4 与 P_gold 共用 ``prompt.py`` 中的
   ``formal_answer_response`` 模板，保证 prompt hash 完全一致；不同路径的差异
   只体现在检索到的证据（``{context_data}``），而不是 prompt 措辞。
3. **路径语义**：
   - P0  : 空 context（llm-only，不检索）。
   - P1  : source-only（仅 chunk 检索）。
   - P2  : hyper-lite（实体线 + 实体邻接扩展；Relation VDB 关闭）。
   - P3  : 标准 hyper。
   - P4  : 扩展 hyper。
   - P_gold : Gold 证据直填（先用测试 fixture；正式 Gold 在 Step 2 接入）。

接口::

    from hyperrag.fixed_route_executor import execute_fixed_route, build_query_param
    result = await execute_fixed_route(
        question="...",
        route_id="P3",
        contract=load_contract(),
        rag=rag,                  # HyperRAG 实例
        repeat_id=1,
        repeat_seed=42,
        system_snapshot_id="...",
    )
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Optional

from .base import QueryParam
from .prompt import PROMPTS
from .experiment_contract import (
    ExperimentContract,
    RouteConfig,
    GOLD_ROUTE_ID,
    ContractError,
)
from .query_modes import hyper_query, naive_query


# ---------------------------------------------------------------------------
# 固定路径 -> 检索模式
# ---------------------------------------------------------------------------
# P0 由 RouteConfig.is_llm_only 单独判定为 "llm"；其余 P1 走 naive、
# P2/P3/P4 走 hyper（relation_vdb_top_k=0 时 hyper_query 内部自动跳过
# Relation VDB，仅实体邻接扩展生成 Relationships —— 即 P2 的 "hyper-lite"）。
_ROUTE_MODE = {
    "P1": "naive",
    "P2": "hyper",
    "P3": "hyper",
    "P4": "hyper",
}


def route_to_mode(route_id: str) -> str:
    """固定路径 -> HyperRAG 检索模式（P0 由调用方按 is_llm_only 处理）。"""
    if route_id not in _ROUTE_MODE:
        raise ContractError(
            f"unknown fixed route_id {route_id!r}; valid: P0..P4, {GOLD_ROUTE_ID}"
        )
    return _ROUTE_MODE[route_id]


def formal_answer_prompt() -> str:
    """唯一正式回答模板；P0-P4+P_gold 共用，保证 prompt hash 一致。"""
    return PROMPTS["formal_answer_response"]


def formal_answer_prompt_hash() -> str:
    """正式回答模板的 sha256；system_snapshot 记录此值以保证可复现。"""
    return hashlib.sha256(formal_answer_prompt().encode("utf-8")).hexdigest()


def build_query_param(
    route_config: RouteConfig,
    contract: ExperimentContract,
    route_id: str,
    repeat_id: Optional[int] = None,
    repeat_seed: Optional[int] = None,
    system_snapshot_id: Optional[str] = None,
) -> QueryParam:
    """从 RouteConfig 构造固定路径检索参数。

    拆分检索字段（chunk/entity/relation_vdb_top_k、各 cap、final cap）与对应的
    legacy 字段（top_k / max_token_for_* / max_total_context_tokens）**同步赋值**，
    既满足 effective_* 优先取拆分字段的契约，又避免 legacy 字段残留脏值。
    系统边界（temperature / max_response_tokens）取自契约，固定路径额外关闭
    router / type-aware weighting（由 1.5 CLI 强约束）。
    """
    sb = contract.system_boundary
    mode = "llm" if route_config.is_llm_only else route_to_mode(route_id)

    representative_top_k = max(
        route_config.chunk_vdb_top_k,
        route_config.entity_vdb_top_k,
        route_config.relation_vdb_top_k,
    )

    return QueryParam(
        mode=mode,
        # legacy 字段与拆分字段保持一致（合法 int，无脏值）
        top_k=representative_top_k,
        max_token_for_text_unit=route_config.source_text_cap,
        max_token_for_entity_context=route_config.entity_description_cap,
        max_token_for_relation_context=route_config.relation_description_cap,
        max_total_context_tokens=route_config.final_context_hard_cap,
        # 拆分检索参数（唯一来源：契约）
        chunk_vdb_top_k=route_config.chunk_vdb_top_k,
        entity_vdb_top_k=route_config.entity_vdb_top_k,
        relation_vdb_top_k=route_config.relation_vdb_top_k,
        entity_description_cap=route_config.entity_description_cap,
        relation_description_cap=route_config.relation_description_cap,
        source_text_cap=route_config.source_text_cap,
        final_context_hard_cap=route_config.final_context_hard_cap,
        # 系统边界（冻结）
        max_response_tokens=sb.max_response_tokens,
        llm_temperature=sb.temperature,
        # 固定路径：关闭 router / type-aware（由 1.5 CLI 强约束 disable）
        router_policy="fixed",
        enable_type_aware_weighting=False,
        # 实验可追溯字段（写入 trace 记录；不影响检索）
        route_id=route_id,
        repeat_id=repeat_id,
        repeat_seed=repeat_seed,
        system_snapshot_id=system_snapshot_id,
        contract_version=contract.contract_version,
        # 契约区段配比（来自 contract.section_allocation_ratios，小写键）；
        # 截断时优先使用，回退到 context_budget 模块常量。
        section_allocation_ratios=contract.section_allocation_ratios,
    )


# ---------------------------------------------------------------------------
# 检索子步骤（only_need_context=True 只取 context 字符串，不做回答）
# ---------------------------------------------------------------------------
async def _retrieve_context(
    question: str,
    route_id: str,
    param: QueryParam,
    rag,
) -> str:
    """按路径执行检索，返回合并后的 context 字符串（不含 LLM 回答）。"""
    gcfg = asdict(rag)
    if route_id == "P1":
        ctx = await naive_query(
            question, rag.chunks_vdb, rag.text_chunks, param, gcfg
        )
        # naive_query 在 only_need_context=True 且 save_trace 时返回
        # {"context": str, "trace_data": dict}；与 hyper 分支同样处理，
        # 绝不把真实检索结果丢弃成空 context（曾导致 P1 退化为 P0）。
        if isinstance(ctx, dict):
            return ctx.get("context", "")
        return ctx if isinstance(ctx, str) else ""
    # P2 / P3 / P4 -> hyper 双线检索
    ctx = await hyper_query(
        question,
        rag.chunk_entity_relation_hypergraph,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.text_chunks,
        param,
        gcfg,
    )
    if isinstance(ctx, dict):
        return ctx.get("context", "")
    return ctx if isinstance(ctx, str) else ""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _make_llm_tracer(original_llm, collector, repeat_seed):
    """包装 rag.llm_model_func：采集真实 token 用量 + 计数 llm_calls，
    并把 repeat_seed 注入每次模型调用（保证正式重复运行可复现）。

    collector 字段（由 execute_fixed_route 初始化）：input_tokens, output_tokens,
    llm_calls, end_to_end_latency_ms（端到端由调用方计时）, seed_supported,
    finish_reason（最后一次调用 = 最终回答）, finish_reasons（全部调用）。
    """
    async def _wrapped(prompt, system_prompt=None, history_messages=None, **kwargs):
        collector["llm_calls"] += 1
        usage = {}
        kwargs["_record_usage_into"] = usage
        # 问题 3：采集真实 finish_reason（缓存命中/未知记 None，绝不伪造 stop）
        finish = {}
        kwargs["_record_finish_reason_into"] = finish
        # 注入 seed：vLLM / OpenAI 兼容 endpoint 支持 seed 保证可复现；
        # 若 endpoint 不支持，调用会失败并上浮，不会被静默忽略。
        if repeat_seed is not None and "seed" not in kwargs:
            kwargs["seed"] = repeat_seed
        try:
            answer = await original_llm(
                prompt, system_prompt=system_prompt,
                history_messages=history_messages or [], **kwargs
            )
        except Exception:
            collector["finish_reasons"].append(
                finish.get("finish_reason") or "error")
            collector["finish_reason"] = collector["finish_reasons"][-1]
            raise
        # 最后一次调用即最终回答（关键词抽取在前），故直接覆盖为"本次路径"的值。
        collector["finish_reasons"].append(finish.get("finish_reason"))
        collector["finish_reason"] = finish.get("finish_reason")
        if usage.get("prompt_tokens") is not None:
            collector["input_tokens"] += usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            collector["output_tokens"] += usage["completion_tokens"]
        collector["seed_supported"] = True
        return answer
    return _wrapped


def _count_context_qwen_tokens(context: str) -> int:
    """用权威 Qwen tokenizer 计数 context（P_gold / P0 无检索路径专用）。

    契约 ``units.token_unit = qwen_tokenizer_token``：trace 必须上报真实计数。
    端点不可用时 fail-fast（上浮 ``QwenTokenizerUnavailable``），禁止回退到 0
    —— 那正是"P_gold 实际 300 token 却记 0"的成因。
    """
    if not context:
        return 0
    from .qwen_tokenizer import get_qwen_token_counter
    return get_qwen_token_counter()(context)


async def execute_fixed_route(
    question: str,
    route_id: str,
    contract: ExperimentContract,
    rag,
    repeat_id: Optional[int] = None,
    repeat_seed: Optional[int] = None,
    system_snapshot_id: Optional[str] = None,
    response_type: str = "Multiple Paragraphs",
    gold_context: Optional[str] = None,
    save_trace: bool = False,
    trace_data: Optional[dict] = None,
) -> dict:
    """执行一条固定路径，返回统一结构的回答结果。

    Parameters
    ----------
    question : 用户问题。
    route_id : "P0".."P4" 或 "P_gold"。
    contract : 加载后的 ExperimentContract（参数唯一来源）。
    rag : HyperRAG 实例（提供存储对象与 llm_model_func）。
    gold_context : P_gold 的 oracle 证据文本（fixture）；非 P_gold 忽略。
    save_trace / trace_data : 透传给检索子步骤的 trace 控制。
    """
    prompt_hash = formal_answer_prompt_hash()

    # --- 真实成本采集器（绝不补零；不可采集项记 None） ---
    collector = {
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "seed_supported": False,
        # 问题 3：真实 finish_reason（stop / length / error；未知记 None）
        "finish_reason": None,
        "finish_reasons": [],
    }
    # 包装 rag.llm_model_func：注入 seed + 采集真实 token 用量。
    # 检索子步骤（关键词抽取）与最终回答都经此包装，确保 seed/usage 全覆盖。
    original_llm = rag.llm_model_func
    rag.llm_model_func = _make_llm_tracer(original_llm, collector, repeat_seed)
    run_t0 = time.perf_counter()
    param = None
    try:
        if route_id == GOLD_ROUTE_ID:
            context = gold_context or ""
            mode = "gold"
        else:
            rc = contract.route(route_id)  # 对 P_gold 抛 ContractError
            param = build_query_param(
                rc, contract, route_id,
                repeat_id=repeat_id, repeat_seed=repeat_seed,
                system_snapshot_id=system_snapshot_id,
            )
            param.response_type = response_type
            param.save_trace = save_trace
            if trace_data is not None:
                param.trace_data = trace_data

            if rc.is_llm_only:  # P0
                context = ""
                mode = "llm"
            else:
                param.only_need_context = True
                context = await _retrieve_context(question, route_id, param, rag)
                mode = param.mode

        # 统一回答：所有路径共用同一模板，仅 context 不同 -> prompt hash 一致
        sys_prompt = formal_answer_prompt().format(
            context_data=context, response_type=response_type
        )
        answer = await rag.llm_model_func(
            question,
            system_prompt=sys_prompt,
            max_tokens=contract.system_boundary.max_response_tokens,
            temperature=contract.system_boundary.temperature,
        )
    finally:
        # 还原，避免影响同进程内后续题目 / 其它调用。
        rag.llm_model_func = original_llm

    end_to_end_ms = (time.perf_counter() - run_t0) * 1000.0

    # --- 组装真实成本 ---
    # 检索相关计数来自 trace_data（检索路径填充；P0/P_gold 无检索则为 0）。
    td = (param.trace_data if param is not None else {}) or {}
    cost = {
        "input_tokens": collector["input_tokens"],
        "output_tokens": collector["output_tokens"],
        "retrieval_latency_ms": td.get("retrieval_latency_ms", 0.0) or 0.0,
        "end_to_end_latency_ms": end_to_end_ms,
        "llm_calls": collector["llm_calls"],
        "embedding_calls": td.get("embedding_calls", 0) or 0,
        "reranker_calls": 0,  # 本项目无 reranker —— 真实计数，非缺失
        "graph_expansion_calls": td.get("graph_expansion_calls", 0) or 0,
        "retrieved_candidate_count": td.get("retrieved_candidate_count", 0) or 0,
        # 本部署（本地 vLLM）无法采集 GPU 秒数与 API 账单 —— 显式记 None（契约：缺失即 null，绝不补 0）。
        "gpu_seconds": None,
        "api_cost": None,
    }

    # P_gold / P0 无检索：trace_data 为空 -> 此前 context_tokens 记 0，
    # 但 P_gold 的 gold_context 是**真实进入 prompt 的证据**，记 0 是错误上报。
    # 这里用权威 Qwen tokenizer 补齐真实计数（P0 context="" 时天然为 0）。
    if save_trace and not td.get("context_tokens") and context:
        td = dict(td)
        _qwen_n = _count_context_qwen_tokens(context)
        td.update({
            "context_tokens": _qwen_n,
            "context_tokens_before_truncate": _qwen_n,
            "context_truncated": False,
            "context_hash": hashlib.md5(context.encode("utf-8")).hexdigest(),
            "qwen_count_source": "vllm_tokenize",
        })
    trace_out = dict(td) if save_trace else {}
    return {
        "route_id": route_id,
        "mode": mode,
        "question": question,
        "context": context,
        "answer": answer,
        "prompt_template": "formal_answer_response",
        "prompt_hash": prompt_hash,
        "response_type": response_type,
        "repeat_id": repeat_id,
        "repeat_seed": repeat_seed,
        "seed_supported": collector["seed_supported"],
        # 问题 3：最终回答的真实 finish_reason（未采集到则为 None，不伪造）
        "finish_reason": collector["finish_reason"],
        "finish_reasons": list(collector["finish_reasons"]),
        "system_snapshot_id": system_snapshot_id,
        "contract_version": contract.contract_version,
        # 检索 trace（仅 save_trace 时非空）；供 trace_collector 组装完整记录
        "trace_data": trace_out,
        "cost": cost,
    }
