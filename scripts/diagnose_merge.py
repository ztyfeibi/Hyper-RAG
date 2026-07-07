"""模拟完整 extract_entities 流程，包括 merge 阶段，捕获隐藏异常。"""
import asyncio
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperdb import HypergraphDB
from hyperrag.prompt import PROMPTS
from hyperrag.utils import split_string_by_multi_markers, compute_args_hash
from hyperrag.chunking import chunking_by_token_size
from hyperrag.extraction import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction_low,
    _handle_single_relationship_extraction_high,
)
from hyperrag.graph_upsert import _merge_nodes_then_upsert, _merge_edges_then_upsert
from hyperrag.llm import openai_complete_if_cache
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

CACHE_PATH = "caches/neurology/kv_store_llm_response_cache.json"
CONTEXTS_PATH = "caches/neurology/contexts/neurology_unique_contexts.json"


async def main():
    # 1. 加载缓存
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[1] Cache entries: {len(cache)}")

    # 2. 加载 context 并切块
    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()
    chunks = chunking_by_token_size(
        full_text,
        overlap_token_size=120,
        max_token_size=2400,
        tiktoken_model="gpt-4o",
    )
    print(f"[2] Total chunks: {len(chunks)}")

    # 3. 构造 prompt 模板
    entity_extract_prompt = PROMPTS["entity_extraction"]
    example_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )
    example_str = PROMPTS["entity_extraction_examples"][3].format(**example_base)
    context_base = dict(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        relation_types=",".join(PROMPTS["DEFAULT_RELATION_TYPES"]),
        high_order_relation_types=",".join(PROMPTS["DEFAULT_HIGH_ORDER_RELATION_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=example_str,
    )
    record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]

    # 4. 模拟 _process_single_content（只用前 5 个 chunk，用缓存数据）
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    maybe_edges_low = defaultdict(list)
    maybe_edges_high = defaultdict(list)

    test_chunks = chunks[:5]
    for i, chunk in enumerate(test_chunks):
        content = chunk["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        messages = [{"role": "user", "content": hint_prompt}]
        cache_key = compute_args_hash(LLM_MODEL, messages)

        if cache_key in cache:
            final_result = cache[cache_key]["return"]
        else:
            print(f"  [chunk {i}] Cache MISS, calling LLM...")
            final_result = await openai_complete_if_cache(
                LLM_MODEL, hint_prompt, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
            )

        # 解析
        records = split_string_by_multi_markers(
            final_result, [record_delimiter, completion_delimiter]
        )

        chunk_nodes = defaultdict(list)
        chunk_edges = defaultdict(list)
        chunk_edges_low = defaultdict(list)
        chunk_edges_high = defaultdict(list)

        for record in records:
            record = record.strip()
            if not record:
                continue
            m = re.search(r"\((.*)\)", record)
            if m is None:
                continue
            inner = m.group(1)
            attrs = split_string_by_multi_markers(inner, [tuple_delimiter])

            ent = await _handle_single_entity_extraction(attrs, f"chunk-{i}")
            if ent is not None:
                chunk_nodes[ent["entity_name"]].append(ent)
                continue

            rl = await _handle_single_relationship_extraction_low(attrs, f"chunk-{i}")
            if rl is not None:
                chunk_edges[tuple(rl["entityN"])].append(rl)
                chunk_edges_low[tuple(rl["entityN"])].append(rl)
                continue

            rh = await _handle_single_relationship_extraction_high(attrs, f"chunk-{i}")
            if rh is not None:
                chunk_edges[tuple(rh["entityN"])].append(rh)
                chunk_edges_high[tuple(rh["entityN"])].append(rh)
                continue

        for k, v in chunk_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in chunk_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
        for k, v in chunk_edges_low.items():
            maybe_edges_low[tuple(sorted(k))].extend(v)
        for k, v in chunk_edges_high.items():
            maybe_edges_high[tuple(sorted(k))].extend(v)

        print(f"  [chunk {i}] {len(chunk_nodes)} entities, {len(chunk_edges)} edges")

    print(f"[3] Aggregated: {len(maybe_nodes)} entities, {len(maybe_edges)} edges")

    # 5. 测试 _merge_nodes_then_upsert（这是最可能抛异常的地方）
    hg = HypergraphDB()
    global_config = {
        "llm_model_func": lambda *a, **kw: None,  # dummy, won't be called for short descriptions
        "llm_model_max_token_size": 32768,
        "tiktoken_model_name": "gpt-4o",
        "entity_summary_to_max_tokens": 500,
        "entity_additional_properties_to_max_tokens": 500,
    }

    print(f"[4] Testing _merge_nodes_then_upsert on {len(maybe_nodes)} entities...")
    success = 0
    fail = 0
    first_error = None
    for name, nodes_data in maybe_nodes.items():
        try:
            await _merge_nodes_then_upsert(name, nodes_data, hg, global_config)
            success += 1
        except Exception as e:
            fail += 1
            if first_error is None:
                first_error = traceback.format_exc()
            if fail <= 3:
                print(f"  FAIL on '{name}': {type(e).__name__}: {e}")

    print(f"  Success: {success}, Fail: {fail}")
    print(f"  Hypergraph vertices: {hg.num_v}")

    if first_error:
        print(f"\n[5] First error traceback:\n{first_error}")

    # 6. 如果 merge 成功，测试 _merge_edges_then_upsert
    if fail == 0 and len(maybe_edges) > 0:
        print(f"\n[6] Testing _merge_edges_then_upsert on {len(maybe_edges)} edges...")
        success_e = 0
        fail_e = 0
        first_error_e = None
        for edge_key, edge_data in maybe_edges.items():
            try:
                await _merge_edges_then_upsert(edge_key, edge_data, hg, global_config)
                success_e += 1
            except Exception as e:
                fail_e += 1
                if first_error_e is None:
                    first_error_e = traceback.format_exc()
                if fail_e <= 3:
                    print(f"  FAIL on edge {edge_key}: {type(e).__name__}: {e}")

        print(f"  Success: {success_e}, Fail: {fail_e}")
        print(f"  Hypergraph: {hg.num_v} vertices, {hg.num_e} hyperedges")

        if first_error_e:
            print(f"\n[7] First edge error traceback:\n{first_error_e}")


if __name__ == "__main__":
    asyncio.run(main())
