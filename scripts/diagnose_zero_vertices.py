"""端到端诊断：用缓存数据走完 extract_entities 全流程，找出 0 vertices 的根因。"""
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperdb import HypergraphDB
from hyperrag.prompt import PROMPTS
from hyperrag.utils import split_string_by_multi_markers
from hyperrag.indexing import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction_low,
    _handle_single_relationship_extraction_high,
)
from hyperrag.graph_upsert import _merge_nodes_then_upsert, _merge_edges_then_upsert

CACHE_PATH = "caches/neurology/kv_store_llm_response_cache.json"


async def main():
    # 1. 加载缓存
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[1] Cache entries: {len(cache)}")

    # 2. 解析所有缓存响应（模拟 _process_single_content 的解析部分）
    record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    maybe_edges_low = defaultdict(list)
    maybe_edges_high = defaultdict(list)

    for i, (cache_key, cache_val) in enumerate(cache.items()):
        final_result = cache_val.get("return", "")
        if not final_result:
            continue

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

    print(f"[2] Parsed: {len(maybe_nodes)} unique entities, {len(maybe_edges)} unique edges")
    print(f"    edges_low: {len(maybe_edges_low)}, edges_high: {len(maybe_edges_high)}")

    if not maybe_nodes:
        print("[!] No entities parsed — this is the bug")
        return

    # 3. 创建空超图，测试 upsert
    hg = HypergraphDB()
    print(f"[3] Fresh hypergraph: {hg.num_v} vertices, {hg.num_e} hyperedges")

    # 4. 测试单个 upsert_vertex
    test_name = list(maybe_nodes.keys())[0]
    test_data = maybe_nodes[test_name][0]
    print(f"[4] Testing upsert_vertex with '{test_name}'...")
    try:
        hg.add_v(test_name, {
            "entity_type": test_data.get("entity_type", "OTHER"),
            "description": test_data.get("description", ""),
            "source_id": test_data.get("source_id", ""),
            "additional_properties": test_data.get("additional_properties", ""),
        })
        print(f"    After add_v: {hg.num_v} vertices")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")

    # 5. 批量 upsert（不调 LLM summary，直接写）
    print(f"[5] Bulk upsert {len(maybe_nodes)} vertices (no LLM summary)...")
    success = 0
    fail = 0
    for name, nodes_data in maybe_nodes.items():
        try:
            # 合并同名实体的描述
            descriptions = set()
            entity_types = []
            source_ids = set()
            additional_props = set()
            for dp in nodes_data:
                if dp.get("description"):
                    descriptions.add(dp["description"])
                if dp.get("entity_type"):
                    entity_types.append(dp["entity_type"])
                if dp.get("source_id"):
                    source_ids.add(dp["source_id"])
                if dp.get("additional_properties"):
                    if isinstance(dp["additional_properties"], (list, set)):
                        additional_props.update(dp["additional_properties"])
                    elif isinstance(dp["additional_properties"], str):
                        additional_props.add(dp["additional_properties"])

            from collections import Counter
            entity_type = sorted(Counter(entity_types).items(), key=lambda x: x[1], reverse=True)[0][0] if entity_types else "OTHER"

            hg.add_v(name, {
                "entity_type": entity_type,
                "description": "<SEP>".join(sorted(descriptions)),
                "source_id": "<SEP>".join(source_ids),
                "additional_properties": "<SEP>".join(sorted(additional_props)),
            })
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"    FAIL on '{name}': {type(e).__name__}: {e}")

    print(f"    Success: {success}, Fail: {fail}")
    print(f"    Final hypergraph: {hg.num_v} vertices, {hg.num_e} hyperedges")


if __name__ == "__main__":
    asyncio.run(main())
