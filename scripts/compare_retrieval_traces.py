#!/usr/bin/env python3
"""Compare two retrieval trace files and report the first divergence layer.

Each trace file is a JSONL (one JSON object per line) with fields:
    question_id, query,
    entity_keywords, relation_keywords, keyword_result_hash,
    entity_query_embedding_hash, relation_query_embedding_hash,
    entity_vdb_ids, entity_post_graph_ids, entity_post_truncate_ids,
    entity_line_relation_ids, entity_line_chunk_ids,
    relation_vdb_ids, relation_post_filter_ids, relation_post_truncate_ids,
    relation_line_entity_ids, relation_line_chunk_ids,
    merged_chunk_ids,
    context_tokens, context_tokens_before_truncate, context_truncated,
    context_hash, effective_params, [route, router_policy]

Causal order (earlier = closer to data source, deterministic first):
    keywords
      -> query embedding hash        (diagnose embedding API stability)
      -> VDB IDs                     (raw VDB query results)
      -> 图过滤/截断 (post_graph/post_filter/truncate IDs)
      -> 跨线扩散实体与关系 (entity_line_relation_ids / relation_line_entity_ids)
      -> chunk IDs                   (entity/relation line source chunks)
      -> context                     (merged + budget -> context_hash)

The comparison walks the layers in this order and reports the FIRST divergence.
"""

# Causal order: 关键词 -> query embedding hash -> VDB结果 -> 图过滤/截断
#               -> 跨线扩散实体与关系 -> chunk -> 最终context
TRACE_FIELDS = [
    ("entity_keywords", "Low-level entity keywords"),
    ("relation_keywords", "High-level relation keywords"),
    ("keyword_result_hash", "Raw keyword response hash (sha256)"),
    ("entity_query_embedding_hash", "Entity query embedding hash (sha256)"),
    ("relation_query_embedding_hash", "Relation query embedding hash (sha256)"),
    ("entity_vdb_ids", "Entity VDB IDs (raw VDB query)"),
    ("entity_post_graph_ids", "Entity IDs (after graph lookup)"),
    ("entity_post_truncate_ids", "Entity IDs (after token truncation)"),
    ("entity_line_relation_ids", "Relations diffused from entity-line (written to CSV)"),
    ("entity_line_chunk_ids", "Entity-line chunk IDs (after truncation)"),
    ("relation_vdb_ids", "Relation VDB IDs (raw VDB query)"),
    ("relation_post_filter_ids", "Relation IDs (after graph lookup)"),
    ("relation_post_truncate_ids", "Relation IDs (after sort+truncation)"),
    ("relation_line_entity_ids", "Entities diffused from relation-line (written to CSV)"),
    ("relation_line_chunk_ids", "Relation-line chunk IDs (after truncation)"),
    ("merged_chunk_ids", "Merged chunk IDs (after dedup)"),
    ("context_tokens", "Context tokens (final)"),
    ("context_tokens_before_truncate", "Context tokens (before budget)"),
    ("context_truncated", "Context truncated (bool)"),
    ("context_hash", "Context hash (md5 of final string)"),
    ("route", "Router output (adaptive only)"),
    ("router_policy", "Router policy used"),
    ("effective_params", "Effective query parameters"),
]

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# Ensure UTF-8 stdout so Chinese question text prints without encode errors.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_trace(path: str) -> dict:
    """Load a trace JSONL file into {question_id: trace_dict}."""
    traces = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry.get("question_id", -1)
            traces[qid] = entry
    return traces


def compare_lists(a, b):
    """Compare two lists strictly: must have identical content AND order.

    Returns (match, details):
        - True: identical order & content
        - False: same elements (Counter match) but different order
        - False: different elements (Counter mismatch)
    """
    if a == b:
        return True, "identical (same order & content)"

    counter_a = Counter(a) if isinstance(a, list) else Counter()
    counter_b = Counter(b) if isinstance(b, list) else Counter()

    if counter_a == counter_b:
        return False, f"REORDER: same elements (len={len(a)}), different order"

    only_a = counter_a - counter_b
    only_b = counter_b - counter_a
    common = counter_a & counter_b
    return (
        False,
        f"CONTENT_DIFF: len_A={len(a)}, len_B={len(b)}, "
        f"only_A={dict(only_a)}, only_B={dict(only_b)}, "
        f"common={dict(common)}",
    )


def compare_dicts(a, b):
    """Compare two dicts. Returns (match, details)."""
    if a == b:
        return True, "identical"
    all_keys = set(a or {}) | set(b or {})
    diff_keys = {k for k in all_keys if (a or {}).get(k) != (b or {}).get(k)}
    return False, f"diff keys: {sorted(diff_keys)}"


def compare_scalars(a, b):
    """Compare two scalars (int, bool, str, etc.). Returns (match, details)."""
    if a == b:
        return True, "identical"
    return False, f"A={a}, B={b}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two retrieval trace files and report first divergence"
    )
    parser.add_argument("trace_a", type=str, help="First trace JSONL file")
    parser.add_argument("trace_b", type=str, help="Second trace JSONL file")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full details for each question (not just divergent ones)",
    )
    args = parser.parse_args()

    traces_a = load_trace(args.trace_a)
    traces_b = load_trace(args.trace_b)

    all_qids = sorted(set(traces_a.keys()) | set(traces_b.keys()))

    if not all_qids:
        print("ERROR: No traces found in either file.")
        sys.exit(1)

    print(f"Trace A: {args.trace_a} ({len(traces_a)} questions)")
    print(f"Trace B: {args.trace_b} ({len(traces_b)} questions)")
    print(f"Questions to compare: {len(all_qids)}")
    print("=" * 80)

    # Summary counters
    layer_mismatch_counts = {field: 0 for field, _ in TRACE_FIELDS}
    identical_count = 0
    divergent_count = 0
    first_divergence_layer = {}  # qid -> first divergent field name

    for qid in all_qids:
        ta = traces_a.get(qid)
        tb = traces_b.get(qid)

        if ta is None:
            print(f"\nQ{qid}: MISSING in trace A")
            divergent_count += 1
            continue
        if tb is None:
            print(f"\nQ{qid}: MISSING in trace B")
            divergent_count += 1
            continue

        query = ta.get("query", "?")[:80]
        first_div = None

        for field, label in TRACE_FIELDS:
            va = ta.get(field)
            vb = tb.get(field)

            if field == "effective_params" or field == "route":
                match, details = compare_dicts(va, vb)
            elif isinstance(va, list) or isinstance(vb, list):
                match, details = compare_lists(va or [], vb or [])
            else:
                match, details = compare_scalars(va, vb)

            if not match:
                if first_div is None:
                    first_div = field
                layer_mismatch_counts[field] += 1

            if args.verbose or not match:
                status = "OK" if match else "DIFF"
                if not match:
                    print(f"\nQ{qid} [{status}] {label}")
                    print(f"  query: {query}")
                    print(f"  {details}")
                    # Show list diff details for list fields
                    list_fields = {f for f, _ in TRACE_FIELDS if "ids" in f}
                    if field in list_fields:
                        a_set = set(va or [])
                        b_set = set(vb or [])
                        only_a = a_set - b_set
                        only_b = b_set - a_set
                        if only_a:
                            print(f"  only in A ({len(only_a)}): {sorted(only_a)[:5]}...")
                        if only_b:
                            print(f"  only in B ({len(only_b)}): {sorted(only_b)[:5]}...")

        if first_div is None:
            identical_count += 1
        else:
            divergent_count += 1
            first_divergence_layer[qid] = first_div

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 40)
    print(f"Total questions compared: {len(all_qids)}")
    print(f"Fully identical traces:   {identical_count}/{len(all_qids)}")
    print(f"Divergent traces:         {divergent_count}/{len(all_qids)}")
    print()
    print("Divergence by layer (a question may diverge at multiple layers):")
    for field, label in TRACE_FIELDS:
        count = layer_mismatch_counts[field]
        bar = "#" * count
        print(f"  {field:35s} {count:3d}  {bar}")

    if first_divergence_layer:
        print()
        print("First divergence layer per question:")
        for qid in sorted(first_divergence_layer.keys()):
            print(f"  Q{qid}: {first_divergence_layer[qid]}")

    # Exit code: 0 if all identical, 1 if any divergence
    if divergent_count > 0:
        print(f"\nVERDICT: {divergent_count} question(s) have divergent traces.")
        sys.exit(1)
    else:
        print(f"\nVERDICT: All {identical_count} traces are identical.")
        sys.exit(0)


if __name__ == "__main__":
    main()
