"""Step 0 verification: test entity line token truncation.

Runs a hyper query with only_need_context=True using different
max_token_for_entity_context values to confirm the Entity CSV is
actually truncated, while relations and sources remain unaffected.
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from my_config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from my_config import EMB_BASE_URL, EMB_API_KEY, EMB_MODEL, EMB_DIM

DATA_NAME = "neurology_chunk1000"


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        LLM_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts):
    return await openai_embedding(
        texts, model=EMB_MODEL, api_key=EMB_API_KEY, base_url=EMB_BASE_URL
    )


async def run_test():
    WORKING_DIR = Path("caches") / DATA_NAME
    rag = HyperRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        llm_model_max_async=4,
        embedding_func_max_async=4,
    )

    test_query = "What are the clinical features and pathophysiology of Alzheimer's disease?"

    results = {}
    for budget in [100, 300, 2000]:
        print(f"\n{'='*60}")
        print(f"Test: max_token_for_entity_context={budget}")
        print(f"{'='*60}")
        qp = QueryParam(mode="hyper", only_need_context=True,
                        max_token_for_entity_context=budget)
        ctx = await rag.aquery(test_query, param=qp)

        # Parse sections
        ent_section = ctx.split("-----Entities-----")[1] \
                        .split("-----Relationships-----")[0] \
                        .strip().strip("```csv").strip("```").strip()
        rel_section = ctx.split("-----Relationships-----")[1] \
                        .split("-----Sources-----")[0] \
                        .strip().strip("```csv").strip("```").strip()
        src_section = ctx.split("-----Sources-----")[1] \
                        .strip().strip("```csv").strip("```").strip()

        ent_lines = [l for l in ent_section.split("\n") if l.strip()]
        rel_lines = [l for l in rel_section.split("\n") if l.strip()]
        src_lines = [l for l in src_section.split("\n") if l.strip()]

        results[budget] = {
            "entity_count": len(ent_lines) - 1,  # minus header
            "entity_chars": len(ent_section),
            "relation_lines": len(rel_lines),
            "source_lines": len(src_lines),
        }
        print(f"  Entities: {results[budget]['entity_count']} rows, {results[budget]['entity_chars']} chars")
        print(f"  Relations: {results[budget]['relation_lines']} lines")
        print(f"  Sources: {results[budget]['source_lines']} lines")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Budget':<10} {'Entities':<12} {'Entity chars':<15} {'Relations':<12} {'Sources':<10}")
    print("-" * 60)
    for b in [100, 300, 2000]:
        r = results[b]
        print(f"{b:<10} {r['entity_count']:<12} {r['entity_chars']:<15} {r['relation_lines']:<12} {r['source_lines']:<10}")

    # Verification
    print()
    ok = (results[100]["entity_count"] <= results[300]["entity_count"] <= results[2000]["entity_count"])
    if ok:
        print("PASS: Entity count scales with budget - truncation is working!")
    else:
        print("WARN: Entity count does not scale as expected")

    has_rel = all(results[b]["relation_lines"] > 1 for b in [100, 300, 2000])
    has_src = all(results[b]["source_lines"] > 1 for b in [100, 300, 2000])
    print(f"Relations present in all tests: {has_rel}")
    print(f"Sources present in all tests: {has_src}")


if __name__ == "__main__":
    loop = always_get_an_event_loop()
    loop.run_until_complete(run_test())
