"""Step 2 verification: test Query Router classification and adaptive integration.

Tests:
1. route_query returns valid QueryRoute for 5 question types
2. Fallback handles bad input without crashing
3. adaptive_query with only_need_context=True includes route info
"""

import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import asdict
from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag.query_router import route_query, _parse_route_response, QueryRoute
from hyperrag.utils import logger
from my_config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from my_config import EMB_BASE_URL, EMB_API_KEY, EMB_MODEL, EMB_DIM

DATA_NAME = "neurology_chunk1000"

# Test questions covering all 5 query types
TEST_QUESTIONS = [
    ("What is multiple sclerosis?", "factual"),
    ("What is the relationship between hypertension and stroke?", "relation"),
    ("How does the blood-brain barrier protect the brain?", "mechanism"),
    ("What is the difference between Alzheimer's disease and vascular dementia?", "comparison"),
    ("What treatment options are available for a patient with both epilepsy and depression who also has liver disease?", "complex"),
]


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


async def test_router_classification(rag):
    """Test 1: route_query returns valid classification for each type."""
    print("\n" + "=" * 70)
    print("Test 1: Query Router classification (5 question types)")
    print("=" * 70)

    param = QueryParam(mode="adaptive")

    for i, (question, expected_type) in enumerate(TEST_QUESTIONS):
        print(f"\n--- Q{i+1} (expect ~{expected_type}) ---")
        print(f"  Question: {question[:80]}...")

        route = await route_query(question, param, asdict(rag))

        print(f"  query_type:  {route.query_type}")
        print(f"  complexity:  {route.complexity}")
        print(f"  focus_types: {route.focus_types}")
        print(f"  reason:      {route.reason[:100]}")

        # Validate fields
        assert isinstance(route, QueryRoute), f"Expected QueryRoute, got {type(route)}"
        assert route.query_type in {"factual", "relation", "mechanism", "comparison", "complex"}, \
            f"Invalid query_type: {route.query_type}"
        assert route.complexity in {"simple", "medium", "complex"}, \
            f"Invalid complexity: {route.complexity}"
        assert isinstance(route.focus_types, list) and len(route.focus_types) > 0, \
            f"Invalid focus_types: {route.focus_types}"
        assert isinstance(route.reason, str), f"Invalid reason type: {type(route.reason)}"

        print(f"  PASS")

    print("\nAll 5 questions classified successfully.")


def _get_global_config(rag):
    """Extract global config dict from HyperRAG instance."""
    from dataclasses import asdict
    # HyperRAG is a dataclass; asdict gives us all fields
    config = asdict(rag)
    # The llm_model_func is already wrapped with caching in __post_init__
    # so config["llm_model_func"] has hashing_kv injected
    return config


async def test_fallback():
    """Test 2: Fallback handles bad input without crashing."""
    print("\n" + "=" * 70)
    print("Test 2: Fallback parsing")
    print("=" * 70)

    # Empty string
    route = _parse_route_response("")
    assert route.query_type == "complex"
    assert route.complexity == "medium"
    print(f"  Empty string -> fallback: {route.query_type}/{route.complexity}  PASS")

    # Garbage text
    route = _parse_route_response("This is not JSON at all")
    assert route.query_type == "complex"
    print(f"  Garbage text -> fallback: {route.query_type}/{route.complexity}  PASS")

    # Markdown-fenced JSON
    route = _parse_route_response('```json\n{"query_type": "factual", "complexity": "simple", "focus_types": ["entity"], "reason": "test"}\n```')
    assert route.query_type == "factual"
    assert route.complexity == "simple"
    assert route.focus_types == ["entity"]
    print(f"  Markdown fence -> parsed: {route.query_type}/{route.complexity}  PASS")

    # JSON embedded in text
    route = _parse_route_response('Here is the result:\n{"query_type": "relation", "complexity": "medium", "focus_types": ["relation", "text"], "reason": "test"}\nDone.')
    assert route.query_type == "relation"
    print(f"  JSON in text -> parsed: {route.query_type}/{route.complexity}  PASS")

    # Missing fields -> defaults
    route = _parse_route_response('{"query_type": "invalid_type"}')
    assert route.query_type == "complex"  # invalid -> default
    print(f"  Invalid type -> default: {route.query_type}  PASS")

    print("\nAll fallback tests passed.")


async def test_adaptive_with_context(rag):
    """Test 3: adaptive_query with only_need_context=True includes route info."""
    print("\n" + "=" * 70)
    print("Test 3: adaptive_query only_need_context includes route")
    print("=" * 70)

    question = "What is the relationship between hypertension and stroke?"
    param = QueryParam(mode="adaptive", only_need_context=True)

    result = await rag.aquery(question, param)

    if isinstance(result, dict) and "route" in result:
        route = result["route"]
        context = result.get("context", "")
        print(f"  context length: {len(context)} chars")
        print(f"  route: {json.dumps(route, indent=2, ensure_ascii=False)}")
        assert "query_type" in route
        assert "complexity" in route
        assert "focus_types" in route
        print(f"\n  PASS - route info present in context-only return")
        return True
    else:
        print(f"  FAIL - expected dict with 'route', got {type(result)}")
        if isinstance(result, str):
            print(f"  (got string of length {len(result)})")
        return False


async def main():
    print("Step 2: Query Router Verification")
    print("=" * 70)

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

    # Run tests
    await test_fallback()
    await test_router_classification(rag)
    success = await test_adaptive_with_context(rag)

    print("\n" + "=" * 70)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - see output above")
    print("=" * 70)


if __name__ == "__main__":
    loop = always_get_an_event_loop()
    loop.run_until_complete(main())
