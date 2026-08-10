"""Step 3 verification: test adaptive parameter control.

Tests:
1. apply_adaptive_params returns correct profiles for simple/medium/complex
2. Unknown complexity falls back to medium
3. Original query_param is not mutated (dataclasses.replace)
4. adaptive_query with only_need_context=True includes route + adaptive_params
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag.query_router import QueryRoute
from hyperrag.adaptive_params import (
    apply_adaptive_params,
    get_adaptive_param_dict,
    ADAPTIVE_PARAM_PROFILES,
)
from hyperrag.utils import logger
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


def test_param_profiles():
    """Test 1: Three complexity levels produce correct parameter profiles."""
    print("\n" + "=" * 70)
    print("Test 1: Parameter profiles for simple/medium/complex")
    print("=" * 70)

    base_param = QueryParam(mode="adaptive")

    test_cases = [
        ("simple", 30, 200, 1200, 2500),
        ("medium", 50, 300, 1600, 3500),
        ("complex", 70, 400, 2200, 4000),
    ]

    for complexity, exp_top_k, exp_entity, exp_relation, exp_text in test_cases:
        route = QueryRoute(
            query_type="factual",
            complexity=complexity,
            focus_types=["entity"],
            reason="test",
        )
        new_param = apply_adaptive_params(base_param, route)

        assert new_param.top_k == exp_top_k, \
            f"{complexity}: top_k={new_param.top_k}, expected {exp_top_k}"
        assert new_param.max_token_for_entity_context == exp_entity, \
            f"{complexity}: entity={new_param.max_token_for_entity_context}, expected {exp_entity}"
        assert new_param.max_token_for_relation_context == exp_relation, \
            f"{complexity}: relation={new_param.max_token_for_relation_context}, expected {exp_relation}"
        assert new_param.max_token_for_text_unit == exp_text, \
            f"{complexity}: text={new_param.max_token_for_text_unit}, expected {exp_text}"

        print(f"  {complexity:8s} -> top_k={new_param.top_k:3d} "
              f"entity={new_param.max_token_for_entity_context:4d} "
              f"relation={new_param.max_token_for_relation_context:4d} "
              f"text={new_param.max_token_for_text_unit:4d}  PASS")

    # Verify complex stays at 4000 (not 5000)
    complex_route = QueryRoute(
        query_type="complex", complexity="complex",
        focus_types=["entity", "relation", "text"], reason="overflow safety"
    )
    complex_param = apply_adaptive_params(base_param, complex_route)
    assert complex_param.max_token_for_text_unit == 4000, \
        f"complex text budget must be 4000, got {complex_param.max_token_for_text_unit}"
    print(f"\n  Complex text budget = 4000 (overflow-safe)  PASS")

    print("\nAll profile tests passed.")


def test_fallback_complexity():
    """Test 2: Unknown complexity falls back to medium."""
    print("\n" + "=" * 70)
    print("Test 2: Fallback on unknown complexity")
    print("=" * 70)

    base_param = QueryParam(mode="adaptive")
    route = QueryRoute(
        query_type="factual",
        complexity="very_hard",  # invalid
        focus_types=["entity"],
        reason="test",
    )
    new_param = apply_adaptive_params(base_param, route)

    medium_profile = ADAPTIVE_PARAM_PROFILES["medium"]
    assert new_param.top_k == medium_profile["top_k"], \
        f"Fallback top_k={new_param.top_k}, expected medium {medium_profile['top_k']}"
    assert new_param.max_token_for_text_unit == medium_profile["max_token_for_text_unit"]

    print(f"  unknown -> medium: top_k={new_param.top_k}, "
          f"text={new_param.max_token_for_text_unit}  PASS")

    # Also test get_adaptive_param_dict fallback
    param_dict = get_adaptive_param_dict(route)
    assert param_dict["top_k"] == medium_profile["top_k"]
    print(f"  get_adaptive_param_dict fallback  PASS")


def test_no_mutation():
    """Test 3: Original query_param is not mutated."""
    print("\n" + "=" * 70)
    print("Test 3: Original QueryParam not mutated")
    print("=" * 70)

    base_param = QueryParam(mode="adaptive", top_k=999, max_token_for_text_unit=9999)
    route = QueryRoute(
        query_type="factual", complexity="simple",
        focus_types=["entity"], reason="test"
    )
    _ = apply_adaptive_params(base_param, route)

    assert base_param.top_k == 999, f"Original top_k mutated: {base_param.top_k}"
    assert base_param.max_token_for_text_unit == 9999, \
        f"Original text budget mutated: {base_param.max_token_for_text_unit}"
    print(f"  Original top_k={base_param.top_k}, text={base_param.max_token_for_text_unit}  PASS")


async def test_adaptive_with_context(rag):
    """Test 4: adaptive_query only_need_context includes route + adaptive_params."""
    print("\n" + "=" * 70)
    print("Test 4: adaptive_query only_need_context returns route + adaptive_params")
    print("=" * 70)

    question = "What is the relationship between hypertension and stroke?"
    param = QueryParam(mode="adaptive", only_need_context=True)

    result = await rag.aquery(question, param)

    if isinstance(result, dict) and "route" in result and "adaptive_params" in result:
        route = result["route"]
        adaptive_params = result["adaptive_params"]
        context = result.get("context", "")

        print(f"  context length:     {len(context)} chars")
        print(f"  route:              {json.dumps(route, indent=2, ensure_ascii=False)}")
        print(f"  adaptive_params:    {json.dumps(adaptive_params, indent=2)}")

        assert "query_type" in route
        assert "complexity" in route
        assert "top_k" in adaptive_params
        assert "max_token_for_entity_context" in adaptive_params
        assert "max_token_for_relation_context" in adaptive_params
        assert "max_token_for_text_unit" in adaptive_params

        # Verify adaptive_params matches the profile for this complexity
        expected = ADAPTIVE_PARAM_PROFILES.get(route["complexity"], ADAPTIVE_PARAM_PROFILES["medium"])
        assert adaptive_params["top_k"] == expected["top_k"], \
            f"adaptive_params top_k={adaptive_params['top_k']} != expected {expected['top_k']}"

        print(f"\n  PASS - route + adaptive_params present and consistent")
        return True
    else:
        print(f"  FAIL - expected dict with 'route' and 'adaptive_params'")
        print(f"  Got type={type(result)}")
        if isinstance(result, str):
            print(f"  (string of length {len(result)})")
        return False


async def main():
    print("Step 3: Adaptive Parameter Control Verification")
    print("=" * 70)

    # Unit tests (no LLM needed)
    test_param_profiles()
    test_fallback_complexity()
    test_no_mutation()

    # Integration test (needs LLM + knowledge graph)
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
