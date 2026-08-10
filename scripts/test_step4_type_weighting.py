"""Step 4 verification: test type-aware weighting.

Tests:
1. Entity focus boosts entity weights and re-sorts
2. Relation focus boosts relation weights and re-sorts
3. Mechanism focus gives extra boost to mechanism-related items
4. Disabled / no focus -> results unchanged (no mutation)
5. Original results list not mutated
6. adaptive_query only_need_context returns route with focus_types
"""

import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag.type_aware_weighting import (
    apply_type_aware_weighting,
    TYPE_AWARE_BOOSTS,
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


def test_entity_focus_boost():
    """Test 1: Entity focus boosts entity weights."""
    print("\n" + "=" * 70)
    print("Test 1: Entity focus boosts entity weights")
    print("=" * 70)

    results = [
        {"entity_name": "dopamine", "description": "a neurotransmitter", "weight": 0.8},
        {"entity_name": "stroke", "description": "brain infarction", "weight": 0.9},
        {"entity_name": "hypertension", "description": "high blood pressure", "weight": 0.7},
    ]
    original_weights = [r["weight"] for r in results]

    boosted = apply_type_aware_weighting(
        results, focus_types=["entity"], result_kind="entity", enabled=True
    )

    # All should be boosted by 1.15 — match by name since re-sorting changes order
    for orig in results:
        name = orig["entity_name"]
        boosted_item = next(r for r in boosted if r["entity_name"] == name)
        expected = orig["weight"] * TYPE_AWARE_BOOSTS["entity"]
        assert abs(boosted_item["weight"] - expected) < 0.001, \
            f"Weight {boosted_item['weight']} != expected {expected}"
        print(f"  {name:15s}  {orig['weight']:.4f} -> {boosted_item['weight']:.4f}  PASS")

    # Re-sorted: highest weight first
    weights = [r["weight"] for r in boosted]
    assert weights == sorted(weights, reverse=True), "Results not sorted by weight"
    print(f"  Re-sorted by weight descending  PASS")


def test_relation_focus_boost():
    """Test 2: Relation focus boosts relation weights."""
    print("\n" + "=" * 70)
    print("Test 2: Relation focus boosts relation weights")
    print("=" * 70)

    results = [
        {"id_set": ("A", "B"), "description": "A causes B", "keywords": "causality", "weight": 0.6},
        {"id_set": ("C", "D"), "description": "C is associated with D", "keywords": "association", "weight": 0.5},
    ]
    original_weights = [r["weight"] for r in results]

    boosted = apply_type_aware_weighting(
        results, focus_types=["relation"], result_kind="relation", enabled=True
    )

    for orig in results:
        id_set = orig["id_set"]
        boosted_item = next(r for r in boosted if r["id_set"] == id_set)
        expected = orig["weight"] * TYPE_AWARE_BOOSTS["relation"]
        assert abs(boosted_item["weight"] - expected) < 0.001, \
            f"Weight {boosted_item['weight']} != expected {expected}"
        print(f"  {str(id_set):15s}  {orig['weight']:.4f} -> {boosted_item['weight']:.4f}  PASS")

    list_keyword_results = [
        {
            "id_set": ("E", "F"),
            "description": "E activates F through a pathway",
            "keywords": ["pathway", "activation"],
            "weight": 0.4,
        }
    ]
    list_keyword_boosted = apply_type_aware_weighting(
        list_keyword_results,
        focus_types=["relation", "mechanism"],
        result_kind="relation",
        enabled=True,
    )
    assert list_keyword_boosted[0]["weight"] > list_keyword_results[0]["weight"]
    print(f"  list keywords handled without TypeError  PASS")


def test_mechanism_boost():
    """Test 3: Mechanism focus gives extra boost to mechanism-related items."""
    print("\n" + "=" * 70)
    print("Test 3: Mechanism focus extra boost")
    print("=" * 70)

    results = [
        {"entity_name": "apoptosis", "description": "programmed cell death pathway", "weight": 0.5},
        {"entity_name": "headache", "description": "pain in the head", "weight": 0.5},
    ]

    boosted = apply_type_aware_weighting(
        results, focus_types=["mechanism"], result_kind="entity", enabled=True
    )

    # apoptosis description contains "pathway" -> gets mechanism boost
    # headache description has no mechanism keywords -> no boost
    apoptosis = next(r for r in boosted if r["entity_name"] == "apoptosis")
    headache = next(r for r in boosted if r["entity_name"] == "headache")

    # apoptosis: 0.5 * 1.15 (mechanism) = 0.575
    assert apoptosis["weight"] > 0.5, f"apoptosis not boosted: {apoptosis['weight']}"
    print(f"  apoptosis:  0.5000 -> {apoptosis['weight']:.4f}  (mechanism boost)  PASS")

    # headache: no mechanism keyword -> no boost, stays 0.5
    assert headache["weight"] == 0.5, f"headache should not be boosted: {headache['weight']}"
    print(f"  headache:   0.5000 -> {headache['weight']:.4f}  (no boost)  PASS")

    # apoptosis should be first after re-sort
    assert boosted[0]["entity_name"] == "apoptosis"
    print(f"  Re-sorted: apoptosis first  PASS")


def test_disabled_no_effect():
    """Test 4: Disabled or no focus -> results unchanged."""
    print("\n" + "=" * 70)
    print("Test 4: Disabled / no focus -> no effect")
    print("=" * 70)

    results = [
        {"entity_name": "A", "description": "desc", "weight": 0.8},
        {"entity_name": "B", "description": "desc", "weight": 0.6},
    ]

    # Disabled
    out1 = apply_type_aware_weighting(results, focus_types=["entity"], result_kind="entity", enabled=False)
    assert all(o["weight"] == r["weight"] for o, r in zip(out1, results))
    print(f"  Disabled  -> weights unchanged  PASS")

    # No focus_types
    out2 = apply_type_aware_weighting(results, focus_types=None, result_kind="entity", enabled=True)
    assert all(o["weight"] == r["weight"] for o, r in zip(out2, results))
    print(f"  No focus  -> weights unchanged  PASS")

    # Empty focus_types
    out3 = apply_type_aware_weighting(results, focus_types=[], result_kind="entity", enabled=True)
    assert all(o["weight"] == r["weight"] for o, r in zip(out3, results))
    print(f"  Empty     -> weights unchanged  PASS")

    # Focus doesn't match kind (e.g. focus=["text"], kind="entity", no mechanism)
    out4 = apply_type_aware_weighting(results, focus_types=["text"], result_kind="entity", enabled=True)
    assert all(o["weight"] == r["weight"] for o, r in zip(out4, results))
    print(f"  Mismatch  -> weights unchanged  PASS")


def test_no_mutation():
    """Test 5: Original results list not mutated."""
    print("\n" + "=" * 70)
    print("Test 5: Original results not mutated")
    print("=" * 70)

    results = [
        {"entity_name": "X", "description": "desc", "weight": 0.5},
    ]
    original_weight = results[0]["weight"]

    _ = apply_type_aware_weighting(results, focus_types=["entity"], result_kind="entity", enabled=True)

    assert results[0]["weight"] == original_weight, \
        f"Original mutated: {results[0]['weight']} != {original_weight}"
    print(f"  Original weight preserved: {results[0]['weight']}  PASS")


async def test_adaptive_with_context(rag):
    """Test 6: adaptive_query only_need_context includes route with focus_types."""
    print("\n" + "=" * 70)
    print("Test 6: adaptive_query returns route.focus_types + weighting applied")
    print("=" * 70)

    question = "What is the relationship between hypertension and stroke?"
    param = QueryParam(mode="adaptive", only_need_context=True)

    result = await rag.aquery(question, param)

    if isinstance(result, dict) and "route" in result:
        route = result["route"]
        context = result.get("context", "")

        print(f"  context length: {len(context)} chars")
        print(f"  route: {json.dumps(route, indent=2, ensure_ascii=False)}")

        assert "focus_types" in route
        assert isinstance(route["focus_types"], list)
        assert len(route["focus_types"]) > 0

        print(f"\n  PASS - route.focus_types present: {route['focus_types']}")
        return True
    else:
        print(f"  FAIL - expected dict with 'route'")
        return False


async def main():
    print("Step 4: Type-Aware Weighting Verification")
    print("=" * 70)

    # Unit tests (no LLM needed)
    test_entity_focus_boost()
    test_relation_focus_boost()
    test_mechanism_boost()
    test_disabled_no_effect()
    test_no_mutation()

    # Integration test
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
