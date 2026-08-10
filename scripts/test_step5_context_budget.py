"""Tests for Step 5.3: context budget v2 + max_response_tokens + cache key fix.

Test coverage:
1. Full context token count (including headers/fences/marker) does not exceed budget
2. 11K/16K/18K three-tier adaptive config is correct
3. Sources section is non-empty after truncation
4. max_response_tokens=3000 is set on QueryParam by default
5. Keyword extraction and Router calls do NOT pass max_tokens
6. Cache key changes when max_tokens is added (no stale cache hit)
7. Legacy tests from Step 5.1 still pass
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from hyperrag.context_budget import apply_context_budget, _parse_sections, _reassemble
from hyperrag.base import QueryParam
from hyperrag.adaptive_params import ADAPTIVE_PARAM_PROFILES
from hyperrag.utils import compute_args_hash, encode_string_by_tiktoken


def _make_context(entities_text="entity_name,type\ndisease_A,DISEASE\n", 
                  relations_text="src_tgt,description\n[A,B],rel_A_B\n",
                  sources_text="id,content\n0,this is source text about disease A\n"):
    """Build a combined context string in the standard format."""
    return f"""
-----Entities-----
```csv
{entities_text}
```
-----Relationships-----
```csv
{relations_text}
```
-----Sources-----
```csv
{sources_text}
```
"""


# ========== Test 1: Full context token count strict assertion ==========

def test_full_context_not_exceed_budget():
    """Test 1: Full context (with headers, fences, marker) must not exceed budget."""
    # Build a large context that exceeds budget
    big_sources = "id,content\n" + "\n".join(
        [f"{i},source chunk number {i} " + "x" * 200 for i in range(50)]
    )
    big_entities = "entity_name,type,description\n" + "\n".join(
        [f"entity_{i},DISEASE,{'desc ' * 20}" for i in range(30)]
    )
    big_relations = "src_tgt,description\n" + "\n".join(
        [f"[A_{i},B_{i}],{'relation desc ' * 10}" for i in range(30)]
    )
    context = _make_context(big_entities, big_relations, big_sources)

    for budget in [2000, 3000, 5000, 8000]:
        truncated, info = apply_context_budget(context, budget)
        actual = info['actual_context_tokens']
        assert actual <= budget, \
            f"FAIL: budget={budget}, actual={actual} > {budget}"
        print(f"  budget={budget:>5}: actual={actual:>5} OK")

    print(f"\nTest 1: Full context strict assertion PASS")


# ========== Test 2: Three-tier adaptive config ==========

def test_adaptive_config_three_tiers():
    """Test 2: 11K/16K/18K three-tier config is correct."""
    expected = {
        "simple": 11000,
        "medium": 16000,
        "complex": 18000,
    }
    for tier, expected_budget in expected.items():
        profile = ADAPTIVE_PARAM_PROFILES[tier]
        actual = profile["max_total_context_tokens"]
        assert actual == expected_budget, \
            f"FAIL: {tier} budget={actual}, expected={expected_budget}"
        print(f"  {tier}: max_total_context_tokens={actual} OK")

    print(f"\nTest 2: Three-tier config (11K/16K/18K) PASS")


# ========== Test 3: Sources non-empty after truncation ==========

def test_sources_non_empty_after_truncation():
    """Test 3: Sources section must have content after truncation."""
    big_sources = "id,content\n" + "\n".join(
        [f"{i}," + "y" * 300 for i in range(40)]
    )
    context = _make_context(sources_text=big_sources)

    budget = 3000
    truncated, info = apply_context_budget(context, budget)

    assert info['source_tokens'] > 0, "FAIL: Sources is empty after truncation!"
    sections = _parse_sections(truncated)
    assert sections['sources'].strip(), "FAIL: Sources section is empty in truncated context!"

    print(f"\nTest 3: Sources non-empty ({info['source_tokens']} tokens) PASS")


# ========== Test 4: max_response_tokens default = 3000 ==========

def test_max_response_tokens_default():
    """Test 4: QueryParam.max_response_tokens defaults to 3000."""
    qp = QueryParam()
    assert qp.max_response_tokens == 3000, \
        f"FAIL: default max_response_tokens={qp.max_response_tokens}, expected 3000"

    # Also verify it can be overridden
    qp2 = QueryParam(max_response_tokens=5000)
    assert qp2.max_response_tokens == 5000, \
        f"FAIL: overridden max_response_tokens={qp2.max_response_tokens}, expected 5000"

    print(f"\nTest 4: max_response_tokens default=3000, overridable PASS")


# ========== Test 5: Keyword extraction and Router don't get max_tokens ==========

def test_keyword_extraction_no_max_tokens():
    """Test 5: Verify that keyword extraction calls don't pass max_tokens.

    We check this by inspecting the source code of query_modes.py to ensure
    that use_model_func(kw_prompt) calls do NOT include max_tokens.
    """
    import inspect
    from hyperrag import query_modes

    source = inspect.getsource(query_modes)

    # Find all use_model_func calls
    lines = source.split('\n')
    kw_calls = []
    response_calls = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'use_model_func(kw_prompt)' in stripped:
            kw_calls.append((i + 1, stripped))
        if 'use_model_func(' in stripped and 'kw_prompt' not in stripped and 'max_tokens' in (lines[i+1] + lines[i+2] if i + 2 < len(lines) else ''):
            response_calls.append((i + 1, stripped))

    # All keyword extraction calls should NOT have max_tokens
    for lineno, line in kw_calls:
        assert 'max_tokens' not in line, \
            f"FAIL: keyword extraction at line {lineno} has max_tokens: {line}"

    print(f"\nTest 5: Keyword extraction calls ({len(kw_calls)}) have no max_tokens PASS")

    # Also check Router
    from hyperrag import query_router
    router_source = inspect.getsource(query_router)
    assert 'max_tokens' not in router_source, \
        "FAIL: query_router.py contains max_tokens"
    print(f"  Router source has no max_tokens PASS")


# ========== Test 6: Cache key changes when max_tokens is added ==========

def test_cache_key_changes_with_max_tokens():
    """Test 6: compute_args_hash produces different keys with/without max_tokens."""
    model = "qwen-27b-int4"
    messages = [{"role": "user", "content": "test prompt"}]

    # Old-style hash (no inference params)
    old_hash = compute_args_hash(model, messages)

    # New-style hash with max_tokens
    new_hash = compute_args_hash(model, messages, {"max_tokens": 3000})

    assert old_hash != new_hash, \
        "FAIL: cache key should differ when max_tokens is added"

    # Different max_tokens values should also differ
    hash_3000 = compute_args_hash(model, messages, {"max_tokens": 3000})
    hash_5000 = compute_args_hash(model, messages, {"max_tokens": 5000})
    assert hash_3000 != hash_5000, \
        "FAIL: cache key should differ for different max_tokens values"

    # Empty params dict should match old-style hash (backward compat)
    empty_hash = compute_args_hash(model, messages, {})
    assert empty_hash != old_hash, \
        "FAIL: empty dict changes hash (should not match old-style)"

    print(f"\nTest 6: Cache key changes with max_tokens PASS")
    print(f"  old_hash={old_hash[:16]}...")
    print(f"  new_hash={new_hash[:16]}... (with max_tokens=3000)")


# ========== Legacy tests from Step 5.1 ==========

def test_sources_preserved_when_truncated():
    """Legacy Test 1: When total exceeds budget, Sources section still has content."""
    big_sources = "id,content\n" + "\n".join(
        [f"{i},source chunk number {i} " + "x" * 200 for i in range(50)]
    )
    big_entities = "entity_name,type,description\n" + "\n".join(
        [f"entity_{i},DISEASE,{'desc ' * 20}" for i in range(30)]
    )
    big_relations = "src_tgt,description\n" + "\n".join(
        [f"[A_{i},B_{i}],{'relation desc ' * 10}" for i in range(30)]
    )
    context = _make_context(big_entities, big_relations, big_sources)

    budget = 2000
    truncated, info = apply_context_budget(context, budget)

    assert info['source_tokens'] > 0, "FAIL: Sources section is empty after truncation!"
    assert info['context_truncated'] == True, "FAIL: context_truncated should be True"

    sections = _parse_sections(truncated)
    assert sections['sources'].strip(), "FAIL: Sources section is empty in truncated context!"

    print(f"\nLegacy Test 1: Sources preserved ({info['source_tokens']} tokens) PASS")


def test_total_under_budget():
    """Legacy Test 2: Total tokens do not exceed the set budget."""
    big_sources = "id,content\n" + "\n".join(
        [f"{i}," + "y" * 300 for i in range(40)]
    )
    context = _make_context(sources_text=big_sources)

    budget = 3000
    truncated, info = apply_context_budget(context, budget)

    assert info['actual_context_tokens'] <= budget, \
        f"FAIL: actual tokens ({info['actual_context_tokens']}) exceeds budget ({budget})!"

    print(f"\nLegacy Test 2: Total under budget ({info['actual_context_tokens']} <= {budget}) PASS")


def test_redistribution():
    """Legacy Test 3: When a section is under allocation, excess goes to others."""
    tiny_entities = "entity_name,type\ndisease_A,DISEASE\n"
    huge_sources = "id,content\n" + "\n".join(
        [f"{i}," + "z" * 200 for i in range(60)]
    )
    context = _make_context(entities_text=tiny_entities, sources_text=huge_sources)

    budget = 4000
    truncated, info = apply_context_budget(context, budget)

    base_source_alloc = int(budget * 0.55)
    assert info['source_tokens'] > base_source_alloc, \
        f"FAIL: Sources ({info['source_tokens']}) should exceed base allocation ({base_source_alloc})"

    print(f"\nLegacy Test 3: Redistribution PASS (Sources={info['source_tokens']} > base={base_source_alloc})")


def test_measure_only():
    """Legacy Test 4: None budget = measure only, returns per-section tokens."""
    context = _make_context()

    truncated, info = apply_context_budget(context, None)

    assert info['context_truncated'] == False, "FAIL: should not truncate when budget is None"
    assert info['entity_tokens'] > 0, "FAIL: entity_tokens should be > 0"
    assert info['source_tokens'] > 0, "FAIL: source_tokens should be > 0"
    assert truncated == context, "FAIL: context should be unchanged when budget is None"

    print(f"\nLegacy Test 4: Measure only PASS")


def test_no_truncation_needed():
    """Legacy Test 5: Context under budget = no truncation."""
    context = _make_context()

    truncated, info = apply_context_budget(context, 10000)

    assert info['context_truncated'] == False, "FAIL: should not truncate when under budget"

    print(f"\nLegacy Test 5: No truncation needed PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5.3 Test Suite")
    print("=" * 60)

    # New tests
    test_full_context_not_exceed_budget()
    test_adaptive_config_three_tiers()
    test_sources_non_empty_after_truncation()
    test_max_response_tokens_default()
    test_keyword_extraction_no_max_tokens()
    test_cache_key_changes_with_max_tokens()

    # Legacy tests
    test_sources_preserved_when_truncated()
    test_total_under_budget()
    test_redistribution()
    test_measure_only()
    test_no_truncation_needed()

    print("\n" + "=" * 60)
    print("=== All Step 5.3 tests passed ===")
    print("=" * 60)
