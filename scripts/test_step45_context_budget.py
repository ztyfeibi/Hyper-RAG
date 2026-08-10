"""Step 4.5 test: total context budget control + token observation.

Tests:
1. max_total_context_tokens=None -> measure only, no truncation.
2. max_total_context_tokens=100 -> truncation with metadata.
3. Metadata fields are complete.
4. Adaptive profiles contain max_total_context_tokens.
5. Integration: adaptive only_need_context returns context_budget.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperrag.context_budget import (
    apply_context_budget,
    measure_context_tokens,
    truncate_context_by_tokens,
)
from hyperrag.adaptive_params import ADAPTIVE_PARAM_PROFILES, get_adaptive_param_dict
from hyperrag.query_router import QueryRoute
from hyperrag.base import QueryParam


def test_measure_only():
    """Test 1: max_total_context_tokens=None -> measure only, no truncation."""
    print("\n=== Test 1: None budget (measure only) ===")
    context = "Hello world. " * 100  # ~400 tokens
    result, budget = apply_context_budget(context, max_total_context_tokens=None)

    assert result == context, "Context should not change when budget is None"
    assert budget["context_truncated"] is False
    assert budget["actual_context_tokens"] == budget["context_tokens_before_truncate"]
    assert budget["context_char_length"] == len(context)
    print(f"  tokens={budget['actual_context_tokens']} chars={budget['context_char_length']} truncated={budget['context_truncated']}  PASS")


def test_truncation():
    """Test 2: max_total_context_tokens=100 -> truncation with metadata."""
    print("\n=== Test 2: Truncation with small budget ===")
    context = "Hello world. " * 500  # ~2000 tokens
    result, budget = apply_context_budget(context, max_total_context_tokens=100)

    assert budget["context_truncated"] is True
    assert budget["context_tokens_before_truncate"] > 100
    assert budget["actual_context_tokens"] <= 120  # allow marker overhead
    # Step 5.3 section-aware marker (renamed from legacy "[Context truncated")
    assert "[Context section-aware truncated" in result
    print(f"  before={budget['context_tokens_before_truncate']} after={budget['actual_context_tokens']} truncated={budget['context_truncated']}  PASS")


def test_metadata_fields():
    """Test 3: Metadata fields are complete."""
    print("\n=== Test 3: Metadata completeness ===")
    context = "Some test content here. " * 50
    _, budget = apply_context_budget(context, max_total_context_tokens=None)

    required_keys = {
        "context_tokens_before_truncate",
        "actual_context_tokens",
        "context_char_length",
        "context_truncated",
    }
    assert required_keys.issubset(budget.keys()), f"Missing keys: {required_keys - set(budget.keys())}"
    print(f"  Keys: {sorted(budget.keys())}  PASS")


def test_adaptive_profiles():
    """Test 4: Adaptive profiles contain max_total_context_tokens."""
    print("\n=== Test 4: Adaptive profiles have max_total_context_tokens ===")
    # Step 5.3 raised these from 5K/7K/8K to 11K/16K/18K
    expected = {"simple": 11000, "medium": 16000, "complex": 18000}
    for tier, expected_val in expected.items():
        profile = ADAPTIVE_PARAM_PROFILES[tier]
        assert "max_total_context_tokens" in profile, f"{tier} missing max_total_context_tokens"
        assert profile["max_total_context_tokens"] == expected_val
        print(f"  {tier}: max_total_context_tokens={profile['max_total_context_tokens']}  PASS")


def test_get_adaptive_param_dict():
    """Test 4b: get_adaptive_param_dict includes the field."""
    print("\n=== Test 4b: get_adaptive_param_dict includes max_total ===")
    for complexity in ["simple", "medium", "complex"]:
        route = QueryRoute(
            query_type="factual",
            complexity=complexity,
            focus_types=[],
            reason="test",
        )
        d = get_adaptive_param_dict(route)
        assert "max_total_context_tokens" in d, f"{complexity} dict missing max_total"
        print(f"  {complexity}: max_total={d['max_total_context_tokens']}  PASS")


def test_none_context():
    """Test 5: None context handled gracefully."""
    print("\n=== Test 5: None context ===")
    result, budget = apply_context_budget(None, max_total_context_tokens=1000)
    assert result is None
    assert budget["actual_context_tokens"] == 0
    assert budget["context_truncated"] is False
    print(f"  result=None tokens=0  PASS")


def test_no_truncation_needed():
    """Test 6: Context within budget -> no truncation."""
    print("\n=== Test 6: Context within budget ===")
    context = "Short content. " * 10  # ~40 tokens
    result, budget = apply_context_budget(context, max_total_context_tokens=200)

    assert budget["context_truncated"] is False
    assert result == context
    print(f"  tokens={budget['actual_context_tokens']} <= 200, not truncated  PASS")


def test_query_param_field():
    """Test 7: QueryParam has max_total_context_tokens field."""
    print("\n=== Test 7: QueryParam field ===")
    qp = QueryParam(mode="adaptive")
    assert qp.max_total_context_tokens is None, "Default should be None"

    qp2 = QueryParam(mode="adaptive", max_total_context_tokens=7000)
    assert qp2.max_total_context_tokens == 7000
    print(f"  default=None, custom=7000  PASS")


def test_hyper_unaffected():
    """Test 8: hyper mode QueryParam defaults to None (no truncation)."""
    print("\n=== Test 8: hyper mode unaffected ===")
    qp = QueryParam(mode="hyper")
    assert qp.max_total_context_tokens is None
    assert qp.enable_type_aware_weighting is False
    assert qp.route_focus_types is None
    print(f"  hyper: max_total=None, weighting=False, focus=None  PASS")


if __name__ == "__main__":
    test_measure_only()
    test_truncation()
    test_metadata_fields()
    test_adaptive_profiles()
    test_get_adaptive_param_dict()
    test_none_context()
    test_no_truncation_needed()
    test_query_param_field()
    test_hyper_unaffected()

    print("\n" + "=" * 50)
    print("All Step 4.5 tests PASSED")
    print("=" * 50)
