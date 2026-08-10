"""Adaptive Parameter Control: map QueryRoute to retrieval parameters.

Based on the router's ``complexity`` classification (simple / medium / complex),
this module selects a pre-defined parameter profile that overrides
``QueryParam.top_k`` and token budgets before ``hyper_query`` runs.

Design notes
------------
* ``apply_adaptive_params`` uses ``dataclasses.replace`` so the original
  ``query_param`` passed by the caller is never mutated in place.
* ``complex`` stays at ``max_token_for_text_unit=4000`` (not 5000) to avoid
  the context-overflow issue observed when hyper budget was 8000.
* If the route's ``complexity`` is not one of the three known values, the
  ``medium`` profile is used as a safe fallback.
"""

from dataclasses import replace
from typing import Dict

from .base import QueryParam
from .query_router import QueryRoute
from .utils import logger

# --- Parameter profiles keyed by complexity ---
# Step 5.3: max_total_context_tokens raised from 5K/7K/8K to 11K/16K/18K.
# The old values were far too conservative — the LLM has 24576 context
# and fixed overhead is only ~2600 tokens, leaving ~18000 for retrieval.
ADAPTIVE_PARAM_PROFILES: Dict[str, dict] = {
    "simple": {
        "top_k": 30,
        "max_token_for_entity_context": 200,
        "max_token_for_relation_context": 1200,
        "max_token_for_text_unit": 2500,
        "max_total_context_tokens": 11000,
    },
    "medium": {
        "top_k": 50,
        "max_token_for_entity_context": 300,
        "max_token_for_relation_context": 1600,
        "max_token_for_text_unit": 3500,
        "max_total_context_tokens": 16000,
    },
    "complex": {
        "top_k": 70,
        "max_token_for_entity_context": 400,
        "max_token_for_relation_context": 2200,
        "max_token_for_text_unit": 4000,
        "max_total_context_tokens": 18000,
    },
}

# Fallback when complexity is not recognised.
_FALLBACK_COMPLEXITY = "medium"


def apply_adaptive_params(query_param: QueryParam, route: QueryRoute) -> QueryParam:
    """Return a *new* ``QueryParam`` with adaptive overrides applied.

    Parameters
    ----------
    query_param : QueryParam
        The original parameter object. Not mutated.
    route : QueryRoute
        Classification result from ``route_query``.

    Returns
    -------
    QueryParam
        A new instance with ``top_k`` and token budgets replaced by the
        profile selected from ``route.complexity``.
    """
    complexity = route.complexity
    profile = ADAPTIVE_PARAM_PROFILES.get(complexity)

    if profile is None:
        logger.warning(
            f"Adaptive params: unknown complexity '{complexity}', "
            f"falling back to '{_FALLBACK_COMPLEXITY}'"
        )
        complexity = _FALLBACK_COMPLEXITY
        profile = ADAPTIVE_PARAM_PROFILES[_FALLBACK_COMPLEXITY]

    adaptive_param = replace(query_param, **profile)

    logger.info(
        f"Adaptive params: complexity={complexity} "
        f"top_k={profile['top_k']} "
        f"entity={profile['max_token_for_entity_context']} "
        f"relation={profile['max_token_for_relation_context']} "
        f"text={profile['max_token_for_text_unit']} "
        f"total={profile['max_total_context_tokens']}"
    )

    return adaptive_param


def get_adaptive_param_dict(route: QueryRoute) -> dict:
    """Return the raw parameter dict for a route (for logging / returning).

    Falls back to ``medium`` on unknown complexity, matching
    ``apply_adaptive_params`` behaviour.
    """
    profile = ADAPTIVE_PARAM_PROFILES.get(route.complexity)
    if profile is None:
        profile = ADAPTIVE_PARAM_PROFILES[_FALLBACK_COMPLEXITY]
    return dict(profile)
