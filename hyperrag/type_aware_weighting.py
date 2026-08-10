"""Type-Aware Weighting: soft-boost retrieval results based on route focus_types.

Given the ``focus_types`` from the Query Router (e.g. ``["relation", "mechanism"]``),
this module re-weights the vector-DB query results for entities and relationships
*before* they are sorted and truncated. It does **not** filter out any results —
every retrieved item stays, only its ``weight`` field is multiplied by a boost
factor so that focused-type items tend to rank higher in the final context.

Key design decisions
-------------------
* Returns a **new list**; the original ``results`` is never mutated.
* When ``enable_type_aware_weighting`` is ``False`` or ``focus_types`` is
  ``None``/empty, the original list is returned as-is (shallow copy to keep
  the no-mutation guarantee).
* Boost factors are deliberately small (1.05–1.20) to avoid distorting
  retrieval too aggressively — this is a *re-rank nudge*, not a filter.
"""

from typing import List, Optional

from .utils import logger


def _stable_edge_id(id_set) -> str:
    """Stable key for a relation's id_set (mirrors query_context._stable_edge_id).

    Kept local to avoid a circular import (query_context imports this module).
    """
    return "|".join(sorted(str(x) for x in id_set))

# --- Boost factors ---
# Applied multiplicatively to the ``weight`` field of each vdb result dict.
TYPE_AWARE_BOOSTS = {
    "entity": 1.15,
    "relation": 1.20,
    "mechanism": 1.15,
    "text": 1.05,
}

# When focus includes "mechanism", items whose text contains these keywords
# get an extra small boost on top of the base focus boost.
_MECHANISM_KEYWORDS = (
    "mechanism", "pathway", "process", "cause", "pathophysiology",
    "regulation", "signal", "cascade", "mediator", "induce",
    "activate", "inhibit", "impair", "disrupt", "degrade",
)


def _matches_mechanism(text: str) -> bool:
    """Check if a description/keywords string contains mechanism-related terms."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _MECHANISM_KEYWORDS)


def apply_type_aware_weighting(
    results: List[dict],
    focus_types: Optional[List[str]],
    result_kind: str,
    enabled: bool = True,
) -> List[dict]:
    """Soft-boost vdb query results based on route focus_types.

    Parameters
    ----------
    results : list[dict]
        Output of ``entities_vdb.query()`` or ``relationships_vdb.query()``.
        Each dict is expected to have a ``weight`` field (float).
    focus_types : list[str] or None
        From ``QueryRoute.focus_types`` (e.g. ``["relation", "mechanism"]``).
    result_kind : str
        ``"entity"`` or ``"relation"`` — determines which focus type applies.
    enabled : bool
        Master switch from ``QueryParam.enable_type_aware_weighting``.
        When ``False``, returns a shallow copy without modification.

    Returns
    -------
    list[dict]
        New list with possibly-boosted ``weight`` values, sorted descending
        by ``weight`` so that boosted items float to the top.
    """
    # Fast path: disabled or no focus — return shallow copy (no mutation)
    if not enabled or not focus_types:
        return list(results)

    if not results:
        return results

    # Determine base boost for this result_kind
    kind_key = result_kind  # "entity" or "relation"
    base_boost = 1.0
    if kind_key in focus_types and kind_key in TYPE_AWARE_BOOSTS:
        base_boost = TYPE_AWARE_BOOSTS[kind_key]

    # If "mechanism" is in focus, items with mechanism-related text get extra boost
    mechanism_boost = 1.0
    if "mechanism" in focus_types and "mechanism" in TYPE_AWARE_BOOSTS:
        mechanism_boost = TYPE_AWARE_BOOSTS["mechanism"]

    # If neither this kind nor mechanism is in focus, no-op
    if base_boost == 1.0 and mechanism_boost == 1.0:
        return list(results)

    boosted_count = 0
    new_results = []
    for r in results:
        item = dict(r)  # shallow copy — don't mutate original

        # Start with the current weight (default 1.0 if missing)
        weight = item.get("weight", 1.0)
        total_boost = 1.0

        if base_boost > 1.0:
            total_boost *= base_boost

        # Mechanism boost applies to both entity and relation results
        # if their text mentions mechanism-related terms
        if mechanism_boost > 1.0:
            text_field = ""
            if result_kind == "entity":
                text_field = (item.get("description", "") or "") + " " + (item.get("entity_name", "") or "")
            elif result_kind == "relation":
                keywords = item.get("keywords", "") or ""
                if isinstance(keywords, list):
                    keywords = " ".join(str(x) for x in keywords)
                text_field = (item.get("description", "") or "") + " " + str(keywords)

            if _matches_mechanism(text_field):
                total_boost *= mechanism_boost

        if total_boost > 1.0:
            weight = weight * total_boost
            item["weight"] = weight
            boosted_count += 1

        new_results.append(item)

    # Re-sort by weight descending so boosted items float up.
    # Stable tie-break: entity_name for entities, stable edge id for relations.
    # Without it, equal-weight items keep arbitrary insertion order (depends on
    # set/async ordering) and can produce non-deterministic ranking.
    if result_kind == "entity":
        new_results = sorted(
            new_results,
            key=lambda x: (x.get("weight", 1.0), x.get("entity_name", "")),
            reverse=True,
        )
    else:
        new_results = sorted(
            new_results,
            key=lambda x: (x.get("weight", 1.0), _stable_edge_id(x.get("id_set", []))),
            reverse=True,
        )

    logger.info(
        f"Type-aware weighting: kind={result_kind} focus={focus_types} "
        f"boosted={boosted_count}/{len(new_results)} "
        f"base_boost={base_boost:.2f} mech_boost={mechanism_boost:.2f}"
    )

    return new_results
