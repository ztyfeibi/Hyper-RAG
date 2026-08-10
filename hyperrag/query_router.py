"""Query Router: classify user query into type/complexity for adaptive retrieval.

This module provides ``route_query`` which uses an LLM call to classify a
medical question. The classification result (``QueryRoute``) can later be
used to adjust retrieval parameters like ``top_k``, token budgets, and
entity/relation weighting.

The LLM call is cached via ``global_config["llm_model_func"]`` (which already
has ``hashing_kv`` injected in ``HyperRAG.__init__``), so repeated queries
hit the cache instead of re-calling the model.
"""

import json
from dataclasses import dataclass, field
from typing import List

from .base import QueryParam
from .prompt import PROMPTS
from .utils import logger


@dataclass
class QueryRoute:
    """Classification result for a user query."""

    query_type: str  # factual | relation | mechanism | comparison | complex
    complexity: str  # simple | medium | complex
    focus_types: List[str]  # subset of ["entity", "relation", "mechanism", "text"]
    reason: str  # brief explanation

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "complexity": self.complexity,
            "focus_types": self.focus_types,
            "reason": self.reason,
        }


# Fallback route used when LLM output is unparseable or the call fails.
_FALLBACK_ROUTE = QueryRoute(
    query_type="complex",
    complexity="medium",
    focus_types=["entity", "relation", "text"],
    reason="router fallback — LLM output was not valid JSON",
)


def _parse_route_response(raw: str) -> QueryRoute:
    """Parse LLM JSON output into ``QueryRoute``.

    Returns the fallback route if parsing fails or required fields are missing.
    """
    if not raw or not raw.strip():
        logger.warning("Router: empty LLM response, using fallback")
        return _FALLBACK_ROUTE

    text = raw.strip()

    # Strip markdown code fences if present (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.warning(f"Router: JSON parse failed, using fallback. Raw: {raw[:200]}")
                return _FALLBACK_ROUTE
        else:
            logger.warning(f"Router: no JSON found in response, using fallback. Raw: {raw[:200]}")
            return _FALLBACK_ROUTE

    # Validate and normalize fields
    valid_types = {"factual", "relation", "mechanism", "comparison", "complex"}
    valid_complexities = {"simple", "medium", "complex"}
    valid_focus = {"entity", "relation", "mechanism", "text"}

    query_type = data.get("query_type", "complex")
    if query_type not in valid_types:
        query_type = "complex"

    complexity = data.get("complexity", "medium")
    if complexity not in valid_complexities:
        complexity = "medium"

    focus_types = data.get("focus_types", ["entity", "relation", "text"])
    if isinstance(focus_types, str):
        focus_types = [focus_types]
    focus_types = [f for f in focus_types if f in valid_focus]
    if not focus_types:
        focus_types = ["entity", "relation", "text"]

    reason = data.get("reason", "")

    return QueryRoute(
        query_type=query_type,
        complexity=complexity,
        focus_types=focus_types,
        reason=reason,
    )


async def route_query(query: str, query_param: QueryParam, global_config: dict) -> QueryRoute:
    """Classify a user query via LLM.

    The LLM call is cached through ``global_config["llm_model_func"]`` which
    already has ``hashing_kv`` injected. On any error, returns the fallback
    route so the pipeline never crashes due to routing failure.
    """
    use_model_func = global_config["llm_model_func"]

    prompt = PROMPTS["query_router"].format(query=query)

    try:
        raw = await use_model_func(
            prompt, temperature=query_param.llm_temperature
        )
    except Exception as e:
        logger.warning(f"Router: LLM call failed ({e}), using fallback")
        return _FALLBACK_ROUTE

    route = _parse_route_response(raw)
    logger.info(
        f"Router: type={route.query_type}, complexity={route.complexity}, "
        f"focus={route.focus_types}, reason={route.reason[:80]}"
    )
    return route
