"""Context budget control: section-aware truncation of merged context.

After ``combine_contexts()`` merges entity and relation CSV sections, the
total context can exceed what the LLM can safely handle.  This module
provides a *section-aware budget fuse*: instead of a blind tail-cut that
removes Sources (original text evidence), it allocates the budget across
three sections — **Sources 55%, Relationships 30%, Entities 15%** — and
truncates each section independently.

Design notes
------------
* ``max_total_context_tokens=None`` means *measure only, do not truncate*.
  This is the default so that ``hyper`` baseline behaviour is unchanged.
* When truncation is needed, each section is truncated from the tail
  (head contains the header + highest-ranked items).
* If a section is already under its allocation, the excess is redistributed
  to sections that need more (proportional to their deficit).
* **Step 5.3 fix**: After initial section truncation, the FULL assembled
  context (including headers, CSV fences, and truncation marker) is
  re-encoded and verified.  If it still exceeds the budget, sections are
  iteratively shrunk — Sources first, then Relationships, then Entities —
  until the strict assertion ``actual_context_tokens <= max_total_context_tokens``
  holds.
* Per-section token counts are always returned for debug logging.
"""

import re
from typing import Optional

from .utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    logger,
)
from .tokenizer_calibration import (
    load_calibration,
    qwen_to_tiktoken_budget,
    tiktoken_to_qwen_estimate,
)

# --- Section budget allocation ratios ---
SECTION_BUDGET_RATIOS = {
    "entities": 0.15,
    "relationships": 0.30,
    "sources": 0.55,
}

# Truncation marker appended to the end of truncated context
_TRUNCATION_MARKER = "\n[Context section-aware truncated to max_total_context_tokens]"

_SECTION_PATTERN = {
    "entities": r"-----Entities-----\s*```csv\s*(.*?)\s*```",
    "relationships": r"-----Relationships-----\s*```csv\s*(.*?)\s*```",
    "sources": r"-----Sources-----\s*```csv\s*(.*?)\s*```",
}


def _parse_sections(context: str) -> dict[str, str]:
    """Parse the combined context string into three section bodies."""
    sections = {}
    for name, pattern in _SECTION_PATTERN.items():
        match = re.search(pattern, context, re.DOTALL)
        sections[name] = match.group(1) if match else ""
    return sections


def _reassemble(sections: dict[str, str]) -> str:
    """Reassemble sections into the standard combined-context format."""
    return f"""
-----Entities-----
```csv
{sections["entities"]}
```
-----Relationships-----
```csv
{sections["relationships"]}
```
-----Sources-----
```csv
{sections["sources"]}
```
"""


def _count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    return len(encode_string_by_tiktoken(text, model_name=model_name))


def _truncate_text_to_tokens(
    text: str, max_tokens: int, model_name: str = "gpt-4o-mini"
) -> str:
    """Truncate *text* to at most *max_tokens* tokens (tail cut)."""
    if not text:
        return text
    tokens = encode_string_by_tiktoken(text, model_name=model_name)
    if len(tokens) <= max_tokens:
        return text
    cut = max(1, max_tokens)
    return decode_tokens_by_tiktoken(tokens[:cut], model_name=model_name)


def measure_context_tokens(context: str, model_name: str = "gpt-4o-mini") -> dict:
    """Measure token count and character length of a context string.

    Returns
    -------
    dict
        ``{"context_tokens": int, "context_char_length": int}``
    """
    tokens = encode_string_by_tiktoken(context, model_name=model_name)
    return {
        "context_tokens": len(tokens),
        "context_char_length": len(context),
    }


def truncate_context_by_tokens(
    context: str,
    max_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> tuple:
    """Truncate *context* to at most *max_tokens* tokens (tail cut).

    .. deprecated::
        This is the old tail-cut function kept for backward compatibility.
        New code should use ``apply_context_budget`` which is section-aware.

    Returns
    -------
    (str, dict)
        The (possibly truncated) context string and a metadata dict.
    """
    tokens = encode_string_by_tiktoken(context, model_name=model_name)
    tokens_before = len(tokens)

    if tokens_before <= max_tokens:
        return context, {
            "context_tokens_before_truncate": tokens_before,
            "actual_context_tokens": tokens_before,
            "context_char_length": len(context),
            "context_truncated": False,
        }

    _MARKER_BUFFER = 20
    cut = max(1, max_tokens - _MARKER_BUFFER)
    truncated_text = decode_tokens_by_tiktoken(tokens[:cut], model_name=model_name)
    marker = "\n\n[Context truncated to max_total_context_tokens]"
    truncated_text += marker

    final_tokens = encode_string_by_tiktoken(truncated_text, model_name=model_name)

    logger.info(
        f"Context budget (legacy tail-cut): truncated {tokens_before} -> {len(final_tokens)} tokens "
        f"(limit={max_tokens}, buffer={_MARKER_BUFFER})"
    )

    return truncated_text, {
        "context_tokens_before_truncate": tokens_before,
        "actual_context_tokens": len(final_tokens),
        "context_char_length": len(truncated_text),
        "context_truncated": True,
    }


def apply_context_budget(
    context: str,
    max_total_context_tokens: Optional[int],
    tiktoken_model_name: str = "gpt-4o-mini",
    section_allocation_ratios: Optional[dict] = None,
) -> tuple:
    """Apply section-aware total-context budget control.

    If *max_total_context_tokens* is ``None``, only measure (no truncation).
    Otherwise, parse the context into Entities / Relationships / Sources
    sections, allocate the budget by ratio (55% / 30% / 15%), and truncate
    each section independently — preserving Sources as the priority.

    **Step 5.3**: After initial truncation, the FULL assembled context
    (including headers, CSV fences, and truncation marker) is re-encoded.
    If it still exceeds the budget, sections are iteratively shrunk
    (Sources → Relationships → Entities) until the strict assertion
    ``actual_context_tokens <= max_total_context_tokens`` holds.

    Returns
    -------
    (str, dict)
        The (possibly truncated) context and a budget-info dict containing:
        - ``context_tokens_before_truncate``
        - ``actual_context_tokens``
        - ``context_truncated``  (bool)
        - ``entity_tokens``
        - ``relation_tokens``
        - ``source_tokens``
    """
    model = tiktoken_model_name

    if context is None:
        return context, {
            "context_tokens_before_truncate": 0,
            "actual_context_tokens": 0,
            "context_char_length": 0,
            "context_truncated": False,
            "entity_tokens": 0,
            "relation_tokens": 0,
            "source_tokens": 0,
        }

    # Parse sections
    sections = _parse_sections(context)
    section_tokens = {
        "entities": _count_tokens(sections["entities"], model),
        "relationships": _count_tokens(sections["relationships"], model),
        "sources": _count_tokens(sections["sources"], model),
    }
    # Count the FULL context (including headers, fences) for accurate comparison
    total_before = _count_tokens(context, model)

    # Measure-only mode (no truncation)
    if max_total_context_tokens is None:
        return context, {
            "context_tokens_before_truncate": total_before,
            "actual_context_tokens": total_before,
            "context_char_length": len(context),
            "context_truncated": False,
            "entity_tokens": section_tokens["entities"],
            "relation_tokens": section_tokens["relationships"],
            "source_tokens": section_tokens["sources"],
        }

    # No truncation needed (full context fits within budget)
    if total_before <= max_total_context_tokens:
        return context, {
            "context_tokens_before_truncate": total_before,
            "actual_context_tokens": total_before,
            "context_char_length": len(context),
            "context_truncated": False,
            "entity_tokens": section_tokens["entities"],
            "relation_tokens": section_tokens["relationships"],
            "source_tokens": section_tokens["sources"],
        }

    # --- Section-aware truncation ---
    # Step 1: Allocate budget by ratio (契约 section_allocation_ratios 优先，
    # 否则回退到模块常量，保证旧调用方不变)。
    ratios = section_allocation_ratios or SECTION_BUDGET_RATIOS
    allocated = {
        name: int(max_total_context_tokens * ratio)
        for name, ratio in ratios.items()
    }

    # Step 2: Redistribute excess from sections that are under allocation
    excess = 0
    for name in section_tokens:
        if section_tokens[name] < allocated[name]:
            excess += allocated[name] - section_tokens[name]
            allocated[name] = section_tokens[name]  # No truncation needed

    # Redistribute excess to sections that need more (proportional to deficit)
    deficits = {
        name: section_tokens[name] - allocated[name]
        for name in section_tokens
        if section_tokens[name] > allocated[name]
    }
    total_deficit = sum(deficits.values())
    if total_deficit > 0 and excess > 0:
        for name, deficit in deficits.items():
            extra = int(excess * deficit / total_deficit)
            allocated[name] += extra

    # Step 3: Truncate each section to its allocated budget
    truncated_sections = {}
    for name, text in sections.items():
        if section_tokens[name] > allocated[name]:
            truncated_sections[name] = _truncate_text_to_tokens(
                text, allocated[name], model
            )
        else:
            truncated_sections[name] = text

    # Step 4: Reassemble with truncation marker
    new_context = _reassemble(truncated_sections) + _TRUNCATION_MARKER

    # Step 5: Verify against FULL context token count (headers + fences + marker)
    actual_tokens = _count_tokens(new_context, model)

    # If still over budget (headers/fences/marker pushed it over),
    # iteratively shrink sections: Sources → Relationships → Entities
    if actual_tokens > max_total_context_tokens:
        # Shrink Sources first (usually the largest section)
        if truncated_sections["sources"]:
            overflow = actual_tokens - max_total_context_tokens
            current = _count_tokens(truncated_sections["sources"], model)
            new_budget = max(1, current - overflow)
            truncated_sections["sources"] = _truncate_text_to_tokens(
                truncated_sections["sources"], new_budget, model
            )
            new_context = _reassemble(truncated_sections) + _TRUNCATION_MARKER
            actual_tokens = _count_tokens(new_context, model)

        # If still over, shrink Relationships
        if actual_tokens > max_total_context_tokens and truncated_sections["relationships"]:
            overflow = actual_tokens - max_total_context_tokens
            current = _count_tokens(truncated_sections["relationships"], model)
            new_budget = max(1, current - overflow)
            truncated_sections["relationships"] = _truncate_text_to_tokens(
                truncated_sections["relationships"], new_budget, model
            )
            new_context = _reassemble(truncated_sections) + _TRUNCATION_MARKER
            actual_tokens = _count_tokens(new_context, model)

        # If still over, shrink Entities (shouldn't happen, but for safety)
        if actual_tokens > max_total_context_tokens and truncated_sections["entities"]:
            overflow = actual_tokens - max_total_context_tokens
            current = _count_tokens(truncated_sections["entities"], model)
            new_budget = max(1, current - overflow)
            truncated_sections["entities"] = _truncate_text_to_tokens(
                truncated_sections["entities"], new_budget, model
            )
            new_context = _reassemble(truncated_sections) + _TRUNCATION_MARKER
            actual_tokens = _count_tokens(new_context, model)

    # Final per-section token counts
    final_section_tokens = {
        "entities": _count_tokens(truncated_sections["entities"], model),
        "relationships": _count_tokens(truncated_sections["relationships"], model),
        "sources": _count_tokens(truncated_sections["sources"], model),
    }

    # Strict assertion: full context must not exceed budget
    assert actual_tokens <= max_total_context_tokens, (
        f"Context budget assertion failed: {actual_tokens} > {max_total_context_tokens}. "
        f"Sections: ent={final_section_tokens['entities']}, "
        f"rel={final_section_tokens['relationships']}, "
        f"src={final_section_tokens['sources']}"
    )

    logger.info(
        f"Context budget (section-aware v2): truncated {total_before} -> {actual_tokens} tokens "
        f"(limit={max_total_context_tokens}). "
        f"Sections: entities={section_tokens['entities']}->{final_section_tokens['entities']}, "
        f"relations={section_tokens['relationships']}->{final_section_tokens['relationships']}, "
        f"sources={section_tokens['sources']}->{final_section_tokens['sources']}"
    )

    return new_context, {
        "context_tokens_before_truncate": total_before,
        "actual_context_tokens": actual_tokens,
        "context_char_length": len(new_context),
        "context_truncated": True,
        "entity_tokens": final_section_tokens["entities"],
        "relation_tokens": final_section_tokens["relationships"],
        "source_tokens": final_section_tokens["sources"],
    }


def apply_qwen_context_budget(
    context: str,
    qwen_hard_cap: Optional[int],
    calibration: Optional[dict] = None,
    tiktoken_model_name: str = "gpt-4o-mini",
    section_allocation_ratios: Optional[dict] = None,
    qwen_counter=None,
    max_verify_attempts: int = 6,
) -> tuple:
    """以 **Qwen tokenizer token** 为单位的硬上限做区段感知截断。

    契约 ``units.token_unit = qwen_tokenizer_token`` 要求最终以 **权威 Qwen
    tokenizer 的真实计数** 保证与上报硬上限合规。流程：

    1. tiktoken 只做 **初始近似**：用校准报告把 Qwen 上限保守转为 tiktoken
       预算（``qwen_to_tiktoken_budget``），先做区段感知截断；
    2. 用真实 Qwen 计数（vLLM ``/tokenize``）验证截断结果；仍超上限则按
       实测比例继续收缩 tiktoken 预算重截，直至合规（最多
       ``max_verify_attempts`` 轮）；
    3. info 上报的 ``context_tokens_qwen_before/_after`` 均为 **真实计数**，
       不再是估算值；
    4. tokenizer 端点不可用 → 抛 ``QwenTokenizerUnavailable``，正式路径
       fail-fast，禁止静默回退到估算。

    Parameters
    ----------
    qwen_hard_cap : 契约给定的 Qwen token 硬上限；``None`` 表示只测量不截断。
    calibration : 校准报告 dict；``None`` 时尝试默认路径加载。无报告且
        ``qwen_hard_cap`` 非 ``None`` 会抛 ``ValueError``（禁止静默绕过硬上限）。
    qwen_counter : ``callable(str) -> int`` 真实 Qwen 计数函数；``None`` 时
        用 :func:`hyperrag.qwen_tokenizer.get_qwen_token_counter` 构造。
        测试可注入 fake counter。

    Returns
    -------
    (str, dict)
        与 ``apply_context_budget`` 相同，但 info 额外携带：
        - ``qwen_hard_cap``       : 原始 Qwen 上限
        - ``tiktoken_budget``     : 最终使用的 tiktoken 预算
        - ``context_tokens_qwen_before`` / ``_after`` : 真实 Qwen 计数
        - ``qwen_count_source``   : "vllm_tokenize"（权威来源标记）
        - ``qwen_verify_attempts``: 真实计数验证后追加的收缩轮数
    """
    if calibration is None:
        calibration = load_calibration()
    if qwen_counter is None:
        from .qwen_tokenizer import get_qwen_token_counter
        qwen_counter = get_qwen_token_counter()

    # 原始 context 的真实 Qwen 计数（上报 before 值）
    qwen_before = qwen_counter(context)

    if qwen_hard_cap is None:
        new_context, info = apply_context_budget(
            context, None, tiktoken_model_name=tiktoken_model_name,
            section_allocation_ratios=section_allocation_ratios,
        )
        info["qwen_hard_cap"] = None
        info["tiktoken_budget"] = None
        info["context_tokens_qwen_before"] = qwen_before
        info["context_tokens_qwen_after"] = qwen_before
        info["qwen_count_source"] = "vllm_tokenize"
        info["qwen_verify_attempts"] = 0
        return new_context, info

    # --- 第 1 步：tiktoken 初始近似截断（保守预算） ---
    tiktoken_budget = qwen_to_tiktoken_budget(qwen_hard_cap, calibration)
    new_context, info = apply_context_budget(
        context, tiktoken_budget, tiktoken_model_name=tiktoken_model_name,
        section_allocation_ratios=section_allocation_ratios,
    )

    # --- 第 2 步：真实 Qwen 计数验证；超限则继续收缩 ---
    qwen_after = qwen_counter(new_context)
    attempts = 0
    while qwen_after > qwen_hard_cap and attempts < max_verify_attempts:
        attempts += 1
        # 按实测超出比例收缩，并留 2% 安全边际
        tiktoken_budget = max(
            1, int(tiktoken_budget * qwen_hard_cap / qwen_after * 0.98)
        )
        new_context, info = apply_context_budget(
            context, tiktoken_budget, tiktoken_model_name=tiktoken_model_name,
            section_allocation_ratios=section_allocation_ratios,
        )
        qwen_after = qwen_counter(new_context)

    if qwen_after > qwen_hard_cap:
        raise AssertionError(
            f"Qwen hard cap violated after {attempts} shrink attempts: "
            f"real count {qwen_after} > cap {qwen_hard_cap}. "
            f"tiktoken_budget={tiktoken_budget}"
        )

    info["qwen_hard_cap"] = qwen_hard_cap
    info["tiktoken_budget"] = tiktoken_budget
    info["context_tokens_qwen_before"] = qwen_before
    info["context_tokens_qwen_after"] = qwen_after
    info["qwen_count_source"] = "vllm_tokenize"
    info["qwen_verify_attempts"] = attempts
    logger.info(
        f"Qwen hard cap verified: real {qwen_before} -> {qwen_after} tokens "
        f"(cap={qwen_hard_cap}, verify_attempts={attempts})"
    )
    return new_context, info
