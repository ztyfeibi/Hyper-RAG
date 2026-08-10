# -*- coding: utf-8 -*-
"""真实 Qwen 计数接入 context 截断的单元测试（fake counter 注入，无网络）。

覆盖：
1. 初始 tiktoken 截断后真实计数已合规 -> 不追加收缩，上报真实值。
2. 真实计数超限 -> 验证循环继续收缩直至合规（qwen_verify_attempts > 0）。
3. 收缩多轮仍超限 -> AssertionError（禁止上报超限 context）。
4. qwen_counter 不可用（抛 QwenTokenizerUnavailable）-> 上浮 fail-fast。
5. 上报值来自 counter 而非校准估算（qwen_count_source=vllm_tokenize）。
"""
import pytest

from hyperrag.context_budget import apply_qwen_context_budget
from hyperrag.qwen_tokenizer import QwenTokenizerUnavailable

# 三区段合法 context（区段感知截断要求的结构）
CTX = (
    "-----Entities-----\n```csv\n" + "\n".join(
        f"e{i},ENT{i},desc {i} " + "x" * 40 for i in range(30)) + "\n```\n"
    "-----Relationships-----\n```csv\n" + "\n".join(
        f"r{i},REL{i},rel desc {i} " + "y" * 40 for i in range(30)) + "\n```\n"
    "-----Sources-----\n```csv\n" + "\n".join(
        f"s{i},source text {i} " + "z" * 60 for i in range(40)) + "\n```\n"
)

FAKE_CAL = {  # 校准报告最小字段（初始近似仍需要）
    "ratio_stats": {"mean": 1.03, "p95": 1.146, "max": 1.151},
    "recommended_safety_factor": 0.8726,
}


def _mk_counter(ratio):
    """fake 真实计数：字符数 * ratio（确定性、可控超限）。"""
    calls = []

    def _count(text):
        calls.append(len(text))
        return int(len(text) * ratio)

    _count.calls = calls
    return _count


def test_real_count_within_cap_no_extra_shrink():
    counter = _mk_counter(0.01)  # 计数远小于上限
    ctx, info = apply_qwen_context_budget(
        CTX, qwen_hard_cap=10000, calibration=FAKE_CAL, qwen_counter=counter)
    assert info["qwen_verify_attempts"] == 0
    assert info["qwen_count_source"] == "vllm_tokenize"
    # 上报值 = counter 的真实输出，而非校准估算
    assert info["context_tokens_qwen_after"] == int(len(ctx) * 0.01)
    assert info["context_tokens_qwen_before"] == int(len(CTX) * 0.01)
    assert info["context_tokens_qwen_after"] <= 10000


def test_real_count_over_cap_triggers_shrink_loop():
    # 比例调高使初始 tiktoken 截断后真实计数仍超限
    counter = _mk_counter(3.0)
    cap = int(len(CTX) * 3.0 * 0.3)  # 上限 = 全文计数的 30%，必须收缩
    ctx, info = apply_qwen_context_budget(
        CTX, qwen_hard_cap=cap, calibration=FAKE_CAL, qwen_counter=counter)
    assert info["qwen_verify_attempts"] >= 1
    assert info["context_tokens_qwen_after"] <= cap
    assert len(ctx) < len(CTX)


def test_unshrinkable_raises():
    # 计数与内容长度无关恒超限 -> 收缩无效，必须抛错
    def _always_over(text):
        return 99999 if text else 0

    with pytest.raises(AssertionError):
        apply_qwen_context_budget(
            CTX, qwen_hard_cap=100, calibration=FAKE_CAL,
            qwen_counter=_always_over)


def test_counter_unavailable_fails_fast():
    def _broken(text):
        raise QwenTokenizerUnavailable("endpoint down")

    with pytest.raises(QwenTokenizerUnavailable):
        apply_qwen_context_budget(
            CTX, qwen_hard_cap=10000, calibration=FAKE_CAL,
            qwen_counter=_broken)


def test_no_cap_still_reports_real_counts():
    counter = _mk_counter(0.5)
    ctx, info = apply_qwen_context_budget(
        CTX, qwen_hard_cap=None, calibration=FAKE_CAL, qwen_counter=counter)
    assert info["qwen_hard_cap"] is None
    assert info["context_tokens_qwen_after"] == int(len(ctx) * 0.5)
    assert info["qwen_count_source"] == "vllm_tokenize"
