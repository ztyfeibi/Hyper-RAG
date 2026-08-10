# -*- coding: utf-8 -*-
"""Step 1 收尾：Qwen tokenizer 校准 —— 把 Qwen 预算转换为 tiktoken 预算。

契约 ``units.token_unit = qwen_tokenizer_token``：所有 budget 数值（包括
``final_context_hard_cap``）都是 **Qwen tokenizer token**。而运行时的截断计数
用 tiktoken（``gpt-4o-mini``）近似。为避免"配置 X Qwen token 最坏约对应
X * max_ratio 个 Qwen token 却当成 X tiktoken token"而违反硬上限，本模块
提供：

* ``load_calibration``       —— 读取 ``caches/calibration/tokenizer_calibration.json``。
* ``qwen_to_tiktoken_budget`` —— 把 Qwen 硬上限 **保守** 转换为 tiktoken 预算，
                                保证 tiktoken_budget * max_ratio <= qwen_cap。
* ``tiktoken_to_qwen_estimate`` —— 把 tiktoken 计数 **估算** 为 Qwen 计数，
                                    用于 trace 上报（契约要求上报真实 Qwen 计数）。
* ``qwen_hard_cap_satisfied`` —— 用校准给出的上限比，判断给定 tiktoken 计数
                                对应的 Qwen 计数是否未超过硬上限。

校准报告字段（``tokenizer_calibration.json``）：
    ratio_stats.max     = qwen_tokens / tiktoken_tokens 的样本最大值
    ratio_stats.mean    = 期望比值
    recommended_safety_factor = 1 / p95 （≈ 0.8726）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATH = (
    REPO_ROOT / "caches" / "calibration" / "tokenizer_calibration.json"
)


def load_calibration(path=None) -> dict:
    """读取校准报告；缺失或损坏时返回 ``None``（调用方须决定降级策略）。"""
    p = Path(path) if path else DEFAULT_CALIBRATION_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ratio_stats(calibration: Optional[dict]):
    if not calibration:
        # 无校准报告：无法安全转换。返回 None 让调用方显式报错。
        return None
    stats = calibration.get("ratio_stats") or {}
    if "max" not in stats or "mean" not in stats:
        return None
    return stats


def qwen_to_tiktoken_budget(qwen_budget: int, calibration: Optional[dict]) -> int:
    """把 Qwen-token 硬上限 **保守** 转为 tiktoken 预算（向下取整）。

    保证：tiktoken_budget * ratio_stats.max <= qwen_budget。
    即最坏情况下 tiktoken 计数折算回 Qwen 计数也不会突破硬上限。

    Raises
    ------
    ValueError
        无任何校准报告（无法安全保证硬上限合规）时抛出，禁止静默绕过。
    """
    if qwen_budget is None:
        return None
    stats = _ratio_stats(calibration)
    if stats is None:
        raise ValueError(
            "no tokenizer calibration available; cannot safely convert "
            "Qwen hard cap to tiktoken budget. Run scripts/calibrate_tokenizers.py."
        )
    max_ratio = float(stats["max"])
    if max_ratio <= 0:
        raise ValueError(f"invalid ratio_stats.max={max_ratio}")
    # 保守转换：除以最大比值（最坏情况），再向下取整留 1 token 余量。
    tiktoken_budget = int(qwen_budget // max_ratio)
    if tiktoken_budget < 1 and qwen_budget >= 1:
        tiktoken_budget = 1
    return tiktoken_budget


# --------------------------------------------------------------------------- #
# 预合并 cap 转换（Step 1 收尾 · 问题 2）
# --------------------------------------------------------------------------- #
# 契约 units.token_unit = qwen_tokenizer_token —— **所有** budget（含 pre_merge_caps
# 里的 entity_description_cap / relation_description_cap / source_text_cap）都是
# Qwen token。但热路径 truncate_list_by_token_size 用 tiktoken 计数，直接把 Qwen
# 数值当 tiktoken 预算会让实际保留量偏大（Qwen 分词更细，同文本 Qwen token 更多），
# 从而使 P2–P4 的实际执行参数偏离冻结契约。
#
# 这里提供带进程级缓存的转换入口：每次截断都读磁盘 JSON 不可接受。
_CALIBRATION_CACHE: dict = {"loaded": False, "value": None}
_BUDGET_CACHE: dict = {}


def get_cached_calibration() -> Optional[dict]:
    """进程级缓存的校准报告（避免热路径反复读盘）。"""
    if not _CALIBRATION_CACHE["loaded"]:
        _CALIBRATION_CACHE["value"] = load_calibration()
        _CALIBRATION_CACHE["loaded"] = True
    return _CALIBRATION_CACHE["value"]


def reset_calibration_cache() -> None:
    """清空缓存（测试 / 重新生成校准报告后调用）。"""
    _CALIBRATION_CACHE["loaded"] = False
    _CALIBRATION_CACHE["value"] = None
    _BUDGET_CACHE.clear()


def qwen_cap_to_tiktoken_budget(qwen_cap: Optional[int]) -> Optional[int]:
    """把契约给定的 Qwen-token 预合并 cap 保守转成 tiktoken 预算（带缓存）。

    无校准报告时抛 ``ValueError``（fail-fast），禁止静默按 tiktoken 口径执行。
    """
    if qwen_cap is None:
        return None
    if qwen_cap in _BUDGET_CACHE:
        return _BUDGET_CACHE[qwen_cap]
    budget = qwen_to_tiktoken_budget(qwen_cap, get_cached_calibration())
    _BUDGET_CACHE[qwen_cap] = budget
    return budget


def tiktoken_to_qwen_estimate(tiktoken_count: int, calibration: Optional[dict]) -> int:
    """把 tiktoken 计数估算为 Qwen 计数（用期望比值 mean），用于 trace 上报。"""
    if tiktoken_count is None:
        return 0
    stats = _ratio_stats(calibration)
    if stats is None:
        return int(tiktoken_count)  # 无校准：保守以 tiktoken 数上报（不虚高）
    mean_ratio = float(stats["mean"])
    return int(round(tiktoken_count * mean_ratio))


def qwen_hard_cap_satisfied(tiktoken_count: int, qwen_cap: int,
                            calibration: Optional[dict]) -> bool:
    """给定 tiktoken 计数，判断其折算后的 Qwen 计数是否未超过硬上限。"""
    if qwen_cap is None:
        return True
    stats = _ratio_stats(calibration)
    if stats is None:
        return True  # 无校准：无法判断，保守放行（调用方应已强制报错）
    max_ratio = float(stats["max"])
    return tiktoken_count * max_ratio <= qwen_cap
