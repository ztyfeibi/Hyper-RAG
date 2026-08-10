# -*- coding: utf-8 -*-
"""真实 Qwen token 计数 —— vLLM ``/tokenize`` 端点。

契约 ``units.token_unit = qwen_tokenizer_token``：所有 budget / trace 上报
必须是 **权威 Qwen tokenizer** 的真实计数，tiktoken 只允许做初始近似。
本模块把部署端（vLLM）的 ``/tokenize`` 端点封装成同步计数函数：

- 端点 URL 从 ``QWEN_TOKENIZE_URL`` 环境变量或 ``my_config.LLM_BASE_URL``
  （去掉 ``/v1`` 后缀 + ``/tokenize``）解析；模型名同理。
- 服务不可用时抛 :class:`QwenTokenizerUnavailable` —— 正式路径必须失败，
  **禁止静默回退到估算值**（那会重新引入"Trace 不是实际计数"问题）。
"""

from __future__ import annotations

import os
from typing import Callable, Optional

__all__ = [
    "QwenTokenizerUnavailable",
    "get_qwen_token_counter",
    "resolve_tokenize_config",
]


class QwenTokenizerUnavailable(RuntimeError):
    """权威 Qwen tokenizer 端点不可用（正式路径必须 fail-fast）。"""


def resolve_tokenize_config(
    base_url: Optional[str] = None, model: Optional[str] = None
) -> tuple:
    """解析 (tokenize_url, model)。优先级：显式参数 > 环境变量 > my_config。"""
    url = base_url or os.environ.get("QWEN_TOKENIZE_URL", "").strip() or None
    mdl = model or os.environ.get("QWEN_TOKENIZE_MODEL", "").strip() or None
    if url is None or mdl is None:
        try:
            import my_config  # repo 根目录的部署配置
            if url is None:
                llm_base = getattr(my_config, "LLM_BASE_URL", "").rstrip("/")
                if llm_base.endswith("/v1"):
                    llm_base = llm_base[: -len("/v1")]
                if llm_base:
                    url = llm_base + "/tokenize"
            if mdl is None:
                mdl = getattr(my_config, "LLM_MODEL", None)
        except ImportError:
            pass
    if not url or not mdl:
        raise QwenTokenizerUnavailable(
            "无法解析 Qwen /tokenize 配置：请设置 QWEN_TOKENIZE_URL / "
            "QWEN_TOKENIZE_MODEL 环境变量，或保证 my_config.LLM_BASE_URL / "
            "LLM_MODEL 可导入。"
        )
    return url, mdl


def get_qwen_token_counter(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 60.0,
) -> Callable[[str], int]:
    """返回 ``count(text) -> int`` 的真实 Qwen 计数函数。

    每次调用 POST ``/tokenize``；HTTP 失败/异常响应抛
    :class:`QwenTokenizerUnavailable`（绝不返回估算值）。
    """
    import requests

    url, mdl = resolve_tokenize_config(base_url, model)
    session = requests.Session()

    def _count(text: str) -> int:
        if text == "":
            return 0
        try:
            resp = session.post(
                url, json={"model": mdl, "prompt": text}, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # 网络/HTTP/JSON 全部视为不可用
            raise QwenTokenizerUnavailable(
                f"Qwen /tokenize 调用失败（{url}, model={mdl}）：{e}"
            ) from e
        count = data.get("count")
        if not isinstance(count, int):
            raise QwenTokenizerUnavailable(
                f"Qwen /tokenize 响应缺少整数 count 字段：{data!r}"
            )
        return count

    return _count
