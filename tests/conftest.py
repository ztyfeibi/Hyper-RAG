# -*- coding: utf-8 -*-
"""tests/conftest.py — 跨测试全局状态隔离。

repeat-aware 裁决脚本（calibrate / build / run / merge_supplementary_review）通过模块级
set_context() 改写全局状态（_RC/REPEAT/SEED/OUT_DIR/FINAL/SUP_DIR/ANN/OUT 等）。
若测试调用 set_context(ri) 后不还原，会泄漏到后续 r0 测试，使其误读非默认 repeat 的
D/E 补充审核材料。本 fixture 在每个用例结束后调用 repeat_context.reset_context()，将
所有脚本全局恢复到默认 r0/s42，使测试顺序无关、彼此零污染。
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import repeat_context as rrp  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_repeat_context():
    """每个用例结束后恢复默认 r0/s42 上下文，防止 set_context 全局泄漏。"""
    yield
    rrp.reset_context()
