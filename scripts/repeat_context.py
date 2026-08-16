#!/usr/bin/env python3
"""repeat_context.py -- 统一实验目录参数（repeat-aware pipeline 基础设施）。

所有裁决脚本（calibrate / build_supplementary / run_supplementary / merge）通过
RepeatContext 派生路径，不再硬编码 r0/s42。

目录规则：
    judge/longcat/r{repeat}_s{seed}_{snapshot前8位}/

向后兼容：模块级 DEFAULT_RC 保持 r0/s42，旧测试和直接调用不受影响。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import judge_longcat as jl  # noqa: E402


@dataclass(frozen=True)
class RepeatContext:
    """一次 repeat 实验的完整路径上下文。

    所有路径从 (data_name, repeat, seed, snapshot) 派生，
    不依赖任何模块级可变状态。
    """
    data_name: str = "neurology_chunk1000"
    repeat: int = 0
    seed: int = 42
    snapshot: str = field(default_factory=lambda: jl.SNAPSHOT_DEFAULT)

    # ------------------------------------------------------------------
    # 目录
    # ------------------------------------------------------------------
    @property
    def judge_dir(self) -> Path:
        """judge/longcat/r{r}_s{s}_{snap8}/"""
        return jl.LONGCAT_DIR / f"r{self.repeat}_s{self.seed}_{self.snapshot[:8]}"

    @property
    def blind_dir(self) -> Path:
        return self.judge_dir / "blind_review"

    @property
    def supplementary_dir(self) -> Path:
        return self.judge_dir / "blind_review_supplementary"

    @property
    def response_dir(self) -> Path:
        return jl.RESPONSE_DIR

    @property
    def base_dir(self) -> Path:
        """caches/<data_name>/"""
        return jl.BASE

    @property
    def pilot_dir(self) -> Path:
        return jl.PILOT_DIR

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------
    @property
    def questions_file(self) -> Path:
        return jl.QUESTIONS_FILE

    @property
    def gold_context_file(self) -> Path:
        return jl.GOLD_CTX_FILE

    def result_file(self, route: str) -> Path:
        """fixed_{route}_r{r}_s{s}_{contract}_{snap}_result.jsonl"""
        return jl.result_file(route, self.repeat, self.seed, self.snapshot)

    def verdict_file(self, route: str) -> Path:
        """{route}_verdict.jsonl"""
        return self.judge_dir / f"{route}_verdict.jsonl"

    @property
    def summary_file(self) -> Path:
        return self.judge_dir / "summary.json"

    @property
    def judge_manifest_file(self) -> Path:
        return self.judge_dir / "manifest.json"

    @property
    def dir_tag(self) -> str:
        """r{repeat}_s{seed}_{snapshot[:8]}"""
        return f"r{self.repeat}_s{self.seed}_{self.snapshot[:8]}"

    @property
    def all_routes(self) -> list[str]:
        return list(jl.ROUTES)

    # ------------------------------------------------------------------
    # 补充审核 review_id 编号
    # ------------------------------------------------------------------
    def review_id(self, set_name: str, idx: int) -> str:
        """R{repeat}-{set}-{idx:03d}（r0 向后兼容用旧格式 SR-{set}-{idx:03d}）."""
        if self.repeat == 0:
            return f"SR-{set_name}-{idx:03d}"
        return f"R{self.repeat}-{set_name}-{idx:03d}"

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args) -> "RepeatContext":
        return cls(
            data_name=getattr(args, "data_name", "neurology_chunk1000"),
            repeat=getattr(args, "repeat", 0),
            seed=getattr(args, "seed", 42),
            snapshot=getattr(args, "snapshot", None) or jl.SNAPSHOT_DEFAULT,
        )

    @classmethod
    def default(cls) -> "RepeatContext":
        """r0/s42 默认上下文（向后兼容）."""
        return cls()


DEFAULT_RC = RepeatContext.default()
