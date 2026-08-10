# -*- coding: utf-8 -*-
"""Step 1.7: 初始 System Snapshot —— 冻结可复现实验的系统状态指纹。

把一次正式重复运行所依赖的全部"系统状态"固化为一个 JSON 指纹，使
``system_snapshot_id`` 可被写进 retrieval-trace-v2 记录，保证实验可复现、
可审计、可对比。

核心不变量
----------
**相同系统状态重复生成必须得到相同 snapshot_id。** 实现方式：
``snapshot_id`` = SHA256(规范化 JSON(快照体))，其中规范化 JSON 排除
``snapshot_id``/``generated_at``（非确定性）与 ``git_*``/
``verification_script_hash``（纯审计）字段。只要契约版本、运行时源码指纹
（白名单）、P0-P4 配置、prompt hash、各数据文件哈希、模型标识、tokenizer
报告 hash、规则与版本号不变，snapshot_id 即恒定 —— 修改 tests/ 或验证
脚本不再使已有回答产物失效。

内容范围（来自 PRD 1.7）
-----------------------
contract/schema 版本、Git commit SHA、P0-P4 配置、prompt hashes、
dataset/chunk/entity/relation/hypergraph 文件哈希、回答/Embedding/Judge 模型、
tokenizer 版本与校准报告哈希、temperature/cache/seed 规则、
retriever 与 score normalization 版本。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .experiment_contract import (
    ExperimentContract,
    POLICY_STAGES,
    RouteConfig,
)
from .fixed_route_executor import formal_answer_prompt_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATH = (
    REPO_ROOT / "caches" / "calibration" / "tokenizer_calibration.json"
)

# 计算 snapshot_id 时排除的字段（存在但不参与哈希）：
# - snapshot_id / generated_at：非确定性；
# - git_* / verification_script_hash：纯审计信息。若参与哈希，则提交一个
#   只改 tests/ 或验证脚本的 commit 也会改变 snapshot_id，使回答产物被
#   误判失效 —— 这正是收窄运行时指纹要消除的漂移。运行时身份完全由
#   runtime_source_fingerprint（白名单）+ 契约/数据/模型/规则字段决定。
_NON_HASH_FIELDS = (
    "snapshot_id",
    "generated_at",
    "git_commit_sha",
    "git_dirty",
    "git_dirty_diff_hash",
    "verification_script_hash",
)


def compute_file_hash(path, alg: str = "sha256") -> Optional[str]:
    """返回文件字节的哈希；文件不存在返回 None（不抛错，保证可复现）。"""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.new(alg)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha(repo_root: Path = REPO_ROOT) -> str:
    """当前仓库 HEAD SHA；git 不可用（或无 git）时返回 'unknown'。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_status(repo_root: Path = REPO_ROOT) -> dict:
    """返回 git 状态：HEAD SHA、是否 dirty、dirty diff 的哈希。

    注意：
    - ``--porcelain``（不带 -uno）→ 未跟踪文件也计入 dirty；
    - diff 以 **原始字节** 捕获并直接哈希（text=True 在 Windows 上按 GBK
      解码会失败/失真，曾导致 diff hash 为 null）；
    - git 状态仅作审计信息；源码内容的权威指纹见
      :func:`compute_runtime_source_fingerprint`（含未跟踪文件）。
    """
    sha = get_git_sha(repo_root)
    dirty = False
    dirty_diff_hash = None
    try:
        # 含未跟踪文件（核心新模块在提交前均为 untracked，必须计入 dirty）。
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, timeout=10,
        )
        dirty = bool(status.stdout.strip())
        if dirty:
            diff = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_root), capture_output=True, timeout=10,
            )
            # 直接哈希原始字节，避免任何 locale 解码问题。
            dirty_diff_hash = hashlib.sha256(diff.stdout).hexdigest()
    except Exception:
        pass
    return {"git_commit_sha": sha, "git_dirty": dirty,
            "git_dirty_diff_hash": dirty_diff_hash}


# ---------------------------------------------------------------------------
# 运行时源码指纹：不依赖 git 的确定性内容哈希（tracked + untracked 一视同仁）
# ---------------------------------------------------------------------------
# 显式白名单：**只有会影响回答产物的代码与冻结契约**参与运行时指纹。
# tests/ 与纯验证脚本被排除 —— 它们的变更由 git commit 与测试报告记录，
# 不应导致回答产物失效（否则每改一行测试就要重跑全部路径）。
RUNTIME_SOURCE_DIRS = ("hyperrag",)
RUNTIME_SOURCE_FILES = (
    "reproduce/Step_3_response_question.py",
    "docs/question_set_v2_contract.yaml",
    "docs/schema/retrieval_trace_v2.schema.json",
)
# 验收脚本单独记录版本哈希（审计用），不进入运行时指纹。
VERIFICATION_SCRIPT_FILE = "scripts/verify_trace_completeness.py"


def compute_runtime_source_fingerprint(repo_root: Path = REPO_ROOT) -> dict:
    """对运行时源码白名单计算确定性指纹（与 git 跟踪状态无关）。

    对 RUNTIME_SOURCE_DIRS 下全部 .py 文件与 RUNTIME_SOURCE_FILES 的
    (相对路径, 文件字节 sha256) 排序后串接再取 SHA256。
    任何白名单内源码字节变化（含未跟踪新文件、删除）都会改变指纹；
    白名单外（tests/、验证脚本等）的变化 **不** 影响指纹。
    """
    entries = []
    for d in RUNTIME_SOURCE_DIRS:
        base = Path(repo_root) / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(repo_root).as_posix()
            entries.append((rel, compute_file_hash(p)))
    for rel in RUNTIME_SOURCE_FILES:
        p = Path(repo_root) / rel
        if p.exists():
            entries.append((rel, compute_file_hash(p)))
    entries.sort()
    h = hashlib.sha256()
    for rel, fh in entries:
        h.update(f"{rel}:{fh}\n".encode("utf-8"))
    return {
        "runtime_source_fingerprint": h.hexdigest(),
        "runtime_source_file_count": len(entries),
    }


def _route_config_dict(rc: RouteConfig) -> dict:
    """RouteConfig -> 纯数据字典（契约冻结字段，确定性）。"""
    return {
        "chunk_vdb_top_k": rc.chunk_vdb_top_k,
        "entity_vdb_top_k": rc.entity_vdb_top_k,
        "relation_vdb_top_k": rc.relation_vdb_top_k,
        "entity_description_cap": rc.entity_description_cap,
        "relation_description_cap": rc.relation_description_cap,
        "source_text_cap": rc.source_text_cap,
        "final_context_hard_cap": rc.final_context_hard_cap,
        "is_llm_only": rc.is_llm_only,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_system_snapshot(
    contract: ExperimentContract,
    data_name: str,
    working_dir: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    default_seed: int = 42,
    retriever_version: str = "v1",
    score_normalization_version=None,
    generated_at: Optional[str] = None,
) -> dict:
    """组装 system snapshot 字典（含确定性 ``snapshot_id``）。

    Parameters
    ----------
    contract : 已加载并校验的 ExperimentContract（参数唯一来源）。
    data_name : 数据集名（用于默认 working_dir = caches/<data_name>）。
    working_dir : 已建库目录；默认 caches/<data_name>。
    calibration_path : tokenizer 校准报告路径；默认 caches/calibration/...。
    default_seed : 固定路径默认 seed（写入 rules，供复现）。
    """
    wd = Path(working_dir) if working_dir else (REPO_ROOT / "caches" / data_name)
    cal_path = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH

    routes = {stage: _route_config_dict(contract.route(stage))
              for stage in POLICY_STAGES}

    data_hashes = {
        "dataset_text_chunks": compute_file_hash(wd / "kv_store_text_chunks.json"),
        "vdb_chunks": compute_file_hash(wd / "vdb_chunks.json"),
        "vdb_entities": compute_file_hash(wd / "vdb_entities.json"),
        "vdb_relationships": compute_file_hash(wd / "vdb_relationships.json"),
        "hypergraph": compute_file_hash(wd / "hypergraph_chunk_entity_relation.hgdb"),
    }

    cal_hash = compute_file_hash(cal_path)
    tokenizer_version = None
    if cal_path.exists():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            tokenizer_version = cal.get("qwen_model") or cal.get("qwen_source")
        except Exception:
            tokenizer_version = None

    sb = contract.system_boundary
    git_status = get_git_status()
    src_fp = compute_runtime_source_fingerprint()
    snapshot = {
        "snapshot_id": "",  # 下方填充
        "generated_at": generated_at or _now_iso(),
        "contract_version": contract.contract_version,
        "trace_schema_version": contract.trace_schema_version,
        "success_rule_version": contract.success_rule_version,
        "metric_version": contract.metric_version,
        "cost_schema_version": contract.cost_schema_version,
        "git_commit_sha": git_status["git_commit_sha"],
        "git_dirty": git_status["git_dirty"],
        "git_dirty_diff_hash": git_status["git_dirty_diff_hash"],
        # 运行时源码指纹（白名单：hyperrag/**/*.py + Step_3 + 契约/Schema）：
        # 决定回答产物是否可比较。tests/ 与验证脚本不参与 —— 它们的版本
        # 由 git commit 与下方 verification_script_hash 单独记录。
        "runtime_source_fingerprint": src_fp["runtime_source_fingerprint"],
        "runtime_source_file_count": src_fp["runtime_source_file_count"],
        # 验收脚本版本哈希（审计参考，不参与运行时指纹语义）。
        "verification_script_hash": compute_file_hash(
            Path(REPO_ROOT) / VERIFICATION_SCRIPT_FILE
        ),
        "routes": routes,
        "prompt_hashes": {
            "formal_answer_response": formal_answer_prompt_hash(),
        },
        "data_hashes": data_hashes,
        "models": {
            "answer_model": sb.answer_model,
            "embedding_model": sb.embedding_model,
            "judge_model": sb.judge_model,
        },
        "tokenizer": {
            "version": tokenizer_version,
            "calibration_report_hash": cal_hash,
        },
        "rules": {
            "temperature": sb.temperature,
            "max_response_tokens": sb.max_response_tokens,
            "llm_response_cache": sb.llm_response_cache,
            "router": sb.router,
            "type_aware_weighting": sb.type_aware_weighting,
            "default_seed": default_seed,
            "seed_supported": True,  # 正式路径把 seed 注入模型调用（vLLM 兼容）
        },
        "retriever_version": retriever_version,
        # 本项目不做分数归一化；版本固定为 identity-v1（与 trace_collector 一致），
        # 绝不谎称做了归一化。
        "score_normalization_version": score_normalization_version or "identity-v1",
    }

    # snapshot_id：对排除非确定性字段后的规范化 JSON 取 SHA256 —— 状态不变则 id 不变。
    body = {k: v for k, v in snapshot.items() if k not in _NON_HASH_FIELDS}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    snapshot["snapshot_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return snapshot


def write_snapshot(snapshot: dict, out_dir) -> Path:
    """写出 <snapshot_id>.json；目录不存在自动创建。返回文件路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{snapshot['snapshot_id']}.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
