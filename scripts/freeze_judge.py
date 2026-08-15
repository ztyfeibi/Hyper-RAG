#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 6: Judge 冻结清单 judge_freeze_manifest.json。

聚合并锁定:
  - LongCat judge: 模型 / prompt hash / rule v2 / coverage v3.1 / retry / 脚本哈希
    （引用 judge manifest.json，不重复计算）
  - AI adjudication: rule v3 / v1+v2 四件套哈希 / 318 条标注文件哈希 / review_metadata 哈希
  - 输入题集: questions_v2_manual_final.jsonl / p_gold_contexts.jsonl 哈希
  - Step 5: repeat0_question_eligibility 哈希
冻结后任何一项变更都应表现为 manifest 对不上（人工比对）。
默认拒绝覆盖已有 freeze manifest，--overwrite 才允许。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import judge_longcat as jl
import calibrate_blind_review as cal

REPEAT = 0
SEED = 42
SNAPSHOT = jl.SNAPSHOT_DEFAULT
JUDGE_DIR = jl.LONGCAT_DIR / f"r{REPEAT}_s{SEED}_{SNAPSHOT[:8]}"

QUESTIONS_FILE = jl.QUESTIONS_FILE
GOLD_CTX_FILE = jl.GOLD_CTX_FILE
ANN_ALL = JUDGE_DIR / "blind_review_supplementary" / "verdicts_ai_all_v1.jsonl"
REVIEW_META = JUDGE_DIR / "blind_review_supplementary" / "review_metadata.json"
ELIG = JUDGE_DIR / "repeat0_question_eligibility.jsonl"
ELIG_SUM = JUDGE_DIR / "repeat0_eligibility_summary.json"
OUT = JUDGE_DIR / "judge_freeze_manifest.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6: 生成 judge_freeze_manifest.json")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 freeze manifest")
    args = ap.parse_args()

    if OUT.exists() and not args.overwrite:
        print(f"ERROR: {OUT} 已存在（冻结产物禁止静默覆盖）；确认重冻加 --overwrite",
              file=sys.stderr)
        return 1

    # 前置依赖必须齐全
    required = {
        "judge_manifest": JUDGE_DIR / "manifest.json",
        "v1_final_verdicts": JUDGE_DIR / "ai_adjudication_v1" / "final_verdicts.jsonl",
        "v2_final_verdicts": JUDGE_DIR / "ai_adjudication_v2" / "final_verdicts.jsonl",
        "v2_manifest": JUDGE_DIR / "ai_adjudication_v2" / "manifest.json",
        "annotation_all": ANN_ALL,
        "review_metadata": REVIEW_META,
        "eligibility": ELIG,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        print("ERROR: 冻结前置产物缺失（fail-closed）:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    jm = json.loads((JUDGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    v2m = json.loads((required["v2_manifest"]).read_text(encoding="utf-8"))
    rm = json.loads(REVIEW_META.read_text(encoding="utf-8"))

    # 对账：v1 四件套必须等于 judge_adjudication_log 记录的原版哈希
    v1_log_hashes = {
        "final_verdicts.jsonl": "1ec1afcfe317c77f",
        "final_summary.json": "642302b30bc2f782",
        "pending_supplementary.jsonl": "6fc8fa81ccdeda4a",
        "manifest.json": "44f3056c7f475ac4",
    }
    v1_mismatch = []
    for f, prefix in v1_log_hashes.items():
        got = sha256_file(JUDGE_DIR / "ai_adjudication_v1" / f)
        if not got.startswith(prefix):
            v1_mismatch.append(f"{f}: {got[:16]} != {prefix}")
    if v1_mismatch:
        print("ERROR: ai_adjudication_v1 与实验日志记录的原版哈希不符（可能又被改动）:",
              file=sys.stderr)
        for m in v1_mismatch:
            print(f"  {m}", file=sys.stderr)
        return 1

    freeze = {
        "freeze_version": "v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "repeat": REPEAT, "seed": SEED, "snapshot": SNAPSHOT,
        "judge": {
            "judge_model": jm["judge_model"],
            "service": jm.get("service"),
            "temperature": jm.get("temperature"),
            "max_tokens": jm.get("max_tokens"),
            "judge_version": jm.get("judge_version"),
            "judge_prompt_hash": jm.get("judge_prompt_hash"),
            "json_schema_version": jm.get("json_schema_version"),
            "adjudication_rule_version": jm.get("adjudication_rule_version"),  # judge 层 v2
            "coverage_rule_version": jm.get("coverage_rule_version"),          # v3.1
            "retry_policy": jm.get("retry_policy"),
            "judge_script": jm.get("judge_script"),
        },
        "adjudication": {
            "rule_version": cal.ADJUDICATION_RULE_VERSION,      # v3
            "coverage_rule_version": jl.COVERAGE_RULE_VERSION,  # v3.1
            "frozen_coverage_pass": cal.FROZEN_COVERAGE_PASS,
            "effective_version": "ai_adjudication_v2",
            "counts": v2m.get("counts"),
        },
        "supplementary_review": {
            "model": rm.get("review_model"),
            "provider": rm.get("review_provider"),
            "temperature": rm.get("review_temperature"),
            "disable_thinking": rm.get("review_disable_thinking"),
            "guide_sha256": rm.get("guide_sha256"),
            "metadata_sha256": sha256_file(REVIEW_META),
        },
        "input_sha256": {
            "questions_file": sha256_file(QUESTIONS_FILE),
            "gold_context_file": sha256_file(GOLD_CTX_FILE),
            "annotation_all_v1": sha256_file(ANN_ALL),
            "eligibility": sha256_file(ELIG),
            "eligibility_summary": sha256_file(ELIG_SUM),
            "longcat_verdicts": {r: sha256_file(JUDGE_DIR / f"{r}_verdict.jsonl")
                                 for r in jl.ROUTES},
            "v1_outputs": {f: sha256_file(JUDGE_DIR / "ai_adjudication_v1" / f)
                           for f in v1_log_hashes},
            "v2_outputs": {f: sha256_file(JUDGE_DIR / "ai_adjudication_v2" / f)
                           for f in ("final_verdicts.jsonl", "final_summary.json",
                                     "excluded_records.jsonl", "manifest.json")},
        },
        "script_sha256": {
            "judge_longcat.py": sha256_file(Path(jl.__file__).resolve()),
            "calibrate_blind_review.py": sha256_file(Path(cal.__file__).resolve()),
            "build_repeat0_eligibility.py": sha256_file(
                Path(__file__).resolve().parent / "build_repeat0_eligibility.py"),
            "freeze_judge.py": sha256_file(Path(__file__).resolve()),
        },
        "known_limitations_ref": "docs/judge_adjudication_log.md §3",
        "note": "冻结点：进入 repeat 1/2 前，以上任何哈希/版本变更均需重新裁决并重冻",
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge 已冻结: {OUT}")
    print(f"  judge={jm['judge_model']} prompt_hash={str(jm.get('judge_prompt_hash'))[:16]}…")
    print(f"  rule: judge 层 {jm.get('adjudication_rule_version')} / adjudication {cal.ADJUDICATION_RULE_VERSION} / coverage {jl.COVERAGE_RULE_VERSION}")
    print(f"  v1 原版哈希对账通过（judge_adjudication_log §2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
