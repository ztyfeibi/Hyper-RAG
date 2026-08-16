#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 6: Judge 冻结清单 judge_freeze_manifest.json。

聚合并锁定:
  - LongCat judge: 模型 / prompt hash / rule v2 / coverage v3.1 / retry / 脚本哈希
    （引用 judge manifest.json，不重复计算）
  - AI adjudication: rule v3 / v1+v2 四件套哈希 / 318 条标注文件哈希 / review_metadata 哈希
  - 补充审核 prompt: REVIEW_GUIDE.md 哈希（与 review_metadata.prompt_guide_sha256 对账）
  - 输入题集: questions_v2_manual_final.jsonl / p_gold_contexts.jsonl 哈希
  - Step 5: repeat0_question_eligibility 哈希
冻结后任何一项变更都应表现为 manifest 对不上。

模式:
  默认       生成 freeze manifest（已存在则拒绝，--overwrite 才允许）
  --verify   只读重算全部冻结项哈希，任何漂移打印 expected/actual 并返回 1；
             全部一致打印 "Judge freeze verification: PASS" 返回 0。
             repeat 生成脚本启动前必须先跑该检查（preflight gate）。
  --verify 与 --overwrite 互斥。
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
SUP_DIR = JUDGE_DIR / "blind_review_supplementary"
ANN_ALL = SUP_DIR / "verdicts_ai_all_v1.jsonl"
REVIEW_META = SUP_DIR / "review_metadata.json"
REVIEW_GUIDE = SUP_DIR / "REVIEW_GUIDE.md"
ELIG = JUDGE_DIR / "repeat0_question_eligibility.jsonl"
ELIG_SUM = JUDGE_DIR / "repeat0_eligibility_summary.json"
OUT = JUDGE_DIR / "judge_freeze_manifest.json"

FREEZE_VERSION = "v1.1"

# docs/judge_adjudication_log.md §2 记录的 ai_adjudication_v1 原版哈希前缀
V1_LOG_HASH_PREFIXES = {
    "final_verdicts.jsonl": "1ec1afcfe317c77f",
    "final_summary.json": "642302b30bc2f782",
    "pending_supplementary.jsonl": "6fc8fa81ccdeda4a",
    "manifest.json": "44f3056c7f475ac4",
}

V2_OUTPUT_FILES = ("final_verdicts.jsonl", "final_summary.json",
                   "excluded_records.jsonl", "manifest.json")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_guide(p: Path) -> str:
    """REVIEW_GUIDE.md 的内容哈希。

    与 run_supplementary_review.py 的 prompt_guide_sha256 口径完全一致：
    read_text(encoding="utf-8")（universal newline，CRLF→LF）后再按 UTF-8 编码哈希。
    注意不能直接用 sha256_file：REVIEW_GUIDE.md 是 CRLF 行尾，按原始字节算会与
    review_metadata 记录值恒不相等。
    """
    return hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def required_files() -> dict[str, Path]:
    return {
        "judge_manifest": JUDGE_DIR / "manifest.json",
        "questions_file": QUESTIONS_FILE,
        "gold_context_file": GOLD_CTX_FILE,
        "v1_final_verdicts": JUDGE_DIR / "ai_adjudication_v1" / "final_verdicts.jsonl",
        "v2_final_verdicts": JUDGE_DIR / "ai_adjudication_v2" / "final_verdicts.jsonl",
        "v2_manifest": JUDGE_DIR / "ai_adjudication_v2" / "manifest.json",
        "annotation_all": ANN_ALL,
        "review_metadata": REVIEW_META,
        "review_guide": REVIEW_GUIDE,
        "eligibility": ELIG,
        "eligibility_summary": ELIG_SUM,
        **{f"longcat_{r}": JUDGE_DIR / f"{r}_verdict.jsonl" for r in jl.ROUTES},
    }


def v1_prefix_problems() -> list[str]:
    """对账：v1 四件套必须等于 judge_adjudication_log 记录的原版哈希。"""
    problems = []
    for f, prefix in V1_LOG_HASH_PREFIXES.items():
        got = sha256_file(JUDGE_DIR / "ai_adjudication_v1" / f)
        if not got.startswith(prefix):
            problems.append(f"ai_adjudication_v1/{f}: {got[:16]} != {prefix}")
    return problems


def guide_problems(rm: dict) -> list[str]:
    """REVIEW_GUIDE.md 实际内容哈希必须与 review_metadata 记录一致。"""
    actual = sha256_guide(REVIEW_GUIDE)
    recorded = rm.get("prompt_guide_sha256")
    if recorded != actual:
        return [f"REVIEW_GUIDE.md 哈希与 review_metadata 不一致: "
                f"actual={actual} metadata={recorded}"]
    return []


def current_hash_view() -> dict:
    """重算当前盘上全部冻结项哈希（build 与 verify 共用同一实现）。"""
    return {
        "input_sha256": {
            "questions_file": sha256_file(QUESTIONS_FILE),
            "gold_context_file": sha256_file(GOLD_CTX_FILE),
            "annotation_all_v1": sha256_file(ANN_ALL),
            "eligibility": sha256_file(ELIG),
            "eligibility_summary": sha256_file(ELIG_SUM),
            "longcat_verdicts": {r: sha256_file(JUDGE_DIR / f"{r}_verdict.jsonl")
                                 for r in jl.ROUTES},
            "v1_outputs": {f: sha256_file(JUDGE_DIR / "ai_adjudication_v1" / f)
                           for f in V1_LOG_HASH_PREFIXES},
            "v2_outputs": {f: sha256_file(JUDGE_DIR / "ai_adjudication_v2" / f)
                           for f in V2_OUTPUT_FILES},
        },
        "script_sha256": {
            "judge_longcat.py": sha256_file(Path(jl.__file__).resolve()),
            "calibrate_blind_review.py": sha256_file(Path(cal.__file__).resolve()),
            "build_repeat0_eligibility.py": sha256_file(
                Path(__file__).resolve().parent / "build_repeat0_eligibility.py"),
            "freeze_judge.py": sha256_file(Path(__file__).resolve()),
        },
    }


def _diff_nested(stored: dict, current: dict, prefix: str, problems: list) -> None:
    for k, cur in current.items():
        path = f"{prefix}{k}"
        if k not in stored:
            problems.append(f"{path}: manifest 缺失该字段（current={cur}）")
        elif isinstance(cur, dict):
            _diff_nested(stored[k], cur, f"{path}.", problems)
        elif stored[k] != cur:
            problems.append(f"{path}: manifest={stored[k]} actual={cur}")


def cmd_freeze(overwrite: bool) -> int:
    if OUT.exists() and not overwrite:
        print(f"ERROR: {OUT} 已存在（冻结产物禁止静默覆盖）；确认重冻加 --overwrite",
              file=sys.stderr)
        return 1

    req = required_files()
    missing = [str(p) for p in req.values() if not p.exists()]
    if missing:
        print("ERROR: 冻结前置产物缺失（fail-closed）:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    rm = json.loads(REVIEW_META.read_text(encoding="utf-8"))
    problems = v1_prefix_problems() + guide_problems(rm)
    if problems:
        print("ERROR: 冻结前置校验失败:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    jm = json.loads((JUDGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    v2m = json.loads((JUDGE_DIR / "ai_adjudication_v2" / "manifest.json").read_text(
        encoding="utf-8"))
    hashes = current_hash_view()

    freeze = {
        "freeze_version": FREEZE_VERSION,
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
            "guide_sha256": sha256_guide(REVIEW_GUIDE),  # 实际内容哈希（非 null）
            "metadata_sha256": sha256_file(REVIEW_META),
        },
        **hashes,
        "known_limitations_ref": "docs/judge_adjudication_log.md §3",
        "note": "冻结点：进入 repeat 1/2 前，以上任何哈希/版本变更均需重新裁决并重冻；"
                "repeat 生成脚本启动前必须先跑 freeze_judge.py --verify（preflight gate）",
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge 已冻结: {OUT}")
    print(f"  freeze_version={FREEZE_VERSION}")
    print(f"  judge={jm['judge_model']} prompt_hash={str(jm.get('judge_prompt_hash'))[:16]}…")
    print(f"  rule: judge 层 {jm.get('adjudication_rule_version')} / "
          f"adjudication {cal.ADJUDICATION_RULE_VERSION} / coverage {jl.COVERAGE_RULE_VERSION}")
    print(f"  guide_sha256={freeze['supplementary_review']['guide_sha256'][:16]}…")
    print(f"  v1 原版哈希对账通过（judge_adjudication_log §2）")
    return 0


def cmd_verify() -> int:
    """只读校验：重算全部冻结项哈希与 manifest 对账，漂移返回 1。"""
    if not OUT.exists():
        print(f"ERROR: {OUT} 不存在，先执行 freeze_judge.py 生成", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(OUT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: freeze manifest JSON 解析失败: {e}", file=sys.stderr)
        return 1

    problems: list[str] = []
    if manifest.get("freeze_version") != FREEZE_VERSION:
        problems.append(f"freeze_version: manifest={manifest.get('freeze_version')} "
                        f"expected={FREEZE_VERSION}")

    # 前置文件齐全性（缺失即漂移）
    for name, p in required_files().items():
        if not p.exists():
            problems.append(f"{name}: 文件缺失 {p}")

    # 身份字段：与 judge manifest / v2 manifest / review_metadata 重读值对账
    try:
        jm = json.loads((JUDGE_DIR / "manifest.json").read_text(encoding="utf-8"))
        v2m = json.loads((JUDGE_DIR / "ai_adjudication_v2" / "manifest.json").read_text(
            encoding="utf-8"))
        rm = json.loads(REVIEW_META.read_text(encoding="utf-8"))
        stored_judge = manifest.get("judge", {})
        expected_judge = {
            "judge_model": jm["judge_model"],
            "service": jm.get("service"),
            "temperature": jm.get("temperature"),
            "max_tokens": jm.get("max_tokens"),
            "judge_version": jm.get("judge_version"),
            "judge_prompt_hash": jm.get("judge_prompt_hash"),
            "json_schema_version": jm.get("json_schema_version"),
            "adjudication_rule_version": jm.get("adjudication_rule_version"),
            "coverage_rule_version": jm.get("coverage_rule_version"),
            "retry_policy": jm.get("retry_policy"),
            "judge_script": jm.get("judge_script"),
        }
        for k, cur in expected_judge.items():
            if stored_judge.get(k) != cur:
                problems.append(f"judge.{k}: manifest={stored_judge.get(k)} actual={cur}")
        if manifest.get("adjudication", {}).get("counts") != v2m.get("counts"):
            problems.append(f"adjudication.counts: manifest="
                            f"{manifest.get('adjudication', {}).get('counts')} "
                            f"actual={v2m.get('counts')}")
        sr = manifest.get("supplementary_review", {})
        sr_expected = {
            "model": rm.get("review_model"),
            "provider": rm.get("review_provider"),
            "temperature": rm.get("review_temperature"),
            "disable_thinking": rm.get("review_disable_thinking"),
        }
        for k, cur in sr_expected.items():
            if sr.get(k) != cur:
                problems.append(f"supplementary_review.{k}: manifest={sr.get(k)} actual={cur}")
    except (OSError, json.JSONDecodeError, KeyError) as e:
        problems.append(f"身份字段对账失败（上游 manifest 读取异常）: {e}")

    # 原版 v1 哈希 + guide 对账（与冻结时同一规则）
    problems.extend(v1_prefix_problems())
    problems.extend(guide_problems(rm))
    sr = manifest.get("supplementary_review", {})
    guide_actual = sha256_guide(REVIEW_GUIDE) if REVIEW_GUIDE.exists() else "<缺失>"
    if sr.get("guide_sha256") != guide_actual:
        problems.append(f"supplementary_review.guide_sha256: "
                        f"manifest={sr.get('guide_sha256')} actual={guide_actual}")
    meta_actual = sha256_file(REVIEW_META) if REVIEW_META.exists() else "<缺失>"
    if sr.get("metadata_sha256") != meta_actual:
        problems.append(f"supplementary_review.metadata_sha256: "
                        f"manifest={sr.get('metadata_sha256')} actual={meta_actual}")

    # 输入 / 输出 / 脚本哈希全量重算比对
    try:
        current = current_hash_view()
        for section in ("input_sha256", "script_sha256"):
            _diff_nested(manifest.get(section, {}), current[section],
                         f"{section}.", problems)
    except OSError as e:
        problems.append(f"哈希重算失败（文件读取异常）: {e}")

    if problems:
        print(f"ERROR: Judge 冻结漂移校验失败（{len(problems)} 项不一致）:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("Judge freeze verification: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6: 生成/校验 judge_freeze_manifest.json")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--overwrite", action="store_true",
                   help="允许覆盖已存在的 freeze manifest")
    g.add_argument("--verify", action="store_true",
                   help="只读重算全部冻结项哈希；发现漂移返回非零退出码（preflight gate）")
    args = ap.parse_args()

    if args.verify:
        return cmd_verify()
    return cmd_freeze(args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
