#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 5: repeat 0 题目准入清单。

输入: ai_adjudication_v2/final_verdicts.jsonl（480 条）+ excluded_records.jsonl
规则（用户 2026-08-16 验收结论）:
  1. P_gold fail 的题 → excluded（p_gold_fail）
  2. 涉及 unresolved pending 路径裁决的题 → router_label_indeterminate
     （仍可进 repeat 1/2，但 Router 标签不可确定）
  3. 其余 → eligible
输出: judge 根目录 repeat0_question_eligibility.jsonl + repeat0_eligibility_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import judge_longcat as jl

_ROOT = Path(__file__).resolve().parent.parent
# 与 calibrate_blind_review 同规则推导 judge 目录
REPEAT = 0
SEED = 42
SNAPSHOT = jl.SNAPSHOT_DEFAULT
JUDGE_DIR = jl.LONGCAT_DIR / f"r{REPEAT}_s{SEED}_{SNAPSHOT[:8]}"
V2_DIR = JUDGE_DIR / "ai_adjudication_v2"

ROUTES = jl.ROUTES


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build(dry_run: bool = False) -> int:
    fv_path = V2_DIR / "final_verdicts.jsonl"
    ex_path = V2_DIR / "excluded_records.jsonl"
    if not fv_path.exists():
        print(f"ERROR: 缺少 {fv_path}（先跑 calibrate_blind_review --mode apply --annotation-file ...）",
              file=sys.stderr)
        return 1
    fv = load_jsonl(fv_path)
    ex = load_jsonl(ex_path) if ex_path.exists() else []

    # ---- fail-fast 结构校验 ----
    errs = []
    keys = [(r["route"], r["question_id"]) for r in fv]
    if len(fv) != 480 or len(set(keys)) != 480:
        errs.append(f"final_verdicts 应为 480 条唯一 (route,qid)，实际 {len(fv)}/{len(set(keys))}")
    for route in ROUTES:
        n = sum(1 for r in fv if r["route"] == route)
        if n != 80:
            errs.append(f"{route}: {n} != 80")
    qids = {r["question_id"] for r in fv}
    if len(qids) != 80:
        errs.append(f"唯一题目数 {len(qids)} != 80")
    pending_rows = [r for r in fv if r["route_success"] == "pending"]
    if len(pending_rows) != len(ex):
        errs.append(f"pending {len(pending_rows)} != excluded_records {len(ex)}")
    ex_keys = {(r["route"], r["question_id"]) for r in ex}
    if ex_keys != {(r["route"], r["question_id"]) for r in pending_rows}:
        errs.append("excluded_records 键集合与 pending 行不一致")
    if not all(r.get("exclusion_reason") for r in ex):
        errs.append("excluded_records 存在缺失 exclusion_reason 的行")
    pg = [r for r in fv if r["route"] == "P_gold"]
    if any(r["route_success"] == "pending" for r in pg):
        errs.append("P_gold 存在 pending，准入规则不适用")
    if errs:
        print("ERROR: 结构校验失败（fail-closed）:", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1

    # ---- 逐题准入 ----
    by_q: dict[str, dict] = {}
    for r in fv:
        q = r["question_id"]
        rec = by_q.setdefault(q, {
            "question_id": q,
            "route_success": {},
            "pending_routes": [],
            "successful_routes": [],
            "failed_routes": [],
        })
        rec["route_success"][r["route"]] = r["route_success"]
        if r["route_success"] == "pass":
            rec["successful_routes"].append(r["route"])
        elif r["route_success"] == "fail":
            rec["failed_routes"].append(r["route"])
        else:
            rec["pending_routes"].append(r["route"])

    records = []
    for q in sorted(by_q, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        rec = by_q[q]
        gold = rec["route_success"]["P_gold"]
        if gold == "fail":
            rec["eligibility"] = "excluded"
            rec["reason"] = "p_gold_fail"
        elif rec["pending_routes"]:
            rec["eligibility"] = "router_label_indeterminate"
            rec["reason"] = "unresolved_pending_routes:" + ",".join(rec["pending_routes"])
        else:
            rec["eligibility"] = "eligible"
            rec["reason"] = None
        records.append(rec)

    dist = Counter(r["eligibility"] for r in records)
    if sum(dist.values()) != 80:
        print(f"ERROR: 准入记录 {len(records)} != 80", file=sys.stderr)
        return 1

    out_path = JUDGE_DIR / "repeat0_question_eligibility.jsonl"
    sum_path = JUDGE_DIR / "repeat0_eligibility_summary.json"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repeat": REPEAT, "seed": SEED, "snapshot": SNAPSHOT,
        "total_questions": 80,
        "eligibility_dist": dict(dist),
        "p_gold_dist": dict(Counter(r["route_success"]["P_gold"] for r in records)),
        "indeterminate_questions": [r["question_id"] for r in records
                                    if r["eligibility"] == "router_label_indeterminate"],
        "excluded_questions": [r["question_id"] for r in records
                               if r["eligibility"] == "excluded"],
        "repeat12_candidate_count": dist.get("eligible", 0) + dist.get("router_label_indeterminate", 0),
        "source": {
            "final_verdicts.jsonl": sha256_file(fv_path),
            "excluded_records.jsonl": sha256_file(ex_path),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }

    print(f"准入分布: {dict(dist)}  repeat 1/2 候选: {summary['repeat12_candidate_count']}")
    print(f"  excluded: {summary['excluded_questions']}")
    print(f"  indeterminate: {summary['indeterminate_questions']}")
    if dry_run:
        print("(dry-run，不写盘)")
        return 0

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写盘: {out_path.name} / {sum_path.name} -> {JUDGE_DIR}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Step 5: repeat 0 题目准入清单")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(build(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
