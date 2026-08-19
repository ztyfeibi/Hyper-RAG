#!/usr/bin/env python3
"""汇总 r0/r1/r2 六路径盲审裁决的稳定成功率，并生成 minimum_sufficient_route_by_policy 标签。

输入：caches/.../judge/longcat/{r0_s42,r1_s43,r2_s44}_5c92f17c*/ai_adjudication_v2/final_verdicts.jsonl
输出：同 r2 目录下 stability_report.json / stability_report.md

稳定定义：对某个 (question_id, route)，r0/r1/r2 三份的 route_success 完全相同且为
          pass 或 fail（pending / judge_error 视为未完成，排除在稳定之外）。
minimum_sufficient_route_by_policy（默认 policy A1）：每个 question 按路由优先级
          [P_gold, P0, P1, P2, P3, P4] 选最优先的稳定单路由作为该题的 sufficient route。
"""
import json
import glob
from collections import Counter
from pathlib import Path

ROUTES = ["P_gold", "P0", "P1", "P2", "P3", "P4"]
REPS = {"r0": "r0_s42", "r1": "r1_s43", "r2": "r2_s44"}
BASE = Path("caches/neurology_chunk1000/question_set_v2/pilot_v1/judge/longcat")


def load(rep: str) -> dict:
    pat = str(BASE / f"{REPS[rep]}_5c92f17c*" / "ai_adjudication_v2" / "final_verdicts.jsonl")
    fs = glob.glob(pat)
    if not fs:
        raise FileNotFoundError(f"{rep} final_verdicts not found")
    rows = [json.loads(l) for l in open(fs[0], encoding="utf-8") if l.strip()]
    return {(r["question_id"], r["route"]): r for r in rows}


def main():
    data = {rep: load(rep) for rep in REPS}
    questions = sorted({k[0] for k in data["r0"]})

    def vs(q, rt):
        return {rep: data[rep][(q, rt)].get("route_success") for rep in REPS}

    # 稳定：三 repeat 的 route_success 完全相同且为 pass/fail（排除 pending/judge_error）
    matrix = {}
    for q in questions:
        for rt in ROUTES:
            v = vs(q, rt)
            stable = (v["r0"] == v["r1"] == v["r2"]) and v["r0"] in ("pass", "fail")
            matrix[(q, rt)] = {"verdicts": v, "stable": stable}

    per_route_stable = Counter()
    for (_q, rt), m in matrix.items():
        if m["stable"]:
            per_route_stable[rt] += 1
    total_stable = sum(1 for m in matrix.values() if m["stable"])

    # minimum_sufficient_route_by_policy (policy A1: 最优先的稳定单路由)
    min_route = {}
    for q in questions:
        chosen = next((rt for rt in ROUTES if matrix[(q, rt)]["stable"]), "NONE")
        min_route[q] = chosen

    report = {
        "policy": ("A1: 每个 question 按路由优先级 [P_gold,P0,P1,P2,P3,P4] 选最优先的稳定单路由"
                   "（三方 route_success 一致且为 pass/fail）"),
        "summary": {
            "total_cells": len(matrix),
            "stable_cells": total_stable,
            "overall_stability_rate": total_stable / len(matrix),
            "per_route_stability_rate": {rt: per_route_stable[rt] / 80 for rt in ROUTES},
        },
        "min_route_distribution": dict(Counter(min_route.values())),
        "min_route_per_question": min_route,
        "unstable_cells": [
            {"question_id": q, "route": rt, "verdicts": matrix[(q, rt)]["verdicts"]}
            for (q, rt), m in matrix.items() if not m["stable"]
        ],
    }

    out_dir = Path(glob.glob(str(BASE / "r2_s44_5c92f17c*"))[0])
    (out_dir / "stability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown 报告
    s = report["summary"]
    L = []
    L.append("# Repeat 稳定性汇总报告 (r0/r1/r2 六路径)\n")
    L.append(f"- 总 cell 数 (80 题 × 6 路径): **{s['total_cells']}**")
    L.append(f"- 三方稳定 cell 数: **{s['stable_cells']}**")
    L.append(f"- 总体稳定成功率: **{s['overall_stability_rate']:.1%}**\n")
    L.append("## 每路径稳定成功率\n")
    L.append("| route | 稳定/80 | 率 |")
    L.append("|---|---|---|")
    for rt in ROUTES:
        L.append(f"| {rt} | {per_route_stable[rt]} | {per_route_stable[rt]/80:.1%} |")
    L.append("\n## minimum_sufficient_route_by_policy 分布 (policy A1)\n")
    for k, v in sorted(report["min_route_distribution"].items(), key=lambda x: -x[1]):
        L.append(f"- {k}: **{v}** 题")
    L.append(f"\n## 不稳定 cell 清单 ({len(report['unstable_cells'])} 个，三方不一致或含 pending/judge_error)\n")
    for i, u in enumerate(report["unstable_cells"]):
        L.append(f"- {u['question_id']} / {u['route']}: "
                 f"r0={u['verdicts']['r0']}, r1={u['verdicts']['r1']}, r2={u['verdicts']['r2']}")
        if i >= 80:
            L.append(f"- ...(其余 {len(report['unstable_cells']) - 81} 个见 stability_report.json)")
            break
    (out_dir / "stability_report.md").write_text("\n".join(L), encoding="utf-8")

    print("written:", out_dir / "stability_report.md")
    print("overall stability:", f"{s['overall_stability_rate']:.1%}")
    print("per_route stable:", dict(per_route_stable))
    print("min_route dist:", report["min_route_distribution"])


if __name__ == "__main__":
    main()
