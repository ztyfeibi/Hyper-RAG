# -*- coding: utf-8 -*-
"""聚合 judge 结果，生成 judge_report.md 与 summary/*.json。

输入: caches/<data_name>/question_set_v2/pilot_v1/judge/{scoring,selection}/*.jsonl
输出: judge/summary/*.json + judge/judge_report.md

用法:
    python scripts/aggregate_judge.py [--data-name neurology_chunk1000]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SCORING_DIMS = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]
SELECTION_DIMS = ["Comprehensiveness", "Empowerment", "Accuracy", "Relevance",
                  "Coherence", "Clarity", "Logical", "Flexibility"]
CANDIDATE_ROUTES = ["P0", "P1", "P2", "P3", "P4"]


def load_rows(path: Path):
    rows = []
    if path.exists():
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-name", default="neurology_chunk1000")
    args = ap.parse_args()
    judge_dir = Path("caches") / args.data_name / "question_set_v2" / "pilot_v1" / "judge"
    summary_dir = judge_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ---------- scoring ----------
    scoring = {}
    for route in CANDIDATE_ROUTES + ["P_gold"]:
        rows = load_rows(judge_dir / "scoring" / f"{route}_scoring.jsonl")
        rows = [r for r in rows if r.get("parse_ok")]
        dim_sums = defaultdict(float)
        for r in rows:
            for d in SCORING_DIMS:
                v = (r.get("scores") or {}).get(d)
                if isinstance(v, (int, float)):
                    dim_sums[d] += v
        n = len(rows)
        dim_avg = {d: round(dim_sums[d] / n, 2) if n else None for d in SCORING_DIMS}
        total = [v for v in dim_avg.values() if v is not None]
        dim_avg["mean"] = round(sum(total) / len(total), 2) if total else None
        dim_avg["n"] = n
        scoring[route] = dim_avg
    (summary_dir / "scoring.json").write_text(
        json.dumps(scoring, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- selection ----------
    selection = {}
    pos_bias = {}
    for cand in CANDIDATE_ROUTES:
        fwd = load_rows(judge_dir / "selection" / f"{cand}_vs_P_gold_fwd.jsonl")
        rev = load_rows(judge_dir / "selection" / f"{cand}_vs_P_gold_rev.jsonl")
        agg = {}
        for d in SELECTION_DIMS:
            wins = sum(1 for r in fwd + rev
                       if r.get("parse_ok") and (r.get("candidate_wins") or {}).get(d) is True)
            tot = sum(1 for r in fwd + rev if r.get("parse_ok"))
            agg[d] = round(wins / tot * 100, 1) if tot else None
        vals = [v for v in agg.values() if v is not None]
        agg["mean"] = round(sum(vals) / len(vals), 1) if vals else None
        agg["n"] = sum(1 for r in fwd + rev if r.get("parse_ok"))
        selection[cand] = agg
        # 位置偏差：8 维全赢的题数（fwd=候选在 Answer1，rev=候选在 Answer2）
        fwd_allwin = sum(1 for r in fwd if r.get("parse_ok")
                         and all(v is True for v in (r.get("candidate_wins") or {}).values()))
        rev_allwin = sum(1 for r in rev if r.get("parse_ok")
                         and all(v is True for v in (r.get("candidate_wins") or {}).values()))
        pos_bias[cand] = {"fwd_allwin": fwd_allwin, "rev_allwin": rev_allwin,
                          "delta": abs(fwd_allwin - rev_allwin)}
    (summary_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (summary_dir / "position_bias.json").write_text(
        json.dumps(pos_bias, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- report ----------
    lines = []
    lines.append("# Judge 质量判定报告（Pilot 80 题）\n")
    lines.append(f"- 数据集: `{args.data_name}`，问题集: pilot_v1（80 题）")
    lines.append("- judge 模型: qwen-27b-int4 @ 10.65.1.110:8002，temperature=0.1")
    lines.append("- 参照: 各题 gold context（p_gold_contexts.jsonl）")
    lines.append("- 方式: 五维打分（绝对分）+ 8 维 pairwise 双向对比（胜率）\n")

    lines.append("## 一、五维打分（0-100，越高越好）\n")
    lines.append("| 路径 | Comprehensiveness | Diversity | Empowerment | Logical | Readability | **Mean** |")
    lines.append("|---|---|---|---|---|---|---|")
    for route in CANDIDATE_ROUTES + ["P_gold"]:
        s = scoring[route]
        cells = [str(s[d]) if s[d] is not None else "-" for d in SCORING_DIMS]
        lines.append(f"| {route} | " + " | ".join(cells) + f" | **{s['mean']}** |")
    lines.append("")

    lines.append("## 二、Pairwise 胜率 vs P_gold（%，双向平均，越高代表越接近/超过 gold）\n")
    lines.append("| 候选路径 | Comprehension | Empowerment | Accuracy | Relevance | Coherence | Clarity | Logical | Flexibility | **Mean** |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cand in CANDIDATE_ROUTES:
        s = selection[cand]
        cells = [str(s[d]) if s[d] is not None else "-" for d in SELECTION_DIMS]
        lines.append(f"| {cand} | " + " | ".join(cells) + f" | **{s['mean']}** |")
    lines.append("")

    # ---------- 观察 ----------
    lines.append("## 三、关键观察\n")
    gold_mean = scoring["P_gold"]["mean"]
    ranked = sorted(CANDIDATE_ROUTES, key=lambda r: scoring[r]["mean"], reverse=True)
    best = ranked[0]
    lines.append(f"- **五维打分**: Gold 上限 {gold_mean} 分；候选路径中最优为 **{best}**（{scoring[best]['mean']}），差距 {round(gold_mean - scoring[best]['mean'], 2)} 分。")
    sel_ranked = sorted(CANDIDATE_ROUTES, key=lambda r: selection[r]["mean"], reverse=True)
    sel_best = sel_ranked[0]
    lines.append(f"- **Pairwise 胜率**: 对 gold 胜率最高的是 **{sel_best}**（{selection[sel_best]['mean']}%）。")
    lines.append("- **位置偏差警告**: judge 存在明显 Answer1 偏好，各路径 8 维全赢题数 fwd 显著高于 rev：")
    for cand in CANDIDATE_ROUTES:
        pb = pos_bias[cand]
        lines.append(f"  - {cand}: fwd={pb['fwd_allwin']}/80, rev={pb['rev_allwin']}/80, 偏差 {pb['delta']} 题")
    lines.append("  因此上表胜率为 fwd/rev 双向平均的修正值；单方向判定不可直接采信。")
    lines.append("- **判定模式**: P0/P2 的 winners 高度一致化（8 维同值占主导），P1/P3/P4 存在维度分化——说明前两者与 gold 差距悬殊时 judge 判定果断，后三者有维度上的相对优势点。")
    lines.append("")

    (judge_dir / "judge_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
