#!/usr/bin/env python3
"""Step 2.3-QA: Generate a quality report for the 80-question pilot set.

Reads:
  - questions_v2.jsonl  (final question set)
  - review_raw/*.json    (LongCat review results)

Outputs:
  - quality_report.html  (human-readable HTML report)
  - quality_report.json  (machine-readable summary)

Usage:
  python scripts/generate_quality_report.py --data-name neurology_chunk1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_questions(qfile: Path) -> list[dict]:
    items = []
    with open(qfile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_reviews(review_dir: Path) -> dict[str, dict]:
    reviews = {}
    if not review_dir.exists():
        return reviews
    for fname in os.listdir(review_dir):
        if not fname.endswith(".json"):
            continue
        with open(review_dir / fname, encoding="utf-8") as f:
            d = json.load(f)
        qid = d.get("question_id", fname.replace(".json", ""))
        reviews[qid] = d
    return reviews


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

CHECK_FIELDS = [
    "is_english",
    "is_open_ended",
    "answerable_from_evidence",
    "no_answer_leak",
    "no_external_knowledge",
    "not_overly_broad",
    "not_choice_like",
]

CHECK_LABELS = {
    "is_english": "English",
    "is_open_ended": "Open-ended",
    "answerable_from_evidence": "Answerable",
    "no_answer_leak": "No leak",
    "no_external_knowledge": "No ext. knowledge",
    "not_overly_broad": "Not too broad",
    "not_choice_like": "Not choice-like",
}


def analyze(items: list[dict], reviews: dict[str, dict]) -> dict:
    # Verdict distribution
    verdicts = Counter()
    # Structure × verdict cross-tab
    struct_verdict = defaultdict(lambda: Counter())
    # Check failure counts
    check_failures = Counter()
    # Per-structure stats
    struct_stats = defaultdict(lambda: {
        "count": 0,
        "n_aus": [],
        "n_chunks": [],
        "n_vertices": [],
        "n_hyperedges": [],
        "n_spans": [],
    })
    # needs_revision detail list
    needs_revision_detail = []
    # All items detail for table
    all_items_detail = []

    for q in items:
        qid = q["question_id"]
        struct = q["intended_structure"]
        rev = reviews.get(qid, {})
        parsed = rev.get("parsed", {})
        verdict = parsed.get("verdict", "not_reviewed")

        verdicts[verdict] += 1
        struct_verdict[struct][verdict] += 1

        # Check failures
        checks = parsed.get("checks", {})
        failed_checks = []
        for chk in CHECK_FIELDS:
            if chk in checks and checks[chk] is False:
                failed_checks.append(chk)
                check_failures[chk] += 1

        # Structure stats
        rsg = q.get("required_subgraph", {})
        n_verts = len(rsg.get("required_vertices", []))
        n_hes = len(rsg.get("required_hyperedges", []))
        n_chunks = sum(
            len(er.get("alternative_chunk_ids", []))
            for er in q.get("evidence_requirements", [])
        )
        n_spans = sum(
            len(es.get("evidence_spans", []))
            for es in q.get("evidence_spans", [])
        )
        ss = struct_stats[struct]
        ss["count"] += 1
        ss["n_aus"].append(len(q.get("answer_units", [])))
        ss["n_chunks"].append(n_chunks)
        ss["n_vertices"].append(n_verts)
        ss["n_hyperedges"].append(n_hes)
        ss["n_spans"].append(n_spans)

        # Compact detail
        detail = {
            "question_id": qid,
            "intended_structure": struct,
            "verdict": verdict,
            "question": q["question"],
            "gold_answer": q["gold_answer"][:200] + "..." if len(q["gold_answer"]) > 200 else q["gold_answer"],
            "n_answer_units": len(q.get("answer_units", [])),
            "n_evidence_chunks": n_chunks,
            "n_vertices": n_verts,
            "n_hyperedges": n_hes,
            "n_evidence_spans": n_spans,
            "failed_checks": failed_checks,
            "issues": parsed.get("issues", []),
            "revised_question": parsed.get("revised_question"),
        }
        all_items_detail.append(detail)

        if verdict == "needs_revision":
            needs_revision_detail.append(detail)

    # Aggregate struct stats
    struct_summary = {}
    for struct, ss in struct_stats.items():
        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0
        struct_summary[struct] = {
            "count": ss["count"],
            "avg_aus": avg(ss["n_aus"]),
            "avg_chunks": avg(ss["n_chunks"]),
            "avg_vertices": avg(ss["n_vertices"]),
            "avg_hyperedges": avg(ss["n_hyperedges"]),
            "avg_spans": avg(ss["n_spans"]),
        }

    return {
        "total": len(items),
        "verdicts": dict(verdicts),
        "struct_verdict": {k: dict(v) for k, v in struct_verdict.items()},
        "check_failures": dict(check_failures),
        "struct_summary": struct_summary,
        "needs_revision_detail": needs_revision_detail,
        "all_items_detail": all_items_detail,
    }


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

def _esc(s: str) -> str:
    """Escape HTML special chars."""
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def build_html(report: dict) -> str:
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'><head><meta charset='utf-8'>")
    lines.append("<title>Pilot Question Set — Quality Report</title>")
    lines.append("<style>")
    lines.append("""
    body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px; color: #1a1a1a; background: #fafafa; }
    h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2 { font-size: 18px; margin-top: 32px; color: #333; }
    h3 { font-size: 15px; margin-top: 20px; color: #555; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; background: #fff; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
    th { background: #f0f0f0; font-weight: 600; }
    tr:nth-child(even) { background: #f9f9f9; }
    .verdict-pass { color: #2e7d32; font-weight: 600; }
    .verdict-needs_revision { color: #e65100; font-weight: 600; }
    .verdict-fail { color: #c62828; font-weight: 600; }
    .check-fail { color: #c62828; }
    .check-pass { color: #2e7d32; }
    .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 12px 0; }
    .stat-box { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 16px; text-align: center; }
    .stat-num { font-size: 28px; font-weight: 700; color: #1565c0; }
    .stat-label { font-size: 12px; color: #666; margin-top: 4px; }
    .revision-item { border-left: 3px solid #e65100; padding-left: 12px; margin: 16px 0; }
    .question-text { font-style: italic; color: #424242; margin: 4px 0; }
    .gold-answer { font-size: 12px; color: #757575; margin: 4px 0; }
    .issues { background: #fff3e0; padding: 8px 12px; border-radius: 4px; margin: 6px 0; font-size: 12px; }
    .revised { background: #e8f5e9; padding: 8px 12px; border-radius: 4px; margin: 6px 0; font-size: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .badge-struct { background: #e3f2fd; color: #1565c0; }
    .summary-row { display: flex; gap: 16px; flex-wrap: wrap; }
    """)
    lines.append("</style></head><body>")

    # Header
    lines.append("<h1>Pilot Question Set — Quality Report</h1>")
    lines.append(f"<p>Generated from <code>questions_v2.jsonl</code> ({report['total']} questions)</p>")

    # Overview stats
    lines.append("<div class='stat-grid'>")
    lines.append(f"<div class='stat-box'><div class='stat-num'>{report['total']}</div><div class='stat-label'>Total Questions</div></div>")
    lines.append(f"<div class='stat-box'><div class='stat-num' style='color:#2e7d32'>{report['verdicts'].get('pass', 0)}</div><div class='stat-label'>Pass</div></div>")
    lines.append(f"<div class='stat-box'><div class='stat-num' style='color:#e65100'>{report['verdicts'].get('needs_revision', 0)}</div><div class='stat-label'>Needs Revision</div></div>")
    lines.append(f"<div class='stat-box'><div class='stat-num' style='color:#c62828'>{report['verdicts'].get('fail', 0) + report['verdicts'].get('parse_error', 0)}</div><div class='stat-label'>Fail / Parse Error</div></div>")
    lines.append("</div>")

    # Structure × verdict cross-tab
    lines.append("<h2>Structure × Verdict Cross-Tab</h2>")
    all_verdicts = ["pass", "needs_revision", "fail", "parse_error", "not_reviewed"]
    structs = ["single_fact", "single_high_arity", "multi_edge_chain", "multi_branch", "similar_subgraph_disambiguation"]
    lines.append("<table><tr><th>Structure</th>")
    for v in all_verdicts:
        lines.append(f"<th>{v}</th>")
    lines.append("<th>Total</th></tr>")
    for s in structs:
        sv = report["struct_verdict"].get(s, {})
        total = sum(sv.values())
        lines.append(f"<tr><td><span class='badge badge-struct'>{s}</span></td>")
        for v in all_verdicts:
            count = sv.get(v, 0)
            cls = f"verdict-{v}" if count else ""
            lines.append(f"<td class='{cls}'>{count if count else ''}</td>")
        lines.append(f"<td><b>{total}</b></td></tr>")
    lines.append("</table>")

    # Check failure analysis
    lines.append("<h2>Review Check Failure Analysis</h2>")
    lines.append("<table><tr><th>Check</th><th>Failures</th><th>Failure Rate</th></tr>")
    for chk in CHECK_FIELDS:
        fails = report["check_failures"].get(chk, 0)
        rate = f"{fails / report['total'] * 100:.1f}%" if report["total"] else "N/A"
        label = CHECK_LABELS.get(chk, chk)
        cls = "check-fail" if fails else "check-pass"
        lines.append(f"<tr><td>{label}</td><td class='{cls}'>{fails}</td><td>{rate}</td></tr>")
    lines.append("</table>")

    # Structure statistics
    lines.append("<h2>Per-Structure Statistics</h2>")
    lines.append("<table><tr><th>Structure</th><th>Count</th><th>Avg AUs</th><th>Avg Chunks</th><th>Avg Vertices</th><th>Avg Hyperedges</th><th>Avg Spans</th></tr>")
    for s in structs:
        ss = report["struct_summary"].get(s, {})
        lines.append(
            f"<tr><td><span class='badge badge-struct'>{s}</span></td>"
            f"<td>{ss.get('count', 0)}</td>"
            f"<td>{ss.get('avg_aus', 0)}</td>"
            f"<td>{ss.get('avg_chunks', 0)}</td>"
            f"<td>{ss.get('avg_vertices', 0)}</td>"
            f"<td>{ss.get('avg_hyperedges', 0)}</td>"
            f"<td>{ss.get('avg_spans', 0)}</td></tr>"
        )
    lines.append("</table>")

    # Needs revision detail
    nr_items = report["needs_revision_detail"]
    lines.append(f"<h2>Needs Revision Detail ({len(nr_items)} items)</h2>")
    for item in nr_items:
        lines.append("<div class='revision-item'>")
        lines.append(f"<div><span class='badge badge-struct'>{item['intended_structure']}</span> "
                     f"<b>{item['question_id']}</b> "
                     f"<span class='verdict-needs_revision'>needs_revision</span></div>")
        lines.append(f"<div class='question-text'>Q: {_esc(item['question'])}</div>")
        lines.append(f"<div class='gold-answer'>Gold: {_esc(item['gold_answer'])}</div>")

        if item["failed_checks"]:
            failed_labels = [CHECK_LABELS.get(c, c) for c in item["failed_checks"]]
            lines.append(f"<div>Failed checks: <span class='check-fail'>{', '.join(failed_labels)}</span></div>")

        if item["issues"]:
            lines.append("<div class='issues'>")
            lines.append("<b>Issues:</b><ul>")
            for iss in item["issues"]:
                lines.append(f"<li>{_esc(iss)}</li>")
            lines.append("</ul></div>")

        if item["revised_question"]:
            lines.append(f"<div class='revised'><b>LongCat suggests:</b> {_esc(item['revised_question'])}</div>")

        lines.append(f"<div style='font-size:11px;color:#999;margin-top:4px;'>"
                     f"AUs={item['n_answer_units']}, Chunks={item['n_evidence_chunks']}, "
                     f"Vertices={item['n_vertices']}, Hyperedges={item['n_hyperedges']}, "
                     f"Spans={item['n_evidence_spans']}</div>")
        lines.append("</div>")

    # Full question listing (compact)
    lines.append("<h2>All Questions (Compact Listing)</h2>")
    lines.append("<table><tr><th>ID</th><th>Structure</th><th>Verdict</th><th>Question</th><th>AUs</th><th>Chunks</th><th>HEs</th></tr>")
    for item in report["all_items_detail"]:
        q_short = item["question"][:120] + "..." if len(item["question"]) > 120 else item["question"]
        verdict_cls = f"verdict-{item['verdict']}"
        lines.append(
            f"<tr><td>{item['question_id']}</td>"
            f"<td><span class='badge badge-struct'>{item['intended_structure']}</span></td>"
            f"<td class='{verdict_cls}'>{item['verdict']}</td>"
            f"<td>{_esc(q_short)}</td>"
            f"<td>{item['n_answer_units']}</td>"
            f"<td>{item['n_evidence_chunks']}</td>"
            f"<td>{item['n_hyperedges']}</td></tr>"
        )
    lines.append("</table>")

    lines.append("</body></html>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Generate quality report for pilot question set")
    parser.add_argument("--data-name", default="neurology_chunk1000")
    args = parser.parse_args()

    base = REPO_ROOT / "caches" / args.data_name / "question_set_v2" / "pilot_v1"
    qdir = base / "question_generation"
    qfile = qdir / "questions_v2.jsonl"
    review_dir = qdir / "review_raw"

    if not qfile.exists():
        print(f"ERROR: {qfile} not found", file=sys.stderr)
        return 1

    items = load_questions(qfile)
    reviews = load_reviews(review_dir)
    report = analyze(items, reviews)

    # Write JSON
    json_path = qdir / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report: {json_path}")

    # Write HTML
    html_path = qdir / "quality_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(report))
    print(f"HTML report: {html_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"Quality Report Summary ({report['total']} questions)")
    print(f"{'='*60}")
    print(f"  Pass:           {report['verdicts'].get('pass', 0)}")
    print(f"  Needs revision: {report['verdicts'].get('needs_revision', 0)}")
    print(f"  Fail:           {report['verdicts'].get('fail', 0)}")
    print(f"  Parse error:    {report['verdicts'].get('parse_error', 0)}")
    print(f"\n  Check failures:")
    for chk in CHECK_FIELDS:
        fails = report["check_failures"].get(chk, 0)
        if fails:
            print(f"    {CHECK_LABELS.get(chk, chk):20s}: {fails}")
    print(f"\n  Structure stats:")
    for s in ["single_fact", "single_high_arity", "multi_edge_chain",
              "multi_branch", "similar_subgraph_disambiguation"]:
        ss = report["struct_summary"].get(s, {})
        print(f"    {s:40s}: n={ss.get('count',0)}, avg_AU={ss.get('avg_aus',0)}, "
              f"avg_chunks={ss.get('avg_chunks',0)}, avg_HE={ss.get('avg_hyperedges',0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
