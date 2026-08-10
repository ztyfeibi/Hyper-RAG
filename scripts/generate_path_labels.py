#!/usr/bin/env python3
"""Step 2.3-Label: Generate heuristic path labels for Router experiments.

For each of the 80 verified questions, proposes a minimum sufficient retrieval
route (P0..P4) based on evidence structure.  These are RESEARCHER-ANNOTATED
oracle labels — the "expected" minimum route that the Router should learn to
predict.  They are NOT experimental results (those come from derive_labels()).

Policy ladder recap (from question_set_v2_contract.yaml):
  P0 = zero-shot (no retrieval, parametric knowledge only)
  P1 = naive RAG (chunk VDB only, 4000 token budget)
  P2 = entity-seeded graph retrieval (entity VDB + hyperedge expansion)
  P3 = entity + relation VDB (dual graph lines)
  P4 = full graph retrieval with max budgets + disambiguation

Outputs:
  - path_labels.jsonl       (machine-readable labels, one per question)
  - path_labels_report.html (human-readable report for review/override)

Usage:
  python scripts/generate_path_labels.py --data-name neurology_chunk1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

POLICY_STAGES = ["P0", "P1", "P2", "P3", "P4"]

ROUTE_DESCRIPTIONS = {
    "P0": "Zero-shot (no retrieval, parametric knowledge only)",
    "P1": "Naive RAG (chunk VDB text retrieval)",
    "P2": "Entity-seeded graph retrieval (entity VDB + hyperedge expansion)",
    "P3": "Dual-line graph retrieval (entity + relation VDB)",
    "P4": "Full adaptive graph retrieval (max budgets + disambiguation)",
}

STRUCTURE_DEFAULT_ROUTE = {
    "single_fact": "P1",
    "single_high_arity": "P2",
    "multi_edge_chain": "P2",
    "multi_branch": "P3",
    "similar_subgraph_disambiguation": "P4",
}


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def extract_features(q: dict) -> dict:
    """Extract evidence-structure features relevant to route prediction."""
    rsg = q.get("required_subgraph", {})
    verts = rsg.get("required_vertices", [])
    hes = rsg.get("required_hyperedges", [])
    au_links = rsg.get("answer_unit_links", {})

    n_aus = len(q.get("answer_units", []))
    n_required_aus = sum(1 for au in q.get("answer_units", []) if au.get("required", True))

    # Count unique chunks across all evidence requirements
    all_chunks = set()
    for er in q.get("evidence_requirements", []):
        for cid in er.get("alternative_chunk_ids", []):
            all_chunks.add(cid)
    n_unique_chunks = len(all_chunks)

    # Check if any AU has multiple alternative chunks (disambiguation signal)
    has_alternatives = any(
        len(er.get("alternative_chunk_ids", [])) > 1
        for er in q.get("evidence_requirements", [])
    )

    # Count evidence spans
    n_spans = sum(
        len(es.get("evidence_spans", []))
        for es in q.get("evidence_spans", [])
    )

    # Hyperedge arity distribution
    he_arities = []
    for he in hes:
        he_arities.append(len(he.get("entity_set", [])))

    # AU-to-hyperedge linkage: how many AUs are linked to hyperedges
    aus_with_he_links = sum(
        1 for au_id, links in au_links.items() if links
    )

    # Topology metrics
    topo = rsg.get("topology_metrics", {})

    return {
        "n_answer_units": n_aus,
        "n_required_aus": n_required_aus,
        "n_unique_chunks": n_unique_chunks,
        "n_evidence_spans": n_spans,
        "n_vertices": len(verts),
        "n_hyperedges": len(hes),
        "max_he_arity": max(he_arities) if he_arities else 0,
        "has_alternatives": has_alternatives,
        "aus_with_he_links": aus_with_he_links,
        "he_arities": he_arities,
        "topology": topo,
    }


# --------------------------------------------------------------------------- #
# Heuristic route prediction
# --------------------------------------------------------------------------- #

def predict_route(structure: str, feat: dict) -> tuple[str, str, str]:
    """Predict minimum sufficient route.

    Returns: (route, confidence, rationale)
    """
    base = STRUCTURE_DEFAULT_ROUTE.get(structure, "P2")

    reasons = []

    # --- Structure-based starting point ---
    reasons.append(f"Base route for {structure}: {base}")

    # --- Escalation rules (push higher if complexity warrants) ---

    # Many answer units → need broader retrieval
    if feat["n_answer_units"] >= 5 and base in ("P1",):
        base = "P2"
        reasons.append(f"Escalated to P2: {feat['n_answer_units']} AUs need graph expansion")

    # Multiple unique chunks spread across graph → need entity traversal
    if feat["n_unique_chunks"] >= 4 and base == "P1":
        base = "P2"
        reasons.append(f"Escalated to P2: {feat['n_unique_chunks']} unique chunks need entity seeding")

    # High-arity hyperedges (≥4 entities) → relation VDB helps
    if feat["max_he_arity"] >= 4 and base in ("P1", "P2"):
        base = "P3"
        reasons.append(f"Escalated to P3: max hyperedge arity={feat['max_he_arity']} needs relation VDB")

    # Multiple hyperedges with AU linkage → branching complexity
    if feat["n_hyperedges"] >= 3 and feat["aus_with_he_links"] >= 2 and base in ("P1", "P2"):
        base = "P3"
        reasons.append(f"Escalated to P3: {feat['n_hyperedges']} hyperedges with {feat['aus_with_he_links']} AU links")

    # Disambiguation structures always get P4
    if structure == "similar_subgraph_disambiguation":
        base = "P4"
        reasons.append("Fixed P4: disambiguation structure requires full adaptive retrieval")

    # --- De-escalation rules (push lower if simplicity warrants) ---

    # Single AU, single chunk, no graph → P1 is enough
    if (feat["n_answer_units"] == 1 and feat["n_unique_chunks"] <= 1
            and feat["n_hyperedges"] == 0 and base in ("P2", "P3")):
        base = "P1"
        reasons.append("De-escalated to P1: single AU, single chunk, no graph structure")

    # Two AUs, both from same chunk, no hyperedges → P1
    if (feat["n_answer_units"] <= 2 and feat["n_unique_chunks"] <= 1
            and feat["n_hyperedges"] == 0 and base == "P2"):
        base = "P1"
        reasons.append("De-escalated to P1: ≤2 AUs from same chunk, no hyperedges")

    # --- Confidence ---
    if structure == "similar_subgraph_disambiguation":
        confidence = "high"
    elif base == STRUCTURE_DEFAULT_ROUTE.get(structure):
        confidence = "medium"
    else:
        confidence = "low"

    rationale = "; ".join(reasons)
    return base, confidence, rationale


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

def _esc(s) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def build_html(labels: list[dict], summary: dict) -> str:
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'><head><meta charset='utf-8'>")
    lines.append("<title>Path Labels — Router Oracle Annotations</title>")
    lines.append("<style>")
    lines.append("""
    body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px; color: #1a1a1a; background: #fafafa; }
    h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2 { font-size: 18px; margin-top: 32px; color: #333; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; background: #fff; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
    th { background: #f0f0f0; font-weight: 600; }
    tr:nth-child(even) { background: #f9f9f9; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .badge-struct { background: #e3f2fd; color: #1565c0; }
    .route-P0 { background: #e8eaf6; color: #283593; }
    .route-P1 { background: #e8f5e9; color: #2e7d32; }
    .route-P2 { background: #fff3e0; color: #e65100; }
    .route-P3 { background: #fce4ec; color: #ad1457; }
    .route-P4 { background: #f3e5f5; color: #6a1b9a; }
    .conf-high { color: #2e7d32; }
    .conf-medium { color: #f57f17; }
    .conf-low { color: #c62828; }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin: 12px 0; }
    .stat-box { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 16px; text-align: center; }
    .stat-num { font-size: 24px; font-weight: 700; }
    .stat-label { font-size: 11px; color: #666; margin-top: 4px; }
    .rationale { font-size: 11px; color: #666; margin-top: 2px; }
    .question-text { font-style: italic; color: #424242; }
    """)
    lines.append("</style></head><body>")

    lines.append("<h1>Path Labels — Router Oracle Annotations</h1>")
    lines.append(f"<p>{len(labels)} questions labeled with heuristic minimum sufficient route.</p>")

    # Overview
    lines.append("<div class='stat-grid'>")
    for stage in POLICY_STAGES:
        count = summary["route_counts"].get(stage, 0)
        lines.append(
            f"<div class='stat-box'>"
            f"<div class='stat-num route-{stage}'>{count}</div>"
            f"<div class='stat-label'>{stage}: {ROUTE_DESCRIPTIONS[stage]}</div>"
            f"</div>"
        )
    lines.append("</div>")

    # Cross-tab: structure × route
    lines.append("<h2>Structure × Predicted Route</h2>")
    structs = ["single_fact", "single_high_arity", "multi_edge_chain",
               "multi_branch", "similar_subgraph_disambiguation"]
    lines.append("<table><tr><th>Structure</th>")
    for s in POLICY_STAGES:
        lines.append(f"<th>{s}</th>")
    lines.append("<th>Total</th></tr>")
    for st in structs:
        row = summary["struct_route"].get(st, {})
        total = sum(row.values())
        lines.append(f"<tr><td><span class='badge badge-struct'>{st}</span></td>")
        for s in POLICY_STAGES:
            count = row.get(s, 0)
            cls = f"route-{s}" if count else ""
            lines.append(f"<td class='{cls}'>{count if count else ''}</td>")
        lines.append(f"<td><b>{total}</b></td></tr>")
    lines.append("</table>")

    # Confidence distribution
    lines.append("<h2>Confidence Distribution</h2>")
    lines.append("<table><tr><th>Route</th><th>High</th><th>Medium</th><th>Low</th></tr>")
    for s in POLICY_STAGES:
        conf = summary["route_confidence"].get(s, Counter())
        lines.append(
            f"<tr><td class='route-{s}'>{s}</td>"
            f"<td class='conf-high'>{conf.get('high', 0)}</td>"
            f"<td class='conf-medium'>{conf.get('medium', 0)}</td>"
            f"<td class='conf-low'>{conf.get('low', 0)}</td></tr>"
        )
    lines.append("</table>")

    # Full table
    lines.append("<h2>All Questions</h2>")
    lines.append("<table><tr><th>ID</th><th>Structure</th><th>Route</th><th>Conf</th>"
                 "<th>Question</th><th>AUs</th><th>Chunks</th><th>Verts</th><th>HEs</th>"
                 "<th>Max Arity</th><th>Rationale</th></tr>")
    for lbl in labels:
        q_short = lbl["question"][:100] + "..." if len(lbl["question"]) > 100 else lbl["question"]
        lines.append(
            f"<tr><td>{lbl['question_id']}</td>"
            f"<td><span class='badge badge-struct'>{lbl['intended_structure']}</span></td>"
            f"<td class='route-{lbl['heuristic_min_route']}'>{lbl['heuristic_min_route']}</td>"
            f"<td class='conf-{lbl['confidence']}'>{lbl['confidence']}</td>"
            f"<td class='question-text'>{_esc(q_short)}</td>"
            f"<td>{lbl['features']['n_answer_units']}</td>"
            f"<td>{lbl['features']['n_unique_chunks']}</td>"
            f"<td>{lbl['features']['n_vertices']}</td>"
            f"<td>{lbl['features']['n_hyperedges']}</td>"
            f"<td>{lbl['features']['max_he_arity']}</td>"
            f"<td class='rationale'>{_esc(lbl['rationale'])}</td></tr>"
        )
    lines.append("</table>")

    # Route descriptions
    lines.append("<h2>Route Descriptions</h2>")
    lines.append("<table><tr><th>Route</th><th>Description</th></tr>")
    for s in POLICY_STAGES:
        lines.append(f"<tr><td class='route-{s}'><b>{s}</b></td><td>{ROUTE_DESCRIPTIONS[s]}</td></tr>")
    lines.append("</table>")

    lines.append("</body></html>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Generate heuristic path labels for Router experiments")
    parser.add_argument("--data-name", default="neurology_chunk1000")
    args = parser.parse_args()

    base = REPO_ROOT / "caches" / args.data_name / "question_set_v2" / "pilot_v1"
    qdir = base / "question_generation"
    qfile = qdir / "questions_v2.jsonl"

    if not qfile.exists():
        print(f"ERROR: {qfile} not found", file=sys.stderr)
        return 1

    # Load questions
    items = []
    with open(qfile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # Generate labels
    labels = []
    for q in items:
        feat = extract_features(q)
        route, confidence, rationale = predict_route(q["intended_structure"], feat)
        labels.append({
            "question_id": q["question_id"],
            "intended_structure": q["intended_structure"],
            "question": q["question"],
            "heuristic_min_route": route,
            "confidence": confidence,
            "rationale": rationale,
            "features": feat,
        })

    # Summary
    route_counts = Counter(l["heuristic_min_route"] for l in labels)
    struct_route = defaultdict(Counter)
    route_confidence = defaultdict(Counter)
    for l in labels:
        struct_route[l["intended_structure"]][l["heuristic_min_route"]] += 1
        route_confidence[l["heuristic_min_route"]][l["confidence"]] += 1

    summary = {
        "total": len(labels),
        "route_counts": dict(route_counts),
        "struct_route": {k: dict(v) for k, v in struct_route.items()},
        "route_confidence": {k: dict(v) for k, v in route_confidence.items()},
    }

    # Write JSONL
    jsonl_path = qdir / "path_labels.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"Labels JSONL: {jsonl_path}")

    # Write HTML
    html_path = qdir / "path_labels_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(labels, summary))
    print(f"HTML report: {html_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"Path Labels Summary ({len(labels)} questions)")
    print(f"{'='*60}")
    print(f"\n  Route distribution:")
    for s in POLICY_STAGES:
        count = route_counts.get(s, 0)
        print(f"    {s}: {count}")
    print(f"\n  Structure × Route:")
    for st in ["single_fact", "single_high_arity", "multi_edge_chain",
               "multi_branch", "similar_subgraph_disambiguation"]:
        row = struct_route.get(st, {})
        parts = [f"{r}={row.get(r, 0)}" for r in POLICY_STAGES if row.get(r, 0)]
        print(f"    {st:40s}: {', '.join(parts)}")
    print(f"\n  Confidence:")
    for s in POLICY_STAGES:
        conf = route_confidence.get(s, Counter())
        print(f"    {s}: high={conf.get('high',0)}, medium={conf.get('medium',0)}, low={conf.get('low',0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
