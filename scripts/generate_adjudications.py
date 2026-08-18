"""
Generate human_adjudications.jsonl by applying machine rules to human_review_queue candidates.

Rules (per user spec):
1. Only process gap structures: multi_branch(7), multi_edge_chain(7), single_fact(5)
2. Skip candidates with mechanical errors or contradicted verdicts
3. Skip candidates with chunk collision against already-verified
4. For each AU:
   - supported -> keep as-is
   - partially_supported -> narrow per revision_suggestion
   - unsupported -> fix if "saying too much" (remove qualifier); else delete AU
5. After fixes: remaining AUs >= min_needed (1 for single_fact, 2 for graph-first)
6. Rebuild evidence_groups, qrels, gold_answer for retained AUs only
7. graph-first samples cannot be reduced to single fact
"""
import json, re, collections, os, copy

STEP2_2 = "caches/neurology_chunk1000/question_set_v2/pilot_v1/step2_2"
GAPS = {"multi_branch": 15, "multi_edge_chain": 20, "single_fact": 20}
GRAPH_FIRST = {"single_high_arity", "multi_edge_chain", "multi_branch", "similar_subgraph_disambiguation"}
MECH_ERRORS = {
    "missing_bounds", "not_verbatim", "chunk_missing", "chunk_hash_mismatch",
    "text_mismatch", "gold_span_qwen_tokens > P4 cap", "evidence_cluster_collision",
    "raw_source_qwen_tokens > P4 cap",
}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def apply_revision(statement: str, revision: str, span_text: str, judgment: str) -> str | None:
    """Try to apply revision_suggestion to narrow a statement.

    Returns the narrowed statement, or None if the AU should be deleted.
    """
    if not revision:
        return None

    rev = revision.strip()

    # Pattern 1: "Remove 'X'" / "Remove the phrase 'X'" / "Remove 'X' to match..."
    m = re.search(r"[Rr]emove\s+(?:the\s+(?:phrase|word| qualifier|context)\s+)?['\u201c\"](.+?)['\u201d\"]", rev)
    if m:
        to_remove = m.group(1)
        new_stmt = statement.replace(to_remove, "").strip()
        # Clean up double spaces, leading commas, trailing spaces before period, etc.
        new_stmt = re.sub(r"\s+,", ",", new_stmt)
        new_stmt = re.sub(r",\s*,", ",", new_stmt)
        new_stmt = re.sub(r"^\s*,\s*", "", new_stmt)
        new_stmt = re.sub(r"\s+\.", ".", new_stmt)
        new_stmt = re.sub(r"\s{2,}", " ", new_stmt).strip()
        # Capitalize first letter if we removed a prefix
        if new_stmt and new_stmt[0].islower():
            new_stmt = new_stmt[0].upper() + new_stmt[1:]
        if new_stmt and len(new_stmt) > 10:
            return new_stmt

    # Pattern 2: "Change 'X' to 'Y'"
    m = re.search(r"[Cc]hange\s+['\u201c\"](.+?)['\u201d\"]\s+to\s+['\u201c\"](.+?)['\u201d\"]", rev)
    if m:
        old, new = m.group(1), m.group(2)
        new_stmt = statement.replace(old, new).strip()
        new_stmt = re.sub(r"\s+\.", ".", new_stmt)
        new_stmt = re.sub(r"\s{2,}", " ", new_stmt).strip()
        if new_stmt and new_stmt[0].islower():
            new_stmt = new_stmt[0].upper() + new_stmt[1:]
        if new_stmt and len(new_stmt) > 10:
            return new_stmt

    # Pattern 3: revision is a complete replacement sentence (ends with period, starts with capital)
    # Extract sentences that look like replacements
    sentences = re.findall(r"['\u201c\"]([^'\u201d\"]{20,})['\u201d\"]", rev)
    if sentences:
        for s in sentences:
            s = s.strip()
            # Check if it's a complete statement (not a fragment to remove)
            if s.endswith(".") and s[0].isupper() and "remove" not in s.lower() and "match" not in s.lower():
                return s

    # Pattern 4: revision itself is a complete sentence (no quotes)
    if rev.endswith(".") and rev[0].isupper() and len(rev) > 20:
        # Check it's not an instruction
        lower_rev = rev.lower()
        if not any(kw in lower_rev for kw in ["remove", "add", "change", "rephrase", "replace", "or add"]):
            return rev

    # Pattern 5: "Rephrase to match the source: 'X'" or "X to match the source"
    m = re.search(r"(?:match|align).*?:\s*['\u201c\"]?(.+?)['\u201d\"]?\s*(?:\.|$)", rev, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(".")
        if candidate and len(candidate) > 10:
            if candidate[0].islower():
                candidate = candidate[0].upper() + candidate[1:]
            return candidate + "."

    return None  # Cannot fix -> delete AU


def process_candidate(r: dict, min_needed: int) -> dict | None:
    """Process a single candidate. Returns adjudication dict if accepted, None if skipped."""
    cid = r["candidate_id"]
    verdicts = {v["answer_unit_id"]: v for v in (r.get("review_verdicts") or [])}
    aus = r.get("answer_units") or []
    egs = r.get("evidence_groups") or []
    qrels = r.get("qrels") or []
    spans = {sp["span_id"]: sp for sp in (r.get("spans") or [])}

    kept_aus = []
    deleted_au_ids = set()

    for au in aus:
        au_id = au["unit_id"]
        v = verdicts.get(au_id, {})
        judgment = v.get("judgment", "supported")

        if judgment == "supported":
            kept_aus.append(au)
        elif judgment == "partially_supported":
            # Try to narrow
            rev = v.get("revision_suggestion") or ""
            # Get span text for this AU
            eg_ids = au.get("evidence_group_ids", [])
            span_text = ""
            for eg_id in eg_ids:
                for eg in egs:
                    if eg.get("group_id") == eg_id:
                        for sid in eg.get("span_ids", []):
                            if sid in spans:
                                span_text = spans[sid].get("text", "")
                                break
                if span_text:
                    break

            new_stmt = apply_revision(au.get("statement", ""), rev, span_text, judgment)
            if new_stmt:
                new_au = copy.deepcopy(au)
                new_au["statement"] = new_stmt
                kept_aus.append(new_au)
            else:
                deleted_au_ids.add(au_id)
        elif judgment == "unsupported":
            # Try to fix if "saying too much"
            rev = v.get("revision_suggestion") or ""
            eg_ids = au.get("evidence_group_ids", [])
            span_text = ""
            for eg_id in eg_ids:
                for eg in egs:
                    if eg.get("group_id") == eg_id:
                        for sid in eg.get("span_ids", []):
                            if sid in spans:
                                span_text = spans[sid].get("text", "")
                                break
                if span_text:
                    break

            new_stmt = apply_revision(au.get("statement", ""), rev, span_text, judgment)
            if new_stmt:
                new_au = copy.deepcopy(au)
                new_au["statement"] = new_stmt
                kept_aus.append(new_au)
            else:
                deleted_au_ids.add(au_id)
        else:
            # Unknown judgment, delete
            deleted_au_ids.add(au_id)

    # Check minimum AU count
    if len(kept_aus) < min_needed:
        return None

    # For graph-first, cannot reduce to single fact
    struct = r.get("intended_structure", "")
    if struct in GRAPH_FIRST and len(kept_aus) < 2:
        return None

    # Build updated structures
    kept_au_ids = {au["unit_id"] for au in kept_aus}
    kept_eg_ids = set()
    for au in kept_aus:
        kept_eg_ids.update(au.get("evidence_group_ids", []))

    new_egs = [eg for eg in egs if eg.get("group_id") in kept_eg_ids]
    new_qrels = [q for q in qrels if q.get("answer_unit_id") in kept_au_ids]
    # Keep all spans (they're the evidence source, even if some AUs were deleted)
    new_spans = r.get("spans") or []

    # Rebuild gold_answer
    parts = []
    for au in kept_aus:
        parts.append(f"{au['statement']} [{au['unit_id']}]")
    new_gold = " ".join(parts)

    # Determine if we need updates (if we deleted/narrowed any AU)
    needs_updates = len(deleted_au_ids) > 0 or any(
        au["statement"] != orig["statement"]
        for au, orig in zip(kept_aus, aus)
        if au["unit_id"] == orig["unit_id"]
    )

    adjudication = {
        "candidate_id": cid,
        "decision": "accept",
        "reviewer": "manual-rule-v1",
        "note": f"Kept {len(kept_aus)}/{len(aus)} answer units; narrowed or removed unsupported/partial claims to match cited spans.",
    }

    if needs_updates:
        adjudication["updates"] = {
            "answer_units": kept_aus,
            "evidence_groups": new_egs,
            "qrels": new_qrels,
            "gold_answer": new_gold,
        }

    return adjudication


def main():
    rows = load_jsonl(os.path.join(STEP2_2, "human_review_queue.jsonl"))
    verified = load_jsonl(os.path.join(STEP2_2, "verified_evidence.jsonl"))
    used_chunks = set()
    for v in verified:
        used_chunks.update(v.get("source_chunk_ids", []))

    # Track quota as we go (adjudication order matters)
    quota_filled = collections.Counter(v["intended_structure"] for v in verified)
    adjudications = []
    skipped = []

    for struct, need in GAPS.items():
        remaining = need - quota_filled[struct]
        if remaining <= 0:
            print(f"{struct}: already full ({quota_filled[struct]}/{GAPS[struct]}), skipping")
            continue

        min_needed = 2 if struct in GRAPH_FIRST else 1
        cands = [r for r in rows if r.get("intended_structure") == struct]

        # Filter
        viable = []
        for r in cands:
            cid = r["candidate_id"]
            rr = set(r.get("rejection_reasons") or [])
            if rr & MECH_ERRORS:
                skipped.append((cid, "mech_error")); continue
            if any(sp.get("char_start") is None for sp in (r.get("spans") or [])):
                skipped.append((cid, "missing_bounds")); continue
            vs = r.get("review_verdicts") or []
            if any(v.get("judgment") == "contradicted" for v in vs):
                skipped.append((cid, "contradicted")); continue
            if used_chunks & set(r.get("source_chunk_ids", [])):
                skipped.append((cid, "chunk_collision")); continue
            sup = sum(1 for v in vs if v.get("judgment") == "supported")
            if sup < min_needed:
                skipped.append((cid, f"insufficient_supported({sup})")); continue
            viable.append((r, sup))

        # Sort: most supported first, then fewest total AUs (simpler fixes)
        viable.sort(key=lambda x: (-x[1], len(x[0].get("review_verdicts", []))))

        accepted_for_struct = 0
        for r, sup_count in viable:
            if accepted_for_struct >= remaining:
                break
            # Check chunk collision again (in case previous accept used same chunks)
            if used_chunks & set(r.get("source_chunk_ids", [])):
                skipped.append((r["candidate_id"], "chunk_collision_after_accept")); continue

            adj = process_candidate(r, min_needed)
            if adj:
                adjudications.append(adj)
                used_chunks.update(r.get("source_chunk_ids", []))
                quota_filled[struct] += 1
                accepted_for_struct += 1
                n_kept = len(adj.get("updates", {}).get("answer_units", r.get("answer_units", [])))
                has_updates = "updates" in adj
                print(f"  ACCEPT {r['candidate_id']}: kept {n_kept} AUs, updates={has_updates}")
            else:
                skipped.append((r["candidate_id"], "process_failed"))

        print(f"{struct}: needed {remaining}, accepted {accepted_for_struct}, "
              f"total now {quota_filled[struct]}/{GAPS[struct]}")

    # Write adjudications
    out_path = os.path.join(STEP2_2, "human_adjudications.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for adj in adjudications:
            f.write(json.dumps(adj, ensure_ascii=False) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"Total adjudications: {len(adjudications)}")
    print(f"Skipped: {len(skipped)}")
    for struct in GAPS:
        print(f"  {struct}: {quota_filled[struct]}/{GAPS[struct]}")
    print(f"quota_met: {all(quota_filled[s] >= GAPS[s] for s in GAPS)}")
    print(f"\nSkipped reasons:")
    skip_reasons = collections.Counter(r[1] for r in skipped)
    for r, n in skip_reasons.most_common():
        print(f"  {r}: {n}")


if __name__ == "__main__":
    main()
