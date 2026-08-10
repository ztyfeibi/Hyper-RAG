#!/usr/bin/env python
"""Human review workflow for Step 2.3 generated questions.

Two phases:

  build   Generate ``human_review_worksheet.jsonl`` from the 19 needs_revision
          review records. Each row carries the context you need to decide:
          failed checks, LongCat's issues, and LongCat's revised_question.
          You only edit the ``decision`` / ``new_question`` / ``note`` fields.

  apply   Read the worksheet, apply your decisions onto questions_v2.jsonl
          (with an automatic timestamped backup), then re-run the contract
          validator on the resulting file.

Worksheet row format (one JSON object per line)::

    {
      "question_id": "qv2-0045",
      "intended_structure": "multi_edge_chain",
      "current_question": "<text currently in questions_v2.jsonl>",
      "failed_checks": ["no_answer_leak"],
      "issues": ["LongCat critique ..."],
      "longcat_revised_question": "<LongCat suggested text>",
      "decision": "accept",        # accept | edit | reject
      "new_question": "",          # fill ONLY when decision == edit
      "note": ""
    }

decision semantics:
  accept : keep current_question as-is (already LongCat-revised in the set)
  edit   : replace with new_question (must be non-empty)
  reject : drop the question from the set (quota will drop below 80; you then
           need to draft a replacement separately)

Items NOT in the worksheet (the 61 auto-pass questions) keep their text and
are marked human_review_status = auto_pass.

Usage:
  python scripts/manage_human_review.py build  --data-name neurology_chunk1000
  python scripts/manage_human_review.py apply  --data-name neurology_chunk1000
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

GEN_DIR_DEFAULT = "caches/{data_name}/question_set_v2/pilot_v1/question_generation"
VALID_DECISIONS = {"accept", "edit", "reject"}

REQUIRED_CHECK_FIELDS = [
    "is_english", "is_open_ended", "answerable_from_evidence",
    "no_answer_leak", "no_external_knowledge", "not_overly_broad",
    "not_choice_like",
]


def _gen_dir(data_name: str) -> str:
    return GEN_DIR_DEFAULT.format(data_name=data_name)


def _load_review_records(gen_dir: str):
    records = {}
    for p in glob.glob(os.path.join(gen_dir, "review_raw", "*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        qid = d.get("question_id")
        if qid:
            records[qid] = d
    return records


def cmd_build(args):
    gen_dir = _gen_dir(args.data_name)
    review_records = _load_review_records(gen_dir)

    # Load current questions (already LongCat-revised for needs_revision)
    qpath = os.path.join(gen_dir, "questions_v2.jsonl")
    questions = {}
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions[d["question_id"]] = d

    rows = []
    for qid, rec in review_records.items():
        parsed = rec.get("parsed", {})
        if parsed.get("verdict") != "needs_revision":
            continue
        checks = parsed.get("checks", {}) or {}
        failed = [k for k in REQUIRED_CHECK_FIELDS if checks.get(k) is False]
        current = questions.get(qid, {}).get("question", "")
        rows.append({
            "question_id": qid,
            "intended_structure": questions.get(qid, {}).get("intended_structure", ""),
            "current_question": current,
            "failed_checks": failed,
            "issues": parsed.get("issues", []) or [],
            "longcat_revised_question": parsed.get("revised_question") or "",
            "decision": "accept",
            "new_question": "",
            "note": "",
        })

    rows.sort(key=lambda r: r["question_id"])
    out_path = os.path.join(gen_dir, "human_review_worksheet.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} needs_revision rows to:")
    print(f"  {out_path}")
    print("Edit the 'decision' / 'new_question' / 'note' fields, then run 'apply'.")
    print("Default decision is 'accept' (keep current text).")
    return 0


def cmd_apply(args):
    gen_dir = _gen_dir(args.data_name)
    wspath = os.path.join(gen_dir, "human_review_worksheet.jsonl")
    qpath = os.path.join(gen_dir, "questions_v2.jsonl")
    if not os.path.exists(wspath):
        print(f"ERROR: worksheet not found: {wspath}\nRun 'build' first.", file=sys.stderr)
        return 2

    # Load worksheet decisions
    decisions = {}
    with open(wspath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            decisions[r["question_id"]] = r

    # Load current questions preserving order
    order = []
    questions = {}
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            order.append(d["question_id"])
            questions[d["question_id"]] = d

    edited, rejected, accepted, bad = [], [], [], []
    new_order = []
    for qid in order:
        d = questions[qid]
        if qid in decisions:
            dec = decisions[qid]
            action = (dec.get("decision") or "accept").strip().lower()
            if action not in VALID_DECISIONS:
                bad.append(f"{qid}: invalid decision '{action}' (use accept/edit/reject)")
                d["human_review_status"] = "auto_pass"
                new_order.append(qid)
                continue
            if action == "accept":
                d["human_review_status"] = "approved"
                accepted.append(qid)
                new_order.append(qid)
            elif action == "edit":
                new_q = (dec.get("new_question") or "").strip()
                if not new_q:
                    bad.append(f"{qid}: decision=edit but new_question empty")
                    d["human_review_status"] = "auto_pass"
                    new_order.append(qid)
                    continue
                d["question"] = new_q
                d["human_review_status"] = "edited"
                d["human_review_note"] = dec.get("note", "")
                edited.append(qid)
                new_order.append(qid)
            elif action == "reject":
                d["human_review_status"] = "rejected"
                rejected.append(qid)
                # dropped from set
        else:
            d["human_review_status"] = "auto_pass"
            new_order.append(qid)

    # Backup original
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = qpath + f".bak.{ts}"
    with open(qpath, encoding="utf-8") as f:
        original_text = f.read()
    with open(bak, "w", encoding="utf-8") as f:
        f.write(original_text)

    # Write updated (preserve order, skip rejected)
    with open(qpath, "w", encoding="utf-8") as f:
        for qid in new_order:
            if qid in rejected:
                continue
            f.write(json.dumps(questions[qid], ensure_ascii=False) + "\n")

    remaining = len(new_order) - len(rejected)

    # Re-validate
    validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "validate_question_set_contract.py")
    py = sys.executable
    result = subprocess.run(
        [py, validator, "--question-file", qpath],
        capture_output=True, text=True,
    )
    validation_ok = result.returncode == 0
    validation_tail = (result.stdout + result.stderr).strip().splitlines()[-15:]

    # Report
    report = {
        "timestamp": ts,
        "accepted": accepted,
        "edited": edited,
        "rejected": rejected,
        "invalid_or_skipped": bad,
        "remaining_questions": remaining,
        "validation_passed": validation_ok,
        "backup": bak,
    }
    report_path = os.path.join(gen_dir, "human_review_applied.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Backup: {bak}")
    print(f"Accepted (kept): {len(accepted)}")
    print(f"Edited        : {len(edited)} -> {edited}")
    print(f"Rejected      : {len(rejected)} -> {rejected}")
    if bad:
        print(f"Invalid/skipped: {bad}")
    print(f"Remaining questions: {remaining} (target 80)")
    if remaining != 80:
        print("WARNING: count != 80; quota NOT met. Draft replacements before "
              "running experiments.")
    print(f"Contract validation: {'PASS' if validation_ok else 'FAIL'}")
    if not validation_ok:
        print("--- validator output (tail) ---")
        print("\n".join(validation_tail))
    print(f"Report: {report_path}")
    return 0 if (validation_ok and not bad) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("build", "apply"):
        sp = sub.add_parser(name)
        sp.add_argument("--data-name", default="neurology_chunk1000")
        sp.set_defaults(func=globals()[f"cmd_{name}"])
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
