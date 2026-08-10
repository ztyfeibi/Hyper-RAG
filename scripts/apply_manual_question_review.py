# -*- coding: utf-8 -*-
"""Apply manual question decisions without mutating evidence or Gold fields."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


QUOTA_TARGET = {
    "single_fact": 20,
    "single_high_arity": 15,
    "multi_edge_chain": 20,
    "multi_branch": 15,
    "similar_subgraph_disambiguation": 10,
}
VALID_DECISIONS = frozenset({"accept", "minor_edit", "reject"})


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _index_unique(rows: Sequence[dict], key: str, label: str) -> Dict[str, dict]:
    indexed = {}
    for row in rows:
        value = row.get(key)
        if not value:
            raise ValueError(f"{label} row missing {key}")
        if value in indexed:
            raise ValueError(f"duplicate {key} in {label}: {value}")
        indexed[value] = row
    return indexed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_manual_reviews(
    questions: Sequence[dict],
    reviews: Sequence[dict],
    *,
    allow_rejects: bool = False,
    quota_target: Dict[str, int] | None = None,
) -> Tuple[List[dict], dict]:
    quota_target = dict(quota_target or QUOTA_TARGET)
    question_by_id = _index_unique(questions, "question_id", "questions")
    review_by_id = _index_unique(reviews, "question_id", "manual reviews")
    if set(question_by_id) != set(review_by_id):
        missing = sorted(set(question_by_id) - set(review_by_id))
        extra = sorted(set(review_by_id) - set(question_by_id))
        raise ValueError(f"manual review ID mismatch: missing={missing}, extra={extra}")

    output = []
    decision_counts = collections.Counter()
    for question in questions:
        question_id = question["question_id"]
        review = review_by_id[question_id]
        decision = review.get("decision")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"{question_id}: invalid decision {decision!r}")
        if review.get("verified_structure") != question.get("verified_structure"):
            raise ValueError(f"{question_id}: verified_structure mismatch")
        if bool(review.get("needs_replacement")) != (decision == "reject"):
            raise ValueError(f"{question_id}: needs_replacement is inconsistent")

        edited_question = review.get("edited_question")
        if decision == "accept":
            if edited_question not in (None, ""):
                raise ValueError(f"{question_id}: accept must not contain edited_question")
            output.append(dict(question))
        elif decision == "minor_edit":
            if not isinstance(edited_question, str) or not edited_question.strip():
                raise ValueError(f"{question_id}: minor_edit requires edited_question")
            if edited_question.strip() == question.get("question", "").strip():
                raise ValueError(f"{question_id}: minor_edit did not change the question")
            updated = dict(question)
            updated["question"] = edited_question.strip()
            output.append(updated)
        elif not allow_rejects:
            raise ValueError(f"{question_id}: reject is not allowed in final mode")
        decision_counts[decision] += 1

    normalized_questions = [row["question"].strip().casefold() for row in output]
    if len(normalized_questions) != len(set(normalized_questions)):
        raise ValueError("duplicate question text after manual review")

    structure_counts = collections.Counter(
        row["verified_structure"] for row in output
    )
    quota_met = (
        len(output) == sum(quota_target.values())
        and all(structure_counts.get(key, 0) == value for key, value in quota_target.items())
    )
    if not allow_rejects and not quota_met:
        raise ValueError(
            f"final question quota mismatch: count={len(output)}, "
            f"structures={dict(structure_counts)}"
        )

    summary = {
        "question_count": len(output),
        "decision_distribution": dict(decision_counts),
        "structure_distribution": dict(structure_counts),
        "quota_target": quota_target,
        "quota_met": quota_met,
    }
    return output, summary


def _write_jsonl_atomic(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply manual question reviews and write a hashed manifest."
    )
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--allow-rejects", action="store_true")
    args = parser.parse_args()

    questions = _load_jsonl(args.question_file)
    reviews = _load_jsonl(args.review_file)
    output, summary = apply_manual_reviews(
        questions, reviews, allow_rejects=args.allow_rejects
    )
    _write_jsonl_atomic(args.output_file, output)

    manifest_path = args.manifest_file or args.output_file.with_suffix(".manifest.json")
    manifest = collections.OrderedDict()
    manifest["manual_review_version"] = "manual-question-review-v1"
    manifest["source_question_file"] = str(args.question_file)
    manifest["manual_review_file"] = str(args.review_file)
    manifest["output_question_file"] = str(args.output_file)
    manifest.update(summary)
    manifest["file_hashes"] = {
        "source_question_sha256": _sha256_file(args.question_file),
        "manual_review_sha256": _sha256_file(args.review_file),
        "output_question_sha256": _sha256_file(args.output_file),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[manual-review] questions={summary['question_count']} "
        f"quota_met={summary['quota_met']} output={args.output_file}"
    )


if __name__ == "__main__":
    main()
