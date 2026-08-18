from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


QUESTION_DIR = Path(
    "caches/neurology_chunk1000/question_set_v2/pilot_v1/question_generation"
)
QUESTION_FILE = QUESTION_DIR / "questions_v2.jsonl"
REVIEW_DIR = QUESTION_DIR / "review_raw"


def load_questions() -> list[dict]:
    with QUESTION_FILE.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_reviews() -> dict[str, dict]:
    reviews: dict[str, dict] = {}
    for path in sorted(REVIEW_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        reviews[payload["question_id"]] = payload
    return reviews


def ordered_questions(questions: list[dict], reviews: dict[str, dict]) -> list[dict]:
    def sort_key(question: dict) -> tuple[int, str]:
        verdict = reviews[question["question_id"]]["parsed"]["verdict"]
        priority = 0 if verdict == "needs_revision" else 1
        return priority, question["question_id"]

    return sorted(questions, key=sort_key)


def render_question_block(question: dict, review: dict) -> str:
    lines = [
        "=" * 100,
        f"question_id: {question['question_id']}",
        f"verified_structure: {question['verified_structure']}",
        f"longcat_verdict: {review['parsed']['verdict']}",
        f"longcat_issues: {json.dumps(review['parsed'].get('issues', []), ensure_ascii=False)}",
        f"longcat_revised_question: {review['parsed'].get('revised_question')}",
        "",
        "question:",
        question["question"],
        "",
        "gold_answer:",
        question["gold_answer"],
        "",
        "answer_units:",
    ]

    for unit in question["answer_units"]:
        lines.append(f"- {unit['unit_id']}: {unit['claim']}")

    lines.extend(["", "evidence_spans:"])
    span_map = {
        entry["unit_id"]: entry.get("evidence_spans", [])
        for entry in question["evidence_spans"]
    }
    for unit in question["answer_units"]:
        lines.append(f"- {unit['unit_id']}:")
        for span in span_map.get(unit["unit_id"], []):
            lines.append(f"  - chunk_id: {span['chunk_id']}")
            lines.append(f"    quote: {span['quote']}")

    return "\n".join(lines)


def print_manifest(questions: list[dict], reviews: dict[str, dict]) -> None:
    verdict_counter = Counter(
        reviews[question["question_id"]]["parsed"]["verdict"] for question in questions
    )
    structure_counter = Counter(question["verified_structure"] for question in questions)
    verdict_by_structure: dict[str, Counter] = defaultdict(Counter)
    for question in questions:
        verdict = reviews[question["question_id"]]["parsed"]["verdict"]
        verdict_by_structure[question["verified_structure"]][verdict] += 1

    print("Question count:", len(questions))
    print("Verdict distribution:", json.dumps(verdict_counter, ensure_ascii=False))
    print("Structure distribution:", json.dumps(structure_counter, ensure_ascii=False))
    print("Verdict by structure:")
    for structure in sorted(verdict_by_structure):
        print(
            f"- {structure}: "
            f"{json.dumps(verdict_by_structure[structure], ensure_ascii=False)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect pilot_v1 question records in manual-review order."
    )
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()

    questions = load_questions()
    reviews = load_reviews()
    ordered = ordered_questions(questions, reviews)

    if args.manifest:
        print_manifest(ordered, reviews)
        return

    start = (args.batch - 1) * args.size
    stop = start + args.size
    for question in ordered[start:stop]:
        print(render_question_block(question, reviews[question["question_id"]]))


if __name__ == "__main__":
    main()
