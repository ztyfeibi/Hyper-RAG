from __future__ import annotations

import copy

import pytest

from scripts.apply_manual_question_review import apply_manual_reviews


QUOTA = {"single_fact": 2}


def _question(question_id: str, text: str) -> dict:
    return {
        "question_id": question_id,
        "question": text,
        "gold_answer": "Gold",
        "answer_units": [{"unit_id": "AU1", "claim": "Gold", "required": True}],
        "evidence_requirements": [],
        "evidence_spans": [],
        "required_subgraph": {},
        "intended_structure": "single_fact",
        "verified_structure": "single_fact",
    }


def _review(question_id: str, decision: str, edited_question=None) -> dict:
    return {
        "question_id": question_id,
        "verified_structure": "single_fact",
        "decision": decision,
        "issue_types": [],
        "comment": "reviewed",
        "edited_question": edited_question,
        "needs_replacement": decision == "reject",
    }


def test_manual_review_changes_only_question_text():
    questions = [_question("q1", "Original one?"), _question("q2", "Original two?")]
    original = copy.deepcopy(questions)
    reviews = [
        _review("q1", "minor_edit", "Revised one?"),
        _review("q2", "accept"),
    ]

    output, summary = apply_manual_reviews(questions, reviews, quota_target=QUOTA)

    assert output[0]["question"] == "Revised one?"
    assert {key: value for key, value in output[0].items() if key != "question"} == {
        key: value for key, value in original[0].items() if key != "question"
    }
    assert output[1] == original[1]
    assert summary["quota_met"] is True


def test_final_mode_rejects_incomplete_set():
    questions = [_question("q1", "Original one?"), _question("q2", "Original two?")]
    reviews = [_review("q1", "reject"), _review("q2", "accept")]

    with pytest.raises(ValueError, match="reject is not allowed"):
        apply_manual_reviews(questions, reviews, quota_target=QUOTA)


def test_allow_rejects_produces_pending_summary():
    questions = [_question("q1", "Original one?"), _question("q2", "Original two?")]
    reviews = [_review("q1", "reject"), _review("q2", "accept")]

    output, summary = apply_manual_reviews(
        questions, reviews, allow_rejects=True, quota_target=QUOTA
    )

    assert [row["question_id"] for row in output] == ["q2"]
    assert summary["quota_met"] is False


def test_manual_review_requires_exact_id_set():
    questions = [_question("q1", "Original one?"), _question("q2", "Original two?")]
    reviews = [_review("q1", "accept")]

    with pytest.raises(ValueError, match="ID mismatch"):
        apply_manual_reviews(questions, reviews, quota_target=QUOTA)
