# Step 2.2 evidence verification repair

## Goal

Repair reserve coverage, evidence-group qrels, multilingual graph grounding, adjudication, and resumability for the Pilot evidence verification pipeline.

## Requirements

- The Pilot evidence verification pipeline must process both primary and reserve candidates through prepare, draft, review, and finalize.
- Reserve promotion must only require reviewed reserve items for missing structure quotas; missing reserve data must be reported separately from rejected scientific evidence.
- Qrels must map each answer unit to its own evidence groups, spans, and source chunk IDs. A candidate-level "all chunks for all units" fallback is not acceptable.
- Graph grounding must not reject English source evidence solely because graph entity names are Chinese or absent as exact substrings.
- Human review items must have a stable adjudication file format and an apply step that can turn accepted revisions into verified outputs.
- Resume behavior must validate candidate scope and input metadata instead of trusting file existence alone.
- Changes must be testable offline without invoking Qwen or LongCat.

## Acceptance Criteria

- [ ] `prepare`, `draft`, and `review` accept a scope that can target primary, reserve, or both.
- [ ] `finalize --promote-reserve` can fill missing quotas from reviewed reserve items.
- [ ] Generated qrels preserve answer-unit-specific evidence groups and source spans.
- [ ] Graph grounding records warnings for weak lexical evidence but does not hard-reject cross-lingual candidates before LLM review.
- [ ] Human review queue records can be resolved by an adjudication input and applied deterministically.
- [ ] Existing tests pass, and new tests cover reserve flow, qrels precision, graph grounding behavior, and adjudication.

## Notes

- Keep `prd.md` focused on requirements, constraints, and acceptance criteria.
- Lightweight tasks can remain PRD-only.
- For complex tasks, add `design.md` for technical design and `implement.md` for execution planning before `task.py start`.
