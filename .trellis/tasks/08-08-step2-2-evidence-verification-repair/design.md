# Step 2.2 Evidence Verification Repair Design

## Scope

This task repairs the offline orchestration and data contracts around `scripts/verify_pilot_evidence.py` and `hyperrag/evidence_verification.py`. It does not call external LLMs and does not attempt to produce the final 80 verified records in this implementation pass.

## Data Flow

Candidate pool -> mechanism inputs -> draft records -> review records -> finalize decisions -> verified outputs / human queue.

Reserve candidates use the same record contracts as primary candidates. Scope is represented explicitly in manifests so cached files cannot be reused for a different candidate set.

## Contract Changes

- Qrels are generated from each answer unit's evidence groups and spans.
- Graph grounding produces structured diagnostics. Lexical weakness is a warning unless provenance is missing entirely.
- Human review decisions are stored as JSONL records keyed by candidate ID and applied in a separate command.
- Manifest metadata includes script version, scope, candidate IDs, model/prompt metadata where available, and input hashes.

## Compatibility

Existing primary-only outputs remain readable, but finalize should classify missing reserve records as `reserve_unprocessed` instead of evidence rejection. New runs should use explicit scope arguments.

## Validation

Unit tests should build small synthetic candidate/draft/review records and avoid network access. CLI smoke tests should exercise finalize and adjudication against temporary fixtures.
