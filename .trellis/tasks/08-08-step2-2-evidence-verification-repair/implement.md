# Step 2.2 Evidence Verification Repair Implementation Plan

1. Inspect existing verification script and tests.
2. Add scope-aware candidate selection and manifest validation.
3. Repair qrels construction to use per-answer-unit evidence groups.
4. Replace graph lexical hard rejection with diagnostics and provenance-based blocking.
5. Add human adjudication apply support.
6. Extend tests for reserve, qrels, graph diagnostics, and adjudication.
7. Run targeted tests, then summarize commands for the user to run real LLM stages.
