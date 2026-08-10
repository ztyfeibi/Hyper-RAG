#!/usr/bin/env python3
"""Build a 12-question diagnostic subset for reproducibility testing.

Samples 4 questions per stage (simple/medium/complex) from the mixed_stage
question set using a fixed random seed (42), so the subset is deterministic.

Usage:
    python scripts/build_repro_subset.py [--data-name neurology_chunk1000]
                                         [--source mixed_stage]
                                         [--output repro_12]
                                         [--per-stage 4] [--seed 42]

Output:
    caches/<data_name>/questions/repro_<N>.json       (question list)
    caches/<data_name>/questions/repro_<N>_meta.json  (metadata with stages)
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from scripts.pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    try:
        from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
    except ImportError:
        DEFAULT_DATA_NAME = "neurology_chunk1000"


def main():
    parser = argparse.ArgumentParser(
        description="Build a 12-question diagnostic subset for repro testing"
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=DEFAULT_DATA_NAME,
        help=f"Working directory name (default: {DEFAULT_DATA_NAME})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source question file prefix (e.g., mixed_stage). "
             "Overrides --source-prefix if both given.",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="mixed_stage",
        help="Source question file prefix (default: mixed_stage). "
             "Ignored if --source is given.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file prefix (default: repro_<N>, e.g. repro_12). "
             "If not given, auto-generated from per-stage count.",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=4,
        help="Questions per stage (default: 4, total = 3 * per_stage = 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    # --source overrides --source-prefix
    source_prefix = args.source if args.source else args.source_prefix

    working_dir = Path("caches") / args.data_name
    q_file = working_dir / f"questions/{source_prefix}.json"
    meta_file = working_dir / f"questions/{source_prefix}_meta.json"

    if not q_file.exists():
        print(f"ERROR: Question file not found: {q_file}")
        sys.exit(1)

    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    meta_list = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
    else:
        print(f"WARNING: Meta file not found: {meta_file}")

    # Group question indices by stage
    stage_indices = {}
    for i, m in enumerate(meta_list):
        stage = m.get("stage", m.get("original_stage", 2))
        stage_indices.setdefault(stage, []).append(i)

    # Sample per-stage deterministically
    rng = random.Random(args.seed)
    selected_indices = []
    selected_meta = []

    for stage in sorted(stage_indices.keys()):
        pool = stage_indices[stage]
        n = min(args.per_stage, len(pool))
        sampled = rng.sample(pool, n)
        selected_indices.extend(sampled)
        for idx in sampled:
            m = dict(meta_list[idx])
            m["repro_original_index"] = idx
            selected_meta.append(m)
        print(
            f"Stage {stage}: sampled {n} from {len(pool)} questions "
            f"(indices: {sampled})"
        )

    # Build question subset
    subset_questions = [questions[i] for i in selected_indices]

    total = len(subset_questions)
    out_prefix = args.output if args.output else f"repro_{total}"
    out_q = working_dir / f"questions/{out_prefix}.json"
    out_meta = working_dir / f"questions/{out_prefix}_meta.json"

    with open(out_q, "w", encoding="utf-8") as f:
        json.dump(subset_questions, f, ensure_ascii=False, indent=2)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(selected_meta, f, ensure_ascii=False, indent=2)

    print(f"\nSubset: {total} questions ({args.per_stage} per stage, seed={args.seed})")
    print(f"Questions: {out_q}")
    print(f"Meta:      {out_meta}")
    print(f"\nUsage with Step_3:")
    print(
        f"  python reproduce/Step_3_response_question.py "
        f"--mode adaptive --question-file {out_prefix} "
        f"--router-policy fixed --forced-complexity medium --save-trace"
    )


if __name__ == "__main__":
    main()
