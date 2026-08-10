"""Build a mixed evaluation set from multiple stage question files.

Usage:
    python scripts/build_mixed_questions.py --data-name neurology_chunk1000 --stages 1 2 3

Reads questions/<stage>_stage.json, <stage>_stage_ref.json, <stage>_stage_meta.json
for each specified stage, merges them into a single mixed_stage set with
renumbered question IDs and preserved meta info.
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

try:
    from reproduce.pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME


def merge_stage(data_name: str, stages: list[int], allow_missing: bool = False) -> dict:
    """Merge multiple stage question/ref/meta files into one mixed set."""
    working_dir = Path("caches") / data_name
    questions_dir = working_dir / "questions"

    all_questions = []
    all_refs = []
    all_meta = []
    global_id = 0

    for stage in stages:
        prefix = f"{stage}_stage"
        q_file = questions_dir / f"{prefix}.json"
        ref_file = questions_dir / f"{prefix}_ref.json"
        meta_file = questions_dir / f"{prefix}_meta.json"

        if not q_file.exists():
            if not allow_missing:
                raise FileNotFoundError(f"Stage {stage} questions not found: {q_file}")
            print(f"WARNING: {q_file} not found, skipping stage {stage}")
            continue
        if not ref_file.exists():
            if not allow_missing:
                raise FileNotFoundError(f"Stage {stage} refs not found: {ref_file}")
            print(f"WARNING: {ref_file} not found, skipping stage {stage}")
            continue

        with open(q_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        with open(ref_file, "r", encoding="utf-8") as f:
            refs = json.load(f)

        # Load meta if it exists, otherwise build minimal meta
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metas = json.load(f)
        else:
            from reproduce.Step_2_extract_question import STAGE_COMPLEXITY_MAP, STAGE_STRATEGY_MAP
            metas = [
                {
                    "question_id": i,
                    "stage": stage,
                    "expected_complexity": STAGE_COMPLEXITY_MAP.get(stage, "medium"),
                    "expected_strategy": STAGE_STRATEGY_MAP.get(stage, "unknown"),
                    "ref_index": i,
                }
                for i in range(len(questions))
            ]

        assert len(questions) == len(refs), (
            f"Stage {stage}: questions ({len(questions)}) != refs ({len(refs)})"
        )
        if metas and len(metas) != len(questions):
            print(f"WARNING: Stage {stage} meta count ({len(metas)}) != questions ({len(questions)})")

        for i, (q, r) in enumerate(zip(questions, refs)):
            meta = metas[i] if i < len(metas) else {}
            all_questions.append(q)
            all_refs.append(r)
            all_meta.append({
                "question_id": global_id,
                "original_stage": stage,
                "original_index": i,
                "stage": stage,
                "expected_complexity": meta.get("expected_complexity", "medium"),
                "expected_strategy": meta.get("expected_strategy", "unknown"),
                "ref_index": global_id,
                "source_context_index": meta.get("source_context_index"),
            })
            global_id += 1

    return {
        "questions": all_questions,
        "refs": all_refs,
        "meta": all_meta,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple stage question sets into mixed_stage")
    parser.add_argument(
        "--data-name",
        type=str,
        default=DEFAULT_DATA_NAME,
        help=f"caches/<name>/questions（默认 {DEFAULT_DATA_NAME!r}）",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="要合并的 stage 列表（默认 1 2 3）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mixed_stage",
        help="输出文件前缀（默认 mixed_stage）",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=False,
        help="允许某些 stage 文件缺失时跳过（默认缺失则报错）",
    )
    args = parser.parse_args()

    merged = merge_stage(args.data_name, args.stages, allow_missing=args.allow_missing)

    working_dir = Path("caches") / args.data_name
    questions_dir = working_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix

    with open(questions_dir / f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(merged["questions"], f, ensure_ascii=False, indent=4)
    with open(questions_dir / f"{prefix}_ref.json", "w", encoding="utf-8") as f:
        json.dump(merged["refs"], f, ensure_ascii=False, indent=4)
    with open(questions_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(merged["meta"], f, ensure_ascii=False, indent=4)

    # Print summary
    stage_counts = {}
    for m in merged["meta"]:
        s = m["stage"]
        stage_counts[s] = stage_counts.get(s, 0) + 1

    print(f"Mixed set written to {questions_dir / f'{prefix}.json'}")
    print(f"References written to {questions_dir / f'{prefix}_ref.json'}")
    print(f"Meta written to {questions_dir / f'{prefix}_meta.json'}")
    print(f"Total questions: {len(merged['questions'])}")
    for s in sorted(stage_counts):
        # Find expected_complexity from the first meta entry of this stage
        stage_meta = next((m for m in merged["meta"] if m["stage"] == s), {})
        exp = stage_meta.get("expected_complexity", "?")
        print(f"  stage {s}: {stage_counts[s]} questions (expected_complexity={exp})")
