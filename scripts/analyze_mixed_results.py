"""Analyze mixed evaluation results by stage.

Reads scoring files and meta to produce per-stage breakdowns.

Usage:
    python scripts/analyze_mixed_results.py --data-name neurology_chunk1000 --question-file mixed_stage
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

try:
    from reproduce.pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME

METRICS = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]


def parse_scores_from_response(response_text: str) -> dict[str, float]:
    """Extract 5 metric scores from a scoring response."""
    scores_raw = re.findall(r'"Score":\s*(?:"?(\d+\.?\d*)"?)', response_text)
    if len(scores_raw) < 5:
        return {}
    return {
        METRICS[i]: float(scores_raw[i])
        for i in range(5)
    }


def load_scoring(scoring_path: Path) -> list[dict[str, float]]:
    """Load scoring JSON and parse per-question scores."""
    with open(scoring_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    return [parse_scores_from_response(r) for r in responses]


def load_meta(meta_path: Path) -> list[dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_averages(scored: list[dict[str, float]]) -> dict[str, float]:
    """Compute average for each metric + overall."""
    if not scored:
        return {m: 0.0 for m in METRICS} | {"Overall": 0.0}

    result = {}
    for metric in METRICS:
        vals = [s.get(metric, 0) for s in scored if s]
        result[metric] = sum(vals) / len(vals) if vals else 0.0

    all_vals = [v for s in scored if s for v in s.values()]
    result["Overall"] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    return result


def print_table(title: str, avg: dict[str, float]):
    """Print a formatted table for one group."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for metric in METRICS + ["Overall"]:
        val = avg.get(metric, 0.0)
        print(f"  {metric:20s}: {val:6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mixed evaluation results by stage")
    parser.add_argument(
        "--data-name",
        type=str,
        default=DEFAULT_DATA_NAME,
        help=f"caches/<name>（默认 {DEFAULT_DATA_NAME!r}）",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="mixed_stage",
        help="问题文件前缀（默认 mixed_stage）",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["naive", "hyper", "adaptive"],
        help="要分析的 mode 列表",
    )
    args = parser.parse_args()

    working_dir = Path("caches") / args.data_name
    meta_path = working_dir / "questions" / f"{args.question_file}_meta.json"

    if not meta_path.exists():
        print(f"ERROR: Meta file not found: {meta_path}")
        sys.exit(1)

    metas = load_meta(meta_path)

    # Group question indices by stage
    stage_indices: dict[int, list[int]] = {}
    for i, m in enumerate(metas):
        stage = m.get("stage", m.get("original_stage", 2))
        if stage not in stage_indices:
            stage_indices[stage] = []
        stage_indices[stage].append(i)

    print(f"\nMixed set: {len(metas)} questions")
    for s in sorted(stage_indices):
        print(f"  stage {s}: {len(stage_indices[s])} questions")

    for mode in args.modes:
        scoring_path = working_dir / "evalation" / f"scoring_{args.question_file}_question_{mode}.json"
        if not scoring_path.exists():
            print(f"\nSKIP: {scoring_path} not found for mode={mode}")
            continue

        scored = load_scoring(scoring_path)

        if len(scored) != len(metas):
            print(f"\nWARNING: scoring count ({len(scored)}) != meta count ({len(metas)}) for mode={mode}")

        # Overall
        overall_avg = compute_averages(scored)
        print_table(f"{mode} — Overall ({len(scored)} questions)", overall_avg)

        # Per-stage
        for stage in sorted(stage_indices):
            indices = stage_indices[stage]
            stage_scored = [scored[i] for i in indices if i < len(scored)]
            stage_avg = compute_averages(stage_scored)
            complexity = metas[indices[0]].get("expected_complexity", "?")
            print_table(
                f"{mode} — Stage {stage} ({len(stage_scored)} questions, expected={complexity})",
                stage_avg,
            )

    # Cross-mode comparison table
    print(f"\n{'='*60}")
    print(f"  Cross-Mode Comparison (Overall by Stage)")
    print(f"{'='*60}")
    header = f"  {'Stage':<8} {'Expected':<12}"
    for mode in args.modes:
        header += f" {mode:>12}"
    print(header)
    print(f"  {'-'*50}")

    for stage in sorted(stage_indices):
        indices = stage_indices[stage]
        complexity = metas[indices[0]].get("expected_complexity", "?")
        row = f"  {stage:<8} {complexity:<12}"

        for mode in args.modes:
            scoring_path = working_dir / "evalation" / f"scoring_{args.question_file}_question_{mode}.json"
            if not scoring_path.exists():
                row += f" {'N/A':>12}"
                continue

            scored = load_scoring(scoring_path)
            stage_scored = [scored[i] for i in indices if i < len(scored)]
            avg = compute_averages(stage_scored)
            row += f" {avg.get('Overall', 0):>12.2f}"

        print(row)
