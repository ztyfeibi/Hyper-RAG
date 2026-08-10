"""Evaluate Router accuracy on mixed question set.

Runs the Query Router on each question in the mixed set and compares
route.complexity against expected_complexity from the meta file.

Outputs:
- Router accuracy (exact match)
- Confusion matrix (expected vs predicted complexity)
- Complex over-classification rate

Usage:
    python scripts/evaluate_router_on_mixed.py --data-name neurology_chunk1000 --question-file mixed_stage
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

try:
    from reproduce.pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME

from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag.query_router import route_query


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


COMPLEXITIES = ["simple", "medium", "complex"]


def print_confusion_matrix(expected: list[str], predicted: list[str]):
    """Print a confusion matrix for complexity classification."""
    print(f"\n{'='*50}")
    print("  Confusion Matrix (rows=expected, cols=predicted)")
    print(f"{'='*50}")

    # Header
    header = f"  {'':>12s}"
    for c in COMPLEXITIES:
        header += f" {c:>10s}"
    print(header)

    # Rows
    for exp in COMPLEXITIES:
        row = f"  {exp:>12s}"
        for pred in COMPLEXITIES:
            count = sum(1 for e, p in zip(expected, predicted) if e == exp and p == pred)
            row += f" {count:>10d}"
        print(row)


def compute_metrics(expected: list[str], predicted: list[str]) -> dict:
    """Compute accuracy, per-class precision/recall, over-classification rate."""
    total = len(expected)
    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    accuracy = correct / total if total > 0 else 0.0

    # Over-classification: expected < predicted (e.g., simple->medium, medium->complex)
    over_classified = sum(
        1 for e, p in zip(expected, predicted)
        if COMPLEXITIES.index(e) < COMPLEXITIES.index(p)
    )
    over_rate = over_classified / total if total > 0 else 0.0

    # Under-classification: expected > predicted
    under_classified = sum(
        1 for e, p in zip(expected, predicted)
        if COMPLEXITIES.index(e) > COMPLEXITIES.index(p)
    )
    under_rate = under_classified / total if total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for cls in COMPLEXITIES:
        tp = sum(1 for e, p in zip(expected, predicted) if e == cls and p == cls)
        fp = sum(1 for e, p in zip(expected, predicted) if e != cls and p == cls)
        fn = sum(1 for e, p in zip(expected, predicted) if e == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "count": sum(1 for e in expected if e == cls),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "over_classification_count": over_classified,
        "over_classification_rate": over_rate,
        "under_classification_count": under_classified,
        "under_classification_rate": under_rate,
        "per_class": per_class,
    }


async def run_evaluation(data_name: str, question_file: str):
    working_dir = Path("caches") / data_name
    question_path = working_dir / "questions" / f"{question_file}.json"
    meta_path = working_dir / "questions" / f"{question_file}_meta.json"

    if not question_path.exists():
        print(f"ERROR: Questions file not found: {question_path}")
        return
    if not meta_path.exists():
        print(f"ERROR: Meta file not found: {meta_path}")
        return

    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    print(f"Loaded {len(questions)} questions from {question_file}")
    print(f"Question distribution by expected complexity:")
    for m in metas:
        pass  # Just counting below

    expected_complexities = [m.get("expected_complexity", "medium") for m in metas]
    for c in COMPLEXITIES:
        count = sum(1 for e in expected_complexities if e == c)
        print(f"  {c}: {count}")

    # Initialize HyperRAG for router (need global_config with llm_model_func + hashing_kv)
    rag = HyperRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        llm_model_max_async=4,
        embedding_func_max_async=4,
    )

    # Build global_config like HyperRAG does internally (asdict(self))
    global_config = asdict(rag)
    query_param = QueryParam(mode="adaptive")

    # Run router on each question
    print(f"\nRunning router on {len(questions)} questions...")
    results = []
    for i, (q, exp) in enumerate(zip(questions, expected_complexities)):
        route = await route_query(q, query_param, global_config)
        results.append({
            "question_id": i,
            "question": q[:100],
            "expected_complexity": exp,
            "predicted_complexity": route.complexity,
            "predicted_type": route.query_type,
            "predicted_focus": route.focus_types,
            "correct": exp == route.complexity,
        })
        status = "OK" if exp == route.complexity else "MISS"
        print(f"  [{i+1}/{len(questions)}] {status} expected={exp:>8s} predicted={route.complexity:>8s} type={route.query_type}")

    # Compute metrics
    predicted = [r["predicted_complexity"] for r in results]
    metrics = compute_metrics(expected_complexities, predicted)

    print(f"\n{'='*50}")
    print(f"  Router Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Total questions:    {metrics['total']}")
    print(f"  Correct:            {metrics['correct']}")
    print(f"  Accuracy:           {metrics['accuracy']:.1%}")
    print(f"  Over-classified:    {metrics['over_classification_count']} ({metrics['over_classification_rate']:.1%})")
    print(f"  Under-classified:   {metrics['under_classification_count']} ({metrics['under_classification_rate']:.1%})")

    print(f"\n  Per-class metrics:")
    for cls in COMPLEXITIES:
        pc = metrics["per_class"][cls]
        print(f"    {cls:>8s}: count={pc['count']:3d}  P={pc['precision']:.2f}  R={pc['recall']:.2f}  F1={pc['f1']:.2f}")

    print_confusion_matrix(expected_complexities, predicted)

    # Also show query type distribution
    type_counts = {}
    for r in results:
        t = r["predicted_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n  Query type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:>15s}: {c}")

    # Save detailed results
    output_path = working_dir / "evalation" / f"router_eval_{question_file}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Router on mixed question set")
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
    args = parser.parse_args()

    loop = always_get_an_event_loop()
    loop.run_until_complete(run_evaluation(args.data_name, args.question_file))
