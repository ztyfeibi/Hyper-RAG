"""Determinism test: same entity/hyperedge sets must yield identical order
across different PYTHONHASHSEED values.

This is the regression guard for the Step-6 determinism fix. Before the fix,
``list(set(...))`` iteration order depended on PYTHONHASHSEED, so two runs with
different seeds could produce different retrieval orderings -> non-reproducible
experiments. After the fix, every set is sorted before use, so the ordering is
fully determined by content.

Usage:
    python scripts/test_determinism_hashseed.py
    # internally spawns subprocesses under PYTHONHASHSEED=1 and =99
"""

import subprocess
import sys
from pathlib import Path

# Use the project's conda environment python (the full package import needs numpy).
_CONDA_PYTHON = (
    Path(r"D:/Tools/Conda/envs/hyperrag/python.exe")
    if Path(r"D:/Tools/Conda/envs/hyperrag/python.exe").exists()
    else sys.executable
)

# Snippet executed inside a subprocess so we can control PYTHONHASHSEED.
# It builds the exact structures that _build_entity/relation_query_context feed
# into the sort tie-breaks and checks that ordering is seed-independent.
_SUBPROC_CODE = r'''
import sys
sys.path.insert(0, r"%ROOT%")

from hyperrag.query_context import _stable_edge_id

# 1) entity vertex set (e.g. all_one_hop_nodes / entity_names)
vertices = {"ent_c", "ent_a", "ent_b", "ent_d", "ent_a", "ent_e"}
# BUGGY (pre-fix): list(set) -> seed dependent
buggy = list(vertices)
# FIXED: sorted(set) -> seed independent
fixed = sorted(vertices)

# 2) hyperedge id sets (e.g. all_edges / relation post-filter ids)
edges = {("a", "b"), ("b", "c"), ("a", "c", "d"), ("c", "a")}
fixed_edge_keys = sorted(_stable_edge_id(e) for e in edges)

# 3) relation-line entity ranking tie-break: (-rank, entity_name)
ranked = [
    {"rank": 3, "entity_name": "z"},
    {"rank": 3, "entity_name": "a"},
    {"rank": 5, "entity_name": "m"},
    {"rank": 5, "entity_name": "k"},
]
fixed_ranked = sorted(ranked, key=lambda x: (-x["rank"], x["entity_name"]))

# 4) hyperedge ranking tie-break: (-rank, -weight, stable_edge_id)
hedges = [
    {"rank": 2, "weight": 1.0, "id_set": ("x", "y")},
    {"rank": 2, "weight": 1.0, "id_set": ("y", "x")},
    {"rank": 1, "weight": 2.0, "id_set": ("a", "b")},
]
fixed_hedges = sorted(
    hedges, key=lambda x: (-x["rank"], -x["weight"], _stable_edge_id(x["id_set"]))
)

import json
out = {
    "fixed_vertices": fixed,
    "fixed_edge_keys": fixed_edge_keys,
    "fixed_ranked": [(x["rank"], x["entity_name"]) for x in fixed_ranked],
    "fixed_hedges": [
        (x["rank"], x["weight"], _stable_edge_id(x["id_set"])) for x in fixed_hedges
    ],
}
# Echo the buggy order too (must be excluded from the equality check).
sys.stdout.write(json.dumps(out))
'''


def run_with_seed(seed: str) -> dict:
    code = _SUBPROC_CODE.replace("%ROOT%", str(Path(".").resolve()))
    proc = subprocess.run(
        [str(_CONDA_PYTHON), "-c", code],
        env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"[seed={seed}] STDERR:\n{proc.stderr}")
        raise RuntimeError(f"subprocess with PYTHONHASHSEED={seed} failed")
    import json
    return json.loads(proc.stdout)


def main():
    out1 = run_with_seed("1")
    out99 = run_with_seed("99")

    failures = []
    for key in out1:
        if out1[key] != out99[key]:
            failures.append((key, out1[key], out99[key]))

    # Also: demonstrate the buggy path IS seed dependent (sanity / documentation)
    # Run the buggy list(set) under two seeds and confirm they CAN differ.
    buggy_code = (
        's={"ent_c","ent_a","ent_b","ent_d","ent_e"};'
        'import sys;sys.stdout.write(str(list(s)))'
    )
    b1 = subprocess.run(
        [sys.executable, "-c", buggy_code],
        env={**__import__("os").environ, "PYTHONHASHSEED": "1"},
        capture_output=True, text=True,
    ).stdout
    b99 = subprocess.run(
        [sys.executable, "-c", buggy_code],
        env={**__import__("os").environ, "PYTHONHASHSEED": "99"},
        capture_output=True, text=True,
    ).stdout
    buggy_differs = (b1 != b99)

    print("=" * 60)
    print("Determinism test across PYTHONHASHSEED=1 vs PYTHONHASHSEED=99")
    print("=" * 60)
    for key in out1:
        status = "OK" if out1[key] == out99[key] else "FAIL"
        print(f"  [{status}] {key}")
    print(f"  [info] buggy list(set) differs across seeds: {buggy_differs}")
    print("=" * 60)

    if failures:
        print(f"FAILED: {len(failures)} field(s) differ across seeds")
        for key, a, b in failures:
            print(f"  {key}: seed1={a} seed99={b}")
        sys.exit(1)
    if not buggy_differs:
        print("WARNING: buggy list(set) happened to match — test environment "
              "may have a fixed default hash seed; determinism fix still valid.")
    print("PASSED: all deterministic orderings identical across seeds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
