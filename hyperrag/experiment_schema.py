# -*- coding: utf-8 -*-
"""实验契约共享校验逻辑 (Schema Owner, Step 1.2)。

本模块是 Question Set v2.1 契约的**唯一**校验实现，被以下入口共享：
  - scripts/validate_question_set_contract.py （契约门禁脚本）
  - hyperrag/trace_collector.py               （运行时 trace 写盘前校验）
  - 分析/评测脚本                              （标签推导）

设计约束：
  - 仅依赖 stdlib（不 import yaml/numpy/openai），保证可被 importlib
    以文件路径方式独立加载，而不触发 hyperrag/__init__ 的重依赖。
  - 所有冻结算法（stable-success 判定 / 11 维成本 Pareto / 标签推导）
    只在这里实现一次；任何 runner/Judge/分析脚本禁止复制这些数值与逻辑。
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "docs" / "schema"
QUESTION_SCHEMA_PATH = SCHEMA_DIR / "question_item_v2.schema.json"
TRACE_SCHEMA_PATH = SCHEMA_DIR / "retrieval_trace_v2.schema.json"

POLICY_STAGES = ["P0", "P1", "P2", "P3", "P4"]

# All raw cost fields defined in the contract (cost_vector.raw_fields).
# Pareto comparison and cost aggregation MUST use this full set, not a
# hand-picked subset, so that no genuine cost dimension is silently dropped.
COST_FIELDS = [
    "input_tokens",
    "output_tokens",
    "retrieval_latency_ms",
    "end_to_end_latency_ms",
    "llm_calls",
    "embedding_calls",
    "reranker_calls",
    "graph_expansion_calls",
    "retrieved_candidate_count",
    "gpu_seconds",
    "api_cost",
]


def load_question_schema(path=None):
    """加载 question_item_v2 JSON Schema。"""
    p = Path(path) if path else QUESTION_SCHEMA_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def load_trace_schema(path=None):
    """加载 retrieval_trace_v2 JSON Schema。"""
    p = Path(path) if path else TRACE_SCHEMA_PATH
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (draft 2020-12 subset)
# --------------------------------------------------------------------------- #
class SchemaValidationError(Exception):
    pass


def _type_matches(value, type_name):
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    return False


def _check_type(value, type_spec, path):
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    if any(_type_matches(value, t) for t in types):
        return
    raise SchemaValidationError(
        f"{path}: expected type {type_spec}, got {type(value).__name__}"
    )


def validate_schema(instance, schema, path="$"):
    """Validate an instance against a JSON-Schema (subset).

    Supports: type, required, properties, additionalProperties, items,
    enum, const, pattern, minimum, maximum, minItems, maxItems,
    minLength, maxLength, and the array-of-types form of ``type``.
    """
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{path}: schema node is not an object")

    if "type" in schema:
        _check_type(instance, schema["type"], path)

    if "const" in schema:
        if instance != schema["const"]:
            raise SchemaValidationError(
                f"{path}: expected const {schema['const']!r}, got {instance!r}"
            )

    if "enum" in schema:
        if instance not in schema["enum"]:
            raise SchemaValidationError(
                f"{path}: {instance!r} not in enum {schema['enum']}"
            )

    if "pattern" in schema and isinstance(instance, str):
        if not re.fullmatch(schema["pattern"], instance):
            raise SchemaValidationError(
                f"{path}: {instance!r} does not match pattern {schema['pattern']}"
            )

    for bound, op in (("minimum", lambda a, b: a >= b),
                      ("maximum", lambda a, b: a <= b)):
        if bound in schema and isinstance(instance, (int, float)) \
                and not isinstance(instance, bool):
            if not op(instance, schema[bound]):
                raise SchemaValidationError(
                    f"{path}: {instance} violates {bound}={schema[bound]}"
                )

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise SchemaValidationError(
                f"{path}: length {len(instance)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise SchemaValidationError(
                f"{path}: length {len(instance)} > maxLength {schema['maxLength']}")

    if isinstance(instance, dict):
        if "required" in schema:
            for key in schema["required"]:
                if key not in instance:
                    raise SchemaValidationError(f"{path}.{key}: required field missing")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    raise SchemaValidationError(
                        f"{path}.{key}: additional property not allowed"
                    )
        for key, subschema in props.items():
            if key in instance:
                validate_schema(instance[key], subschema, f"{path}.{key}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise SchemaValidationError(
                f"{path}: expected >= {schema['minItems']} items, got {len(instance)}"
            )
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaValidationError(
                f"{path}: expected <= {schema['maxItems']} items, got {len(instance)}"
            )
        if "items" in schema:
            items = schema["items"]
            if isinstance(items, list):  # tuple-positional (rare)
                for i, item_schema in enumerate(items):
                    if i < len(instance):
                        validate_schema(instance[i], item_schema, f"{path}[{i}]")
            else:
                for i, item in enumerate(instance):
                    validate_schema(item, items, f"{path}[{i}]")


# --------------------------------------------------------------------------- #
# Cross-field checks (question item / trace record)
# --------------------------------------------------------------------------- #
def check_question_cross_field(item, errors):
    """Cross-field rules the JSON type system cannot express.

    - answer_unit unit_id must be unique (no duplicates).
    - evidence_requirement requirement_id must be unique.
    - evidence_requirement.answer_unit_ids must reference a defined AU.
    - evidence_spans.unit_id must reference a defined AU (no dangling AU999).
    - every evidence span must satisfy start_char < end_char (strict; a
      zero-length / equal-offset span is rejected).
    - every REQUIRED answer unit must be backed by >=1 evidence span.
    - every REQUIRED answer unit must be referenced by >=1 evidence requirement.
    """
    aus = [u for u in item.get("answer_units", []) if isinstance(u, dict)]
    au_ids = {u.get("unit_id") for u in aus}
    # duplicate answer_unit_ids
    seen_au = set()
    for u in aus:
        uid = u.get("unit_id")
        if uid in seen_au:
            errors.append(f"duplicate answer_unit_id {uid}")
        seen_au.add(uid)
    # duplicate requirement_ids
    ers = [e for e in item.get("evidence_requirements", []) if isinstance(e, dict)]
    seen_er = set()
    for e in ers:
        eid = e.get("requirement_id")
        if eid in seen_er:
            errors.append(f"duplicate evidence_requirement_id {eid}")
        seen_er.add(eid)
    # ER references + required-AU referenced set
    referenced_by_er = set()
    for er in ers:
        for au in er.get("answer_unit_ids", []):
            if au not in au_ids:
                errors.append(
                    f"evidence_requirement {er.get('requirement_id')} "
                    f"references undefined answer unit {au}")
            else:
                referenced_by_er.add(au)
    # spans reference + strict offsets + required-AU spanned set
    spanned_units = set()
    for sp in item.get("evidence_spans", []):
        uid = sp.get("unit_id")
        if uid not in au_ids:
            errors.append(
                f"evidence_spans references undefined answer unit {uid}")
        else:
            spanned_units.add(uid)
        for seg in sp.get("evidence_spans", []):
            sc, ec = seg.get("start_char"), seg.get("end_char")
            if isinstance(sc, int) and isinstance(ec, int) and sc >= ec:
                errors.append(
                    f"evidence span {uid} has start_char {sc} >= end_char {ec} "
                    f"(must be strictly less)")
    # every required AU must have a span AND be referenced by an ER
    for u in aus:
        uid = u.get("unit_id")
        if u.get("required") is True:
            if uid not in spanned_units:
                errors.append(
                    f"required answer unit {uid} has no evidence span")
            if uid not in referenced_by_er:
                errors.append(
                    f"required answer unit {uid} is not referenced by any "
                    f"evidence_requirement")


def validate_question_item(item, schema, path="$"):
    """Schema validation + cross-field checks; raises on either failure."""
    validate_schema(item, schema, path)
    errs = []
    check_question_cross_field(item, errs)
    if errs:
        raise SchemaValidationError("; ".join(errs))


def check_trace_cross_field(trace, errors):
    """Cross-field rules for a trace record.

    - When cost.reranker_calls == 0, every candidate's reranker_score MUST be
      null (never faked as 0), per design doc 17.2.
    """
    cost = trace.get("cost") or {}
    if cost.get("reranker_calls") == 0:
        for retr in trace.get("retrievers", []):
            for cand in retr.get("candidates", []):
                if cand.get("reranker_score") is not None:
                    errors.append(
                        f"retriever {retr.get('retriever_id')} candidate "
                        f"{cand.get('candidate_id')} has reranker_score="
                        f"{cand.get('reranker_score')} but reranker_calls=0 "
                        f"(must be null)")


def validate_trace_record(trace, schema, path="$"):
    """Schema validation + cross-field checks; raises on either failure."""
    validate_schema(trace, schema, path)
    errs = []
    check_trace_cross_field(trace, errs)
    if errs:
        raise SchemaValidationError("; ".join(errs))


# --------------------------------------------------------------------------- #
# Label derivation (frozen algorithm)
# --------------------------------------------------------------------------- #
def _cost_vector(record):
    """Return the 11-field cost dict for a raw run or an aggregated summary.

    A missing cost component (absent from the raw run, or absent from EVERY
    repeat in a summary) is returned as ``None`` -- it is NEVER defaulted to 0,
    so an unmeasured route cannot be falsely reported as the cheapest.
    """
    if "n_runs" in record:  # aggregated summary already carries a median vector
        return record["cost"]
    cost = record.get("cost") or {}
    return {k: cost.get(k) for k in COST_FIELDS}


def _is_cost_complete(record):
    """True iff the record carries all 11 cost fields with non-null values.

    A route missing ANY cost component has an INCOMPLETE cost vector and is
    NOT eligible for Pareto dominance comparison: an unmeasured route must
    never be declared cheaper than a fully-measured one.
    """
    vec = _cost_vector(record)
    return all(vec.get(k) is not None for k in COST_FIELDS)


def _dominates(a, b):
    """Return True iff cost-vector *a* dominates *b*: a <= b on every cost
    component and a < b on at least one, over the FULL 11-field cost vector.

    Both operands MUST carry a complete cost vector (see _is_cost_complete);
    if either is incomplete, domination is False -- neither route can claim to
    be cheaper on incomplete data.
    """
    if not _is_cost_complete(a) or not _is_cost_complete(b):
        return False
    ca, cb = _cost_vector(a), _cost_vector(b)
    le = True
    lt = False
    for k in COST_FIELDS:
        av, bv = ca[k], cb[k]
        if av > bv:
            le = False
        if av < bv:
            lt = True
    return le and lt


def _stable_success(pass_count, n):
    """operational_stable_success_v1 decision.

    Only the explicit, frozen stopping points are accepted. Any
    (pass_count, n) that is NOT a stopping point returns None (undecided):
    the route must NOT be counted as successful or failed until more runs are
    collected.

    Stopping points (frozen in contract stability_rule.decision):
      n = 3 : 3/3 -> success ; 0/3, 1/3 -> fail ; 2/3 -> undecided
      n = 5 : 4/5, 5/5 -> success ; 0/5..3/5 -> fail
      n = 10: 8/10..10/10 -> success ; 0/10..7/10 -> fail
      any other n -> undecided (None)
    """
    if n == 3:
        if pass_count == 3:
            return True
        if pass_count <= 1:
            return False
        return None  # 2/3 -> inconclusive, run 2 more
    if n == 5:
        if pass_count >= 4:  # 4/5 or 5/5
            return True
        return False  # 0/5..3/5 -> fail
    if n == 10:
        if pass_count >= 8:  # 8/10..10/10
            return True
        return False  # 0/10..7/10 -> fail
    return None  # not a defined stopping point -> undecided


def _median(values):
    s = sorted(values)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return s[m]
    return (s[m - 1] + s[m]) / 2


def _median_cost_vector(recs):
    """Per-component median across repeats. A component absent from EVERY run
    is ``None`` (genuinely unavailable), never defaulted to 0."""
    result = {}
    for k in COST_FIELDS:
        present = [r.get("cost", {}).get(k)
                   for r in recs if isinstance(r.get("cost"), dict)]
        present = [v for v in present if v is not None]
        result[k] = _median(present) if present else None
    return result


def _aggregate_routes(runs):
    """Group raw repeat runs by (question_id, route_id), apply the stable
    success rule, and collapse to ONE summary record per route carrying the
    pass rate and a median cost vector."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        groups[(r["question_id"], r["route_id"])].append(r)
    summaries = []
    for (qid, route), recs in groups.items():
        successes = [bool(r.get("success")) for r in recs]
        n = len(successes)
        pass_count = sum(successes)
        summaries.append({
            "question_id": qid,
            "route_id": route,
            "n_runs": n,
            "pass_count": pass_count,
            "stable_success": _stable_success(pass_count, n),
            "cost": _median_cost_vector(recs),
        })
    return summaries


def derive_labels(runs):
    """Derive frozen label outputs from a flat list of per-route repeat runs.

    Pipeline: raw repeats -> aggregate per (question_id, route_id) with the
    stable-success rule -> one summary per route -> successful set /
    minimum-sufficient-by-policy / Pareto over the FULL 11-field cost vector.
    """
    summaries = _aggregate_routes(runs)
    by_q = {}
    for s in summaries:
        by_q.setdefault(s["question_id"], []).append(s)

    out = {}
    for qid, sums in by_q.items():
        successful = [s for s in sums if s["stable_success"] is True]
        successful_routes = [s["route_id"] for s in successful]
        min_sufficient = None
        for stage in POLICY_STAGES:
            if stage in successful_routes:
                min_sufficient = stage
                break
        # Pareto: among successful routes with a COMPLETE cost vector, those
        # not dominated by another complete-cost route. Routes with an
        # incomplete cost vector are excluded from Pareto (they stay in
        # successful_route_set, but cannot claim to be cheapest).
        successful_complete = [s for s in successful if _is_cost_complete(s)]
        pareto = []
        for s in successful_complete:
            if not any(_dominates(other, s)
                       for other in successful_complete if other is not s):
                pareto.append(s["route_id"])
        out[qid] = {
            "successful_route_set": sorted(successful_routes,
                                           key=POLICY_STAGES.index),
            "minimum_sufficient_route_by_policy": min_sufficient,
            "pareto_optimal_routes": sorted(pareto, key=POLICY_STAGES.index),
        }
    return out


# 公开别名（新代码建议使用无下划线名字；旧调用方兼容下划线名字）
stable_success = _stable_success
is_cost_complete = _is_cost_complete
dominates = _dominates
median_cost_vector = _median_cost_vector
aggregate_routes = _aggregate_routes
