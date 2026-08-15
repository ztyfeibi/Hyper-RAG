#!/usr/bin/env python3
"""calibrate_blind_review.py — 盲审校准脚本（validate / report / apply 三模式）。

阶段一（validate）：校验 AI 标注文件结构完整性 + 机械派生 claim fatality 生成
                   verdicts_ai_annotated_v2.jsonl（不覆盖原文件）+ 生成 ai_annotation_manifest.json。
阶段三（report）：计算 LongCat 与独立 AI 的一致性指标 + bootstrap 95% CI。
阶段五（apply）：逐条计算真实 ER 级 evidence coverage（复用 judge_longcat 同一实现，
                 并与冻结统计对账），应用 adjudication_rule_v3 生成 480 条最终裁决。

依赖：复用 scripts/judge_longcat.py 的纯函数（calc_source_evidence_coverage /
extract_sources_section / load_result_contexts / load_evidence_requirements /
load_evidence_spans / result_file）。judge_longcat 的 import 链含 openai/my_config，
但本脚本只调用其纯计算函数，不发任何 LLM 请求。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


def _import_judge_longcat():
    """导入 judge_longcat 以复用其纯 coverage 函数。

    judge_longcat 的 import 链会拉起 hyperrag/__init__（aioboto3/aiohttp 等重依赖）
    与 openai——本脚本只调用纯计算函数（calc_source_evidence_coverage /
    load_result_contexts / load_evidence_* / result_file），不发任何 LLM 请求，
    故对重依赖注入轻量 stub（若真实可导入则优先用真实模块）。
    """
    import importlib
    import types

    def _stub(name: str, **attrs):
        try:
            importlib.import_module(name)
        except Exception:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod

    _stub("openai", OpenAI=object)
    if "hyperrag" not in sys.modules:
        try:
            importlib.import_module("hyperrag")
        except Exception:
            pkg = types.ModuleType("hyperrag")
            pkg.__path__ = []  # 标记为 package，允许 import hyperrag.env
            env = types.ModuleType("hyperrag.env")
            env.normalize_proxy_env = lambda *a, **k: None
            pkg.env = env
            sys.modules["hyperrag"] = pkg
            sys.modules["hyperrag.env"] = env
    return importlib.import_module("judge_longcat")


jl = _import_judge_longcat()  # 复用同一 coverage 实现（不重写）

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 路径 / 运行参数常量
# ---------------------------------------------------------------------------
REPEAT = 0
SEED = 42
SNAPSHOT = jl.SNAPSHOT_DEFAULT
_JUDGE_DIR = _ROOT / jl.LONGCAT_DIR / f"r{REPEAT}_s{SEED}_{SNAPSHOT[:8]}"
BLIND_DIR = _JUDGE_DIR / "blind_review"
QUESTIONS_FILE = _ROOT / jl.QUESTIONS_FILE
ROUTES = list(jl.ROUTES)

VALID_VERDICTS = {"pass", "fail", "uncertain"}
VALID_AU_STATUS = {"supported", "missing", "contradicted"}
VALID_CLAIM_STATUS = {"supported_by_source", "unsupported_noncritical", "contradicted", "unverifiable"}
# v2 起记录级/claim 级 fatality 均含 unresolved（unverifiable 无法确认为无害）
VALID_CLAIM_FATALITY = {"fatal", "harmless", "none", "unresolved"}
VALID_FATALITY = {"none", "harmless", "fatal", "unresolved"}

# adjudication_rule_v3 常量
ADJUDICATION_RULE_VERSION = "v3"

# claim status -> claim fatality 机械派生映射（v2，区别于旧版 unverifiable->harmless）
CLAIM_FATALITY_MAP = {
    "contradicted": "fatal",
    "unsupported_noncritical": "harmless",
    "supported_by_source": "none",
    "unverifiable": "unresolved",
}
# record-level fatality 由 claim-level 重新派生的优先级（高 -> 低）
RECORD_FATALITY_PRIORITY = ("fatal", "unresolved", "harmless", "none")

# 冻结 coverage 统计（summary.json，r0_s42_5c92f17c）——apply 必须逐条复算后与此对账
FROZEN_COVERAGE_PASS = {"P1": 48, "P2": 6, "P3": 24, "P4": 34, "P_gold": 80}


class AlignError(Exception):
    """fail-closed 对齐错误（映射缺失/重复/记录缺失/parse_ok 非 true/AU ID 不一致）。"""


# ---------------------------------------------------------------------------
# 通用 IO
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _set_of(bid: str) -> str:
    n = int(bid.split("-")[1])
    if n <= 100:
        return "A"
    if n <= 143:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
ANN_V1 = "verdicts_ai_annotated.jsonl"
ANN_V2 = "verdicts_ai_annotated_v2.jsonl"


def load_annotated() -> list[dict]:
    return load_jsonl(BLIND_DIR / ANN_V1)


def load_annotated_v2() -> list[dict]:
    path = BLIND_DIR / ANN_V2
    if not path.exists():
        raise AlignError(f"{ANN_V2} 不存在：请先运行 --mode validate 生成 v2 标注文件")
    return load_jsonl(path)


def load_template() -> list[dict]:
    return load_jsonl(BLIND_DIR / "verdicts_template.jsonl")


def load_sample_manifest() -> dict:
    return json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))


def load_longcat_verdicts() -> dict[str, dict[str, dict]]:
    """route -> {qid -> best_record}（去重，优先 parse_ok）。"""
    out: dict[str, dict[str, dict]] = {}
    for route in ROUTES:
        path = _JUDGE_DIR / f"{route}_verdict.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        best: dict[str, dict] = {}
        for r in rows:
            q = r["question_id"]
            if q not in best or (r.get("parse_ok") and not best[q].get("parse_ok")):
                best[q] = r
        out[route] = best
    return out


def load_questions() -> dict[str, dict]:
    return {r["question_id"]: r for r in load_jsonl(QUESTIONS_FILE)}


def load_judge_manifest() -> dict:
    mp = _JUDGE_DIR / "manifest.json"
    if not mp.exists():
        return {}
    return json.loads(mp.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# fatality 机械派生（v2）
# ---------------------------------------------------------------------------
def _derive_claim_fatality(status: str) -> str:
    """从 claim status 派生 fatality。

    v2 映射（与旧版区别：unverifiable -> unresolved，无法确认为无害即悬置）。
    未知 status 一律 unresolved（fail-safe：不得默认无害）。
    """
    return CLAIM_FATALITY_MAP.get(status, "unresolved")


def derive_record_fatality(claims: list[dict]) -> str:
    """record-level fatality 由 claim-level 重新派生：fatal > unresolved > harmless > none。

    无 claim 时为 none。
    """
    fatal_rank = {f: i for i, f in enumerate(RECORD_FATALITY_PRIORITY)}
    best = "none"
    for c in claims or []:
        f = c.get("fatality") or _derive_claim_fatality(c.get("status", ""))
        if fatal_rank.get(f, 99) < fatal_rank.get(best, 99):
            best = f
    return best


def annotate_v2(src_path: Path, dst_path: Path) -> dict:
    """机械派生生成 v2 标注文件（不覆盖原文件）。

    每条 claim 补显式 fatality；record-level unsupported_fatality 由 claim-level
    重新派生（原值保留为 unsupported_fatality_original）。
    """
    rows = load_jsonl(src_path)
    out = []
    claim_fat_dist = Counter()
    n_derived = 0
    for r in rows:
        r2 = dict(r)
        claims = []
        for c in r.get("unsupported_claims", []):
            c2 = dict(c)
            f = c2.get("fatality") or _derive_claim_fatality(c2.get("status", ""))
            c2["fatality"] = f
            claim_fat_dist[f] += 1
            n_derived += 1
            claims.append(c2)
        r2["unsupported_claims"] = claims
        r2["unsupported_fatality_original"] = r.get("unsupported_fatality")
        r2["unsupported_fatality"] = derive_record_fatality(claims)
        r2["fatality_derivation"] = "claim_status_mechanical_v2"
        out.append(r2)
    with open(dst_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "records": len(out),
        "claims_derived": n_derived,
        "claim_fatality_dist": dict(claim_fat_dist),
    }


# ---------------------------------------------------------------------------
# 阶段一：validate
# ---------------------------------------------------------------------------
def validate_annotations(rows: list[dict], require_fatality: bool = False) -> dict:
    """校验 AI 标注结构完整性。require_fatality=True 时 claim 缺/非法 fatality 为 ERROR。"""
    tmpl = load_template()
    errors: list[str] = []
    warnings: list[str] = []

    # 1. 恰好 163 条
    if len(rows) != 163:
        errors.append(f"记录数 {len(rows)} != 163")

    # 2. blind_id 唯一
    ids = [r.get("blind_id", "") for r in rows]
    dupes = [bid for bid, c in Counter(ids).items() if c > 1]
    if dupes:
        errors.append(f"重复 blind_id: {dupes[:5]}")

    # 3. 与 template ID 集合一致
    tmpl_ids = {r["blind_id"] for r in tmpl}
    ann_ids = set(ids)
    if ann_ids != tmpl_ids:
        missing = tmpl_ids - ann_ids
        extra = ann_ids - tmpl_ids
        if missing:
            errors.append(f"缺失 ID: {sorted(missing)[:5]}")
        if extra:
            errors.append(f"多余 ID: {sorted(extra)[:5]}")

    # 4. A/B/C 分别 100/43/20
    set_counts = Counter(_set_of(bid) for bid in ids if bid)
    if set_counts.get("A", 0) != 100:
        errors.append(f"set_A={set_counts.get('A', 0)} != 100")
    if set_counts.get("B", 0) != 43:
        errors.append(f"set_B={set_counts.get('B', 0)} != 43")
    if set_counts.get("C", 0) != 20:
        errors.append(f"set_C={set_counts.get('C', 0)} != 20")

    # 5. verdict / au_status / record fatality 枚举
    for r in rows:
        bid = r.get("blind_id", "?")
        if not r.get("verdict"):
            errors.append(f"{bid}: 空 verdict")
        elif r["verdict"] not in VALID_VERDICTS:
            errors.append(f"{bid}: 非法 verdict={r['verdict']}")
        if not r.get("au_status"):
            errors.append(f"{bid}: 空 au_status")
        else:
            for au_id, status in r["au_status"].items():
                if status not in VALID_AU_STATUS:
                    errors.append(f"{bid}: 非法 au_status={status} (AU={au_id})")
        fat = r.get("unsupported_fatality", "")
        if fat not in VALID_FATALITY:
            errors.append(f"{bid}: 非法 unsupported_fatality={fat}")

    # 6. claim status / fatality
    claims_no_status = 0
    claims_bad_fatality = 0
    for r in rows:
        for c in r.get("unsupported_claims", []):
            if "status" not in c:
                claims_no_status += 1
            elif c["status"] not in VALID_CLAIM_STATUS:
                errors.append(f"{r.get('blind_id', '?')}: 非法 claim status={c.get('status')}")
            fat = c.get("fatality")
            if require_fatality:
                if fat is None:
                    claims_bad_fatality += 1
                elif fat not in VALID_CLAIM_FATALITY:
                    errors.append(f"{r.get('blind_id', '?')}: 非法 claim fatality={fat}")
    if claims_no_status:
        errors.append(f"{claims_no_status} 条 claim 缺 status")
    if claims_bad_fatality:
        errors.append(
            f"{claims_bad_fatality} 条 claim 缺独立 fatality 字段"
            f"（require_fatality=True 时不允许，需 v2 机械派生）"
        )

    # 7. record-level fatality 与 claim-level 派生一致性（v2 应严格一致）
    if require_fatality:
        for r in rows:
            derived = derive_record_fatality(r.get("unsupported_claims", []))
            if r.get("unsupported_fatality") != derived:
                errors.append(
                    f"{r.get('blind_id', '?')}: record fatality={r.get('unsupported_fatality')}"
                    f" != claim 派生={derived}"
                )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "total": len(rows),
            "set_A": set_counts.get("A", 0),
            "set_B": set_counts.get("B", 0),
            "set_C": set_counts.get("C", 0),
            "verdict_dist": dict(Counter(r.get("verdict") for r in rows)),
            "fatality_dist": dict(Counter(r.get("unsupported_fatality") for r in rows)),
            "au_status_dist": dict(
                Counter(v for r in rows for v in r.get("au_status", {}).values())
            ),
            "total_claims": sum(len(r.get("unsupported_claims", [])) for r in rows),
            "claim_status_dist": dict(
                Counter(c.get("status") for r in rows for c in r.get("unsupported_claims", []))
            ),
        },
    }


def generate_manifest(args) -> dict:
    """生成 ai_annotation_manifest.json（审核元数据来自 CLI；不可证明写 unknown）。"""
    ann_path = BLIND_DIR / ANN_V1
    v2_path = BLIND_DIR / ANN_V2
    prompt_path = Path(args.review_prompt_path) if getattr(args, "review_prompt_path", None) else None
    provenance_complete = all([
        getattr(args, "review_model", None),
        getattr(args, "review_provider", None),
        getattr(args, "review_temperature", None) is not None,
        prompt_path is not None and prompt_path.exists(),
    ])
    manifest = {
        "annotation_type": getattr(args, "annotation_type", None) or "unknown",
        "disclaimer": "本标注由 AI 独立完成，不是人工 Gold。仅用于与 LongCat Judge 的一致性校准。",
        "annotated_file": ANN_V1,
        "annotated_file_sha256": sha256_file(ann_path),
        "annotated_record_count": len(load_annotated()),
        "annotated_v2_file": ANN_V2,
        "annotated_v2_sha256": sha256_file(v2_path) if v2_path.exists() else None,
        # 审核元数据（CLI 提供；未提供记 unknown，不伪造）
        "review_model": getattr(args, "review_model", None) or "unknown",
        "review_provider": getattr(args, "review_provider", None) or "unknown",
        "review_temperature": (getattr(args, "review_temperature", None)
                               if getattr(args, "review_temperature", None) is not None else "unknown"),
        "review_prompt_source": (str(prompt_path) if prompt_path else "unknown"),
        "review_prompt_sha256": (sha256_file(prompt_path) if prompt_path and prompt_path.exists() else "unknown"),
        "provenance_complete": provenance_complete,
        "provenance_note": (
            "审核模型/服务/温度/prompt 哈希由 CLI 参数提供；无法证明的字段记 unknown，"
            "provenance_complete=false。禁止从旧 manifest 抄袭未经证实的元数据。"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_fatality_derivation": {
            "note": "v2 文件的 claim/record fatality 均为机械派生，非独立人工判定",
            "claim_mapping": dict(CLAIM_FATALITY_MAP),
            "record_priority": list(RECORD_FATALITY_PRIORITY),
            "migration_warning": (
                f"verdicts_ai_annotated.jsonl 原文件中 unverifiable 曾映射为 harmless（v1）；"
                f"v2 起映射为 unresolved。原始文件未改动。"
            ),
        },
    }
    out_path = BLIND_DIR / "ai_annotation_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# 数据对齐（fail-closed）
# ---------------------------------------------------------------------------
def _build_bl_map(manifest: dict, errors: list[str]) -> dict[str, dict]:
    bl_map: dict[str, dict] = {}
    seen_keys: set = set()
    for set_name, key in [
        ("A", "set_A_main_100"),
        ("B", "set_B_pgold_pending_43"),
        ("C", "set_C_neg_control_20"),
    ]:
        for entry in manifest.get(key, []):
            bid = entry["blind_id"]
            if bid in bl_map:
                errors.append(f"sample_manifest 重复 blind_id: {bid}")
                continue
            bl_map[bid] = entry
            rk = (entry["route"], entry["qid"])
            if rk in seen_keys:
                errors.append(f"sample_manifest 重复 (route, qid): {rk}")
            seen_keys.add(rk)
    return bl_map


def align_records() -> list[dict]:
    """对齐 AI 标注(v2) ↔ LongCat 原始判定。任何不一致直接抛 AlignError（fail-closed）。"""
    ann = load_annotated_v2()
    manifest = load_sample_manifest()
    longcat = load_longcat_verdicts()
    questions = load_questions()

    errors: list[str] = []
    if len(ann) != 163:
        errors.append(f"v2 标注记录数 {len(ann)} != 163")
    bl_map = _build_bl_map(manifest, errors)

    records = []
    for a in ann:
        bid = a.get("blind_id", "?")
        info = bl_map.get(bid)
        if not info:
            errors.append(f"{bid}: sample_manifest 无映射（fail-closed）")
            continue
        route = info["route"]
        qid = info["qid"]
        route_map = longcat.get(route)
        if route_map is None:
            errors.append(f"{bid}: LongCat 无 {route}_verdict.jsonl 记录")
            continue
        lc = route_map.get(qid)
        if lc is None:
            errors.append(f"{bid}: LongCat {route} 找不到 qid={qid}")
            continue
        if not lc.get("parse_ok"):
            errors.append(f"{bid}: LongCat 记录 parse_ok 非 true（qid={qid}）")
            continue
        q = questions.get(qid)
        if q is None:
            errors.append(f"{bid}: 冻结题集找不到 qid={qid}")
            continue
        all_aus = {u["unit_id"] for u in q.get("answer_units", [])}
        ai_au = a.get("au_status", {})
        unknown_ai = set(ai_au) - all_aus
        if unknown_ai:
            errors.append(f"{bid}: AI AU ID 不在题集定义: {sorted(unknown_ai)}")
            continue
        lc_au_ids = {au.get("unit_id") for au in lc.get("answer_units", []) or []}
        unknown_lc = lc_au_ids - all_aus
        if unknown_lc:
            errors.append(f"{bid}: LongCat AU ID 不在题集定义: {sorted(unknown_lc)}")
            continue

        lc_au = {au["unit_id"]: au.get("status", "") for au in lc.get("answer_units", [])}
        lc_unsup = lc.get("unsupported_claims", [])
        ai_unsup = a.get("unsupported_claims", [])
        required_aus = {u["unit_id"] for u in q.get("answer_units", []) if u.get("required", True)}

        records.append({
            "blind_id": bid,
            "set": _set_of(bid),
            "route": route,
            "qid": qid,
            "au_shape": info.get("au_shape", "single"),
            "cov_hit": info.get("cov_hit", True),
            "n_unsupported_lc": info.get("n_unsupported", 0),
            "has_contradicted": info.get("has_contradicted", False),
            # LongCat side
            "lc_verdict": lc.get("verdict", ""),
            "lc_derived": lc.get("derived_verdict", ""),
            "lc_human_review": lc.get("human_review", False),
            "lc_au_status": lc_au,
            "lc_unsup_count": len(lc_unsup) if isinstance(lc_unsup, list) else 0,
            # AI side（v2：fatality 为机械派生值）
            "ai_verdict": a.get("verdict", ""),
            "ai_fatality": a.get("unsupported_fatality", "none"),
            "ai_fatality_original": a.get("unsupported_fatality_original"),
            "ai_au_status": dict(ai_au),
            "ai_unsup_claims": ai_unsup,
            "ai_unsup_count": len(ai_unsup),
            # question structure
            "required_aus": required_aus,
            "all_aus": all_aus,
            "n_aus": len(all_aus),
            "is_multi_au": len(all_aus) > 1,
        })

    if errors:
        raise AlignError("align_records fail-closed:\n  " + "\n  ".join(errors))
    if len(records) != 163:
        raise AlignError(f"对齐后记录数 {len(records)} != 163")
    return records


# ---------------------------------------------------------------------------
# 真实 coverage 计算（复用 judge_longcat 同一实现）
# ---------------------------------------------------------------------------
def compute_coverage_map(longcat: dict[str, dict[str, dict]] | None = None) -> dict:
    """逐 (route, qid) 计算 ER 级 evidence coverage。

    规则与 judge_longcat.build_summary 完全一致（同一函数、同一输入文件）：
    - P0 无证据门槛（coverage_hit=None）
    - P2/P3/P4 只搜 -----Sources----- 区段（search="sources"）
    - P1/P_gold 全文（search="full"）
    - context 缺失（is None）记 no_context（coverage_hit=None）
    禁止从 summary 总数反推。
    """
    if longcat is None:
        longcat = load_longcat_verdicts()
    spans = jl.load_evidence_spans()
    ers = jl.load_evidence_requirements()
    cov_map: dict[tuple, dict] = {}
    for route in ROUTES:
        route_map = longcat.get(route, {})
        has_gate = route != "P0"
        contexts = None
        if has_gate:
            rf = _ROOT / jl.result_file(route, REPEAT, SEED, SNAPSHOT)
            contexts = jl.load_result_contexts(rf) if rf.exists() else None
        search = "sources" if route in ("P2", "P3", "P4") else "full"
        for qid in route_map:
            if not has_gate:
                cov_map[(route, qid)] = {
                    "coverage_hit": None,
                    "coverage_rule_version": None,
                    "no_context": False,
                    "evidence_requirements_total": None,
                    "evidence_requirements_hit": None,
                    "er_recall": None,
                    "coverage_details": None,
                    "context_hash": None,
                }
                continue
            ctx = contexts.get(qid) if contexts is not None else None
            if ctx is None:
                cov_map[(route, qid)] = {
                    "coverage_hit": None,
                    "coverage_rule_version": jl.COVERAGE_RULE_VERSION,
                    "no_context": True,
                    "evidence_requirements_total": len(ers.get(qid, [])),
                    "evidence_requirements_hit": None,
                    "er_recall": None,
                    "coverage_details": None,
                    "context_hash": None,
                }
                continue
            result = jl.calc_source_evidence_coverage(
                ctx, ers.get(qid, []), spans.get(qid, {}), search=search)
            total = len(ers.get(qid, []))
            hit = sum(1 for v in result["requirement_hits"].values() if v)
            cov_map[(route, qid)] = {
                "coverage_hit": result["complete_evidence_hit"],
                "coverage_rule_version": jl.COVERAGE_RULE_VERSION,
                "no_context": False,
                "evidence_requirements_total": total,
                "evidence_requirements_hit": hit,
                "er_recall": result["er_recall"],
                "coverage_details": result["requirement_hits"],
                "context_hash": sha256_str(ctx),
            }
    return cov_map


def coverage_route_stats(cov_map: dict) -> dict:
    """按路径统计 coverage（pass/fail/no_context，含 P0 无门槛）。"""
    stats = {}
    for route in ROUTES:
        items = [v for (r, _q), v in cov_map.items() if r == route]
        stats[route] = {
            "n": len(items),
            "pass": sum(1 for v in items if v["coverage_hit"] is True),
            "fail": sum(1 for v in items if v["coverage_hit"] is False),
            "no_context": sum(1 for v in items if v["no_context"]),
            "not_gated": sum(1 for v in items if v["coverage_rule_version"] is None),
        }
    return stats


def verify_coverage_against_frozen(cov_map: dict) -> list[str]:
    """逐条复算的全量 coverage 统计必须与冻结 summary 一致，否则 fail-closed。"""
    errors = []
    stats = coverage_route_stats(cov_map)
    # 与内置冻结常量对账
    for route, expected in FROZEN_COVERAGE_PASS.items():
        actual = stats[route]["pass"]
        if actual != expected:
            errors.append(f"{route}: 复算 coverage pass={actual} != 冻结 {expected}")
    # 与 summary.json 对账（若存在）
    sp = _JUDGE_DIR / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
        for route, expected in FROZEN_COVERAGE_PASS.items():
            frozen = summary.get("routes", {}).get(route, {}) \
                .get("source_evidence_coverage", {}).get("pass")
            if frozen is not None and frozen != expected:
                errors.append(f"{route}: 内置冻结常量 {expected} 与 summary.json {frozen} 不一致")
    return errors


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def _binary(status: str) -> str:
    return "supported" if status == "supported" else "not_supported"


def au_binary_prf(records: list[dict]) -> dict:
    """AU supported-vs-not-supported P/R/F1（AI=reference, LongCat=system）。"""
    tp = fp = fn = tn = 0
    for r in records:
        for au_id in r["all_aus"]:
            lc_b = _binary(r["lc_au_status"].get(au_id, "missing"))
            ai_b = _binary(r["ai_au_status"].get(au_id, "missing"))
            if lc_b == "supported" and ai_b == "supported":
                tp += 1
            elif lc_b == "supported" and ai_b == "not_supported":
                fp += 1
            elif lc_b == "not_supported" and ai_b == "supported":
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_au": tp + fp + fn + tn,
    }


def au_confusion_matrix(records: list[dict]) -> dict:
    labels = ["supported", "missing", "contradicted"]
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for r in records:
        for au_id in r["all_aus"]:
            ai_s = r["ai_au_status"].get(au_id, "missing")
            lc_s = r["lc_au_status"].get(au_id, "missing")
            if ai_s in labels and lc_s in labels:
                matrix[ai_s][lc_s] += 1
    return {"labels": labels, "matrix": matrix}


def cohen_kappa(labels1: list, labels2: list, categories: list) -> float | None:
    """Cohen's kappa。退化情形（pe>=1，即按类别分布完全可预测 / 类别无变化）返回 None，
    不得虚构 1.0 或 0.0。"""
    n = len(labels1)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    pe = sum((c1.get(c, 0) / n) * (c2.get(c, 0) / n) for c in categories)
    if pe >= 1.0:
        return None  # 退化：类别无变化/完全可预测 → N/A
    return round((po - pe) / (1 - pe), 4)


def verdict_agreement(records: list[dict]) -> dict:
    cats3 = ["pass", "fail", "uncertain"]
    lc_v = [r["lc_verdict"] for r in records]
    ai_v = [r["ai_verdict"] for r in records]
    n = len(records)

    acc3 = sum(1 for a, b in zip(ai_v, lc_v) if a == b) / n if n else 0.0
    kappa3 = cohen_kappa(ai_v, lc_v, cats3)

    pairs2 = [(a, b) for a, b in zip(ai_v, lc_v) if a != "uncertain" and b != "uncertain"]
    cats2 = ["pass", "fail"]
    acc2 = sum(1 for a, b in pairs2 if a == b) / len(pairs2) if pairs2 else 0.0
    kappa2 = cohen_kappa([a for a, _ in pairs2], [b for _, b in pairs2], cats2) if pairs2 else None

    false_pass = sum(1 for a, b in zip(ai_v, lc_v) if b == "pass" and a == "fail")
    false_fail = sum(1 for a, b in zip(ai_v, lc_v) if b == "fail" and a == "pass")
    lc_unc = sum(1 for v in lc_v if v == "uncertain") / n if n else 0.0
    ai_unc = sum(1 for v in ai_v if v == "uncertain") / n if n else 0.0

    cm = {a: {b: 0 for b in cats3} for a in cats3}
    for a, b in zip(ai_v, lc_v):
        if a in cm and b in cm[a]:
            cm[a][b] += 1

    return {
        "n": n,
        "three_class": {
            "accuracy": round(acc3, 4),
            "cohen_kappa": kappa3,
            "confusion_matrix": {"labels": cats3, "matrix": cm},
        },
        "binary_drop_uncertain": {
            "n": len(pairs2),
            "accuracy": round(acc2, 4),
            "cohen_kappa": kappa2,
        },
        "false_pass": false_pass,
        "false_fail": false_fail,
        "uncertain_rate_longcat": round(lc_unc, 4),
        "uncertain_rate_ai": round(ai_unc, 4),
    }


def unsupported_metrics(records: list[dict]) -> dict:
    lc_detected = sum(1 for r in records if r["lc_unsup_count"] > 0)
    ai_detected = sum(1 for r in records if r["ai_unsup_count"] > 0)
    both_detected = sum(1 for r in records if r["lc_unsup_count"] > 0 and r["ai_unsup_count"] > 0)
    neither = sum(1 for r in records if r["lc_unsup_count"] == 0 and r["ai_unsup_count"] == 0)
    lc_only = lc_detected - both_detected
    ai_only = ai_detected - both_detected

    # claim-level fatality 分布（v2：claim 自带显式 fatality，缺失时按 status 派生）
    fatality_dist = Counter()
    total_ai_claims = 0
    for r in records:
        for c in r["ai_unsup_claims"]:
            total_ai_claims += 1
            fatality_dist[c.get("fatality") or _derive_claim_fatality(c.get("status", ""))] += 1

    set_c = [r for r in records if r["set"] == "C"]
    set_c_ai_found = sum(1 for r in set_c if r["ai_unsup_count"] > 0)
    set_c_total_claims = sum(r["ai_unsup_count"] for r in set_c)

    return {
        "record_level": {
            "n": len(records),
            "longcat_detected": lc_detected,
            "ai_detected": ai_detected,
            "both_detected": both_detected,
            "longcat_only": lc_only,
            "ai_only": ai_only,
            "neither": neither,
            "longcat_detection_rate": round(lc_detected / len(records), 4) if records else 0.0,
            "ai_detection_rate": round(ai_detected / len(records), 4) if records else 0.0,
        },
        "claim_level": {
            "total_ai_claims": total_ai_claims,
            "fatality_dist": dict(fatality_dist),
        },
        "set_c_negative_control": {
            "n": len(set_c),
            "ai_found_unsupported": set_c_ai_found,
            "ai_total_claims": set_c_total_claims,
            "longcat_false_negative_rate": round(set_c_ai_found / len(set_c), 4) if set_c else 0.0,
        },
    }


def bootstrap_metric(records: list[dict], metric_fn, n_boot: int = 1000,
                     confidence: float = 0.95) -> tuple:
    """Bootstrap 95% CI。返回 (point, lo, hi, n_valid)。

    - 采样使 metric 退化（返回 None，如 kappa pe=1）时忽略该轮并计入 n_valid 缺口；
    - 全部退化时 lo=hi=None（报告 N/A），不虚构数值。
    """
    n = len(records)
    if n == 0:
        return (None, None, None, 0)
    point = metric_fn(records)
    rng = random.Random(42)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(records) for _ in range(n)]
        val = metric_fn(sample)
        if val is not None:
            boots.append(val)
    if point is None or not boots:
        return (None if point is None else round(point, 4), None, None, len(boots))
    boots.sort()
    alpha = (1 - confidence) / 2
    lo = boots[int(alpha * len(boots))]
    hi = boots[min(int((1 - alpha) * len(boots)), len(boots) - 1)]
    return (round(point, 4), round(lo, 4), round(hi, 4), len(boots))


def _group_by(records: list[dict], key_fn) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(key_fn(r))].append(r)
    return dict(groups)


def compute_all_metrics(records: list[dict]) -> dict:
    au_prf = au_binary_prf(records)
    au_cm = au_confusion_matrix(records)
    vagree = verdict_agreement(records)
    unsup = unsupported_metrics(records)

    def _precision(rs):
        return au_binary_prf(rs)["precision"]

    def _recall(rs):
        return au_binary_prf(rs)["recall"]

    def _f1(rs):
        return au_binary_prf(rs)["f1"]

    def _acc3(rs):
        return verdict_agreement(rs)["three_class"]["accuracy"]

    def _kappa3(rs):
        return verdict_agreement(rs)["three_class"]["cohen_kappa"]  # 可为 None

    p_ci = bootstrap_metric(records, _precision)
    r_ci = bootstrap_metric(records, _recall)
    f1_ci = bootstrap_metric(records, _f1)
    acc_ci = bootstrap_metric(records, _acc3)
    kappa_ci = bootstrap_metric(records, _kappa3)

    def _by_set(r):
        return r["set"]

    def _by_route(r):
        return r["route"]

    def _by_au_shape(r):
        return "multi" if r["is_multi_au"] else "single"

    def _by_cov(r):
        return "pass" if r["cov_hit"] else "fail"

    groups = {}
    for name, fn in [("set", _by_set), ("route", _by_route),
                     ("au_shape", _by_au_shape), ("coverage", _by_cov)]:
        grp = _group_by(records, fn)
        groups[name] = {}
        for k, rs in sorted(grp.items()):
            groups[name][k] = {
                "n": len(rs),
                "au_prf": au_binary_prf(rs),
                "verdict": verdict_agreement(rs),
            }

    return {
        "overall": {
            "n_records": len(records),
            "au_binary_prf": au_prf,
            "au_binary_prf_ci": {"precision": p_ci, "recall": r_ci, "f1": f1_ci},
            "au_confusion_matrix": au_cm,
            "verdict_agreement": vagree,
            "verdict_ci": {"accuracy": acc_ci, "kappa": kappa_ci},
            "unsupported_metrics": unsup,
        },
        "grouped": groups,
    }


# ---------------------------------------------------------------------------
# 阶段三：report
# ---------------------------------------------------------------------------
def _fmt_val(v):
    return "N/A" if v is None else v


def _fmt_ci(ci) -> str:
    if ci is None or ci[1] is None:
        return "N/A"
    return f"[{ci[1]}, {ci[2]}] (n_valid={ci[3]}/{1000})"


def cmd_report(args) -> int:
    try:
        records = align_records()
    except AlignError as e:
        print(f"ERROR: 对齐失败（fail-closed）:\n{e}", file=sys.stderr)
        return 1

    metrics = compute_all_metrics(records)

    recs_path = BLIND_DIR / "calibration_records.jsonl"
    with open(recs_path, "w", encoding="utf-8") as f:
        for r in records:
            out = {k: (list(v) if isinstance(v, set) else v) for k, v in r.items()}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    provenance = _provenance(args)
    met_path = BLIND_DIR / "calibration_metrics.json"
    met_path.write_text(
        json.dumps({**metrics, "provenance": provenance}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    report = _render_report(metrics, records, provenance)
    rep_path = BLIND_DIR / "calibration_report.md"
    rep_path.write_text(report, encoding="utf-8")

    print("校准报告已生成:")
    print(f"  {met_path}")
    print(f"  {rep_path}")
    print(f"  {recs_path}")
    return 0


def _provenance(args) -> dict:
    judge_manifest = load_judge_manifest()
    judge_model = judge_manifest.get("judge_model", "unknown")
    review_model = getattr(args, "review_model", None) or "unknown"
    return {
        "review_model": review_model,
        "review_provider": getattr(args, "review_provider", None) or "unknown",
        "review_temperature": getattr(args, "review_temperature", None),
        "annotation_type": getattr(args, "annotation_type", None) or "unknown",
        "judge_model": judge_model,
        "same_model": (review_model != "unknown" and review_model == judge_model),
        "provenance_complete": all([
            getattr(args, "review_model", None),
            getattr(args, "review_provider", None),
            getattr(args, "review_temperature", None) is not None,
        ]),
    }


def _render_report(m: dict, records: list[dict], provenance: dict) -> str:
    o = m["overall"]
    same_model = provenance.get("same_model")
    if same_model:
        scope = "同模型不同提示词的二次裁决一致性（审核模型与 Judge 模型相同，仅提示词不同）"
    else:
        scope = "LongCat Judge 与独立 AI 审核的一致性"
    lines = [
        f"# 盲审校准报告：{scope}",
        "",
        f"**记录数**: {o['n_records']}",
        f"**生成时间**: {datetime.now(timezone.utc).isoformat()}",
        f"**审核模型**: {provenance.get('review_model')} "
        f"(provider={provenance.get('review_provider')}, "
        f"temperature={_fmt_val(provenance.get('review_temperature'))})",
        f"**Judge 模型**: {provenance.get('judge_model')}",
        f"**同模型**: {'是（结论仅支持二次裁决一致性，不支持跨模型独立性）' if same_model else '否/无法证明'}",
        "",
        "## 1. AU 级指标（supported vs not-supported）",
        "",
        "以独立 AI 标注为参考（reference），LongCat Judge 为系统输出（system）。",
        "",
        f"| 指标 | 值 | Bootstrap 95% CI |",
        f"|---|---|---|",
        f"| Precision | {o['au_binary_prf']['precision']} | {_fmt_ci(o['au_binary_prf_ci']['precision'])} |",
        f"| Recall | {o['au_binary_prf']['recall']} | {_fmt_ci(o['au_binary_prf_ci']['recall'])} |",
        f"| F1 | {o['au_binary_prf']['f1']} | {_fmt_ci(o['au_binary_prf_ci']['f1'])} |",
        f"| TP / FP / FN / TN | {o['au_binary_prf']['tp']} / {o['au_binary_prf']['fp']} / {o['au_binary_prf']['fn']} / {o['au_binary_prf']['tn']} | |",
        f"| 总 AU 数 | {o['au_binary_prf']['n_au']} | |",
        "",
        "## 2. AU 三分类混淆矩阵（行=AI reference, 列=LongCat）",
        "",
    ]
    cm = o["au_confusion_matrix"]
    labels = cm["labels"]
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    for a in labels:
        row = cm["matrix"][a]
        lines.append(f"| {a} | " + " | ".join(str(row[b]) for b in labels) + " |")

    lines += [
        "",
        "## 3. Verdict 级一致性（三分类 pass/fail/uncertain）",
        "",
        f"| 指标 | 值 | Bootstrap 95% CI |",
        f"|---|---|---|",
        f"| Accuracy | {o['verdict_agreement']['three_class']['accuracy']} | {_fmt_ci(o['verdict_ci']['accuracy'])} |",
        f"| Cohen's kappa | {_fmt_val(o['verdict_agreement']['three_class']['cohen_kappa'])} | {_fmt_ci(o['verdict_ci']['kappa'])} |",
        "",
        f"**二分类**（去掉 uncertain, n={o['verdict_agreement']['binary_drop_uncertain']['n']}）:",
        f"- Accuracy: {o['verdict_agreement']['binary_drop_uncertain']['accuracy']}",
        f"- Cohen's kappa: {_fmt_val(o['verdict_agreement']['binary_drop_uncertain']['cohen_kappa'])}",
        "",
        f"**误差分析**:",
        f"- false-pass（LongCat=pass, AI=fail）: {o['verdict_agreement']['false_pass']}",
        f"- false-fail（LongCat=fail, AI=pass）: {o['verdict_agreement']['false_fail']}",
        f"- uncertain rate（LongCat）: {o['verdict_agreement']['uncertain_rate_longcat']}",
        f"- uncertain rate（AI）: {o['verdict_agreement']['uncertain_rate_ai']}",
        "",
        "## 4. Unsupported claim 检出",
        "",
    ]
    u = o["unsupported_metrics"]
    rl = u["record_level"]
    lines += [
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| LongCat 检出率 | {rl['longcat_detection_rate']} ({rl['longcat_detected']}/{rl['n']}) |",
        f"| AI 检出率 | {rl['ai_detection_rate']} ({rl['ai_detected']}/{rl['n']}) |",
        f"| 双方均检出 | {rl['both_detected']} |",
        f"| 仅 LongCat | {rl['longcat_only']} |",
        f"| 仅 AI | {rl['ai_only']} |",
        f"| 均未检出 | {rl['neither']} |",
        "",
        f"**Claim 级 fatality 分布**（v2 机械派生：contradicted→fatal, "
        f"unsupported_noncritical→harmless, supported_by_source→none, unverifiable→unresolved）:",
    ]
    for k, v in sorted(u["claim_level"]["fatality_dist"].items()):
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        f"**set_C 阴性对照**（LongCat 判定无 unsupported）:",
        f"- AI 在 {u['set_c_negative_control']['ai_found_unsupported']}/{u['set_c_negative_control']['n']} 条中发现 unsupported",
        f"- 共 {u['set_c_negative_control']['ai_total_claims']} 条 AI 检出 claim",
        f"- LongCat false-negative rate: {u['set_c_negative_control']['longcat_false_negative_rate']}",
        "",
        "## 5. 分组指标",
        "",
    ]
    for grp_name, grp_data in m["grouped"].items():
        lines.append(f"### 按 {grp_name} 分组")
        lines.append("")
        lines.append(f"| {grp_name} | n | AU P | AU R | AU F1 | Verdict Acc | Kappa | FP | FF |")
        lines.append(f"|---|---|---|---|---|---|---|---|---|")
        for k, v in sorted(grp_data.items()):
            prf = v["au_prf"]
            va = v["verdict"]
            lines.append(
                f"| {k} | {v['n']} | {prf['precision']} | {prf['recall']} | {prf['f1']} | "
                f"{va['three_class']['accuracy']} | {_fmt_val(va['three_class']['cohen_kappa'])} | "
                f"{va['false_pass']} | {va['false_fail']} |"
            )
        lines.append("")

    lines += [
        "## 6. 说明",
        "",
        f"- 本报告衡量 {scope}，不是人机一致性。",
        f"- 审核元数据由 CLI 提供（无法证明的字段记 unknown，provenance_complete="
        f"{provenance.get('provenance_complete')}），不沿用旧 manifest 中未经证实的硬编码值。",
        "- claim 级 fatality 为 v2 机械派生: contradicted→fatal, unsupported_noncritical→harmless, "
        "supported_by_source→none, unverifiable→unresolved（v1 曾把 unverifiable 映射为 harmless）。",
        "- kappa 退化（pe=1，类别无变化/完全可预测）返回 N/A，不虚构 1.0；bootstrap 退化轮忽略并报告 n_valid。",
        "- Bootstrap CI 使用 seed=42, n_boot=1000。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段五：apply（adjudication_rule_v3 + 真实 coverage）
# ---------------------------------------------------------------------------
def apply_rule_v3(lc_record: dict, ai_record: dict | None, coverage_hit: bool | None,
                  has_evidence_gate: bool, required_aus: set | None = None) -> dict:
    """应用 adjudication_rule_v3 返回最终裁决。

    规则:
    1. AI 终审存在时:
       - required AU contradicted 或 missing（unsupported）-> answer_correctness=fail
       - 任一 claim fatality=fatal（contradicted claim）-> fail
       - 全部 required AU supported 且无 fatal/unresolved claim -> pass
       - 存在 unresolved claim（unverifiable）-> pending（进补充审核）
    2. 无 AI 终审（longcat_only）: answer_correctness 一律 pending，
       不得因 LongCat AU 全 supported 自动 pass，也不得外推 163 条结论。
    3. P0 无证据门槛: route_success = answer_correctness。
    4. P1-P4/P_gold: coverage=False 是 route_success 硬 fail（answer_correctness 保持原值，
       未审核时保持 pending）。
    """
    if ai_record is not None:
        au_status = ai_record.get("au_status", {})
        ai_verdict = ai_record.get("verdict", "")
        claims = ai_record.get("unsupported_claims", [])
        req = required_aus if required_aus is not None else set(au_status)
        req_status = {u: au_status.get(u, "missing") for u in req}
        claim_fatalities = [
            c.get("fatality") or _derive_claim_fatality(c.get("status", "")) for c in claims
        ]
        has_required_contradicted = any(s == "contradicted" for s in req_status.values())
        has_required_missing = any(s == "missing" for s in req_status.values())
        has_fatal_claim = "fatal" in claim_fatalities
        has_unresolved_claim = "unresolved" in claim_fatalities
        all_required_supported = bool(req_status) and all(
            s == "supported" for s in req_status.values())
        source = "ai_review"

        if has_required_contradicted or has_required_missing or has_fatal_claim:
            answer_correctness = "fail"
        elif all_required_supported and not has_unresolved_claim:
            answer_correctness = "pass"
        else:  # unresolved claim 或 AU 集为空 -> 悬置
            answer_correctness = "pending"
    else:
        au_status = {au["unit_id"]: au.get("status", "")
                     for au in lc_record.get("answer_units", []) or []}
        ai_verdict = None
        req = required_aus or set()
        claim_fatalities = []
        has_required_contradicted = None
        has_required_missing = None
        has_fatal_claim = None
        has_unresolved_claim = None
        answer_correctness = "pending"  # 未审核一律 pending，不自动 pass
        source = "longcat_only"

    # route_success
    if not has_evidence_gate:
        route_success = answer_correctness
    elif coverage_hit is False:
        route_success = "fail"  # coverage fail 硬 fail，不被 pending 覆盖
    else:
        route_success = answer_correctness

    return {
        "question_id": lc_record.get("question_id"),
        "route": lc_record.get("route"),
        "answer_correctness": answer_correctness,
        "route_success": route_success,
        "source": source,
        "au_status": au_status,
        "required_aus": sorted(req) if isinstance(req, (set, list)) else list(req or []),
        "has_required_contradicted": has_required_contradicted,
        "has_required_missing": has_required_missing,
        "has_fatal_claim": has_fatal_claim,
        "has_unresolved_claim": has_unresolved_claim,
        "ai_verdict": ai_verdict,
        "coverage_hit": coverage_hit,
    }


def cmd_apply(args) -> int:
    """阶段五：真实 coverage + adjudication_rule_v3 生成 480 条最终裁决（fail-closed）。"""
    # 1) AI 对齐（fail-closed，v2 标注）
    try:
        records163 = align_records()
    except AlignError as e:
        print(f"ERROR: 对齐失败（fail-closed）:\n{e}", file=sys.stderr)
        return 1
    ai_by_route_qid = {(r["route"], r["qid"]): r for r in records163}

    longcat = load_longcat_verdicts()
    questions = load_questions()

    # 2) 真实 coverage（逐条复算，与冻结统计对账）
    cov_map = compute_coverage_map(longcat)
    cov_errors = verify_coverage_against_frozen(cov_map)
    if cov_errors:
        print("ERROR: coverage 复算与冻结统计不一致（fail-closed）:", file=sys.stderr)
        for e in cov_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    # 3) 结构检查：六路径各 80、parse_ok 全 true
    structural_errors = []
    for route in ROUTES:
        route_map = longcat.get(route, {})
        if len(route_map) != 80:
            structural_errors.append(f"{route}: {len(route_map)} 条 != 80")
        for qid, lc in route_map.items():
            if not lc.get("parse_ok"):
                structural_errors.append(f"{route}/{qid}: parse_ok 非 true")
    if structural_errors:
        print("ERROR: LongCat 记录结构不满足 480 条约束（fail-closed）:", file=sys.stderr)
        for e in structural_errors[:20]:
            print(f"  {e}", file=sys.stderr)
        return 1

    # 4) 候选答案（补充审核材料用）
    answers_by_route = {}
    for route in ROUTES:
        rf = _ROOT / jl.result_file(route, REPEAT, SEED, SNAPSHOT)
        answers_by_route[route] = jl.load_results(rf) if rf.exists() else {}

    final_records = []
    pending_supplementary = []
    ai_reviewed_pending = 0

    for route in ROUTES:
        has_gate = route != "P0"
        for qid in sorted(longcat[route]):
            lc = longcat[route][qid]
            cov = cov_map[(route, qid)]
            ai_r = ai_by_route_qid.get((route, qid))
            q = questions[qid]
            required = {u["unit_id"] for u in q.get("answer_units", [])
                        if u.get("required", True)}
            ai_input = None
            if ai_r is not None:
                ai_input = {
                    "au_status": ai_r["ai_au_status"],
                    "verdict": ai_r["ai_verdict"],
                    "unsupported_claims": ai_r["ai_unsup_claims"],
                    "unsupported_fatality": ai_r["ai_fatality"],
                }
            lc_input = dict(lc)
            lc_input["route"] = route
            v = apply_rule_v3(lc_input, ai_input, cov["coverage_hit"], has_gate,
                              required_aus=required)
            final_records.append({
                "route": route,
                "question_id": qid,
                "blind_id": ai_r["blind_id"] if ai_r else None,
                "source": v["source"],
                "answer_correctness": v["answer_correctness"],
                "route_success": v["route_success"],
                "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
                # 真实 coverage 字段（问题一）
                "coverage_hit": cov["coverage_hit"],
                "coverage_rule_version": cov["coverage_rule_version"],
                "no_context": cov["no_context"],
                "evidence_requirements_total": cov["evidence_requirements_total"],
                "evidence_requirements_hit": cov["evidence_requirements_hit"],
                "er_recall": cov["er_recall"],
                "coverage_details": cov["coverage_details"],
                "context_hash": cov["context_hash"],
                # 裁决依据
                "required_aus": v["required_aus"],
                "has_required_contradicted": v["has_required_contradicted"],
                "has_required_missing": v["has_required_missing"],
                "has_fatal_claim": v["has_fatal_claim"],
                "has_unresolved_claim": v["has_unresolved_claim"],
                "ai_verdict": v["ai_verdict"],
                "lc_verdict": lc.get("verdict"),
                "lc_derived_verdict": lc.get("derived_verdict"),
            })
            if v["route_success"] == "pending":
                if ai_r is None:
                    # 补充审核队列：不含 163 条已审核；隐藏 route/blind_id
                    pending_supplementary.append({
                        "question_id": qid,
                        "question": q.get("question", ""),
                        "answer_units": [
                            {"unit_id": u["unit_id"], "claim": u.get("claim", ""),
                             "required": bool(u.get("required", True))}
                            for u in q.get("answer_units", [])
                        ],
                        "evidence_spans": {
                            es["unit_id"]: es["evidence_spans"]
                            for es in q.get("evidence_spans", [])
                        },
                        "candidate_answer": answers_by_route.get(route, {}).get(qid, ""),
                        "lc_verdict": lc.get("verdict"),
                        "lc_derived_verdict": lc.get("derived_verdict"),
                        "coverage_hit": cov["coverage_hit"],
                        "evidence_requirements_hit": cov["evidence_requirements_hit"],
                        "evidence_requirements_total": cov["evidence_requirements_total"],
                        "reason": "no_ai_review",
                    })
                else:
                    ai_reviewed_pending += 1

    # 5) 480 条约束校验
    keys = [(r["route"], r["question_id"]) for r in final_records]
    if len(final_records) != 480 or len(set(keys)) != 480:
        print(f"ERROR: final_verdicts {len(final_records)} 条 / 唯一键 {len(set(keys))} != 480",
              file=sys.stderr)
        return 1
    for route in ROUTES:
        n = sum(1 for r in final_records if r["route"] == route)
        if n != 80:
            print(f"ERROR: {route} {n} 条 != 80", file=sys.stderr)
            return 1

    # 6) 输出四件套
    out_dir = _JUDGE_DIR / "ai_adjudication_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    fv_path = out_dir / "final_verdicts.jsonl"
    with open(fv_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cov_stats = coverage_route_stats(cov_map)
    rs_dist = Counter(r["route_success"] for r in final_records)
    ac_dist = Counter(r["answer_correctness"] for r in final_records)
    by_route = {}
    for route in ROUTES:
        rs = [r for r in final_records if r["route"] == route]
        by_route[route] = {
            "n": len(rs),
            "route_success": dict(Counter(r["route_success"] for r in rs)),
            "answer_correctness": dict(Counter(r["answer_correctness"] for r in rs)),
            "coverage": {k: cov_stats[route][k] for k in ("pass", "fail", "no_context", "not_gated")},
            "ai_reviewed": sum(1 for r in rs if r["source"] == "ai_review"),
        }
    summary_out = {
        "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
        "coverage_rule_version": jl.COVERAGE_RULE_VERSION,
        "total_records": len(final_records),
        "route_success_dist": dict(rs_dist),
        "answer_correctness_dist": dict(ac_dist),
        "by_route": by_route,
        "ai_annotated_records": len(records163),
        "ai_reviewed_pending": ai_reviewed_pending,
        "pending_supplementary_count": len(pending_supplementary),
        "coverage_frozen_check": {
            "expected_pass": FROZEN_COVERAGE_PASS,
            "recomputed_pass": {r: cov_stats[r]["pass"] for r in FROZEN_COVERAGE_PASS},
            "match": True,
        },
    }
    fs_path = out_dir / "final_summary.json"
    fs_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    sup_path = out_dir / "pending_supplementary.jsonl"
    with open(sup_path, "w", encoding="utf-8") as f:
        for r in pending_supplementary:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 7) manifest（全量 SHA-256 溯源）
    input_hashes = {
        "questions_file": sha256_file(QUESTIONS_FILE),
        "ai_annotated_v1": sha256_file(BLIND_DIR / ANN_V1),
        "ai_annotated_v2": sha256_file(BLIND_DIR / ANN_V2),
        "sample_manifest": sha256_file(BLIND_DIR / "sample_manifest.json"),
        "judge_summary": sha256_file(_JUDGE_DIR / "summary.json"),
        "longcat_verdicts": {route: sha256_file(_JUDGE_DIR / f"{route}_verdict.jsonl")
                             for route in ROUTES},
        "result_files": {
            route: sha256_file(_ROOT / jl.result_file(route, REPEAT, SEED, SNAPSHOT))
            for route in ROUTES
        },
    }
    provenance = _provenance(args)
    manifest_out = {
        "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
        "coverage_rule_version": jl.COVERAGE_RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repeat": REPEAT, "seed": SEED, "snapshot": SNAPSHOT,
        "counts": {
            "total_records": len(final_records),
            "per_route": {route: 80 for route in ROUTES},
            "ai_annotated": len(records163),
            "ai_reviewed_pending": ai_reviewed_pending,
            "pending_supplementary": len(pending_supplementary),
        },
        "coverage_recomputation": {
            "method": "judge_longcat.calc_source_evidence_coverage 逐条复算（P2-P4 sources 区段，P1/P_gold 全文，P0 无门槛）",
            "frozen_match": True,
            "expected_pass": FROZEN_COVERAGE_PASS,
        },
        "provenance": provenance,
        "claim_fatality_derivation": {
            "claim_mapping": dict(CLAIM_FATALITY_MAP),
            "record_priority": list(RECORD_FATALITY_PRIORITY),
            "note": "fatality 为机械派生，非独立人工判定",
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "judge_script_sha256": sha256_file(Path(jl.__file__).resolve()),
        "input_sha256": input_hashes,
        "output_sha256": {
            "final_verdicts.jsonl": sha256_file(fv_path),
            "final_summary.json": sha256_file(fs_path),
            "pending_supplementary.jsonl": sha256_file(sup_path),
        },
        "note": "final_verdicts.jsonl 不覆盖原始 LongCat 判定；原始数据保留在 *_verdict.jsonl",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"最终裁决已生成: {out_dir}/")
    print(f"  final_verdicts.jsonl: {len(final_records)} 条（六路径各 80，(route,qid) 唯一）")
    print(f"  final_summary.json")
    print(f"  pending_supplementary.jsonl: {len(pending_supplementary)} 条（不含已审核 163 条，隐藏 route）")
    print(f"  manifest.json（全量 SHA-256）")
    print(f"  coverage 复算与冻结统计一致: {FROZEN_COVERAGE_PASS}")
    print(f"  已审核但仍 pending（unresolved 悬置）: {ai_reviewed_pending} 条")
    return 0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def cmd_validate(args) -> int:
    # 原文件结构校验（不含 fatality 要求）
    orig = load_annotated()
    r1 = validate_annotations(orig, require_fatality=False)
    for e in r1["errors"]:
        print(f"  ERROR: {e}")
    for w in r1["warnings"]:
        print(f"  WARN: {w}")
    if not r1["passed"]:
        print("=== 验收失败（原文件结构错误） ===")
        return 1

    # 机械派生生成 v2（不覆盖原文件）
    v2_stats = annotate_v2(BLIND_DIR / ANN_V1, BLIND_DIR / ANN_V2)
    v2 = load_annotated_v2()
    r2 = validate_annotations(v2, require_fatality=True)
    for e in r2["errors"]:
        print(f"  ERROR(v2): {e}")

    passed = r1["passed"] and r2["passed"]
    if passed:
        print("=== 验收通过（原文件 + v2 派生文件） ===")
    else:
        print("=== 验收失败 ===")
    print()
    print("统计（v2）:")
    for k, v in r2["stats"].items():
        print(f"  {k}: {v}")
    print(f"  claim fatality 派生: {v2_stats['claims_derived']} 条 -> {v2_stats['claim_fatality_dist']}")

    if passed:
        manifest = generate_manifest(args)
        print()
        print("ai_annotation_manifest.json 已生成:")
        print(f"  原文件 SHA-256: {manifest['annotated_file_sha256']}")
        print(f"  v2 文件 SHA-256: {manifest['annotated_v2_sha256']}")
        print(f"  review_model: {manifest['review_model']} ({manifest['review_provider']}), "
              f"temperature={manifest['review_temperature']}")
        print(f"  provenance_complete: {manifest['provenance_complete']}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="盲审校准脚本（validate / report / apply）")
    parser.add_argument("--mode", required=True, choices=["validate", "report", "apply"],
                        help="运行模式: validate(校验+v2派生) / report(指标) / apply(最终裁决)")
    # 审核元数据（问题五：禁止硬编码，不可证明写 unknown + provenance_complete=false）
    parser.add_argument("--review-model", default=None,
                        help="AI 审核所用模型名；无法证明则不传（记 unknown）")
    parser.add_argument("--review-provider", default=None,
                        help="AI 审核服务提供方；无法证明则不传")
    parser.add_argument("--review-temperature", type=float, default=None,
                        help="AI 审核温度；无法证明则不传")
    parser.add_argument("--review-prompt-path", default=None,
                        help="AI 审核 prompt 文件路径（取 SHA-256）；无法证明则不传")
    parser.add_argument("--annotation-type", default=None,
                        help="标注类型（如 independent_ai_review）；无法证明则不传")
    args = parser.parse_args()

    if args.mode == "validate":
        return cmd_validate(args)
    elif args.mode == "report":
        return cmd_report(args)
    elif args.mode == "apply":
        return cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
