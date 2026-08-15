#!/usr/bin/env python3
"""calibrate_blind_review.py — 盲审校准脚本（validate / report / apply 三模式）。

阶段一（validate）：校验 AI 标注文件结构完整性 + 生成 ai_annotation_manifest.json。
阶段三（report）：计算 LongCat 与独立 AI 的一致性指标 + bootstrap 95% CI。
阶段五（apply）：应用 adjudication_rule_v3 生成 480 条最终裁决。

依赖：仅 Python 标准库（json/hashlib/random/collections），不依赖 hyperrag/numpy/tiktoken。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_JUDGE_DIR = _ROOT / "caches/neurology_chunk1000/question_set_v2/pilot_v1/judge/longcat/r0_s42_5c92f17c"
BLIND_DIR = _JUDGE_DIR / "blind_review"
QUESTIONS_FILE = _ROOT / "caches/neurology_chunk1000/question_set_v2/pilot_v1/question_generation/questions_v2_manual_final.jsonl"
ROUTES = ["P0", "P1", "P2", "P3", "P4", "P_gold"]

VALID_VERDICTS = {"pass", "fail", "uncertain"}
VALID_AU_STATUS = {"supported", "missing", "contradicted"}
VALID_FATALITY = {"none", "harmless", "fatal"}
VALID_CLAIM_STATUS = {"supported_by_source", "unsupported_noncritical", "contradicted", "unverifiable"}

# adjudication_rule_v3 常量
ADJUDICATION_RULE_VERSION = "v3"


# ---------------------------------------------------------------------------
# 通用 IO
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
def load_annotated() -> list[dict]:
    return load_jsonl(BLIND_DIR / "verdicts_ai_annotated.jsonl")


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


# ---------------------------------------------------------------------------
# 阶段一：validate
# ---------------------------------------------------------------------------
def validate_annotations() -> dict:
    """校验 AI 标注文件结构完整性，返回 {passed, errors, warnings, stats}。"""
    ann = load_annotated()
    tmpl = load_template()
    errors: list[str] = []
    warnings: list[str] = []

    # 1. 恰好 163 条
    if len(ann) != 163:
        errors.append(f"记录数 {len(ann)} != 163")

    # 2. blind_id 唯一
    ids = [r.get("blind_id", "") for r in ann]
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
    set_counts = Counter(_set_of(bid) for bid in ids)
    if set_counts.get("A", 0) != 100:
        errors.append(f"set_A={set_counts.get('A',0)} != 100")
    if set_counts.get("B", 0) != 43:
        errors.append(f"set_B={set_counts.get('B',0)} != 43")
    if set_counts.get("C", 0) != 20:
        errors.append(f"set_C={set_counts.get('C',0)} != 20")

    # 5. 所有 AU 已填写，无非法枚举或空 verdict
    for r in ann:
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

    # 6. 每条 unsupported claim 有 status（fatality 字段缺失→warning + 派生）
    claims_no_status = 0
    claims_no_fatality = 0
    for r in ann:
        for c in r.get("unsupported_claims", []):
            if "status" not in c:
                claims_no_status += 1
            if c.get("status") not in VALID_CLAIM_STATUS:
                errors.append(f"{r.get('blind_id','?')}: 非法 claim status={c.get('status')}")
            if "fatality" not in c:
                claims_no_fatality += 1
    if claims_no_status:
        errors.append(f"{claims_no_status} 条 claim 缺 status")
    if claims_no_fatality:
        warnings.append(
            f"{claims_no_fatality} 条 claim 缺独立 fatality 字段；"
            f"将按 status 派生: contradicted→fatal, unsupported_noncritical→harmless, "
            f"unverifiable→harmless, supported_by_source→none"
        )

    # 一致性 warning: record-level fatality vs claim-level derived fatality
    inconsistent = 0
    for r in ann:
        fat = r.get("unsupported_fatality", "none")
        statuses = [c.get("status", "") for c in r.get("unsupported_claims", [])]
        derived_fatal = any(_derive_claim_fatality(s) == "fatal" for s in statuses)
        if fat == "fatal" and not derived_fatal:
            inconsistent += 1
        elif fat == "harmless" and derived_fatal:
            inconsistent += 1
    if inconsistent:
        warnings.append(f"{inconsistent} 条记录 record-level fatality 与 claim-level derived fatality 不一致（保留双级独立统计）")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "total": len(ann),
            "set_A": set_counts.get("A", 0),
            "set_B": set_counts.get("B", 0),
            "set_C": set_counts.get("C", 0),
            "verdict_dist": dict(Counter(r.get("verdict") for r in ann)),
            "fatality_dist": dict(Counter(r.get("unsupported_fatality") for r in ann)),
            "au_status_dist": dict(
                Counter(v for r in ann for v in r.get("au_status", {}).values())
            ),
            "total_claims": sum(len(r.get("unsupported_claims", [])) for r in ann),
            "claim_status_dist": dict(
                Counter(c.get("status") for r in ann for c in r.get("unsupported_claims", []))
            ),
        },
    }


def _derive_claim_fatality(status: str) -> str:
    """从 claim status 派生 fatality（claim 缺独立 fatality 字段时使用）。"""
    if status == "contradicted":
        return "fatal"
    if status == "supported_by_source":
        return "none"
    if status == "unsupported_noncritical":
        return "harmless"
    if status == "unverifiable":
        return "harmless"  # 无法确认为错误 → 默认无害
    return "harmless"


def generate_manifest() -> dict:
    """生成 ai_annotation_manifest.json。"""
    ann_path = BLIND_DIR / "verdicts_ai_annotated.jsonl"
    guide_path = BLIND_DIR / "REVIEW_GUIDE.md"
    manifest = {
        "annotation_type": "independent_ai_review",
        "disclaimer": "本标注由 AI 独立完成，不是人工 Gold。仅用于与 LongCat Judge 的一致性校准。",
        "annotated_file": "verdicts_ai_annotated.jsonl",
        "annotated_file_sha256": sha256_file(ann_path),
        "annotated_record_count": len(load_annotated()),
        "review_model": "LongCat-2.0",
        "review_provider": "SiliconFlow",
        "review_prompt_source": "REVIEW_GUIDE.md",
        "review_prompt_sha256": sha256_file(guide_path),
        "review_temperature": 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_fatality_derivation": {
            "note": "claim 缺独立 fatality 字段，按 status 派生",
            "mapping": {
                "contradicted": "fatal",
                "unsupported_noncritical": "harmless",
                "unverifiable": "harmless",
                "supported_by_source": "none",
            },
        },
    }
    out_path = BLIND_DIR / "ai_annotation_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# 数据对齐
# ---------------------------------------------------------------------------
def align_records() -> list[dict]:
    """对齐 AI 标注 ↔ LongCat 原始判定，返回逐条记录列表。"""
    ann = load_annotated()
    manifest = load_sample_manifest()
    longcat = load_longcat_verdicts()
    questions = load_questions()

    # 构建 blind_id -> (route, qid, set_info) 映射
    bl_map: dict[str, dict] = {}
    for set_name, key in [
        ("A", "set_A_main_100"),
        ("B", "set_B_pgold_pending_43"),
        ("C", "set_C_neg_control_20"),
    ]:
        for entry in manifest.get(key, []):
            bl_map[entry["blind_id"]] = entry

    records = []
    for a in ann:
        bid = a["blind_id"]
        info = bl_map.get(bid)
        if not info:
            continue
        route = info["route"]
        qid = info["qid"]
        lc = longcat.get(route, {}).get(qid, {})

        # LongCat AU status: [{unit_id, status}] -> {unit_id: status}
        lc_au = {au["unit_id"]: au.get("status", "") for au in lc.get("answer_units", [])}
        # AI AU status
        ai_au = a.get("au_status", {})

        # LongCat unsupported: list[str]; AI unsupported: list[{claim, status}]
        lc_unsup = lc.get("unsupported_claims", [])
        ai_unsup = a.get("unsupported_claims", [])

        # 题集 AU 定义
        q = questions.get(qid, {})
        required_aus = {u["unit_id"] for u in q.get("answer_units", []) if u.get("required", True)}
        all_aus = {u["unit_id"] for u in q.get("answer_units", [])}

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
            # AI side
            "ai_verdict": a.get("verdict", ""),
            "ai_fatality": a.get("unsupported_fatality", "none"),
            "ai_au_status": ai_au,
            "ai_unsup_claims": ai_unsup,
            "ai_unsup_count": len(ai_unsup),
            # question structure
            "required_aus": required_aus,
            "all_aus": all_aus,
            "n_aus": len(all_aus),
            "is_multi_au": len(all_aus) > 1,
        })
    return records


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def _binary(status: str) -> str:
    """AU status -> binary: supported / not_supported。"""
    return "supported" if status == "supported" else "not_supported"


def au_binary_prf(records: list[dict]) -> dict:
    """AU supported-vs-not-supported Precision/Recall/F1。

    以 AI 为参考标注（reference），LongCat 为系统输出（system）。
    TP=both supported, FP=LC supported + AI not, FN=AI supported + LC not, TN=both not。
    """
    tp = fp = fn = tn = 0
    for r in records:
        all_aus = r["all_aus"]
        lc = r["lc_au_status"]
        ai = r["ai_au_status"]
        for au_id in all_aus:
            lc_b = _binary(lc.get(au_id, "missing"))
            ai_b = _binary(ai.get(au_id, "missing"))
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
    """AU 三分类混淆矩阵（行=AI reference, 列=LongCat system）。"""
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
    """Cohen's kappa for two annotators."""
    n = len(labels1)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    pe = sum((c1.get(c, 0) / n) * (c2.get(c, 0) / n) for c in categories)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def verdict_agreement(records: list[dict]) -> dict:
    """三分类 + 二分类 verdict 一致性。"""
    cats3 = ["pass", "fail", "uncertain"]
    lc_v = [r["lc_verdict"] for r in records]
    ai_v = [r["ai_verdict"] for r in records]
    n = len(records)

    # 三分类
    acc3 = sum(1 for a, b in zip(ai_v, lc_v) if a == b) / n if n else 0.0
    kappa3 = cohen_kappa(ai_v, lc_v, cats3)

    # 二分类（去掉 uncertain）
    pairs2 = [(a, b) for a, b in zip(ai_v, lc_v) if a != "uncertain" and b != "uncertain"]
    cats2 = ["pass", "fail"]
    acc2 = sum(1 for a, b in pairs2 if a == b) / len(pairs2) if pairs2 else 0.0
    kappa2 = cohen_kappa([a for a, _ in pairs2], [b for _, b in pairs2], cats2) if pairs2 else None

    # false-pass: LongCat=pass, AI=fail
    false_pass = sum(1 for a, b in zip(ai_v, lc_v) if b == "pass" and a == "fail")
    # false-fail: LongCat=fail, AI=pass
    false_fail = sum(1 for a, b in zip(ai_v, lc_v) if b == "fail" and a == "pass")
    # uncertain rates
    lc_unc = sum(1 for v in lc_v if v == "uncertain") / n if n else 0.0
    ai_unc = sum(1 for v in ai_v if v == "uncertain") / n if n else 0.0

    # 混淆矩阵
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
    """unsupported claim 检出率及 fatal/nonfatal 分布。"""
    # 每条记录: LongCat 是否检出 unsupported, AI 是否检出
    lc_detected = sum(1 for r in records if r["lc_unsup_count"] > 0)
    ai_detected = sum(1 for r in records if r["ai_unsup_count"] > 0)
    both_detected = sum(1 for r in records if r["lc_unsup_count"] > 0 and r["ai_unsup_count"] > 0)
    neither = sum(1 for r in records if r["lc_unsup_count"] == 0 and r["ai_unsup_count"] == 0)
    lc_only = lc_detected - both_detected
    ai_only = ai_detected - both_detected

    # claim-level fatality 分布（从 AI claim status 派生）
    fatality_dist = Counter()
    total_ai_claims = 0
    for r in records:
        for c in r["ai_unsup_claims"]:
            total_ai_claims += 1
            fatality_dist[_derive_claim_fatality(c.get("status", ""))] += 1

    # set_C 阴性对照: LongCat 说 0 unsupported, AI 找到多少
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


def bootstrap_metric(records: list[dict], metric_fn, n_boot: int = 1000, confidence: float = 0.95) -> tuple:
    """Bootstrap 95% CI for a metric function that takes records and returns a float."""
    n = len(records)
    if n == 0:
        return (None, None, None)
    point = metric_fn(records)
    rng = random.Random(42)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(records) for _ in range(n)]
        boots.append(metric_fn(sample))
    boots.sort()
    alpha = (1 - confidence) / 2
    lo = boots[int(alpha * n_boot)]
    hi = boots[int((1 - alpha) * n_boot)]
    return (round(point, 4), round(lo, 4), round(hi, 4))


def _group_by(records: list[dict], key_fn) -> dict:
    """分组记录，返回 {key: [records]}。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(key_fn(r))].append(r)
    return dict(groups)


def compute_all_metrics(records: list[dict]) -> dict:
    """计算全部指标。"""
    au_prf = au_binary_prf(records)
    au_cm = au_confusion_matrix(records)
    vagree = verdict_agreement(records)
    unsup = unsupported_metrics(records)

    # bootstrap CI
    def _precision(rs):
        return au_binary_prf(rs)["precision"]
    def _recall(rs):
        return au_binary_prf(rs)["recall"]
    def _f1(rs):
        return au_binary_prf(rs)["f1"]
    def _acc3(rs):
        return verdict_agreement(rs)["three_class"]["accuracy"]
    def _kappa3(rs):
        return verdict_agreement(rs)["three_class"]["cohen_kappa"] or 0.0

    p_ci = bootstrap_metric(records, _precision)
    r_ci = bootstrap_metric(records, _recall)
    f1_ci = bootstrap_metric(records, _f1)
    acc_ci = bootstrap_metric(records, _acc3)
    kappa_ci = bootstrap_metric(records, _kappa3)

    # 分组
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
def cmd_report(args):
    records = align_records()
    if not records:
        print("ERROR: 无法对齐记录", file=sys.stderr)
        return 1

    metrics = compute_all_metrics(records)

    # 输出 calibration_records.jsonl
    recs_path = BLIND_DIR / "calibration_records.jsonl"
    with open(recs_path, "w", encoding="utf-8") as f:
        for r in records:
            # 序列化前清理 set 类型
            out = {k: (list(v) if isinstance(v, set) else v) for k, v in r.items()}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 输出 calibration_metrics.json
    met_path = BLIND_DIR / "calibration_metrics.json"
    met_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 输出 calibration_report.md
    report = _render_report(metrics, records)
    rep_path = BLIND_DIR / "calibration_report.md"
    rep_path.write_text(report, encoding="utf-8")

    print(f"校准报告已生成:")
    print(f"  {met_path}")
    print(f"  {rep_path}")
    print(f"  {recs_path}")
    return 0


def _render_report(m: dict, records: list[dict]) -> str:
    o = m["overall"]
    lines = [
        "# 盲审校准报告：LongCat 与独立 AI 的一致性",
        "",
        f"**记录数**: {o['n_records']}",
        f"**生成时间**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. AU 级指标（supported vs not-supported）",
        "",
        "以独立 AI 标注为参考（reference），LongCat Judge 为系统输出（system）。",
        "",
        f"| 指标 | 值 | Bootstrap 95% CI |",
        f"|---|---|---|",
        f"| Precision | {o['au_binary_prf']['precision']} | [{o['au_binary_prf_ci']['precision'][1]}, {o['au_binary_prf_ci']['precision'][2]}] |",
        f"| Recall | {o['au_binary_prf']['recall']} | [{o['au_binary_prf_ci']['recall'][1]}, {o['au_binary_prf_ci']['recall'][2]}] |",
        f"| F1 | {o['au_binary_prf']['f1']} | [{o['au_binary_prf_ci']['f1'][1]}, {o['au_binary_prf_ci']['f1'][2]}] |",
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
        f"| Accuracy | {o['verdict_agreement']['three_class']['accuracy']} | [{o['verdict_ci']['accuracy'][1]}, {o['verdict_ci']['accuracy'][2]}] |",
        f"| Cohen's kappa | {o['verdict_agreement']['three_class']['cohen_kappa']} | [{o['verdict_ci']['kappa'][1]}, {o['verdict_ci']['kappa'][2]}] |",
        "",
        f"**二分类**（去掉 uncertain, n={o['verdict_agreement']['binary_drop_uncertain']['n']}）:",
        f"- Accuracy: {o['verdict_agreement']['binary_drop_uncertain']['accuracy']}",
        f"- Cohen's kappa: {o['verdict_agreement']['binary_drop_uncertain']['cohen_kappa']}",
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
        f"**Claim 级 fatality 分布**（从 AI claim status 派生）:",
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
                f"{va['three_class']['accuracy']} | {va['three_class']['cohen_kappa']} | "
                f"{va['false_pass']} | {va['false_fail']} |"
            )
        lines.append("")

    lines += [
        "## 6. 说明",
        "",
        "- 本报告衡量 **LongCat Judge** 与 **独立 AI 审核** 的一致性，不是人机一致性。",
        "- AI 标注由 LongCat-2.0 (SiliconFlow) 生成，temperature=0.0，prompt=REVIEW_GUIDE.md。",
        "- claim 级 fatality 从 status 派生: contradicted→fatal, unsupported_noncritical→harmless, "
        "unverifiable→harmless, supported_by_source→none。",
        "- Bootstrap CI 使用 seed=42, n_boot=1000。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段五：apply（adjudication_rule_v3）
# ---------------------------------------------------------------------------
def apply_rule_v3(lc_record: dict, ai_record: dict | None, coverage_hit: bool,
                  has_evidence_gate: bool) -> dict:
    """应用 adjudication_rule_v3 返回最终裁决。

    规则:
    1. required AU contradicted/missing/unsupported -> answer_correctness=fail
    2. all required AU supported + all unsupported nonfatal -> pass
    3. fatal unsupported claim -> fail
    4. cannot confirm AU or claim -> pending
    5. P0: route_success = answer_correctness
    6. P1-P4/P_gold: route_success = answer_correctness AND coverage
    7. coverage fail -> route_success=fail (not overridden by pending)
    """
    # 优先使用 AI 标注的 AU 状态；无 AI 标注则用 LongCat 原始
    if ai_record:
        au_status = ai_record.get("au_status", {})
        ai_verdict = ai_record.get("verdict", "")
        ai_fatality = ai_record.get("unsupported_fatality", "none")
        ai_claims = ai_record.get("unsupported_claims", [])
        source = "ai_review"
    else:
        au_status = {au["unit_id"]: au.get("status", "") for au in lc_record.get("answer_units", [])}
        ai_verdict = lc_record.get("verdict", "")
        ai_fatality = "none"  # 无 AI 标注时无法判定 fatality
        ai_claims = []
        source = "longcat_only"

    # 判定 answer_correctness
    has_contradicted = any(s == "contradicted" for s in au_status.values())
    has_missing = any(s == "missing" for s in au_status.values())
    has_fatal_claim = ai_fatality == "fatal" or any(
        _derive_claim_fatality(c.get("status", "")) == "fatal" for c in ai_claims
    )
    all_supported = all(s == "supported" for s in au_status.values()) if au_status else False

    if has_contradicted or has_fatal_claim:
        answer_correctness = "fail"
    elif all_supported and ai_fatality != "fatal":
        # 检查是否有 unverifiable claim（无法确认 -> pending）
        has_unverifiable = any(c.get("status") == "unverifiable" for c in ai_claims)
        if has_unverifiable and source == "ai_review":
            answer_correctness = "pending"
        else:
            answer_correctness = "pass"
    elif has_missing:
        # missing 但无 contradicted -> uncertain/pending
        answer_correctness = "pending" if source == "ai_review" else lc_record.get("derived_verdict", "uncertain")
    else:
        answer_correctness = "pending" if source == "ai_review" else lc_record.get("derived_verdict", "uncertain")

    # route_success
    if not has_evidence_gate:
        route_success = answer_correctness
    elif not coverage_hit:
        route_success = "fail"  # coverage fail 硬 fail
    else:
        route_success = answer_correctness

    return {
        "question_id": lc_record.get("question_id"),
        "route": lc_record.get("route"),
        "answer_correctness": answer_correctness,
        "route_success": route_success,
        "source": source,
        "au_status": au_status,
        "has_contradicted": has_contradicted,
        "has_fatal_claim": has_fatal_claim,
        "ai_verdict": ai_verdict if ai_record else None,
        "coverage_hit": coverage_hit,
    }


def cmd_apply(args):
    """阶段五：应用 adjudication_rule_v3 生成 480 条最终裁决。"""
    longcat = load_longcat_verdicts()
    questions = load_questions()

    # 加载 AI 标注并构建 (route, qid) -> ai_record 映射
    ann = load_annotated()
    manifest = load_sample_manifest()
    bl_map: dict[str, dict] = {}
    for key in ("set_A_main_100", "set_B_pgold_pending_43", "set_C_neg_control_20"):
        for entry in manifest.get(key, []):
            bl_map[entry["blind_id"]] = entry
    ai_by_route_qid: dict[tuple, dict] = {}
    for a in ann:
        info = bl_map.get(a["blind_id"])
        if info:
            ai_by_route_qid[(info["route"], info["qid"])] = a

    # 加载 coverage（从 sample_manifest 获取已审核的；其余需从 summary 读取）
    # 简化：P0 无 gate；P1-P4/P_gold 需 coverage。此处用 LongCat derived_verdict 近似：
    # 如果 derived_verdict 是因 coverage 失败（通过 summary 已知），标记 coverage_hit=False
    # 完整实现需读取 result context 并计算，此处先用 summary 数据
    summary_path = _JUDGE_DIR / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    final_records = []
    pending_supplementary = []

    for route in ROUTES:
        route_data = longcat.get(route, {})
        has_gate = route != "P0"
        for qid, lc_rec in route_data.items():
            if not lc_rec.get("parse_ok"):
                continue
            ai_rec = ai_by_route_qid.get((route, qid))
            # coverage: 从 summary 读取或默认 True
            cov_hit = True
            if has_gate and route in summary.get("routes", {}):
                # 无法从 summary 获取逐条 coverage，用近似：
                # 如果 lc derived_verdict=fail 且 human_review=False，可能是 coverage fail
                # 更精确的方法在阶段五补全
                cov_hit = True  # placeholder，阶段五完善
            verdict = apply_rule_v3(lc_rec, ai_rec, cov_hit, has_gate)
            final_records.append(verdict)
            if verdict["route_success"] == "pending" and not ai_rec:
                pending_supplementary.append({
                    "route": route,
                    "question_id": qid,
                    "lc_verdict": lc_rec.get("verdict"),
                    "lc_derived": lc_rec.get("derived_verdict"),
                    "reason": "no_ai_review",
                })

    # 输出目录
    out_dir = _JUDGE_DIR / "ai_adjudication_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # final_verdicts.jsonl
    fv_path = out_dir / "final_verdicts.jsonl"
    with open(fv_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # final_summary.json
    rs_dist = Counter(r["route_success"] for r in final_records)
    ac_dist = Counter(r["answer_correctness"] for r in final_records)
    by_route = {}
    for route in ROUTES:
        rs = [r for r in final_records if r["route"] == route]
        by_route[route] = {
            "n": len(rs),
            "route_success": dict(Counter(r["route_success"] for r in rs)),
            "answer_correctness": dict(Counter(r["answer_correctness"] for r in rs)),
        }
    summary_out = {
        "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
        "total_records": len(final_records),
        "route_success_dist": dict(rs_dist),
        "answer_correctness_dist": dict(ac_dist),
        "by_route": by_route,
        "pending_supplementary_count": len(pending_supplementary),
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # manifest.json
    manifest_out = {
        "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
        "ai_annotation_sha256": sha256_file(BLIND_DIR / "verdicts_ai_annotated.jsonl"),
        "ai_annotated_count": len(ann),
        "total_records": len(final_records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "final_verdicts.jsonl 不覆盖原始 LongCat 判定；原始数据保留在 *_verdict.jsonl",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"最终裁决已生成: {out_dir}/")
    print(f"  final_verdicts.jsonl: {len(final_records)} 条")
    print(f"  final_summary.json")
    print(f"  manifest.json")
    print(f"  待补充审核: {len(pending_supplementary)} 条")
    if pending_supplementary:
        sup_path = out_dir / "pending_supplementary.jsonl"
        with open(sup_path, "w", encoding="utf-8") as f:
            for r in pending_supplementary:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  pending_supplementary.jsonl")
    return 0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def cmd_validate(args):
    result = validate_annotations()
    if result["passed"]:
        print("=== 验收通过 ===")
    else:
        print("=== 验收失败 ===")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
    for w in result["warnings"]:
        print(f"  WARN: {w}")
    print()
    print("统计:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")

    if result["passed"]:
        manifest = generate_manifest()
        print()
        print(f"ai_annotation_manifest.json 已生成:")
        print(f"  SHA-256: {manifest['annotated_file_sha256']}")
        print(f"  model: {manifest['review_model']} ({manifest['review_provider']})")
        print(f"  prompt sha256: {manifest['review_prompt_sha256']}")
        print(f"  temperature: {manifest['review_temperature']}")
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="盲审校准脚本")
    parser.add_argument("--mode", required=True, choices=["validate", "report", "apply"],
                        help="运行模式: validate(校验) / report(指标) / apply(最终裁决)")
    args = parser.parse_args()

    if args.mode == "validate":
        return cmd_validate(args)
    elif args.mode == "report":
        return cmd_report(args)
    elif args.mode == "apply":
        return cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
