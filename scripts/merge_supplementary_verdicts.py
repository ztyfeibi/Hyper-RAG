# -*- coding: utf-8 -*-
"""Step 3: 合并原盲审标注 + 补充审核（set_D 新审 + set_E 复核）为统一标注全集。

产出（写入 blind_review_supplementary/）：
  - verdicts_ai_all_v1.jsonl        318 条唯一 (route, question_id) 标注
  - verdicts_ai_all_v1_manifest.json 输入哈希 / 计数 / 分布 / E 复核翻转统计

合并语义：
  - 原 163 条（verdicts_ai_annotated.jsonl, blind_id 键）为基线，review_source=original
  - set_E 12 条复核**覆盖**对应 blind_id 的原记录（保留 superseded 溯源），review_source=recheck_E
  - set_D 155 条新增（blind_id=null, review_id 键），review_source=supplementary_D

验收（fail-fast）：
  1. final_verdicts 中 blind_id 集合 == 原 163 条 annotated 集合
  2. sample_manifest set_D 的 (route,qid) == final_verdicts 中 pending 且 blind_id=null 的集合
  3. set_E 的 blind_id 全部存在于原 annotated
  4. 输出 (route,qid) 唯一且 = 163 + 155 = 318
  5. 每条输出 verdict/fatality/claim status 枚举合法
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import run_supplementary_review as rsr  # noqa: E402  (复用 SET_* 常量与枚举)

BASE = _ROOT / "caches/neurology_chunk1000/question_set_v2/pilot_v1/judge/longcat/r0_s42_5c92f17c"
SUP_DIR = rsr.SUP_DIR
ANN = BASE / "blind_review/verdicts_ai_annotated.jsonl"
FINAL = BASE / "ai_adjudication_v1/final_verdicts.jsonl"
SM = SUP_DIR / "sample_manifest.json"
OUT = SUP_DIR / "verdicts_ai_all_v1.jsonl"
OUT_MANIFEST = SUP_DIR / "verdicts_ai_all_v1_manifest.json"

ANNOTATION_FIELDS = ("au_status", "verdict", "unsupported_fatality",
                     "unsupported_claims", "notes")


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def validate_annotation(rec: dict, ctx: str) -> list:
    """枚举级校验（merge 阶段复核，写入时已强校验过）。"""
    errs = []
    if rec.get("verdict") not in rsr.VERDICT_OPTIONS:
        errs.append(f"{ctx}: verdict 非法 {rec.get('verdict')!r}")
    if rec.get("unsupported_fatality") not in rsr.FATALITY_OPTIONS:
        errs.append(f"{ctx}: fatality 非法 {rec.get('unsupported_fatality')!r}")
    au = rec.get("au_status")
    if not isinstance(au, dict) or not au:
        errs.append(f"{ctx}: au_status 缺失/非法")
    else:
        bad = [f"{k}={v}" for k, v in au.items() if v not in rsr.AU_STATUS_OPTIONS]
        if bad:
            errs.append(f"{ctx}: au_status 非法取值 {bad}")
    claims = rec.get("unsupported_claims")
    if not isinstance(claims, list):
        errs.append(f"{ctx}: unsupported_claims 不是列表")
    else:
        for c in claims:
            if not isinstance(c, dict) or not c.get("claim"):
                errs.append(f"{ctx}: claim 结构非法 {str(c)[:60]}")
            elif c.get("status") not in rsr.CLAIM_STATUS_OPTIONS:
                errs.append(f"{ctx}: claim status 非法 {c.get('status')!r}")
    return errs


def merge(base_dir: Path = BASE, write: bool = True) -> dict:
    ann = load_jsonl(ANN)
    fv = load_jsonl(FINAL)
    sm = json.loads(SM.read_text(encoding="utf-8"))
    rec_d = {r["review_id"]: r for r in load_jsonl(rsr.SET_OUTPUT["D"])}
    rec_e = {r["review_id"]: r for r in load_jsonl(rsr.SET_OUTPUT["E"])}

    errors: list[str] = []

    # --- 验收 1: blind_id 集合对齐 ---
    bid_to_route_qid = {}
    for r in fv:
        if r.get("blind_id"):
            key = (r["route"], r["question_id"])
            if r["blind_id"] in bid_to_route_qid:
                errors.append(f"final_verdicts 重复 blind_id: {r['blind_id']}")
            bid_to_route_qid[r["blind_id"]] = key
    ann_bids = {r["blind_id"] for r in ann}
    if ann_bids != set(bid_to_route_qid):
        errors.append(f"blind_id 集合不一致: annotated-only={sorted(ann_bids - set(bid_to_route_qid))[:5]} "
                      f"fv-only={sorted(set(bid_to_route_qid) - ann_bids)[:5]}")

    # --- 验收 2: set_D == pending 无 blind_id 集合 ---
    pending_nobid = {(r["route"], r["question_id"]) for r in fv
                     if r.get("route_success") == "pending" and not r.get("blind_id")}
    d_meta = {e["review_id"]: e for e in sm["set_D_unreviewed_155"]}
    d_keys = {(e["route"], e["question_id"]) for e in d_meta.values()}
    if d_keys != pending_nobid:
        errors.append(f"set_D keys != pending-no-bid keys "
                      f"(D-only={sorted(d_keys - pending_nobid)[:3]}, "
                      f"pending-only={sorted(pending_nobid - d_keys)[:3]})")
    if set(d_meta) != set(rec_d):
        errors.append(f"set_D 记录缺失: {sorted(set(d_meta) - set(rec_d))[:5]}")

    # --- 验收 3: set_E blind_id 都在原 annotated ---
    e_meta = {e["review_id"]: e for e in sm["set_E_recheck_12"]}
    if set(e_meta) != set(rec_e):
        errors.append(f"set_E 记录缺失: {sorted(set(e_meta) - set(rec_e))[:5]}")
    for rid, e in e_meta.items():
        if e.get("blind_id") not in ann_bids:
            errors.append(f"{rid}: blind_id {e.get('blind_id')!r} 不在原 annotated")

    if errors:
        raise SystemExit("输入一致性校验失败:\n  " + "\n  ".join(errors))

    # --- 合并 ---
    ann_by_bid = {r["blind_id"]: r for r in ann}
    e_by_bid = {e["blind_id"]: (rid, rec_e[rid]) for rid, e in e_meta.items()}
    merged: list[dict] = []
    e_flips = {"verdict": [], "fatality": []}

    for r in ann:  # 保持原顺序
        bid = r["blind_id"]
        route, qid = bid_to_route_qid[bid]
        out = {"route": route, "question_id": qid, "blind_id": bid,
               "review_id": None, "review_source": "original",
               "response_schema": None, "superseded": None}
        if bid in e_by_bid:
            rid, erec = e_by_bid[bid]
            out.update({k: erec[k] for k in ANNOTATION_FIELDS})
            out["review_source"] = "recheck_E"
            out["review_id"] = rid
            out["response_schema"] = erec.get("response_schema")
            out["superseded"] = {"original_verdict": r["verdict"],
                                 "original_fatality": r["unsupported_fatality"],
                                 "original_review": "blind_review/verdicts_ai_annotated.jsonl"}
            if r["verdict"] != erec["verdict"]:
                e_flips["verdict"].append(f"{bid}:{r['verdict']}->{erec['verdict']}")
            if r["unsupported_fatality"] != erec["unsupported_fatality"]:
                e_flips["fatality"].append(
                    f"{bid}:{r['unsupported_fatality']}->{erec['unsupported_fatality']}")
        else:
            out.update({k: r[k] for k in ANNOTATION_FIELDS})
        merged.append(out)

    for rid, e in sorted(d_meta.items()):  # SR-D-001..155
        d = rec_d[rid]
        out = {"route": e["route"], "question_id": e["question_id"], "blind_id": None,
               "review_id": rid, "review_source": "supplementary_D",
               "response_schema": d.get("response_schema"), "superseded": None}
        out.update({k: d[k] for k in ANNOTATION_FIELDS})
        merged.append(out)

    # --- 验收 4: 唯一性 + 总数 ---
    keys = [(m["route"], m["question_id"]) for m in merged]
    if len(keys) != len(set(keys)):
        dupes = [k for k, n in Counter(keys).items() if n > 1]
        raise SystemExit(f"输出 (route,qid) 重复: {dupes[:5]}")
    expected = len(ann) + len(d_meta)
    if len(merged) != expected:
        raise SystemExit(f"输出条数 {len(merged)} != 预期 {expected}")

    # --- 验收 5: 枚举复核 ---
    for m in merged:
        errs = validate_annotation(m, f"{m['route']}/{m['question_id']}")
        if errs:
            raise SystemExit("输出校验失败:\n  " + "\n  ".join(errs))

    stats = {
        "total": len(merged),
        "by_source": dict(Counter(m["review_source"] for m in merged)),
        "verdict_dist": dict(Counter(m["verdict"] for m in merged)),
        "fatality_dist": dict(Counter(m["unsupported_fatality"] for m in merged)),
        "verdict_dist_by_source": {
            s: dict(Counter(m["verdict"] for m in merged if m["review_source"] == s))
            for s in ("original", "recheck_E", "supplementary_D")},
        "e_recheck_flips": e_flips,
        "unreviewed_remaining": len(fv) - len(merged),
    }

    manifest = {
        "annotation_type": "merged_ai_annotations_v1",
        "inputs": {
            "verdicts_ai_annotated": {"path": _rel_or_abs(ANN),
                                      "n": len(ann), "sha256": sha256_file(ANN)},
            "final_verdicts": {"path": _rel_or_abs(FINAL),
                               "n": len(fv), "sha256": sha256_file(FINAL)},
            "verdicts_ai_supplementary_D": {"n": len(rec_d),
                                            "sha256": sha256_file(rsr.SET_OUTPUT["D"])},
            "verdicts_ai_recheck_E": {"n": len(rec_e),
                                      "sha256": sha256_file(rsr.SET_OUTPUT["E"])},
            "sample_manifest": {"sha256": sha256_file(SM)},
        },
        "review_model_lineage": {
            "original_163": "见 blind_review/ai_annotation_manifest.json",
            "supplementary": "见 blind_review_supplementary/review_metadata.json",
        },
        "merge_rule": "recheck_E overrides blind_id record (superseded kept); "
                      "supplementary_D appended as new (route,question_id)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stats": stats,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if write:
        OUT.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in merged),
                       encoding="utf-8")
        OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Step 3: 合并补充审核为 verdicts_ai_all_v1.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="只校验不写盘")
    args = ap.parse_args()
    manifest = merge(write=not args.dry_run)
    s = manifest["stats"]
    print(f"合并完成: {s['total']} 条 = {s['by_source']}")
    print(f"verdict: {s['verdict_dist']}")
    print(f"fatality: {s['fatality_dist']}")
    print(f"E 复核翻转: verdict {len(s['e_recheck_flips']['verdict'])} 条, "
          f"fatality {len(s['e_recheck_flips']['fatality'])} 条")
    print(f"未审核剩余: {s['unreviewed_remaining']} (480 - {s['total']})")
    if not args.dry_run:
        print(f"输出: {OUT}")
        print(f"清单: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
