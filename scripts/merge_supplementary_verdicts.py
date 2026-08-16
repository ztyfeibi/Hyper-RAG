# -*- coding: utf-8 -*-
"""Step 3: 合并 AI 标注 + 补充审核（set_D 新审 + set_E 复核）为统一标注全集。

r0 模式（--base-annotation original163，默认 repeat=0）：
  - 原 163 条（verdicts_ai_annotated.jsonl, blind_id 键）为基线，review_source=original
  - set_E 12 条复核**覆盖**对应 blind_id 的原记录（保留 superseded 溯源），review_source=recheck_E
  - set_D 155 条新增（blind_id=null, review_id 键），review_source=supplementary_D
  - 产出 verdicts_ai_all_v1.jsonl（318 条）

r1+ 模式（--base-annotation none，新 repeat 不读原 163 标注）：
  - final_verdicts 来源 ai_adjudication_pre/（--no-annotations apply 产物）
  - set_D 为基线（review_id 键 R{repeat}-D-xxx），review_source=supplementary_D
  - set_E 复核**覆盖**对应 (route, question_id) 的 set_D 记录（E 覆盖 D，不重复计数）
  - 产出 verdicts_ai_all_r{repeat}.jsonl（条数 = len(set_D)）

验收（fail-fast）：
  r0:
    1. final_verdicts 中 blind_id 集合 == 原 163 条 annotated 集合
    2. sample_manifest set_D 的 (route,qid) == final_verdicts 中 pending 且 blind_id=null 的集合
    3. set_E 的 blind_id 全部存在于原 annotated
    4. 输出 (route,qid) 唯一且 = 163 + |set_D|
  r1+:
    1. set_D 的 (route,qid) == final_verdicts(pre) 中 pending 且无 AI 审核的集合
    2. set_E 的 (route,qid) ⊆ set_D（E 是 D 的复核，覆盖不重复计数）
  所有 repeat:
    3. 输出 (route,qid) 唯一
    4. 每条输出 verdict/fatality/claim status 枚举合法
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
from repeat_context import RepeatContext, DEFAULT_RC  # noqa: E402

_RC: RepeatContext = DEFAULT_RC
BASE = _ROOT / "caches/neurology_chunk1000/question_set_v2/pilot_v1/judge/longcat/r0_s42_5c92f17c"
SUP_DIR = rsr.SUP_DIR
ANN = BASE / "blind_review/verdicts_ai_annotated.jsonl"
FINAL = BASE / "ai_adjudication_v1/final_verdicts.jsonl"
SM = SUP_DIR / "sample_manifest.json"
OUT = SUP_DIR / "verdicts_ai_all_v1.jsonl"
OUT_MANIFEST = SUP_DIR / "verdicts_ai_all_v1_manifest.json"

ANNOTATION_FIELDS = ("au_status", "verdict", "unsupported_fatality",
                     "unsupported_claims", "notes")


def set_context(rc: RepeatContext) -> None:
    """切换活跃 RepeatContext（CLI main 调用；测试可重置）.

    r0: ANN/FINAL 指向原 163 标注 + ai_adjudication_v1；
    r1+: ANN 不适用（置 None），FINAL 指向 ai_adjudication_pre。
    """
    global _RC, BASE, SUP_DIR, ANN, FINAL, SM, OUT, OUT_MANIFEST
    _RC = rc
    rsr.set_context(rc)
    BASE = rc.judge_dir
    SUP_DIR = rc.supplementary_dir
    if rc.repeat == 0:
        ANN = BASE / "blind_review/verdicts_ai_annotated.jsonl"
        FINAL = BASE / "ai_adjudication_v1/final_verdicts.jsonl"
        OUT = SUP_DIR / "verdicts_ai_all_v1.jsonl"
        OUT_MANIFEST = SUP_DIR / "verdicts_ai_all_v1_manifest.json"
    else:
        ANN = None  # r1+ 无原 163 标注，禁止读取
        FINAL = BASE / "ai_adjudication_pre/final_verdicts.jsonl"
        OUT = SUP_DIR / f"verdicts_ai_all_r{rc.repeat}.jsonl"
        OUT_MANIFEST = SUP_DIR / f"verdicts_ai_all_r{rc.repeat}_manifest.json"
    SM = SUP_DIR / "sample_manifest.json"


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _manifest_key(sm: dict, prefix: str) -> str:
    """前缀查找动态键（set_D_unreviewed_{N} / set_E_recheck_{N}）."""
    matches = [k for k in sm if k.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"sample_manifest 需要恰好 1 个 {prefix}* 键，实际 {matches}")
    return matches[0]


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


def merge(base_dir: Path = None, write: bool = True,
          base_annotation: str | None = None,
          supplementary_d: Path | None = None,
          recheck_e: Path | None = None,
          output: Path | None = None) -> dict:
    """合并统一标注全集。

    base_annotation: original163（r0，原 163 条为基线） / none（r1+，set_D 为基线）。
                     默认按 _RC.repeat 推断（r0→original163，r1+→none）。
    supplementary_d / recheck_e / output: 路径覆盖（默认 rsr.SET_OUTPUT / 模块 OUT）。
    """
    base_mode = base_annotation or ("original163" if _RC.repeat == 0 else "none")
    if base_mode not in ("original163", "none"):
        raise SystemExit(f"未知 --base-annotation: {base_mode}")
    if base_mode == "original163" and _RC.repeat != 0:
        raise SystemExit("--base-annotation original163 仅限 r0（新 repeat 禁止读原 163 标注）")
    if base_mode == "none" and _RC.repeat == 0 and base_annotation == "none":
        raise SystemExit("r0 必须以原 163 标注为基线（repeat-aware 契约）")

    fv = load_jsonl(FINAL)
    sm = json.loads(SM.read_text(encoding="utf-8"))
    d_path = Path(supplementary_d) if supplementary_d else rsr.SET_OUTPUT["D"]
    e_path = Path(recheck_e) if recheck_e else rsr.SET_OUTPUT.get("E")
    out_path = Path(output) if output else OUT

    if not d_path.exists():
        raise SystemExit(f"set_D 审核输出不存在: {d_path}")
    rec_d = {r["review_id"]: r for r in load_jsonl(d_path)}
    # r1+ 允许无 E 复核轮（第一次审核无 uncertain 时）；r0 必须有
    if e_path is not None and e_path.exists():
        rec_e = {r["review_id"]: r for r in load_jsonl(e_path)}
    elif _RC.repeat == 0:
        raise SystemExit(f"set_E 审核输出不存在: {e_path}")
    else:
        rec_e = {}

    d_key_name = _manifest_key(sm, "set_D_unreviewed_")
    e_key_name = _manifest_key(sm, "set_E_recheck_")
    d_meta = {e["review_id"]: e for e in sm[d_key_name]}
    e_meta = {e["review_id"]: e for e in sm[e_key_name]}

    errors: list[str] = []

    # --- 公共验收: set_D 与 final_verdicts pending 无 AI 审核集合对齐 ---
    pending_nobid = {(r["route"], r["question_id"]) for r in fv
                     if r.get("route_success") == "pending" and not r.get("blind_id")}
    d_keys = {(e["route"], e["question_id"]) for e in d_meta.values()}
    if d_keys != pending_nobid:
        errors.append(f"set_D keys != pending-no-bid keys "
                      f"(D-only={sorted(d_keys - pending_nobid)[:3]}, "
                      f"pending-only={sorted(pending_nobid - d_keys)[:3]})")
    if set(d_meta) != set(rec_d):
        errors.append(f"set_D 记录缺失: {sorted(set(d_meta) - set(rec_d))[:5]}")
    if set(e_meta) != set(rec_e):
        errors.append(f"set_E 记录缺失: {sorted(set(e_meta) - set(rec_e))[:5]}")

    ann = None
    merged: list[dict] = []
    e_flips = {"verdict": [], "fatality": []}

    if base_mode == "original163":
        # ------- r0: 原 163 条为基线 -------
        ann = load_jsonl(ANN)
        bid_to_route_qid = {}
        for r in fv:
            if r.get("blind_id"):
                key = (r["route"], r["question_id"])
                if r["blind_id"] in bid_to_route_qid:
                    errors.append(f"final_verdicts 重复 blind_id: {r['blind_id']}")
                bid_to_route_qid[r["blind_id"]] = key
        ann_bids = {r["blind_id"] for r in ann}
        if ann_bids != set(bid_to_route_qid):
            errors.append(f"blind_id 集合不一致: "
                          f"annotated-only={sorted(ann_bids - set(bid_to_route_qid))[:5]} "
                          f"fv-only={sorted(set(bid_to_route_qid) - ann_bids)[:5]}")
        for rid, e in e_meta.items():
            if e.get("blind_id") not in ann_bids:
                errors.append(f"{rid}: blind_id {e.get('blind_id')!r} 不在原 annotated")

        if errors:
            raise SystemExit("输入一致性校验失败:\n  " + "\n  ".join(errors))

        ann_by_bid = {r["blind_id"]: r for r in ann}
        e_by_bid = {e["blind_id"]: (rid, rec_e[rid]) for rid, e in e_meta.items()}
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

        for rid, e in sorted(d_meta.items()):
            d = rec_d[rid]
            out = {"route": e["route"], "question_id": e["question_id"], "blind_id": None,
                   "review_id": rid, "review_source": "supplementary_D",
                   "response_schema": d.get("response_schema"), "superseded": None}
            out.update({k: d[k] for k in ANNOTATION_FIELDS})
            merged.append(out)

        expected = len(ann) + len(d_meta)
    else:
        # ------- r1+: set_D 为基线，E 覆盖 D（不重复计数）-------
        e_by_route_qid = {}
        for rid, e in e_meta.items():
            k = (e["route"], e["question_id"])
            if k in e_by_route_qid:
                errors.append(f"set_E 重复 (route,qid): {k}")
            e_by_route_qid[k] = (rid, e)
        # E 必须是 D 的复核子集
        d_by_key = {(e["route"], e["question_id"]): rid for rid, e in d_meta.items()}
        for k in e_by_route_qid:
            if k not in d_by_key:
                errors.append(f"set_E 复核了不在 set_D 中的记录: {k}")

        if errors:
            raise SystemExit("输入一致性校验失败:\n  " + "\n  ".join(errors))

        for rid, e in sorted(d_meta.items()):
            d = rec_d[rid]
            k = (e["route"], e["question_id"])
            out = {"route": e["route"], "question_id": e["question_id"], "blind_id": None,
                   "review_id": rid, "review_source": "supplementary_D",
                   "response_schema": d.get("response_schema"), "superseded": None}
            out.update({k: d[k] for k in ANNOTATION_FIELDS})
            if k in e_by_route_qid:
                erid, _ = e_by_route_qid[k]
                erec = rec_e[erid]
                out.update({k2: erec[k2] for k2 in ANNOTATION_FIELDS})
                out["review_source"] = "recheck_E"
                out["review_id"] = erid
                out["response_schema"] = erec.get("response_schema")
                out["superseded"] = {"original_verdict": d["verdict"],
                                     "original_fatality": d["unsupported_fatality"],
                                     "original_review": "verdicts_ai_supplementary_D.jsonl"}
                if d["verdict"] != erec["verdict"]:
                    e_flips["verdict"].append(f"{erid}:{d['verdict']}->{erec['verdict']}")
                if d["unsupported_fatality"] != erec["unsupported_fatality"]:
                    e_flips["fatality"].append(
                        f"{erid}:{d['unsupported_fatality']}->{erec['unsupported_fatality']}")
            merged.append(out)

        expected = len(d_meta)

    # --- 输出唯一性 + 总数 ---
    keys = [(m["route"], m["question_id"]) for m in merged]
    if len(keys) != len(set(keys)):
        dupes = [k for k, n in Counter(keys).items() if n > 1]
        raise SystemExit(f"输出 (route,qid) 重复: {dupes[:5]}")
    if len(merged) != expected:
        raise SystemExit(f"输出条数 {len(merged)} != 预期 {expected}")

    # --- 枚举复核 ---
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
            for s in set(m["review_source"] for m in merged)},
        "e_recheck_flips": e_flips,
        "unreviewed_remaining": len(fv) - len(merged),
    }

    inputs = {
        "final_verdicts": {"path": _rel_or_abs(FINAL),
                           "n": len(fv), "sha256": sha256_file(FINAL)},
        "verdicts_ai_supplementary_D": {"n": len(rec_d),
                                        "sha256": sha256_file(d_path)},
        "sample_manifest": {"sha256": sha256_file(SM)},
    }
    if e_path is not None and e_path.exists():
        inputs["verdicts_ai_recheck_E"] = {"n": len(rec_e), "sha256": sha256_file(e_path)}
    if ann is not None:
        inputs["verdicts_ai_annotated"] = {"path": _rel_or_abs(ANN),
                                           "n": len(ann), "sha256": sha256_file(ANN)}

    manifest = {
        "annotation_type": (f"merged_ai_annotations_r{_RC.repeat}"
                            if _RC.repeat > 0 else "merged_ai_annotations_v1"),
        "repeat": _RC.repeat,
        "seed": _RC.seed,
        "snapshot": _RC.snapshot,
        "base_annotation": base_mode,
        "inputs": inputs,
        "review_model_lineage": {
            "original_163": ("r0: 见 blind_review/ai_annotation_manifest.json"
                             if ann is not None else "none（r1+ 不使用原 163 标注）"),
            "supplementary": "见 blind_review_supplementary/review_metadata.json",
        },
        "merge_rule": ("recheck_E overrides blind_id record (superseded kept); "
                       "supplementary_D appended as new (route,question_id)"
                       if base_mode == "original163" else
                       "set_D as baseline; recheck_E overrides matching set_D "
                       "(route,question_id) (superseded kept, no double counting)"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stats": stats,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in merged),
            encoding="utf-8")
        out_manifest = (out_path.parent / (out_path.stem + "_manifest.json"))
        out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Step 3: 合并补充审核为统一标注全集")
    ap.add_argument("--data-name", default="neurology_chunk1000")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot", default=None,
                    help="系统快照 ID（默认 jl.SNAPSHOT_DEFAULT）")
    ap.add_argument("--base-annotation", choices=["original163", "none"], default=None,
                    help="original163=r0 原 163 条为基线；none=r1+ set_D 为基线"
                         "（默认按 repeat 推断）")
    ap.add_argument("--supplementary-D", default=None,
                    help="set_D 审核输出路径覆盖（默认 rsr.SET_OUTPUT['D']）")
    ap.add_argument("--recheck-E", default=None,
                    help="set_E 复核输出路径覆盖（默认 rsr.SET_OUTPUT['E']）")
    ap.add_argument("--output", default=None,
                    help="输出路径覆盖（默认 verdicts_ai_all_v1.jsonl / "
                         f"verdicts_ai_all_r{{repeat}}.jsonl）")
    ap.add_argument("--dry-run", action="store_true", help="只校验不写盘")
    args = ap.parse_args()

    rc = RepeatContext(
        data_name=args.data_name,
        repeat=args.repeat,
        seed=args.seed,
        snapshot=args.snapshot or rsr.bsr.jl.SNAPSHOT_DEFAULT,
    )
    set_context(rc)

    manifest = merge(write=not args.dry_run,
                     base_annotation=args.base_annotation,
                     supplementary_d=Path(args.supplementary_D) if args.supplementary_D else None,
                     recheck_e=Path(args.recheck_E) if args.recheck_E else None,
                     output=Path(args.output) if args.output else None)
    s = manifest["stats"]
    print(f"合并完成 (repeat={_RC.repeat}, base={manifest['base_annotation']}): "
          f"{s['total']} 条 = {s['by_source']}")
    print(f"verdict: {s['verdict_dist']}")
    print(f"fatality: {s['fatality_dist']}")
    print(f"E 复核翻转: verdict {len(s['e_recheck_flips']['verdict'])} 条, "
          f"fatality {len(s['e_recheck_flips']['fatality'])} 条")
    print(f"未审核剩余: {s['unreviewed_remaining']} "
          f"({manifest['inputs']['final_verdicts']['n']} - {s['total']})")
    if not args.dry_run:
        print(f"输出: {OUT if not args.output else args.output}")


if __name__ == "__main__":
    main()
