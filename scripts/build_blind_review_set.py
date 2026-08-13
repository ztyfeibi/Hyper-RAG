"""步骤 4：盲审校准样本生成（§12.2 分层抽样，契约问题集 v2.1-v1）。

从 r0_s42_5c92f17c 的 480 条 Judge 原始判定中生成两套审核材料：

- set_A：主集 100 条，P0-P4 每路径 20 条；路径内按 verdict（pass/fail/uncertain）
  分层，强制覆盖 单/多 AU、证据完整/缺失（source_evidence_coverage pass/fail）、
  unsupported 有/无、contradicted AU。
- set_B：P_gold 专项 43 条 pending（契约 §12.1：全部进人工复核，一条不漏）。

盲审约束（§12.2）：
- 材料**不暴露路径名**：审核 ID 用 BL-xxx，qid/route 映射只写在内部 manifest。
- 人工审核者对照 answer units + 精确 evidence spans 做 source-grounded 判定，
  材料中不预填 Judge 的 verdict（避免锚定）。
- 每条附人工判定栏：AU 状态（supported/missing/contradicted）+ 整体 verdict
  （pass/fail/uncertain）+ unsupported claims 致命性（harmless/fatal）。

纯本地确定性操作，不调用 LLM。抽样 seed=42 可复现。
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import judge_longcat as j  # noqa: E402

OUT_DIR = j.out_dir(0, 42, j.SNAPSHOT_DEFAULT)
BLIND_DIR = OUT_DIR / "blind_review"

SEED = 42


def load_questions_with_spans():
    """qid -> {question, answer_units, evidence_spans_by_au}"""
    out = {}
    for line in open(j.QUESTIONS_FILE, encoding="utf-8"):
        r = json.loads(line)
        spans_by_au = {es["unit_id"]: es.get("evidence_spans", [])
                       for es in r.get("evidence_spans", [])}
        out[r["question_id"]] = {
            "question": r["question"],
            "answer_units": r.get("answer_units", []),
            "evidence_spans": spans_by_au,
        }
    return out


def load_candidate_answers():
    """route -> qid -> candidate answer 文本（result.jsonl 的 result 字段）。"""
    out = {}
    for route in j.ROUTES:
        rf = j.result_file(route, 0, 42, j.SNAPSHOT_DEFAULT)
        if not rf.exists():
            continue
        out[route] = {r["question_id"]: r.get("result") or ""
                      for r in (json.loads(l) for l in open(rf, encoding="utf-8"))}
    return out


def compute_coverage(route, qid, question_meta, evidence_by_qid):
    """source_evidence_coverage 命中（与 judge_longcat summary 同规则）。"""
    if route == "P0":
        return True  # 无证据门槛
    rf = j.result_file(route, 0, 42, j.SNAPSHOT_DEFAULT)
    ctx = None
    for line in open(rf, encoding="utf-8"):
        r = json.loads(line)
        if r["question_id"] == qid:
            ctx = r.get("context") or ""
            break
    if ctx is None:
        return False
    required = {u["unit_id"] for u in question_meta["answer_units"] if u.get("required", True)}
    _, all_hit = j.calc_source_evidence_coverage(
        ctx, evidence_by_qid.get(qid, {}), required,
        search="sources" if route in ("P2", "P3", "P4") else "full")
    return all_hit


def describe_au_counts(n_au):
    return "multi" if n_au > 1 else "single"


def collect_records(questions, candidates, evidence_by_qid):
    """收集全部 parse_ok 记录并附加分层特征。"""
    recs = []
    for route in j.ROUTES:
        out = OUT_DIR / f"{route}_verdict.jsonl"
        if not out.exists():
            continue
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("parse_ok"):
                continue
            qid = r["question_id"]
            meta = questions[qid]
            aus = r.get("answer_units") or []
            statuses = [au.get("status") for au in aus]
            recs.append({
                "route": route, "qid": qid,
                "verdict": r.get("verdict"),
                "derived": r.get("derived_verdict"),
                "human_review": bool(r.get("human_review")),
                "hr_reasons": r.get("human_review_reason") or [],
                "n_units": len(meta["answer_units"]),
                "au_shape": describe_au_counts(len(meta["answer_units"])),
                "has_contradicted": "contradicted" in statuses,
                "has_missing": "missing" in statuses,
                "n_unsupported": len(r.get("unsupported_claims") or []),
                "cov_hit": compute_coverage(route, qid, meta, evidence_by_qid),
                "candidate": (candidates.get(route) or {}).get(qid, ""),
            })
    return recs


def stratify_sample(recs, rng, per_route=20, min_per_verdict=2):
    """主集分层抽样：P0-P4 每路径 per_route 条，路径内覆盖 verdict/AU 数/cov/unsupported。

    注意：同一 qid 在六路径都出现（同一题集），去重键必须是 (route, qid)。
    """
    picked, picked_keys = [], set()
    for route in ["P0", "P1", "P2", "P3", "P4"]:
        pool = [r for r in recs if r["route"] == route]
        route_pick = []
        # 1) 每 verdict 至少 min_per_verdict 条（若该 verdict 存在）
        for v in ("pass", "fail", "uncertain"):
            vpool = [r for r in pool if r["verdict"] == v
                     and (r["route"], r["qid"]) not in picked_keys]
            take = min(min_per_verdict, len(vpool))
            route_pick += rng.sample(vpool, take)
        picked_keys |= {(r["route"], r["qid"]) for r in route_pick}
        # 2) 补齐到 per_route：优先未选中的记录，保持 verdict 分布
        rest = [r for r in pool if (r["route"], r["qid"]) not in picked_keys]
        rng.shuffle(rest)
        need = per_route - len(route_pick)
        route_pick += rest[:need]
        picked_keys |= {(r["route"], r["qid"]) for r in route_pick}
        picked.extend(route_pick)
    # 3) 强制特征覆盖检查（不足则替换同路径低价值样本——保持 per_route 不变）
    ensure_coverage(picked, recs, rng, picked_keys)
    return picked


def ensure_coverage(picked, recs, rng, picked_keys):
    """强制覆盖：多 AU / 单 AU / cov fail / unsupported 有 / contradicted。"""
    targets = {
        "multi": 20, "single": 20, "cov_fail": 25,
        "unsupported": 30, "contradicted": 5,
    }
    for feat, need in targets.items():
        have = sum(1 for r in picked if feat_of(r, feat))
        missing = need - have
        if missing <= 0:
            continue
        # 从 P0-P4 未选记录里补
        cand = [r for r in recs if r["route"] in ("P0", "P1", "P2", "P3", "P4")
                and (r["route"], r["qid"]) not in picked_keys and feat_of(r, feat)]
        rng.shuffle(cand)
        for r in cand[:missing]:
            # 替换一条同路径、无该特征、且不破坏其他关键覆盖的记录
            swap_candidates = [x for x in picked if x["route"] == r["route"]
                               and x["qid"] != r["qid"] and not feat_of(x, feat)]
            if not swap_candidates:
                continue
            victim = rng.choice(swap_candidates)
            picked[picked.index(victim)] = r
            picked_keys.discard((victim["route"], victim["qid"]))
            picked_keys.add((r["route"], r["qid"]))


def feat_of(r, feat):
    return {
        "multi": r["au_shape"] == "multi",
        "single": r["au_shape"] == "single",
        "cov_fail": not r["cov_hit"],
        "unsupported": r["n_unsupported"] > 0,
        "contradicted": r["has_contradicted"],
    }[feat]


def render_record(idx, r, questions, show_judge=False):
    """渲染一条审核材料（不暴露路径名）。"""
    meta = questions[r["qid"]]
    lines = []
    lines.append(f"### BL-{idx:03d}")
    lines.append(f"**Question**: {meta['question']}")
    lines.append("")
    lines.append("**Answer units**（判定基准）:")
    for u in meta["answer_units"]:
        req = "required" if u.get("required", True) else "optional"
        lines.append(f"- {u['unit_id']} ({req}): {u['claim']}")
    lines.append("")
    lines.append("**Evidence spans**（source-grounded 对照基准）:")
    for u in meta["answer_units"]:
        spans = meta["evidence_spans"].get(u["unit_id"], [])
        if not spans:
            lines.append(f"- {u['unit_id']}: (无 evidence span)")
            continue
        for s in spans:
            quote = (s.get("quote") or "").strip()
            lines.append(f"- {u['unit_id']}: \"{quote}\"")
    lines.append("")
    lines.append("**Candidate answer**:")
    lines.append(r["candidate"].strip() or "(空回答)")
    lines.append("")
    if show_judge:
        lines.append("**Judge 原始判定**（仅供事后对比，非审核输入）:")
        lines.append(f"- verdict={r['verdict']}, AU statuses="
                     f"{[(au.get('unit_id'), au.get('status')) for au in (r.get('answer_units') or [])]}")
        lines.append(f"- unsupported_claims={r.get('unsupported_claims') or []}")
        lines.append(f"- critical_error={r.get('critical_error')}, "
                     f"evidence_sufficient={r.get('evidence_sufficient')}")
        lines.append("")
    lines.append("**人工判定栏**（请填写）:")
    lines.append("- AU 状态: " + " ".join(
        f"{u['unit_id']}=[supported|missing|contradicted]" for u in meta["answer_units"]))
    lines.append("- 整体 verdict: [pass|fail|uncertain]")
    lines.append("- unsupported 致命性: [无|全部无害|有致命]（若候选答案含证据外的陈述，"
                 "判断其为无害背景信息还是致命错误）")
    lines.append("- 备注: ")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="生成步骤 4 盲审材料（§12.2 分层抽样）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    questions = load_questions_with_spans()
    candidates = load_candidate_answers()
    recs = collect_records(questions, candidates, j.load_evidence_spans())
    print(f"收集 parse_ok 记录: {len(recs)} 条")

    # ---- set_A：主集 100 条（P0-P4 每路径 20）----
    set_a = stratify_sample(recs, rng)
    assert len(set_a) == 100, f"set_A 应为 100 条，实际 {len(set_a)}"
    # ---- set_B：P_gold 43 条 pending 全量 ----
    set_b = [r for r in recs if r["route"] == "P_gold" and r["human_review"]]
    assert len(set_b) == 43, f"P_gold pending 应为 43，实际 {len(set_b)}"

    BLIND_DIR.mkdir(parents=True, exist_ok=True)

    def write_md(path, rows, start_idx=1, show_judge=False):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 盲审校准材料\n\n"
                    "判定规则（§12.2 source-grounded）：\n"
                    "- supported = 候选答案陈述了该 answer unit 且与证据一致；\n"
                    "- missing = 候选答案未覆盖该 answer unit；\n"
                    "- contradicted = 候选答案与该 answer unit 的证据相矛盾；\n"
                    "- 整体 verdict：全部 required AU supported 且无致命 unsupported -> pass；\n"
                    "  存在 contradicted / 致命 unsupported -> fail；证据不足或不确定 -> uncertain。\n\n")
            for i, r in enumerate(rows, start=start_idx):
                f.write(render_record(i, r, questions, show_judge=show_judge))

    write_md(BLIND_DIR / "set_A_main_100.md", set_a, start_idx=1)
    write_md(BLIND_DIR / "set_B_pgold_pending_43.md", set_b, start_idx=101)

    # ---- manifest：审核 ID <-> qid/route 映射（内部，不随材料分发）----
    manifest = {"seed": args.seed, "generated_at": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()}
    for name, rows in (("set_A_main_100", set_a), ("set_B_pgold_pending_43", set_b)):
        manifest[name] = [{"blind_id": f"BL-{i:03d}",
                           "route": r["route"], "qid": r["qid"],
                           "verdict": r["verdict"], "derived": r["derived"],
                           "human_review": r["human_review"],
                           "hr_reasons": r["hr_reasons"],
                           "au_shape": r["au_shape"],
                           "cov_hit": r["cov_hit"],
                           "n_unsupported": r["n_unsupported"],
                           "has_contradicted": r["has_contradicted"]}
                          for i, r in enumerate(rows, start=101 if "B" in name else 1)]
    (BLIND_DIR / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 分层覆盖统计 ----
    def stats(rows):
        c = Counter()
        for r in rows:
            c["route_" + r["route"]] += 1
            c["verdict_" + r["verdict"]] += 1
            c["au_" + r["au_shape"]] += 1
            c["cov_" + ("pass" if r["cov_hit"] else "fail")] += 1
            c["unsup_" + ("yes" if r["n_unsupported"] else "no")] += 1
            c["contra_" + ("yes" if r["has_contradicted"] else "no")] += 1
            c["hr_" + ("yes" if r["human_review"] else "no")] += 1
        return c

    print("\n=== set_A（主集 100 条）分层覆盖 ===")
    for k, v in sorted(stats(set_a).items()):
        print(f"  {k}: {v}")
    print("\n=== set_B（P_gold pending 43 条）分层覆盖 ===")
    for k, v in sorted(stats(set_b).items()):
        print(f"  {k}: {v}")
    print(f"\n产物: {BLIND_DIR}/")
    print("  set_A_main_100.md / set_B_pgold_pending_43.md / sample_manifest.json")


if __name__ == "__main__":
    main()
