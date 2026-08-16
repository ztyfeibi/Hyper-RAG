"""补充盲审集构建（Step 1：关闭 repeat 0 的 167 条 route_success=pending）。

从 ai_adjudication_v1/final_verdicts.jsonl 构建：

- set_D：155 条 source=longcat_only && route_success=pending（未审核悬置）。
- set_E：12 条 source=ai_review && route_success=pending（已审核但 unresolved 悬置，复核）。

盲审约束（沿用 build_blind_review_set.py §12.2 约定）：
- 材料**不暴露 route / question_id**：审核 ID 用 SR-D-xxx / SR-E-xxx，
  review_id -> (route, qid, blind_id) 映射只写在 sample_manifest.json（内部，不随材料分发）。
- 每条包含：Question / Answer units（required 标注）/ Evidence spans / Candidate answer /
  待判 claims（set_D 用 Judge 标记的 unsupported_claims 文本预填，set_E 用原 AI 标注 claims 预填）。
- set_E 为复核集：材料附原 AI 审核判定（verdict / au_status / fatality / claims 状态），
  复核结果在 Step 3 合并时优先级高于原 163 条中的对应记录。

纯本地确定性操作，不调用 LLM，无随机性（按 (route, qid) 排序可复现）。

后续模式（Step 3 实现，当前未启用）：
- --mode validate：校验 verdicts_ai_supplementary_D.jsonl / verdicts_ai_recheck_E.jsonl。
- --mode merge：合并原 163 + D155 + E12(覆盖) -> verdicts_ai_all_v1.jsonl。
"""

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))


def _import_judge_longcat():
    """轻量 stub 绕过 judge_longcat 的重依赖 import 链（与 calibrate_blind_review 相同手法）。"""
    import importlib
    import types
    try:
        import openai  # noqa: F401
    except Exception:
        mod = types.ModuleType("openai")
        mod.OpenAI = object
        sys.modules["openai"] = mod
    if "hyperrag" not in sys.modules:
        try:
            importlib.import_module("hyperrag")
        except Exception:
            pkg = types.ModuleType("hyperrag")
            pkg.__path__ = []
            env = types.ModuleType("hyperrag.env")
            env.normalize_proxy_env = lambda *a, **k: None
            pkg.env = env
            sys.modules["hyperrag"] = pkg
            sys.modules["hyperrag.env"] = env
    return importlib.import_module("judge_longcat")


jl = _import_judge_longcat()

from repeat_context import RepeatContext, DEFAULT_RC  # noqa: E402

# ---------------------------------------------------------------------------
# 路径 / 运行参数（RepeatContext 派生；r0/s42 默认值向后兼容）
# ---------------------------------------------------------------------------
_RC: RepeatContext = DEFAULT_RC
REPEAT, SEED = 0, 42
OUT_DIR = DEFAULT_RC.judge_dir
# 新 repeat 用 ai_adjudication_pre（--no-annotations 产物）；r0 用 ai_adjudication_v1
_PRE_DIR_NAME = "ai_adjudication_pre"
_V1_DIR_NAME = "ai_adjudication_v1"
FV_FILE = OUT_DIR / _V1_DIR_NAME / "final_verdicts.jsonl"
SUP_DIR = DEFAULT_RC.supplementary_dir
AI_ANNOTATED = OUT_DIR / "blind_review" / "verdicts_ai_annotated.jsonl"


def set_context(rc: RepeatContext) -> None:
    """切换活跃 RepeatContext."""
    global _RC, REPEAT, SEED, OUT_DIR, FV_FILE, SUP_DIR, AI_ANNOTATED
    _RC = rc
    REPEAT = rc.repeat
    SEED = rc.seed
    OUT_DIR = rc.judge_dir
    # r0 用 v1（有 163 条 blind_id 对齐产物）；r1+ 用 pre（--no-annotations 产物）
    vname = _V1_DIR_NAME if rc.repeat == 0 else _PRE_DIR_NAME
    FV_FILE = OUT_DIR / vname / "final_verdicts.jsonl"
    SUP_DIR = rc.supplementary_dir
    AI_ANNOTATED = rc.blind_dir / "verdicts_ai_annotated.jsonl"


def _fv_path_for_rc(rc: RepeatContext) -> Path:
    """获取当前 repeat 的 final_verdicts 路径（r0=v1, r1+=pre）."""
    vname = _V1_DIR_NAME if rc.repeat == 0 else _PRE_DIR_NAME
    return rc.judge_dir / vname / "final_verdicts.jsonl"


# r0 冻结值（仅 r0 校验用；r1+ 从 final_verdicts 动态派生）
EXPECTED_D, EXPECTED_E = 155, 12

# 路径泄漏检查：审核材料中不允许出现 route 名 / qid / 本地路径碎片
_LEAK_PATTERNS = [
    re.compile(r"qv2-\d"),
    re.compile(r"\bP_gold\b"),
    re.compile(r"\bP[0-4]\b"),
    re.compile(r"blind_review|\.jsonl|caches[/\\]|fixed_"),
]

CLAIM_STATUS_OPTIONS = "supported_by_source|unsupported_noncritical|contradicted|unverifiable"
# 与 run_supplementary_review.FATALITY_OPTIONS / calibrate.VALID_FATALITY 统一（v2: 含 unresolved）
FATALITY_OPTIONS = "none|harmless|fatal|unresolved"


def load_final_verdicts(path=None, rc=None):
    """加载 final_verdicts，按 source/route_success 拆分 set_D / set_E。

    r0: 校验 EXPECTED_D=155 / EXPECTED_E=12（冻结值）。
    r1+: 从实际 pending 记录动态派生，不校验固定计数。
    """
    ctx = rc or _RC
    fv_path = Path(path) if path else _fv_path_for_rc(ctx)
    rows = [json.loads(l) for l in open(fv_path, encoding="utf-8")]
    set_d = sorted((r for r in rows
                    if r.get("source") == "longcat_only"
                    and r.get("route_success") == "pending"),
                   key=lambda r: (r["route"], r["question_id"]))
    set_e = sorted((r for r in rows
                    if r.get("source") == "ai_review"
                    and r.get("route_success") == "pending"),
                   key=lambda r: (r["route"], r["question_id"]))
    if ctx.repeat == 0:
        if len(set_d) != EXPECTED_D:
            raise SystemExit(f"set_D 应为 {EXPECTED_D} 条，实际 {len(set_d)}")
        if len(set_e) != EXPECTED_E:
            raise SystemExit(f"set_E 应为 {EXPECTED_E} 条，实际 {len(set_e)}")
    keys = [(r["route"], r["question_id"]) for r in set_d + set_e]
    if len(set(keys)) != len(keys):
        raise SystemExit("(route, question_id) 存在重复")
    return set_d, set_e


def load_questions_with_spans():
    out = {}
    for line in open(jl.QUESTIONS_FILE, encoding="utf-8"):
        r = json.loads(line)
        spans_by_au = {es["unit_id"]: es.get("evidence_spans", [])
                       for es in r.get("evidence_spans", [])}
        out[r["question_id"]] = {
            "question": r["question"],
            "answer_units": r.get("answer_units", []),
            "evidence_spans": spans_by_au,
        }
    return out


def load_candidate_answers(rc=None):
    ctx = rc or _RC
    out = {}
    for route in jl.ROUTES:
        rf = _ROOT / ctx.result_file(route)
        if not rf.exists():
            continue
        out[route] = {r["question_id"]: r.get("result") or ""
                      for r in (json.loads(l) for l in open(rf, encoding="utf-8"))}
    return out


def load_longcat_verdicts(rc=None):
    """(route, qid) -> verdict 记录（取 unsupported_claims 文本列表）。"""
    ctx = rc or _RC
    out = {}
    for route in jl.ROUTES:
        f = ctx.verdict_file(route)
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[(route, r["question_id"])] = r
    return out


def load_ai_annotations():
    """blind_id -> 原 AI 标注记录（set_E 复核用）。"""
    return {json.loads(l)["blind_id"]: json.loads(l)
            for l in open(AI_ANNOTATED, encoding="utf-8")}


def review_id(prefix, idx, rc=None):
    """R{repeat}-{prefix}-{idx:03d}（r0 向后兼容用 SR-{prefix}-{idx:03d}）."""
    ctx = rc or _RC
    if ctx.repeat == 0:
        return f"SR-{prefix}-{idx:03d}"
    return f"R{ctx.repeat}-{prefix}-{idx:03d}"


def render_record(rid, meta, candidate, claims_text, lc_required_aus,
                  original=None):
    """渲染一条审核材料（不暴露 route/qid）。

    claims_text: 待判 claim 文本列表（set_D=Judge unsupported_claims；
                 set_E=原 AI 标注 claims，另在 original 区块展示原判定）。
    lc_required_aus: final_verdicts 的 required_aus（unit_id 列表）。
    original: set_E 的原 AI 标注记录（复核参考）。
    """
    lines = [f"### {rid}"]
    lines.append(f"**Question**: {meta['question']}")
    lines.append("")
    lines.append("**Answer units**（判定基准，required 为必判）:")
    for u in meta["answer_units"]:
        req = "required" if u.get("required", True) else "optional"
        marker = " *" if u["unit_id"] in (lc_required_aus or []) else ""
        lines.append(f"- {u['unit_id']} ({req}){marker}: {u['claim']}")
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
    lines.append((candidate or "").strip() or "(空回答)")
    lines.append("")
    if claims_text:
        lines.append("**待判 claims**（逐条判定 status）:")
        for c in claims_text:
            lines.append(f"- \"{c}\"")
        lines.append(f"  status 取值: {CLAIM_STATUS_OPTIONS}")
        lines.append("")
    if original is not None:
        lines.append("**原审核判定**（本条为复核：请独立重判，可维持或推翻）:")
        lines.append(f"- verdict: {original.get('verdict')}")
        lines.append(f"- au_status: {json.dumps(original.get('au_status') or {}, ensure_ascii=False)}")
        lines.append(f"- unsupported_fatality: {original.get('unsupported_fatality')}")
        for c in original.get("unsupported_claims") or []:
            lines.append(f"  - claim \"{(c.get('claim') or '')[:120]}...\" -> {c.get('status')}"
                         if len(c.get("claim") or "") > 120 else
                         f"  - claim \"{c.get('claim') or ''}\" -> {c.get('status')}")
        lines.append("")
    lines.append("**人工判定栏**（请填写）:")
    lines.append("- AU 状态: " + " ".join(
        f"{u['unit_id']}=[supported|missing|contradicted]" for u in meta["answer_units"]))
    lines.append("- 整体 verdict: [pass|fail|uncertain]")
    lines.append(f"- unsupported 致命性: [{FATALITY_OPTIONS}]")
    lines.append("- unsupported claims 详情（逐条，含上面待判 claims 及候选答案中其他超出证据的陈述）:")
    lines.append(f"  - claim: \"...\" status: [{CLAIM_STATUS_OPTIONS}]")
    lines.append("  （若无超出证据的陈述，填空列表；可多行）")
    lines.append("- 备注: ")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


REVIEW_GUIDE_TEXT = f"""# 补充盲审审核指引（set_D 未审核悬置 + set_E 复核）

## 任务
对照每条的 Question / Answer units / Evidence spans，对 Candidate answer 做 source-grounded 判定。
**不要看 set 文件以外的东西**，不知道路径名（P0-P4/P_gold）与模型来源。

## 判定规则（与契约 §12.2/§8.3 对齐，沿用主盲审 REVIEW_GUIDE）
1. **AU 状态**（逐 AU 判）：
   - supported: 候选答案陈述了该 AU 的 claim 且与 evidence span 一致；
   - missing: 候选答案未覆盖该 AU 的 claim（或只有含糊提及、无实质内容）；
   - contradicted: 候选答案陈述与 evidence span 直接矛盾（如错误年份/作者/结论）。
2. **整体 verdict**：
   - pass: 全部 required AU 均 supported，且无致命 unsupported claim；
   - fail: 任一 required AU contradicted 或 missing，或存在致命 unsupported claim；
   - uncertain: 证据不足（无法从 evidence 判定）、或候选答案含糊无法判定。
   **不得把 uncertain 强行改成 fail。**
3. **unsupported 致命性**（{FATALITY_OPTIONS}）：
   - none: 候选答案没有超出证据的陈述；
   - harmless: 有超出证据的陈述，但是无害背景/常识性补充（不误导）；
   - fatal: 有超出证据且错误/误导的陈述（虚假引用、错误数据、与证据矛盾）。
4. **unsupported_claims 详情**（逐条，status 取值 {CLAIM_STATUS_OPTIONS}）：
   - supported_by_source: 该陈述实际可在 evidence span 中找到依据（Judge 误判为 unsupported）；
   - unsupported_noncritical: 超出证据但无害（背景信息、常识）；
   - contradicted: 与证据直接矛盾；
   - unverifiable: 无法从给定 evidence 判定真伪。
   （verifier 派生时 unverifiable 记为 unresolved 悬置，不会强制归 pass/fail。）

## set_D（未审核悬置）
Judge 已判定但从未经独立审核。材料中"待判 claims"为 Judge 标记的 unsupported 陈述，
逐条给 status；也请自行检查候选答案中其他超出证据的陈述。

## set_E（复核）
这批此前已有一次 AI 审核（结果为 uncertain / unresolved 悬置）。材料附"原审核判定"，
请**独立重判**（不锚定原结论），复核结果将覆盖原记录。

## 填写方式
分别编辑 verdicts_template_D.jsonl / verdicts_template_E.jsonl，每行填：
- au_status: {{AU1: supported, AU2: missing, ...}}
- verdict: pass/fail/uncertain
- unsupported_fatality: {FATALITY_OPTIONS}
- unsupported_claims: [{{"claim": "...", "status": "{CLAIM_STATUS_OPTIONS}"}}]
  （预填 claim 文本的条目请直接补 status；若无超出证据的陈述，填空列表 []）
- notes: 可选备注

填完存为 verdicts_ai_supplementary_D.jsonl / verdicts_ai_recheck_E.jsonl。
"""


def check_no_leak(text, where):
    for pat in _LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            raise SystemExit(f"材料泄漏检查失败 [{where}]: 命中 {pat.pattern!r} -> {m.group(0)!r}")


def run_build(out_dir=None, rc=None):
    ctx = rc or _RC
    out_dir = Path(out_dir) if out_dir else ctx.supplementary_dir
    set_d, set_e = load_final_verdicts(rc=ctx)
    questions = load_questions_with_spans()
    candidates = load_candidate_answers(ctx)
    lc_verdicts = load_longcat_verdicts(ctx)
    ai_ann = load_ai_annotations() if ctx.repeat == 0 else {}

    out_dir.mkdir(parents=True, exist_ok=True)

    n_d, n_e = len(set_d), len(set_e)
    d_ids, e_ids = [], []
    for i in range(1, n_d + 1):
        d_ids.append(review_id("D", i, ctx))
    for i in range(1, n_e + 1):
        e_ids.append(review_id("E", i, ctx))
    all_ids = d_ids + e_ids
    if len(set(all_ids)) != len(all_ids):
        raise SystemExit("review_id 存在重复")

    # 动态文件名：set_D_unreviewed_{N}.md / set_E_recheck_{N}.md
    d_md_name = f"set_D_unreviewed_{n_d}.md"
    e_md_name = f"set_E_recheck_{n_e}.md"
    d_tpl_name = "verdicts_template_D.jsonl"
    e_tpl_name = "verdicts_template_E.jsonl"

    # ---- set_D md + 模板 ----
    d_md = [f"# 补充盲审材料 set_D（未审核悬置 {n_d} 条）\n\n"
            "判定规则见 REVIEW_GUIDE.md（source-grounded；uncertain 不得强行改 fail）。\n\n"]
    d_template = []
    for rid, r in zip(d_ids, set_d):
        meta = questions[r["question_id"]]
        cand = (candidates.get(r["route"]) or {}).get(r["question_id"], "")
        lc = lc_verdicts[(r["route"], r["question_id"])]
        claims = lc.get("unsupported_claims") or []
        d_md.append(render_record(rid, meta, cand, claims, r.get("required_aus")))
        d_template.append({
            "review_id": rid,
            "au_status": {},
            "verdict": "",
            "unsupported_fatality": "",
            "unsupported_claims": [{"claim": c, "status": ""} for c in claims],
            "notes": "",
        })

    # ---- set_E md + 模板（复核：附原 AI 判定；r1+ 无原标注则跳过 original）----
    e_md = [f"# 补充盲审材料 set_E（复核 {n_e} 条，原审核结果悬置）\n\n"
            "判定规则见 REVIEW_GUIDE.md。本集为复核：请独立重判，结果将覆盖原记录。\n\n"]
    e_template = []
    for rid, r in zip(e_ids, set_e):
        meta = questions[r["question_id"]]
        cand = (candidates.get(r["route"]) or {}).get(r["question_id"], "")
        orig = ai_ann.get(r.get("blind_id")) if r.get("blind_id") else None
        if orig is not None:
            claims = [c.get("claim") for c in (orig.get("unsupported_claims") or [])]
        else:
            # r1+ 无原标注：用 final_verdicts 的 AI claims
            claims = [c.get("claim") for c in (r.get("unsupported_claims") or [])] if isinstance(r.get("unsupported_claims"), list) else []
            # fallback: 从 lc_verdicts 取
            if not claims:
                lc = lc_verdicts.get((r["route"], r["question_id"]), {})
                claims = lc.get("unsupported_claims") or []
        e_md.append(render_record(rid, meta, cand, claims, r.get("required_aus"),
                                  original=orig))
        e_template.append({
            "review_id": rid,
            "au_status": {},
            "verdict": "",
            "unsupported_fatality": "",
            "unsupported_claims": [{"claim": c, "status": ""} for c in claims],
            "notes": "",
        })

    # ---- 泄漏检查（md 与模板正文，manifest 是内部文件不检查）----
    for name, text in [("set_D", "\n".join(d_md)), ("set_E", "\n".join(e_md)),
                       ("template_D", json.dumps(d_template, ensure_ascii=False)),
                       ("template_E", json.dumps(e_template, ensure_ascii=False))]:
        check_no_leak(text, name)

    (out_dir / d_md_name).write_text("\n".join(d_md), encoding="utf-8")
    (out_dir / e_md_name).write_text("\n".join(e_md), encoding="utf-8")
    with open(out_dir / d_tpl_name, "w", encoding="utf-8") as f:
        for e in d_template:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out_dir / e_tpl_name, "w", encoding="utf-8") as f:
        for e in e_template:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out_dir / "REVIEW_GUIDE.md").write_text(REVIEW_GUIDE_TEXT, encoding="utf-8")

    # ---- sample_manifest.json：动态键名 ----
    fv_source = "ai_adjudication_v1/final_verdicts.jsonl" if ctx.repeat == 0 else "ai_adjudication_pre/final_verdicts.jsonl"
    manifest = {
        "source_final_verdicts": fv_source,
        "adjudication_rule_version": "v3",
        "repeat": ctx.repeat,
        "seed": ctx.seed,
        f"set_D_unreviewed_{n_d}": [
            {"review_id": rid, "route": r["route"], "question_id": r["question_id"],
             "blind_id": None, "lc_verdict": r.get("lc_verdict"),
             "required_aus": r.get("required_aus")}
            for rid, r in zip(d_ids, set_d)],
        f"set_E_recheck_{n_e}": [
            {"review_id": rid, "route": r["route"], "question_id": r["question_id"],
             "blind_id": r.get("blind_id"), "lc_verdict": r.get("lc_verdict"),
             "ai_verdict": r.get("ai_verdict"),
             "required_aus": r.get("required_aus")}
            for rid, r in zip(e_ids, set_e)],
    }
    (out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": out_dir,
        "set_d": n_d, "set_e": n_e,
        "review_ids": len(all_ids),
        "d_claims_prefilled": sum(len(e["unsupported_claims"]) for e in d_template),
        "e_claims_prefilled": sum(len(e["unsupported_claims"]) for e in e_template),
        "d_md_name": d_md_name,
        "e_md_name": e_md_name,
    }


def main():
    ap = argparse.ArgumentParser(description="构建补充盲审集（set_D/set_E）")
    ap.add_argument("--mode", default="build", choices=["build"],
                    help="build=生成材料（validate/merge 在 Step 3 启用）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 judge 目录下 blind_review_supplementary/；测试重定向用）")
    ap.add_argument("--data-name", default="neurology_chunk1000")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot", default=None,
                    help="系统快照 ID（默认 jl.SNAPSHOT_DEFAULT）")
    args = ap.parse_args()

    rc = RepeatContext(
        data_name=args.data_name,
        repeat=args.repeat,
        seed=args.seed,
        snapshot=args.snapshot or jl.SNAPSHOT_DEFAULT,
    )
    set_context(rc)

    info = run_build(out_dir=args.out_dir, rc=rc)
    print(f"补充盲审集已生成: {info['out_dir']}/")
    print(f"  set_D（未审核悬置）: {info['set_d']} 条 -> {info['d_md_name']} / verdicts_template_D.jsonl")
    print(f"  set_E（复核）: {info['set_e']} 条 -> {info['e_md_name']} / verdicts_template_E.jsonl")
    print(f"  review_id 总数: {info['review_ids']}（唯一）")
    print(f"  预填待判 claims: D={info['d_claims_prefilled']} 条, E={info['e_claims_prefilled']} 条")
    print(f"  REVIEW_GUIDE.md / sample_manifest.json（内部映射，不随材料分发）")


if __name__ == "__main__":
    main()
