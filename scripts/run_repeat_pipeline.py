#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_repeat_pipeline.py — repeat-aware 实验总控（r1+ 固定 11 步流水线）。

设计约束：
  - 只允许 repeat >= 1（r0 为冻结基线，禁止通过本脚本重跑/覆盖）。
  - preflight 先行：freeze_judge.py --verify 必须 PASS，题集 SHA-256 必须锁定，
    P_gold 必须有 80 条 gold context；任何失败在零写盘状态下 fail-fast。
  - P_gold 强约束：禁止 --max-questions；必须 --gold-context-file、
    --disable-llm-cache、--save-trace、--validate-trace、--expected-question-sha256。
  - 所有子步骤 subprocess.run(check=True)，失败即中止（不继续后续步骤）。
  - --dry-run 只打印命令不执行；--resume 跳过“产物已覆盖全部请求路线”的步骤
    （单文件存在但只覆盖部分路线时不跳过，确保 P_gold-only 扩展到六路径会重跑合并裁决）。

11 步固定顺序：
  1  preflight     freeze verify + 题集/gold 数量与哈希门禁（零写盘）
  2  generate      每路径 Step_3 fixed-route 生成（LLM，长任务）
  3  judge         judge_longcat --mode judge（LLM，长任务）
  4  preapply      calibrate --mode apply --no-annotations → ai_adjudication_pre
  5  build-review  build_supplementary_review 生成 set_D/set_E 材料
  6  review        run_supplementary_review --set D（LLM，长任务）
  7  recheck       run_supplementary_review --set E（仅 E 材料存在时）
  8  merge         merge_supplementary_verdicts --base-annotation none
  9  final-apply   calibrate --mode apply --annotation-file <merged> --output-version ai_adjudication_v2
  10 verify        只读验收：条数 / 路由 / P_gold / coverage 自洽对账
  11 report        汇总打印（不写盘）

用法（由用户在终端手动执行，长任务不由 AI 代跑）：
  python scripts/run_repeat_pipeline.py --repeat 1 --seed 43 --routes P_gold --phase all
  python scripts/run_repeat_pipeline.py --repeat 1 --seed 43 --phase preflight
  python scripts/run_repeat_pipeline.py --repeat 1 --seed 43 --phase all --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


def _import_judge_longcat():
    """导入 judge_longcat 复用 ROUTES/SNAPSHOT_DEFAULT/QUESTIONS_FILE 等常量。

    judge_longcat 的 import 链会拉起 hyperrag（aioboto3 等重依赖）——本脚本
    只用常量，不发 LLM 请求，故对重依赖注入轻量 stub（可导入则用真实模块）。
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
            pkg.__path__ = []
            env = types.ModuleType("hyperrag.env")
            env.normalize_proxy_env = lambda *a, **k: None
            pkg.env = env
            sys.modules["hyperrag"] = pkg
            sys.modules["hyperrag.env"] = env
    return importlib.import_module("judge_longcat")


jl = _import_judge_longcat()

from repeat_context import RepeatContext  # noqa: E402

# 题集锁定哈希（questions_v2_manual_final.jsonl；与 r0 冻结值一致）
QUESTIONS_SHA256 = "b1067566df3303197e991d56873b9406ad860e15f4c25ec67852a3939a8a7ed2"
P_GOLD_N_QUESTIONS = 80

PHASES = ["preflight", "generate", "judge", "preapply", "build-review",
          "review", "recheck", "merge", "final-apply", "verify", "report", "all"]
STEP_ORDER = ["preflight", "generate", "judge", "preapply", "build-review",
              "review", "recheck", "merge", "final-apply", "verify", "report"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class Step:
    """一个可执行步骤：name + 命令 + 完成标记（--resume 判断用）。"""
    name: str
    cmd: list[str]
    marker: Path | None = None          # 存在即视为已完成（resume 跳过）
    description: str = ""
    kind: str = "subprocess"            # subprocess / check / report


def _py(script: str) -> list[str]:
    return [sys.executable, str(_ROOT / script)]


def preflight_problems(rc: RepeatContext, routes: list[str],
                       questions_sha: str) -> list[str]:
    """只读门禁检查（零写盘）。返回问题列表，空列表 = 通过。"""
    problems: list[str] = []
    if rc.repeat < 1:
        problems.append(f"repeat={rc.repeat} 不允许：r0 为冻结基线，本脚本仅服务 r1+")

    qfile = rc.questions_file
    if not qfile.exists():
        problems.append(f"题集文件缺失: {qfile}")
    else:
        actual = sha256_file(qfile)
        if actual != questions_sha:
            problems.append(f"题集 SHA-256 漂移: expected={questions_sha} actual={actual}")

    if "P_gold" in routes:
        gfile = rc.gold_context_file
        if not gfile.exists():
            problems.append(f"P_gold 需要 gold context 文件: {gfile}")
        else:
            n = sum(1 for line in gfile.read_text(encoding="utf-8").splitlines()
                    if line.strip())
            if n != P_GOLD_N_QUESTIONS:
                problems.append(f"P_gold gold context 条数 {n} != {P_GOLD_N_QUESTIONS}")
    return problems


def build_steps(rc: RepeatContext, routes: list[str],
                questions_sha: str) -> list[Step]:
    """构造 11 步固定流水线（纯函数，不执行）。"""
    sup = rc.supplementary_dir
    routes_csv = ",".join(routes)
    repeat_args = ["--repeat", str(rc.repeat), "--seed", str(rc.seed),
                   "--snapshot", rc.snapshot]
    steps: list[Step] = []

    # 1 preflight —— freeze verify + 只读门禁（零写盘）
    steps.append(Step(
        "preflight",
        _py("scripts/freeze_judge.py") + ["--verify"],
        marker=None,  # 每次都执行（只读）
        description="freeze manifest 漂移校验（preflight gate）",
        kind="check",
    ))

    # 2 generate —— 每路径一次（P_gold 带强约束）
    for route in routes:
        cmd = _py("reproduce/Step_3_response_question.py") + [
            "--fixed-route", route,
            "--repeat-id", str(rc.repeat),
            "--seed", str(rc.seed),
            "--system-snapshot", rc.snapshot,
            # 显式锁定 data_name = chunk=1000 规范建库（caches/neurology_chunk1000），
            # 否则 Step_3 回退默认 pipeline_defaults.DATA_NAME="neurology"（chunk=2400 旧超图），
            # 既会写到 judge 找不到的目录，更会用错超图导致与 r0 基线不可比。
            "--data-name", rc.data_name,
            # 显式锁定题集文件路径（80 题 questions_v2_manual_final.jsonl），
            # 否则 Step_3 回退到 --question-stage 默认 2_stage.json（43 题旧文件），
            # 与冻结 SHA b1067566 不符会被 --expected-question-sha256 门禁拒绝。
            "--question-file-path", str(jl.QUESTIONS_FILE),
            "--disable-llm-cache", "--save-trace", "--validate-trace",
            "--expected-question-sha256", questions_sha,
        ]
        if route == "P_gold":
            cmd += ["--gold-context-file", str(rc.gold_context_file)]
            # 禁止 --max-questions：全量 80 题，无截断
        steps.append(Step(
            f"generate[{route}]", cmd,
            marker=_ROOT / rc.result_file(route),
            description=f"Step_3 fixed-route {route}（LLM，长任务）",
        ))

    # 3 judge
    steps.append(Step(
        "judge",
        _py("scripts/judge_longcat.py")
        + ["--mode", "judge", "--routes", *routes] + repeat_args,
        marker=rc.verdict_file(routes[0]),
        description="LongCat judge（LLM，长任务）",
    ))

    # 4 preapply —— 无标注初裁 → ai_adjudication_pre
    steps.append(Step(
        "preapply",
        _py("scripts/calibrate_blind_review.py")
        + ["--mode", "apply", "--no-annotations", "--overwrite", "--routes", routes_csv]
        + repeat_args,
        marker=rc.judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl",
        description="no-annotations 初裁（全 pending，输出 ai_adjudication_pre）",
    ))

    # 5 build-review
    steps.append(Step(
        "build-review",
        _py("scripts/build_supplementary_review.py") + repeat_args,
        marker=sup / "sample_manifest.json",
        description="生成 set_D/set_E 补充审核材料",
    ))

    # 6 review（set_D）
    steps.append(Step(
        "review",
        _py("scripts/run_supplementary_review.py")
        + ["--set", "D"] + repeat_args,
        marker=sup / "verdicts_ai_supplementary_D.jsonl",
        description="set_D AI 审核（LLM，长任务）",
    ))

    # 7 recheck（set_E，仅材料存在时执行；否则跳过）
    steps.append(Step(
        "recheck",
        _py("scripts/run_supplementary_review.py")
        + ["--set", "E"] + repeat_args,
        marker=sup / "verdicts_ai_recheck_E.jsonl",
        description="set_E 复核（LLM；E 材料不存在时自动跳过）",
    ))

    # 8 merge
    merged_out = sup / f"verdicts_ai_all_r{rc.repeat}.jsonl"
    steps.append(Step(
        "merge",
        _py("scripts/merge_supplementary_verdicts.py")
        + repeat_args + ["--base-annotation", "none"],
        marker=merged_out,
        description="合并 set_D(+E) 为统一标注（不读原 163 标注）",
    ))

    # 9 final-apply
    steps.append(Step(
        "final-apply",
        _py("scripts/calibrate_blind_review.py")
        + ["--mode", "apply", "--annotation-file", str(merged_out),
           "--output-version", "ai_adjudication_v2", "--overwrite", "--routes", routes_csv]
        + repeat_args,
        marker=rc.judge_dir / "ai_adjudication_v2" / "final_verdicts.jsonl",
        description="应用合并标注生成最终裁决 ai_adjudication_v2",
    ))

    # 10 verify —— 只读验收
    steps.append(Step(
        "verify",
        [],  # 内联只读检查，不启子进程
        marker=None,
        description="只读验收：条数/路由/P_gold/coverage 自洽",
        kind="check",
    ))

    # 11 report —— 汇总打印
    steps.append(Step(
        "report",
        [],
        marker=None,
        description="汇总打印",
        kind="report",
    ))
    return steps


# ---------------------------------------------------------------------------
# 只读验收
# ---------------------------------------------------------------------------
def verify_problems(rc: RepeatContext, routes: list[str]) -> list[str]:
    problems: list[str] = []
    expected_total = P_GOLD_N_QUESTIONS * len(routes)

    fv_path = rc.judge_dir / "ai_adjudication_v2" / "final_verdicts.jsonl"
    if not fv_path.exists():
        return [f"最终裁决缺失: {fv_path}"]
    rows = [json.loads(l) for l in fv_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    keys = [(r.get("route"), r.get("question_id")) for r in rows]
    if len(rows) != expected_total:
        problems.append(f"final_verdicts {len(rows)} 条 != {expected_total}")
    if len(set(keys)) != len(keys):
        problems.append("final_verdicts (route,question_id) 存在重复")
    for route in routes:
        n = sum(1 for k in keys if k[0] == route)
        if n != P_GOLD_N_QUESTIONS:
            problems.append(f"{route}: {n} 条 != {P_GOLD_N_QUESTIONS}")

    fs_path = rc.judge_dir / "ai_adjudication_v2" / "final_summary.json"
    if fs_path.exists():
        s = json.loads(fs_path.read_text(encoding="utf-8"))
        if s.get("total_records") != expected_total:
            problems.append(f"final_summary.total_records={s.get('total_records')} "
                            f"!= {expected_total}")
        if s.get("repeat") != rc.repeat:
            problems.append(f"final_summary.repeat={s.get('repeat')} != {rc.repeat}")
        pg = s.get("P_gold_route_success")
        if "P_gold" in routes and pg:
            total_pg = sum(pg.values())
            if total_pg != P_GOLD_N_QUESTIONS:
                problems.append(f"P_gold route_success 总和 {total_pg} "
                                f"!= {P_GOLD_N_QUESTIONS}")
    else:
        problems.append(f"final_summary 缺失: {fs_path}")
    return problems


def report(rc: RepeatContext, routes: list[str]) -> None:
    fs_path = rc.judge_dir / "ai_adjudication_v2" / "final_summary.json"
    print("=" * 70)
    print(f"[report] repeat={rc.repeat} seed={rc.seed} snapshot={rc.snapshot[:8]}")
    print(f"[report] judge_dir: {_rel(rc.judge_dir)}")
    if fs_path.exists():
        s = json.loads(fs_path.read_text(encoding="utf-8"))
        print(f"[report] total={s.get('total_records')} "
              f"route_success={s.get('route_success_dist')}")
        print(f"[report] P_gold={s.get('P_gold_route_success')}")
        print(f"[report] excluded={s.get('excluded_count')} "
              f"reasons={s.get('exclusion_reason_dist')}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------
def _e_material_exists(rc: RepeatContext) -> bool:
    return bool(list(rc.supplementary_dir.glob("set_E_recheck_*.md")))


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT))
    except ValueError:
        return str(p)


def _combined_route_counts(path: Path) -> dict:
    """读取 combined 裁决 JSONL（含 route 字段），返回 {route: 条数}。不存在返回 {}。"""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        rt = r.get("route")
        if rt:
            counts[rt] = counts.get(rt, 0) + 1
    return counts


def _covers_routes(rc: RepeatContext, routes: list[str], path: Path) -> bool:
    """path 是 combined 裁决 JSONL；是否覆盖全部请求路线（每路线恰好 80 条）。"""
    if not path.exists():
        return False
    counts = _combined_route_counts(path)
    return all(counts.get(r, 0) == P_GOLD_N_QUESTIONS for r in routes)


def _judge_covers(rc: RepeatContext, routes: list[str]) -> bool:
    """所有请求路线的 judge verdict 文件存在且各含 80 条 distinct cell。

    允许 parse_ok=false（确定性 judge 失败视为已完成，不再重试）。
    """
    for r in routes:
        vf = rc.verdict_file(r)
        if not vf.exists():
            return False
        done = jl.read_done_any(vf)
        if len(done) != P_GOLD_N_QUESTIONS:
            return False
    return True


def _pending_cells(rc: RepeatContext) -> tuple[set, set]:
    """当前 preapply 产物的 pending cell 集合，返回 (no_bid, with_bid)。

    no_bid  -> 未经 AI 审核的悬置（set_D 的来源）
    with_bid-> 已审核但 unresolved 的悬置（set_E 的来源）
    """
    fv = rc.judge_dir / "ai_adjudication_pre" / "final_verdicts.jsonl"
    no_bid: set = set()
    with_bid: set = set()
    if not fv.exists():
        return no_bid, with_bid
    for line in fv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("route_success") != "pending":
            continue
        key = (r.get("route"), r.get("question_id"))
        (with_bid if r.get("blind_id") else no_bid).add(key)
    return no_bid, with_bid


def _manifest_cells(rc: RepeatContext) -> tuple[set, set]:
    """sample_manifest.json 记录的 (set_D, set_E) cell 集合。"""
    sm = rc.judge_dir / "blind_review_supplementary" / "sample_manifest.json"
    d_cells: set = set()
    e_cells: set = set()
    if not sm.exists():
        return d_cells, e_cells
    try:
        obj = json.loads(sm.read_text(encoding="utf-8"))
    except Exception:
        return d_cells, e_cells
    if not isinstance(obj, dict):
        return d_cells, e_cells
    for key, val in obj.items():
        if not isinstance(val, list):
            continue
        if key.startswith("set_D_unreviewed"):
            target = d_cells
        elif key.startswith("set_E_recheck"):
            target = e_cells
        else:
            continue
        for e in val:
            if isinstance(e, dict):
                target.add((e.get("route"), e.get("question_id")))
    return d_cells, e_cells


def _build_review_covers(rc: RepeatContext) -> bool:
    """sample_manifest 的 cell 集合与当前 preapply pending 一致才可跳过。

    仅判断“文件存在”会在 preapply 从 P_gold-only 重生成为六路径后误跳过，
    导致模板仍是陈旧的 80 条、review 空转、merge 一致性校验才爆。
    """
    no_bid, with_bid = _pending_cells(rc)
    d_cells, e_cells = _manifest_cells(rc)
    if not d_cells and not e_cells:
        return False
    return d_cells == no_bid and e_cells == with_bid


def _review_ids(path: Path) -> set:
    """JSONL 的 review_id 集合。不存在返回空集。"""
    ids: set = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rid = json.loads(line).get("review_id")
        except Exception:
            continue
        if rid:
            ids.add(rid)
    return ids


def _review_output_covers(rc: RepeatContext, which: str) -> bool:
    """set_D/set_E 的 AI 裁决是否覆盖对应模板的全部 review_id。

    盲审输出刻意不含 route 字段（review_id 隐藏 route/qid），因此不能用路线
    覆盖度判据——_covers_routes 对它恒为 False。以模板 review_id 为准。
    """
    sup = rc.judge_dir / "blind_review_supplementary"
    tpl = sup / f"verdicts_template_{which}.jsonl"
    out = sup / ("verdicts_ai_supplementary_D.jsonl" if which == "D"
                 else "verdicts_ai_recheck_E.jsonl")
    if not tpl.exists() or not out.exists():
        return False
    return _review_ids(tpl).issubset(_review_ids(out))


# 盲审链步骤：执行前统一做一次陈旧产物 fail-closed 检查
_BLIND_REVIEW_STEPS = {"build-review", "review", "recheck", "merge"}


def stale_blind_review_problems(rc: RepeatContext) -> list[str]:
    """陈旧盲审产物 fail-closed 检查（返回问题列表，空=通过）。

    review_id 是序号编码（R{n}-D-001…），随 pending 集合的大小与排序漂移。
    若 sample_manifest 的 cell 集合与当前 preapply pending 不一致，而目录里
    已存在非空 AI 裁决，则重建模板会让旧 review_id 的裁决被复用到**不同
    cell** 上（run_supplementary_review 按 review_id 断点续跑）——静默数据
    污染。此时必须人工清理，不允许自动重建。
    """
    sup = rc.judge_dir / "blind_review_supplementary"
    if not (sup / "sample_manifest.json").exists():
        return []
    no_bid, with_bid = _pending_cells(rc)
    if not no_bid and not with_bid:
        return []  # preapply 尚未产出/无 pending，交由前序步骤处理
    d_cells, e_cells = _manifest_cells(rc)
    if d_cells == no_bid and e_cells == with_bid:
        return []  # 一致
    stale = [p.name for p in (sup / "verdicts_ai_supplementary_D.jsonl",
                              sup / "verdicts_ai_recheck_E.jsonl")
             if p.exists() and p.stat().st_size > 0]
    if not stale:
        return []  # 仅 manifest 陈旧、无裁决可污染 -> 允许 build-review 重建
    return [
        f"sample_manifest 记录 set_D={len(d_cells)} / set_E={len(e_cells)} cell，"
        f"当前 preapply pending 为 no-bid={len(no_bid)} / with-bid={len(with_bid)}（不一致）",
        f"已存在 AI 裁决 {stale}；review_id 为序号编码，重建模板会使旧裁决错配到"
        f"不同 cell（静默数据污染）",
        f"处理：确认放弃旧盲审结果后，删除整个目录再重跑 -> {_rel(sup)}",
    ]


# 这些步骤的输出是“覆盖全部请求路线的合并产物”，resume 必须以路线覆盖度为判据，
# 而非单文件存在——否则从 P_gold-only 扩展到六路径时会错误跳过，只产出 80 条。
# 注意：review/recheck 的输出无 route 字段，不能用此判据（见 _review_output_covers）。
_ROUTE_AWARE_COMBINED = {"preapply", "final-apply", "merge"}


def _should_skip(step: Step, rc: RepeatContext, routes: list[str]) -> bool:
    """--resume 时该步骤是否可跳过。

    - 无 marker（preflight/verify/report）：从不跳过。
    - judge：所有请求路线 verdict 完整才跳过（per-route 文件，增量可重跑）。
    - 合并裁决步骤（preapply/final-apply/merge）：输出覆盖全部请求路线才跳过。
    - build-review：sample_manifest 的 cell 集合与当前 preapply pending 一致才跳过。
    - review/recheck：AI 裁决覆盖模板全部 review_id 才跳过（输出无 route 字段）。
    - 其余（generate[route] 每路径单文件）：单文件存在即跳过。
    """
    if step.marker is None:
        return False
    if step.name == "judge":
        return _judge_covers(rc, routes)
    if step.name in _ROUTE_AWARE_COMBINED:
        return _covers_routes(rc, routes, step.marker)
    if step.name == "build-review":
        return _build_review_covers(rc)
    if step.name == "review":
        return _review_output_covers(rc, "D")
    if step.name == "recheck":
        return _review_output_covers(rc, "E")
    return step.marker.exists()


def run_pipeline(steps: list[Step], rc: RepeatContext, routes: list[str],
                 phase: str = "all", dry_run: bool = False,
                 resume: bool = False, questions_sha: str = QUESTIONS_SHA256,
                 runner=None) -> int:
    """执行流水线。runner 注入点供测试（默认 subprocess.run）。"""
    run = runner or subprocess.run

    if phase == "all":
        selected = steps
    else:
        selected = [s for s in steps if s.name == phase or s.name.startswith(phase + "[")]

    for step in selected:
        # preflight 只读门禁：dry-run 也必须执行（零写盘，纯检查）
        if step.name == "preflight":
            print(f"[run ] {step.name}: {step.description}")
            problems = preflight_problems(rc, routes, questions_sha)
            if problems:
                print("ERROR: preflight 门禁失败（零写盘 fail-fast）:", file=sys.stderr)
                for p in problems:
                    print(f"  {p}", file=sys.stderr)
                return 1
            if dry_run:
                print("        " + " ".join(step.cmd))
                continue
            proc = run(step.cmd, cwd=str(_ROOT))
            if proc.returncode != 0:
                print(f"ERROR: preflight 子步骤失败: {step.name}", file=sys.stderr)
                return 1
            continue

        # 盲审链 fail-closed：preapply pending 已变但目录残留旧 AI 裁决时，
        # 重建模板会使 review_id 序号错配到不同 cell（静默污染），必须人工清理。
        # 置于 skip 判断之前——陈旧态不得被 resume 跳过而掩盖。
        if step.name in _BLIND_REVIEW_STEPS:
            stale = stale_blind_review_problems(rc)
            if stale:
                print(f"ERROR: 陈旧盲审产物（fail-closed，禁止静默重建）: {step.name}",
                      file=sys.stderr)
                for s in stale:
                    print(f"  {s}", file=sys.stderr)
                return 1

        # resume：仅当产物覆盖全部请求路线时跳过（防止 P_gold-only 扩展到六路径时
        # 错误跳过 preapply/final-apply 等合并步骤，导致最终只 80 条而非 480）。
        if resume and _should_skip(step, rc, routes):
            tag = _rel(step.marker) if step.marker else step.name
            print(f"[skip] {step.name}: 产物已完整（覆盖当前请求范围） {tag}")
            continue

        # recheck：E 材料不存在则跳过（第一次审核无 uncertain 属正常）
        if step.name == "recheck" and not _e_material_exists(rc):
            print(f"[skip] {step.name}: 无 set_E 材料（无 uncertain 待复核）")
            continue

        print(f"[run ] {step.name}: {step.description}")
        if dry_run:
            print("        " + " ".join(step.cmd))
            continue

        if step.kind == "check" and step.name == "verify":
            problems = verify_problems(rc, routes)
            if problems:
                print(f"ERROR: verify 验收失败（{len(problems)} 项）:", file=sys.stderr)
                for p in problems:
                    print(f"  {p}", file=sys.stderr)
                return 1
            print("[ok  ] verify: 全部通过")
            continue

        if step.kind == "report":
            report(rc, routes)
            continue

        proc = run(step.cmd, cwd=str(_ROOT))
        if proc.returncode != 0:
            print(f"ERROR: 步骤失败，流水线中止: {step.name} "
                  f"(exit={proc.returncode})", file=sys.stderr)
            return 1

    print(f"[done] phase={phase} 完成（repeat={rc.repeat}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="repeat-aware 实验总控（r1+ 11 步流水线）")
    ap.add_argument("--data-name", default="neurology_chunk1000")
    ap.add_argument("--repeat", type=int, required=True,
                    help="repeat 编号（必须 >= 1；r0 为冻结基线禁止重跑）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snapshot", default=None,
                    help="系统快照 ID（默认 judge_longcat.SNAPSHOT_DEFAULT）")
    ap.add_argument("--routes", default="P_gold",
                    help="路由子集（逗号分隔；默认 P_gold）")
    ap.add_argument("--phase", default="all", choices=PHASES,
                    help="执行阶段（默认 all；单阶段如 preflight/generate/judge…）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    ap.add_argument("--resume", action="store_true",
                    help="跳过产物已存在的步骤")
    ap.add_argument("--questions-sha256", default=QUESTIONS_SHA256,
                    help="题集 SHA-256 锁定值（默认 r0 冻结值）")
    args = ap.parse_args()

    rc = RepeatContext(
        data_name=args.data_name,
        repeat=args.repeat,
        seed=args.seed,
        snapshot=args.snapshot or jl.SNAPSHOT_DEFAULT,
    )
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    for r in routes:
        if r not in jl.ROUTES:
            print(f"ERROR: 未知路由 {r}", file=sys.stderr)
            return 1

    steps = build_steps(rc, routes, args.questions_sha256)
    return run_pipeline(steps, rc, routes, phase=args.phase,
                        dry_run=args.dry_run, resume=args.resume,
                        questions_sha=args.questions_sha256)


if __name__ == "__main__":
    sys.exit(main())
