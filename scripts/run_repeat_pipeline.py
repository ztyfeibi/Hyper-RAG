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
  - --dry-run 只打印命令不执行；--resume 跳过产物已存在的步骤。

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
        + ["--mode", "apply", "--no-annotations", "--routes", routes_csv]
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
           "--output-version", "ai_adjudication_v2", "--routes", routes_csv]
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
    print(f"[report] judge_dir: {rc.judge_dir.relative_to(_ROOT)}")
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

        # resume：产物已存在则跳过（preflight/verify/report 每次都跑）
        if resume and step.marker is not None and step.marker.exists():
            print(f"[skip] {step.name}: 产物已存在 {_rel(step.marker)}")
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
