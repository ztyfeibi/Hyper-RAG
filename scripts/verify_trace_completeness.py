# -*- coding: utf-8 -*-
"""Step 1 收尾验收：校验固定路径产物的 Trace 完整性与快照一致性。

替代此前一次性的 ``_smoke_verify*.py`` 调试脚本（那三个脚本本身还污染了
源码指纹）。本脚本对一批已产出的固定路径结果做 **五项** 硬性检查：

1. **快照一致性** —— 当前源码指纹是否等于产物文件名里的 snapshot 指纹；
   不一致说明产物不是当前代码跑出来的，结论不可引用。
2. **Qwen 计数真实性** —— 直接对 result 里的最终 context 调用 vLLM
   ``/tokenize``，与 trace 上报的 ``stages.tokens_after_truncation`` 比对；
   同时校验未突破契约 ``final_context_hard_cap``。
3. **finish_reason** —— 顶层 ``finish_reason`` 必须被真实采集（非 null）。
4. **source_provenance_ids** —— 每个 retriever 的候选必须带真实 provenance。
5. **stages.provenance / graph_lineage** —— 证据来源与超图扩展路径必须存在。
6. **实验输入锁定** —— 运行记录必须带 question_file_hash；P_gold 还必须带
   gold_context_file_hash（锁定实际问题集与 Gold 证据字节）。

用法::

    python scripts/verify_trace_completeness.py \\
        --data-name neurology_chunk1000 --snapshot <fingerprint> \\
        --routes P0 P1 P2 P3 P4 P_gold

任一路径不通过 -> 退出码 1（可直接接 CI / 收尾 gate）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperrag.experiment_contract import load_contract, GOLD_ROUTE_ID
from hyperrag.qwen_tokenizer import get_qwen_token_counter
from hyperrag.system_snapshot import compute_runtime_source_fingerprint

# 上报值与真实计数允许的绝对偏差（0 表示要求完全一致）
TOKEN_TOLERANCE = 0


def _load_route(base: Path, route: str, snapshot: str, repeat: int, seed: int,
                contract_tag: str):
    """加载一条路径的全部 result / trace 记录（JSONL 逐行 or 旧 JSON 数组）。

    Returns
    -------
    (results, traces, err) —— 均为**全量**列表（不取第一条）；result 与 trace
    行数必须一一对应，这是本脚本检查的一部分，由调用方比对。
    """
    stem = f"fixed_{route}_r{repeat}_s{seed}_{contract_tag}_{snapshot}"
    # 优先查找 JSONL 格式（新），回退到 JSON 数组（旧）
    res_path_jsonl = base / f"{stem}_result.jsonl"
    res_path_json = base / f"{stem}_result.json"
    trace_path = base / f"{stem}_trace.jsonl"

    results = []
    if res_path_jsonl.exists():
        # JSONL: 逐行读取全部有效记录
        for line in res_path_jsonl.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
        if not results:
            return [], [], f"结果文件为空 {res_path_jsonl.name}"
    elif res_path_json.exists():
        # 旧格式: JSON 数组
        data = json.loads(res_path_json.read_text(encoding="utf-8"))
        results = data if isinstance(data, list) else [data]
        if not results:
            return [], [], f"结果文件为空 {res_path_json.name}"
    else:
        return [], [], f"缺少结果文件 {stem}_result.jsonl/.json"

    traces = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    if not traces:
        return results, [], f"缺少 trace {trace_path.name}"
    return results, traces, None


def _check_provenance(trace: dict) -> list:
    """候选级 source_provenance_ids 必须真实填充（至少一条 retriever 有值）。"""
    problems = []
    retrievers = trace.get("retrievers") or []
    if not retrievers:
        return problems  # P0 / P_gold 无检索器，属正常
    for r in retrievers:
        cands = r.get("candidates") or []
        if not cands:
            continue
        missing = sum(1 for c in cands if c.get("source_provenance_ids") is None)
        if missing == len(cands):
            problems.append(
                f"retriever={r.get('retriever_id')} 全部 {missing} 个候选 "
                f"source_provenance_ids 为 null")
    return problems


def _check_route_identity(results: list, traces: list) -> list:
    """result 与 trace 的 question_id 必须**完全一致**（含顺序）且无重复。

    result 是唯一事实源，trace 是其派生副本：任何错位/缺失/重复都说明
    两文件未对齐，产物不可引用。
    """
    problems = []
    res_ids, trc_ids = [], []
    for i, r in enumerate(results):
        qid = r.get("question_id")
        if qid is None:
            problems.append(f"result[{i}] 缺 question_id")
        else:
            res_ids.append(str(qid))
    for i, t in enumerate(traces):
        qid = t.get("question_id")
        if qid is None:
            problems.append(f"trace[{i}] 缺 question_id")
        else:
            trc_ids.append(str(qid))
    if res_ids and res_ids != trc_ids:
        problems.append(
            f"result/trace question_id 不一致: result={res_ids} trace={trc_ids}")
    if len(res_ids) != len(set(res_ids)):
        dup = sorted({x for x in res_ids if res_ids.count(x) > 1})
        problems.append(f"result 存在重复 question_id: {dup}")
    return problems


def _check_manifest(base: Path, stem: str, route: str) -> list:
    """manifest 必须存在且 status == complete（incomplete 批次不得验收）。"""
    mpath = base / f"{stem}_manifest.json"
    if not mpath.exists():
        return [f"{route}:manifest 缺失（无法确认批次状态）"]
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{route}:manifest 无法解析: {e}"]
    if m.get("status") != "complete":
        return [f"{route}:manifest.status={m.get('status')} != complete"]
    return []


def _check_stages(trace: dict) -> list:
    problems = []
    stages = trace.get("stages") or {}
    for key in ("provenance", "graph_lineage"):
        v = stages.get(key)
        if not isinstance(v, dict) or not v:
            problems.append(f"stages.{key} 缺失或为空")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-name", default="neurology_chunk1000")
    ap.add_argument("--snapshot", required=True, help="产物文件名里的 snapshot 指纹")
    ap.add_argument("--routes", nargs="+",
                    default=["P0", "P1", "P2", "P3", "P4", GOLD_ROUTE_ID])
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contract-tag", default="v2.1-v1",
                    help="产物文件名中的契约标记段")
    ap.add_argument("--contract", default=None)
    ap.add_argument("--expected-count", type=int, default=None,
                    help="每条路径的预期记录数（正式全量=80；smoke 传 3）。"
                         "缺省不检查条数。")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    base = repo / "caches" / args.data_name / "response"
    contract = load_contract(args.contract)
    counter = get_qwen_token_counter()

    failures = []

    # --- 检查 1：快照一致性 ---
    # 注意：产物文件名里的是 snapshot_id（对整份快照 JSON 取哈希），
    # 而源码指纹是快照内的 runtime_source_fingerprint 字段。二者不可直接
    # 相等，必须读回快照文件比较 runtime_source_fingerprint。
    # 指纹范围 = 运行时白名单（hyperrag/ + Step_3 + 契约/Schema）；
    # 修改 tests/ 或本验证脚本 **不** 影响一致性判定。
    fp = compute_runtime_source_fingerprint()["runtime_source_fingerprint"]
    snap_path = (repo / "caches" / args.data_name / "snapshots"
                 / f"{args.snapshot}.json")
    print("[1] 快照一致性（运行时白名单指纹）")
    if not snap_path.exists():
        print(f"    快照文件缺失: {snap_path}")
        failures.append("snapshot_file_missing")
        snap_ok = False
    else:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        recorded_fp = snap.get("runtime_source_fingerprint")
        snap_ok = fp == recorded_fp
        print(f"    snapshot_id                          = {args.snapshot}")
        print(f"    snapshot.runtime_source_fingerprint  = {recorded_fp}")
        print(f"    current  runtime_source_fingerprint  = {fp}")
        print(f"    snapshot.runtime_source_file_count   = "
              f"{snap.get('runtime_source_file_count')}")
        print(f"    snapshot.verification_script_hash    = "
              f"{snap.get('verification_script_hash')}")
        if not snap_ok:
            failures.append("snapshot_mismatch")
    print(f"    -> {'一致' if snap_ok else '不一致（产物与当前源码不匹配）'}")
    print()

    # --- 检查 2-5：逐路径、逐条记录 ---
    header = (f"{'#':>3} {'qid':>12} {'ctx_qwen':>9} {'reported':>9} "
              f"{'hard_cap':>9} {'finish':>8} {'prov':>5} {'stages':>7}  结论")
    print("[2-5] 逐路径 Trace 完整性（全部记录逐条检查）")
    print("    " + header)
    for route in args.routes:
        results, traces, err = _load_route(
            base, route, args.snapshot, args.repeat, args.seed, args.contract_tag)
        if err:
            print(f"    {route:8} {err}")
            failures.append(f"{route}:{err}")
            continue

        # 0) 行数一致性：result 与 trace 必须一一对应
        if len(results) != len(traces):
            msg = (f"result {len(results)} 行 != trace {len(traces)} 行"
                   f"（应一一对应）")
            print(f"    {route:8} {msg}")
            failures.append(f"{route}:{msg}")
            continue

        # 0.1) ID 一致性：result/trace 的 question_id 完全相等（含顺序）、无重复
        id_problems = _check_route_identity(results, traces)
        # 0.2) 预期条数：正式全量必须为 80 条（--expected-count 指定时）
        if args.expected_count is not None and len(results) != args.expected_count:
            id_problems.append(
                f"记录数 {len(results)} != 预期 {args.expected_count}")
        # 0.3) manifest 状态：status 必须为 complete
        stem = f"fixed_{route}_r{args.repeat}_s{args.seed}_{args.contract_tag}_{args.snapshot}"
        id_problems.extend(_check_manifest(base, stem, route))
        if id_problems:
            for p in id_problems:
                print(f"    {route:8} {p}")
                failures.append(f"{route}:{p}")
            continue

        route_fail = 0
        for idx, (res, trace) in enumerate(zip(results, traces)):
            problems = []
            ctx = res.get("context") or ""
            real = counter(ctx)
            reported = (trace.get("stages") or {}).get("tokens_after_truncation")
            if reported is None:
                problems.append("tokens_after_truncation 缺失")
            elif abs(real - reported) > TOKEN_TOLERANCE:
                problems.append(f"上报 token {reported} != 真实 {real}")

            cap = None
            if route != GOLD_ROUTE_ID:
                cap = contract.route(route).final_context_hard_cap
                if cap and real > cap:
                    problems.append(f"超硬上限 {real} > {cap}")

            finish = trace.get("finish_reason")
            if finish is None:
                problems.append("finish_reason 为 null")

            # 实验输入锁定：每条运行记录必须带问题集哈希；P_gold 还必须带 gold 哈希
            if res.get("question_file_hash") is None:
                problems.append("question_file_hash 缺失（实验输入未锁定）")
            if route == GOLD_ROUTE_ID and res.get("gold_context_file_hash") is None:
                problems.append("gold_context_file_hash 缺失（P_gold 输入未锁定）")

            prov_problems = _check_provenance(trace)
            problems.extend(prov_problems)
            stage_problems = _check_stages(trace)
            problems.extend(stage_problems)

            qid = res.get("question_id") or trace.get("question_id") or "?"
            verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
            print(f"    {idx:>3} {str(qid):>12} {real:>9} {str(reported):>9} "
                  f"{str(cap):>9} {str(finish):>8} "
                  f"{'OK' if not prov_problems else 'NG':>5} "
                  f"{'OK' if not stage_problems else 'NG':>7}  {verdict}")
            if problems:
                route_fail += 1
                failures.append(f"{route}[{qid}]: {'; '.join(problems)}")

        print(f"    {route:8} 汇总: {len(results)} 条记录，"
              f"{'全部通过' if route_fail == 0 else f'{route_fail} 条失败'}")
        if route_fail == 0:
            print()

    print()
    if failures:
        print(f"验收未通过（{len(failures)} 项）：")
        for f in failures:
            print("  -", f)
        return 1
    print("全部检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
