import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 让脚本可以直接导入项目根目录下的 hyperrag 包和 my_config.py。
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop, limit_async_func_call
from hyperrag.experiment_contract import load_contract
from hyperrag.fixed_route_executor import execute_fixed_route
from hyperrag.trace_collector import (
    build_trace_record,
    validate_and_finalize,
    SchemaValidationError,
)
from hyperrag.system_snapshot import compute_file_hash
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
try:
    from .pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """给 HyperRAG 查询阶段注入的 LLM 调用函数。

    查询时 LLM 会被用于关键词抽取、上下文组织后的最终回答生成等步骤。
    openai_complete_if_cache 会结合 HyperRAG 的 llm_response_cache 做缓存。
    """
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    """给 HyperRAG 查询阶段注入的 embedding 函数。

    查询问题会被 embedding 后拿去查 chunks/entities/relationships 向量库。
    这里的 EMB_DIM 必须和 Step_1 建库时使用的 embedding 维度一致。
    """
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def extract_queries(file_path):
    """读取 Step_2 生成的问题列表。"""
    with open(file_path, "r", encoding="utf-8") as file:
        query_list = json.load(file)
    return query_list


def extract_queries_v2(file_path):
    """读取最终锁定题集（JSONL 或 JSON 数组），返回 (queries, question_ids)。

    - ``.jsonl``：逐行解析，每行是一个完整 question 对象（含 question_id、question 等）。
    - ``.json``  ：JSON 数组；元素为 dict 时取 question_id / question，
                   元素为 str 时退化为旧格式（question_id 用整数索引）。

    Returns
    -------
    queries : list[str]  — 每题的问题文本
    question_ids : list[str] — 每题的真实 question_id（旧格式退化为 str(index)）
    """
    p = Path(file_path)
    queries: list[str] = []
    question_ids: list[str] = []
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = obj.get("question_id") or str(idx)
                q = obj.get("question") or ""
                if not q:
                    raise ValueError(f"question_id={qid} (line {idx}) has empty question text")
                queries.append(q)
                question_ids.append(str(qid))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                qid = item.get("question_id") or str(idx)
                q = item.get("question") or ""
                if not q:
                    raise ValueError(f"question_id={qid} (index {idx}) has empty question text")
            else:
                qid = str(idx)
                q = str(item)
            queries.append(q)
            question_ids.append(qid)
    return queries, question_ids


def _write_run_manifest(manifest_path: str, data: dict):
    """写出每路径的实验 manifest（题集哈希、代码指纹、参数、计数、时间）。"""
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 断点续跑 / 运行身份 / 产物一致性（Step 1.5 实验运行基础设施）
#
# 断点续跑必须按 **run_identity**（六要素）判定，而非只看 question_id：
# 同一 qid 但换 snapshot / seed / 题集 / 契约时，旧记录是其它实验的产物，
# 不得复用，必须清理后重跑；同 qid 且六要素全一致，才是真正的中断续跑。
#
# result.jsonl 每行内嵌完整 trace 记录（"trace" 字段），它是唯一事实源；
# trace.jsonl 仅为冗余派生副本，任何时刻可由 result 重建（_rebuild_*）。
# 因此写入顺序固定为"先 result 后 trace"：若进程在两者之间崩溃，resume
# 阶段以 result 为准重建 trace 行，绝不产生"有 trace 无 result"的悬空记录。
# ---------------------------------------------------------------------------

def _run_identity(route_id, repeat_id, seed, contract_version,
                  system_snapshot_id, question_file_hash):
    """当前运行的身份元组：六要素完全一致才算同一批实验的断点续跑。"""
    return (
        route_id, repeat_id, seed, contract_version,
        system_snapshot_id, question_file_hash,
    )


def _record_identity(rec: dict):
    """从一条 result 记录提取身份元组；字段不全（无法判定）返回 None。"""
    keys = ("route_id", "repeat_id", "seed", "contract_version",
            "system_snapshot_id", "question_file_hash")
    try:
        vals = tuple(rec.get(k) for k in keys)
    except AttributeError:
        return None
    if any(v is None for v in vals):
        return None
    return vals


def _load_result_records(jsonl_path: str) -> list:
    """读取 result JSONL 全部有效记录（损坏行跳过），保持文件内顺序。"""
    records = []
    p = Path(jsonl_path)
    if not p.exists():
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _classify_existing_records(records: list, identity):
    """把已有 result 记录分成两批：

    - processed：身份与当前 run 完全一致 -> 断点续跑时跳过（不重跑）；
    - stale：身份不一致（或字段不全无法判定）-> 属其它实验产物，须清理重跑。
    """
    processed, stale = set(), set()
    for rec in records:
        qid = rec.get("question_id")
        if qid is None:
            continue  # 无 qid 的行无法归属，交由 _rewrite_jsonl_keep 剔除
        if _record_identity(rec) == identity:
            processed.add(str(qid))
        else:
            stale.add(str(qid))
    return processed, stale


def _rewrite_jsonl_keep(jsonl_path: str, keep_qids: set):
    """重写 JSONL，仅保留 question_id 在 keep_qids 中的行（文件不存在则无操作）。

    用于清理异身份旧记录：result / trace / errors 三个文件都按 qid 对齐。
    """
    p = Path(jsonl_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    kept = []
    for ln in lines:
        try:
            rec = json.loads(ln)
            qid = rec.get("question_id")
        except json.JSONDecodeError:
            continue
        if qid is not None and str(qid) in keep_qids:
            kept.append(ln)
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(kept)


def _rebuild_trace_from_results(result_file: str, trace_file: str):
    """让 trace.jsonl 与 result.jsonl 完全对齐（result 内嵌 trace 为唯一事实源）。

    - **始终** 以 result 行内嵌的 "trace" 字段重建每一行；已有 trace 文件
      的内容一律忽略——它是纯派生副本，任何时刻都可由 result 完整重建，
      不得存在"以旧 trace 优先"的语义。
    - result 中缺失内嵌 trace 的记录不产出 trace 行（事实源缺失必须暴露，
      由调用方校验两文件行数是否对齐）。
    - trace_file 不存在时由 result 全量派生。
    """
    if not trace_file:
        return
    records = _load_result_records(result_file)
    lines = []
    for rec in records:
        qid = rec.get("question_id")
        if qid is None:
            continue
        tr = rec.get("trace")
        if isinstance(tr, dict) and tr:
            lines.append(json.dumps(tr, ensure_ascii=False) + "\n")
    with open(trace_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_record_pair(rf, tf, out: dict, record: dict):
    """成对写入 result + trace（先 result 后 trace，返回前双 flush）。

    result 行内嵌完整 trace（事实源）；trace.jsonl 为冗余副本，任何时刻
    可由 result 重建。若进程在两次写入之间崩溃，resume 阶段以 result 为准
    重建 trace 行，保证两文件永远一一对应。
    """
    rf.write(json.dumps(out, ensure_ascii=False) + "\n")
    rf.flush()
    if record is not None and tf is not None:
        tf.write(json.dumps(record, ensure_ascii=False) + "\n")
        tf.flush()


def _compute_run_status(cumulative_processed: int, requested_count: int,
                        error_count: int) -> str:
    """实验批次状态：累计成功数覆盖本次配置题数且无错误 -> complete。

    - 被 --max-questions 截断的 smoke 批次：requested_count 是截断后的题数，
      状态可为 complete，但 manifest.truncated=True 明确标记不可当全量。
    - 全量批次中途失败：error_count>0 或累计数不足 -> incomplete。
    """
    if cumulative_processed >= requested_count and error_count == 0:
        return "complete"
    return "incomplete"


def _check_locked_question_hash(expected_sha256: str, actual_sha256: str) -> bool:
    """题集 SHA-256 锁定校验（fail-fast 的前置判定）。

    一致返回 True（并打印锁定确认）；不一致打印 FATAL 并返回 False，
    由调用方立即 sys.exit(2)——必须在任何 stale 清理/续跑逻辑之前调用。
    """
    if expected_sha256 == actual_sha256:
        print(f"[locked] question file SHA-256 verified: {actual_sha256}")
        return True
    print(f"FATAL: question file SHA-256 mismatch — "
          f"expected {expected_sha256}, actual {actual_sha256}")
    print("       拒绝运行：题集内容与锁定哈希不符，不产出也不清理任何实验产物。")
    return False


async def process_query(query_text, rag_instance, query_param):
    """对单个问题调用 HyperRAG.aquery，并把成功/失败结果拆开返回。

    当 query_param.save_trace=True 时，还会返回 trace 数据。
    """
    try:
        # 这是 Step_3 进入核心方法本体的位置。
        # query_param.mode 决定会走 hyper、hyper-lite 还是 naive。
        result = await rag_instance.aquery(query_text, param=query_param)
        # 提取 trace 数据并清空，防止下一题串数据
        trace = None
        if query_param.save_trace and query_param.trace_data:
            trace = dict(query_param.trace_data)
            query_param.trace_data.clear()
        return {"query": query_text, "result": result}, None, trace
    except Exception as e:
        print("error", e)
        # 清空可能残留的 trace
        query_param.trace_data.clear()
        return None, {"query": query_text, "error": str(e)}, None


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file, meta_list=None,
    trace_file=None,
):
    """批量执行问题，并把正常结果和错误结果分别写入文件。

    输出目录默认是 caches/<data_name>/response/。
    例如 mode=hyper 时，会写出 hyper_2_stage_result.json 和
    hyper_2_stage_errors.json。

    当 meta_list 不为 None 时（oracle 模式），每题从 meta 读取
    expected_complexity 并注入 query_param.forced_complexity。

    当 trace_file 不为 None 时，每题的检索 trace 写入 JSONL 文件。
    """
    loop = always_get_an_event_loop()

    with open(output_file, "w", encoding="utf-8") as result_file, open(
        error_file, "w", encoding="utf-8"
    ) as err_file:
        trace_f = open(trace_file, "w", encoding="utf-8") if trace_file else None

        # 手动流式写 JSON 数组，避免所有结果都积在内存里。
        result_file.write("[\n")
        first_entry = True

        for i, query_text in enumerate(tqdm(queries, desc="Processing queries", unit="query")):
            # oracle 模式：逐题注入 forced_complexity
            if meta_list and hasattr(query_param, "forced_complexity"):
                expected = meta_list[i].get("expected_complexity", "medium") if i < len(meta_list) else "medium"
                query_param.forced_complexity = expected

            result, error, trace = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )
            if result:
                # 正常回答写入 result 文件。
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                # 单题失败不会中断整个批处理，而是写入 errors 文件方便排查。
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

            # 写入 trace 数据（如果启用了 save_trace）
            if trace and trace_f:
                trace["question_id"] = i
                trace["query"] = query_text
                trace_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_f.flush()

        result_file.write("\n]")
        if trace_f:
            trace_f.close()


def run_fixed_route_and_save(
    queries, rag_instance, contract, route_id, repeat_id, seed,
    system_snapshot_id, output_file, error_file, trace_file=None,
    save_trace=False, validate_trace=False, gold_context=None,
    input_hashes=None,
):
    """固定路径批量执行：调用 execute_fixed_route，组装并（可选）校验 trace。

    与 run_queries_and_save_to_json 的区别：
    - 检索 + 统一回答由 execute_fixed_route 内部完成（共享 prompt hash）。
    - 每题从 retrieve 阶段采集的 trace_data 组装成 retrieval-trace-v2 记录；
      当 --validate-trace 开启且记录不合法时，该题标记失败（写入 errors），
      禁止静默保存残缺 trace。
    - 输出文件名由调用方带 route / repeat / 契约版本 / snapshot 标识。
    - input_hashes：{"question_file_hash", "gold_context_file_hash"}，锁定
      本次运行的实际实验输入（问题集与 P_gold 证据），写入每条运行记录。
    """
    input_hashes = input_hashes or {}
    loop = always_get_an_event_loop()
    cv = contract.contract_version

    with open(output_file, "w", encoding="utf-8") as result_file, open(
        error_file, "w", encoding="utf-8"
    ) as err_file:
        trace_f = open(trace_file, "w", encoding="utf-8") if trace_file else None

        result_file.write("[\n")
        first_entry = True

        for i, query_text in enumerate(tqdm(queries, desc="Fixed route", unit="query")):
            trace_data = {}
            try:
                res = loop.run_until_complete(
                    execute_fixed_route(
                        query_text, route_id, contract, rag_instance,
                        repeat_id=repeat_id, repeat_seed=seed,
                        system_snapshot_id=system_snapshot_id,
                        save_trace=save_trace, trace_data=trace_data,
                        gold_context=gold_context,
                    )
                )
                # 组装 trace 记录（仅 save_trace 时有意义数据）
                record = None
                if save_trace:
                    record = build_trace_record(
                        trace_data,
                        question_id=str(i),
                        run_id=f"{route_id}_r{repeat_id}_s{seed}_{i}",
                        route_id=route_id,
                        seed=seed,
                        system_snapshot_id=system_snapshot_id or "",
                        answer_text=res["answer"],
                        finish_reason=res.get("finish_reason"),
                        cost=res.get("cost"),
                    )
                    if validate_trace:
                        try:
                            validate_and_finalize(record)
                        except SchemaValidationError as e:
                            # 残缺 trace 禁止保存：该题标记失败
                            err_file.write(json.dumps(
                                {"query": query_text, "error": f"trace_schema: {e}"},
                                ensure_ascii=False, indent=4,
                            ))
                            err_file.write("\n")
                            continue

                out = {
                    "query": query_text,
                    "result": res["answer"],
                    "route_id": route_id,
                    "mode": res["mode"],
                    "context": res["context"],
                    "prompt_template": res["prompt_template"],
                    "prompt_hash": res["prompt_hash"],
                    "contract_version": cv,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "seed_supported": res.get("seed_supported"),
                    "system_snapshot_id": system_snapshot_id,
                    "question_file_hash": input_hashes.get("question_file_hash"),
                    "gold_context_file_hash": input_hashes.get("gold_context_file_hash"),
                    "trace": record,
                }
                if not first_entry:
                    result_file.write(",\n")
                json.dump(out, result_file, ensure_ascii=False, indent=4)
                first_entry = False

                if record and trace_f:
                    trace_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    trace_f.flush()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("error", e)
                err_file.write(json.dumps(
                    {"query": query_text, "error": str(e)},
                    ensure_ascii=False, indent=4,
                ))
                err_file.write("\n")

        result_file.write("\n]")
        if trace_f:
            trace_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对 questions 调用 HyperRAG.aquery，并写入 response/")
    parser.add_argument(
        "--data-name",
        type=str,
        default=DEFAULT_DATA_NAME,
        help=f"HyperRAG working_dir = caches/<name>（默认 {DEFAULT_DATA_NAME!r}）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        choices=["naive", "hyper", "hyper-lite", "adaptive"],
        help="QueryParam.mode（可多次运行本脚本，分别生成各 mode 结果）",
    )
    parser.add_argument(
        "--max-token-for-text-unit",
        type=int,
        default=None,
        help="检索上下文的最大 token 数。不指定时按 mode 自动选择："
             "naive=12000, hyper/hyper-lite=4000",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="问题文件名前缀（不含 .json），如 2_stage / mixed_stage。"
             "默认根据 --question-stage 自动生成",
    )
    parser.add_argument(
        "--question-file-path",
        type=str,
        default=None,
        help="直接指定问题文件完整路径（支持 .jsonl 和 .json）。"
             "指定后忽略 --question-file / --question-stage。"
             "用于读取最终锁定题集等任意路径的题集文件。",
    )
    parser.add_argument(
        "--question-stage",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="问题阶段，对应 <n>_stage.json（默认 2，当 --question-file 指定时被忽略）",
    )
    parser.add_argument(
        "--router-policy",
        type=str,
        default="llm",
        choices=["llm", "fixed", "oracle"],
        help="adaptive 模式下的路由策略：llm=调用Router, fixed=强制固定档位, oracle=按meta的stage设定",
    )
    parser.add_argument(
        "--forced-complexity",
        type=str,
        default=None,
        choices=["simple", "medium", "complex"],
        help="router_policy=fixed 时强制使用的复杂度档位",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="输出文件后缀，如 fixed_medium / oracle。不加时行为与之前一致",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        default=False,
        help="启用后每题记录检索 trace（entity_seed_ids, chunk_ids, context_hash 等），"
             "写入 *_trace.jsonl 文件",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Step 6: 统一 LLM 温度（关键词抽取 / Router / 最终回答）。"
             "固定值使检索管线可复现；该值进入 LLM cache key，不会误用旧缓存。",
    )

    # ------------------------------------------------------------------
    # Step 1.5: 固定路径重复运行控制（P0-P4 + P_gold 独立、稳定、可复现运行）
    # ------------------------------------------------------------------
    parser.add_argument(
        "--fixed-route",
        type=str,
        default=None,
        choices=["P0", "P1", "P2", "P3", "P4", "P_gold"],
        help="Step 1.5: 固定路径模式。指定后从冻结契约加载 P0-P4 检索配置，"
             "经 execute_fixed_route 共享同一 formal_answer_response 模板运行。",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="固定路径模式使用的契约 YAML 路径；默认加载 docs 下冻结契约。",
    )
    parser.add_argument(
        "--repeat-id",
        type=int,
        default=0,
        help="固定路径模式下的 repeat 编号（每次正式重复运行递增）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="固定路径模式下的随机种子（写入 trace / system_snapshot 用于复现）。",
    )
    parser.add_argument(
        "--disable-llm-cache",
        action="store_true",
        default=False,
        help="固定路径模式：重建 rag.llm_model_func 不注入 hashing_kv，"
             "强制每次 LLM 调用直连模型、不命中 llm_response_cache（契约要求 disabled）。",
    )
    parser.add_argument(
        "--system-snapshot",
        type=str,
        default=None,
        help="固定路径模式：写入 trace 的 system_snapshot_id（与 1.7 生成的 snapshot 对应）。",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="固定路径模式：截断问题数（仅 smoke test / 调试用，如 1）。",
    )
    parser.add_argument(
        "--validate-trace",
        action="store_true",
        default=False,
        help="固定路径模式：对组装的 retrieval-trace-v2 记录做 Schema 校验，"
             "不合法则该题标记失败（禁止静默保存残缺 trace）。",
    )
    parser.add_argument(
        "--debug-relax-constraints",
        action="store_true",
        default=False,
        help="【仅调试】固定路径模式默认强制：关闭 LLM cache + 保存并校验 trace + "
             "必填 --system-snapshot。加此 flag 可放宽以上约束，输出文件名带 DEBUG 前缀，"
             "产物不得用于正式实验。",
    )
    parser.add_argument(
        "--gold-context-file",
        type=str,
        default=None,
        help="固定路径模式 P_gold：指定 gold 证据文本文件（每行一题，或单个字符串）。"
             "非 P_gold 忽略；缺省时 P_gold 以空 context 运行（fixture 占位）。",
    )
    parser.add_argument(
        "--expected-question-sha256",
        type=str,
        default=None,
        help="固定路径模式：题集文件 SHA-256 锁定值。提供后与题集文件实际哈希"
             "强制比对，不匹配立即退出（fail-fast，exit 2），绝不清理旧结果后继续运行。"
             "用于正式实验锁定题集字节；smoke 亦应传入以提前暴露题集漂移。",
    )
    args = parser.parse_args()

    # ==================================================================
    # Step 1.5: 固定路径模式（P0-P4 + P_gold）—— 从冻结契约独立、稳定运行
    # ==================================================================
    if args.fixed_route:
        route_id = args.fixed_route
        contract = load_contract(args.contract)  # 默认冻结契约；版本不符抛 ContractError

        # ---- 正式固定路径强制约束（operational-stable-success-v1）----
        # 契约要求：llm_response_cache=disabled、trace 必存必校验、绑定真实 snapshot。
        # 默认强制开启；--debug-relax-constraints 仅供本地调试（输出带 DEBUG 前缀）。
        if not args.debug_relax_constraints:
            if not args.disable_llm_cache:
                print("[enforce] --disable-llm-cache 强制开启（契约要求 llm_response_cache=disabled）")
                args.disable_llm_cache = True
            if not args.save_trace:
                print("[enforce] --save-trace 强制开启（正式运行必须保存 trace）")
                args.save_trace = True
            if not args.validate_trace:
                print("[enforce] --validate-trace 强制开启（禁止静默保存残缺 trace）")
                args.validate_trace = True
            if not args.system_snapshot or args.system_snapshot.strip().lower() == "nosnap":
                print(
                    "ERROR: 固定路径正式运行必须提供 --system-snapshot <snapshot_id>"
                    "（拒绝空值 / 'nosnap' 占位）。\n"
                    "请先生成 system snapshot（hyperrag/system_snapshot.py），"
                    "或加 --debug-relax-constraints 明确进入非正式调试模式。"
                )
                sys.exit(2)

        # 问题文件加载：优先使用 --question-file-path（支持 JSONL 最终锁定题集）
        WORKING_DIR = Path("caches") / args.data_name

        if args.question_file_path:
            question_file_path = Path(args.question_file_path)
            queries, question_ids = extract_queries_v2(question_file_path)
            print(f"Question file: {question_file_path} ({len(queries)} questions, JSONL/dict mode)")
        else:
            # 回退到旧逻辑：--question-file / --question-stage + .json 数组
            if args.question_file:
                question_prefix = args.question_file
            else:
                question_prefix = f"{args.question_stage}_stage"
            question_file_path = WORKING_DIR / f"questions/{question_prefix}.json"
            queries = extract_queries(question_file_path)
            question_ids = [str(i) for i in range(len(queries))]
            print(f"Question file: {question_file_path} ({len(queries)} questions, legacy mode)")

        # 锁定实验输入：问题集文件哈希（gold 文件哈希在加载 gold 后补充）。
        # 写入每条运行记录，保证产物可追溯到确切的输入字节。
        input_hashes = {
            "question_file_hash": compute_file_hash(question_file_path),
            "gold_context_file_hash": None,
        }

        # 题集哈希强制锁定（fail-fast）：必须在任何 stale 清理/续跑逻辑之前。
        # 不匹配说明题集内容已漂移，当前产物不得产出，也绝不能"清掉旧结果继续跑"。
        if args.expected_question_sha256 and not _check_locked_question_hash(
            args.expected_question_sha256, input_hashes["question_file_hash"]
        ):
            sys.exit(2)

        # 题集总题数（截断前）——manifest 用它区分 smoke 截断批次与正式全量批次
        total_question_count = len(queries)

        # smoke test / 调试用：截断问题数
        if args.max_questions is not None:
            queries = queries[: args.max_questions]
            question_ids = question_ids[: args.max_questions]
            print(f"--max-questions={args.max_questions}: truncated to {len(queries)} questions")

        rag = HyperRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
            ),
            llm_model_max_async=32,
            embedding_func_max_async=4,
        )

        # 契约要求 llm_response_cache=disabled：重建 llm_model_func 不注入 hashing_kv，
        # 强制每次 LLM 调用直连模型，不命中历史缓存（保证官方重复运行的真实复现）。
        if args.disable_llm_cache:
            rag.llm_model_func = limit_async_func_call(rag.llm_model_max_async)(
                llm_model_func
            )
            print("LLM response cache DISABLED (no hashing_kv injected)")

        # P_gold 证据文本（fixture 占位；正式 per-question gold 在 Step 2 接入）。
        # 约定：若文件行数 == 题目数，则按行一一对应；否则整文件内容作为共享 gold。
        # fail-fast：Gold 证据缺失/为空时 P_gold 会退化为 P0（空上下文 +
        # 相同回答），产物具有欺骗性。除 --debug-relax-constraints 外一律拒跑。
        gold_context = None
        if route_id == "P_gold":
            def _gold_fatal(msg: str):
                if args.debug_relax_constraints:
                    print(f"WARNING (debug-relax): {msg}; P_gold 将以空 gold 运行，产物仅供调试")
                    return False
                print(f"FATAL: {msg}")
                print("       P_gold 没有非空 Gold 证据时与 P0 行为完全相同，禁止生成正式产物。")
                print("       请提供 --gold-context-file（每题一行或共享文本，均须非空）。")
                sys.exit(2)

            if not args.gold_context_file:
                _gold_fatal("P_gold 需要 --gold-context-file，但未提供")
            else:
                gpath = Path(args.gold_context_file)
                if not gpath.exists():
                    _gold_fatal(f"gold-context-file 不存在: {gpath}")
                else:
                    input_hashes["gold_context_file_hash"] = compute_file_hash(gpath)
                    lines = gpath.read_text(encoding="utf-8").splitlines()
                    if len(lines) == len(queries) and len(queries) > 0:
                        gold_context = [ln.strip() for ln in lines]
                        empty_idx = [i for i, g in enumerate(gold_context) if not g]
                        if empty_idx:
                            _gold_fatal(
                                f"per-question gold 中第 {empty_idx} 行为空："
                                f"每个选中问题都必须有非空 Gold 证据")
                        else:
                            print(f"P_gold: per-question gold context loaded ({len(gold_context)} entries)")
                    else:
                        gold_text = gpath.read_text(encoding="utf-8").strip()
                        if not gold_text:
                            _gold_fatal(f"gold-context-file 内容为空: {gpath}")
                        else:
                            gold_context = gold_text
                            print(f"P_gold: shared gold context loaded ({len(gold_text)} chars)")

        # 输出文件名：含 route / repeat / 契约版本 / snapshot，便于独立归档。
        cv_short = contract.contract_version.replace("question-set-", "").replace("contract-", "")
        snap = args.system_snapshot or "nosnap"
        out_base = f"fixed_{route_id}_r{args.repeat_id}_s{args.seed}_{cv_short}_{snap}"
        if args.debug_relax_constraints:
            out_base = f"DEBUG_{out_base}"

        OUT_DIR = WORKING_DIR / "response"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        save_trace = args.save_trace
        validate_trace = args.validate_trace
        # 结果改 JSONL（每行一条），支持断点续跑追加写
        result_file = OUT_DIR / f"{out_base}_result.jsonl"
        error_file = OUT_DIR / f"{out_base}_errors.jsonl"
        trace_file = str(OUT_DIR / f"{out_base}_trace.jsonl") if save_trace else None
        manifest_file = OUT_DIR / f"{out_base}_manifest.json"

        # --- 断点续跑：按 run_identity（六要素）判定，而非只看 question_id ---
        # 同一 qid 但 identity 不同 -> 旧记录是其它实验的产物，清理后重跑；
        # 同 qid 且 identity 一致 -> 真正的中断续跑，跳过不重跑。
        identity = _run_identity(
            route_id, args.repeat_id, args.seed, contract.contract_version,
            args.system_snapshot, input_hashes.get("question_file_hash"),
        )
        existing = _load_result_records(str(result_file))
        processed_ids, stale_ids = _classify_existing_records(existing, identity)
        if stale_ids:
            print(f"Resume: {len(stale_ids)} stale records (run identity mismatch, "
                  f"e.g. different snapshot/seed/question set), cleaning and re-running")
            _rewrite_jsonl_keep(str(result_file), processed_ids)
            if trace_file:
                _rewrite_jsonl_keep(trace_file, processed_ids)
            _rewrite_jsonl_keep(str(error_file), processed_ids)
            # 清理后重新加载，确认无残留异身份记录
            existing = _load_result_records(str(result_file))
            processed_ids, stale_ids = _classify_existing_records(existing, identity)
            assert not stale_ids, "stale cleanup failed; aborting to avoid duplicate records"
        # trace 与 result 对齐（清孤儿行、补缺失行），保证一一对应
        if trace_file:
            _rebuild_trace_from_results(str(result_file), trace_file)
        if processed_ids:
            print(f"Resume: {len(processed_ids)} questions already processed "
                  f"(same run identity), skipping")
        pending = [(q, question_ids[i], i) for i, q in enumerate(queries)
                   if question_ids[i] not in processed_ids]
        skipped_count = len(processed_ids)

        # --- 逐题执行（统一 JSONL 输出，支持 per-question gold 和 shared/none gold）---
        loop = always_get_an_event_loop()
        file_mode = "a" if processed_ids else "w"
        error_count = 0
        with open(result_file, file_mode, encoding="utf-8") as rf, open(
            error_file, file_mode, encoding="utf-8"
        ) as ef:
            tf = open(trace_file, file_mode, encoding="utf-8") if trace_file else None
            for q, qid, orig_idx in tqdm(pending, desc="Fixed route", unit="query"):
                # gold_context: per-question list -> 按原始索引取；否则整值
                gc = gold_context[orig_idx] if isinstance(gold_context, list) else gold_context
                try:
                    res = loop.run_until_complete(
                        execute_fixed_route(
                            q, route_id, contract, rag,
                            repeat_id=args.repeat_id, repeat_seed=args.seed,
                            system_snapshot_id=(args.system_snapshot or ""),
                            save_trace=save_trace, trace_data={},
                            gold_context=gc,
                        )
                    )
                    record = None
                    if save_trace:
                        # 用执行器返回的规范化 trace_out（res["trace_data"]），
                        # 而非外部容器 —— 它含 P0/P_gold 的 token 补齐等最终修正。
                        record = build_trace_record(
                            res["trace_data"],
                            question_id=str(qid),
                            run_id=f"{route_id}_r{args.repeat_id}_s{args.seed}_{qid}",
                            route_id=route_id, seed=args.seed,
                            system_snapshot_id=(args.system_snapshot or ""),
                            answer_text=res["answer"],
                            finish_reason=res.get("finish_reason"),
                            cost=res.get("cost"),
                        )
                        if validate_trace:
                            try:
                                validate_and_finalize(record)
                            except SchemaValidationError as e:
                                ef.write(json.dumps(
                                    {"question_id": str(qid), "query": q,
                                     "error": f"trace_schema: {e}"},
                                    ensure_ascii=False))
                                ef.write("\n")
                                ef.flush()
                                error_count += 1
                                continue
                    out = {
                        "question_id": str(qid),
                        "query": q, "result": res["answer"], "route_id": route_id,
                        "mode": res["mode"], "context": res["context"],
                        "prompt_template": "formal_answer_response",
                        "prompt_hash": res["prompt_hash"],
                        "contract_version": contract.contract_version,
                        "repeat_id": args.repeat_id, "seed": args.seed,
                        "seed_supported": res.get("seed_supported"),
                        "system_snapshot_id": args.system_snapshot,
                        "question_file_hash": input_hashes.get("question_file_hash"),
                        "gold_context_file_hash": input_hashes.get("gold_context_file_hash"),
                        "trace": record,
                    }
                    # result + trace 成对写入（先 result 后 trace，双 flush）
                    _write_record_pair(rf, tf, out, record)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("error", e)
                    ef.write(json.dumps(
                        {"question_id": str(qid), "query": q, "error": str(e)},
                        ensure_ascii=False))
                    ef.write("\n")
                    ef.flush()
                    error_count += 1
            if tf:
                tf.close()

        # 运行收尾：trace 再对齐一次（覆盖"result 写完、trace 未写"即崩溃的窗口）
        if trace_file:
            _rebuild_trace_from_results(str(result_file), trace_file)

        # --- 写实验 manifest（题集哈希、代码指纹、参数、累计计数、状态、时间）---
        from datetime import datetime, timezone
        from hyperrag.system_snapshot import compute_runtime_source_fingerprint
        from hyperrag.fixed_route_executor import formal_answer_prompt_hash
        src_fp = compute_runtime_source_fingerprint()
        # 累计完成数 = 文件内成功记录数（含本次续跑前已完成的），以 result 为准
        cumulative_processed = len(_load_result_records(str(result_file)))
        requested_count = len(queries)          # 本次运行配置的问题数（截断后）
        truncated = args.max_questions is not None
        status = _compute_run_status(cumulative_processed, requested_count, error_count)
        manifest = {
            "route_id": route_id,
            "repeat_id": args.repeat_id,
            "seed": args.seed,
            "contract_version": contract.contract_version,
            "system_snapshot_id": args.system_snapshot,
            "question_file_path": str(question_file_path),
            "question_file_sha256": input_hashes.get("question_file_hash"),
            "question_file_total": total_question_count,   # 题集总题数（未截断）
            "requested_count": requested_count,            # 本次配置处理题数（截断后）
            "truncated": truncated,                        # True=smoke 截断批次，不可当全量
            "cumulative_processed": cumulative_processed,  # 文件累计成功记录数
            "skipped_count": skipped_count,                # 本次跳过（身份一致已存在）
            "error_count": error_count,                    # 本次失败数
            "status": status,                              # complete / incomplete
            "temperature": contract.system_boundary.temperature,
            "max_response_tokens": contract.system_boundary.max_response_tokens,
            "llm_response_cache": contract.system_boundary.llm_response_cache,
            "runtime_source_fingerprint": src_fp["runtime_source_fingerprint"],
            "runtime_source_file_count": src_fp["runtime_source_file_count"],
            "prompt_hash": formal_answer_prompt_hash(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_run_manifest(str(manifest_file), manifest)

        print(f"Fixed route {route_id} done. Results: {result_file}")
        if trace_file:
            print(f"Trace: {trace_file}")
        print(f"Manifest: {manifest_file}")
        # 运行状态门禁：status=incomplete 必须以非零退出码结束，
        # 让调用方（CI / 批量脚本）能明确感知"本批实验未完整完成"。
        if status != "complete":
            print(f"ERROR: run status={status} (incomplete) — "
                  f"cumulative_processed={cumulative_processed}, "
                  f"requested_count={requested_count}, error_count={error_count}")
            sys.exit(1)
        sys.exit(0)

    data_name = args.data_name
    mode = args.mode

    # 确定问题文件前缀
    if args.question_file:
        question_prefix = args.question_file
    else:
        question_prefix = f"{args.question_stage}_stage"

    # Step_1 建好的全部索引和超图都在这个目录。
    WORKING_DIR = Path("caches") / data_name

    # Step_2 生成的问题文件，是本脚本的输入。
    question_file_path = Path(
        WORKING_DIR / f"questions/{question_prefix}.json"
    )
    queries = extract_queries(question_file_path)

    # 重新实例化 HyperRAG 时，storage 会从 WORKING_DIR 里加载已有的
    # kv_store_*.json、vdb_*.json 和 hypergraph_*.hgdb，而不是重新建库。
    rag = HyperRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        llm_model_max_async=32,
        embedding_func_max_async=4,
    )

    # mode 是实验对照的核心开关：
    # - naive：只查 chunk 向量库，单线检索预算充足；
    # - hyper：查实体、关系和超图上下文，双线各独立检索后合并去重；
    # - hyper-lite：只查实体相关上下文，速度更轻。
    if args.max_token_for_text_unit is not None:
        budget = args.max_token_for_text_unit
    elif mode == "naive":
        budget = 12000
    else:
        # hyper/hyper-lite 双线检索会额外带上实体和关系表。
        # chunk=1000 时每线 8000 在部分问题上仍会超过 qwen-27b 的 24K context，
        # 4000 更稳，必要时可用 --max-token-for-text-unit 显式调参。
        budget = 4000
    query_param = QueryParam(mode=mode, max_token_for_text_unit=budget)

    # Step 6: 统一 LLM 温度
    query_param.llm_temperature = args.temperature

    # Step 5.4: 启用 trace 收集
    if args.save_trace:
        query_param.save_trace = True

    # adaptive 模式下注入 router_policy 和 forced_complexity
    if mode == "adaptive":
        query_param.router_policy = args.router_policy
        if args.router_policy == "fixed":
            query_param.forced_complexity = args.forced_complexity or "medium"

    # oracle 模式需要读取 meta 文件，按每题的 expected_complexity 注入
    meta_list = None
    if mode == "adaptive" and args.router_policy == "oracle":
        meta_path = WORKING_DIR / f"questions/{question_prefix}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
            print(f"Oracle mode: loaded {len(meta_list)} meta entries")
        else:
            print(f"WARNING: oracle mode but meta file not found: {meta_path}")

    # 输出文件名：有 suffix 时追加，如 adaptive_mixed_stage_fixed_medium_result.json
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_name = f"{mode}_{question_prefix}{suffix}"

    OUT_DIR = WORKING_DIR / "response"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # trace 文件路径（仅当 --save-trace 启用时使用）
    trace_path = OUT_DIR / f"{output_name}_trace.jsonl" if args.save_trace else None

    run_queries_and_save_to_json(
        queries,
        rag,
        query_param,
        OUT_DIR / f"{output_name}_result.json",
        OUT_DIR / f"{output_name}_errors.json",
        meta_list=meta_list,
        trace_file=str(trace_path) if trace_path else None,
    )
    print(f"Results saved to {OUT_DIR / f'{output_name}_result.json'}")
    if trace_path:
        print(f"Trace saved to {trace_path}")
