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

        # 问题文件前缀（与原始路径一致）
        if args.question_file:
            question_prefix = args.question_file
        else:
            question_prefix = f"{args.question_stage}_stage"

        WORKING_DIR = Path("caches") / args.data_name
        question_file_path = WORKING_DIR / f"questions/{question_prefix}.json"
        queries = extract_queries(question_file_path)

        # 锁定实验输入：问题集文件哈希（gold 文件哈希在加载 gold 后补充）。
        # 写入每条运行记录，保证产物可追溯到确切的输入字节。
        input_hashes = {
            "question_file_hash": compute_file_hash(question_file_path),
            "gold_context_file_hash": None,
        }

        # smoke test / 调试用：截断问题数
        if args.max_questions is not None:
            queries = queries[: args.max_questions]
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
            # 调试产物显式打标，禁止混入正式实验结果
            out_base = f"DEBUG_{out_base}"

        OUT_DIR = WORKING_DIR / "response"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        save_trace = args.save_trace
        validate_trace = args.validate_trace
        trace_file = str(OUT_DIR / f"{out_base}_trace.jsonl") if save_trace else None

        # gold_context 支持 per-question（list）或共享（str）；
        # run_fixed_route_and_save 当前按"本次运行统一 gold_context"传入，
        # 当 gold_context 为 list 时逐题切片（Step 2 正式映射复用此逻辑）。
        result_file = OUT_DIR / f"{out_base}_result.json"
        error_file = OUT_DIR / f"{out_base}_errors.json"

        # 逐题执行（gold_context 为列表时按索引切片）
        if isinstance(gold_context, list):
            loop = always_get_an_event_loop()
            with open(result_file, "w", encoding="utf-8") as rf, open(
                error_file, "w", encoding="utf-8"
            ) as ef:
                tf = open(trace_file, "w", encoding="utf-8") if trace_file else None
                rf.write("[\n")
                first = True
                for i, q in enumerate(tqdm(queries, desc="Fixed route", unit="query")):
                    try:
                        res = loop.run_until_complete(
                            execute_fixed_route(
                                q, route_id, contract, rag,
                                repeat_id=args.repeat_id, repeat_seed=args.seed,
                                system_snapshot_id=(args.system_snapshot or ""),
                                save_trace=save_trace, trace_data={},
                                gold_context=gold_context[i],
                            )
                        )
                        record = None
                        if save_trace:
                            record = build_trace_record(
                                dict(res["trace_data"]),
                                question_id=str(i),
                                run_id=f"{route_id}_r{args.repeat_id}_s{args.seed}_{i}",
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
                                        {"query": q, "error": f"trace_schema: {e}"},
                                        ensure_ascii=False, indent=4))
                                    ef.write("\n")
                                    continue
                        out = {
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
                        if not first:
                            rf.write(",\n")
                        json.dump(out, rf, ensure_ascii=False, indent=4)
                        first = False
                        if record and tf:
                            tf.write(json.dumps(record, ensure_ascii=False) + "\n")
                            tf.flush()
                    except Exception as e:
                        print("error", e)
                        ef.write(json.dumps({"query": q, "error": str(e)},
                                            ensure_ascii=False, indent=4))
                        ef.write("\n")
                rf.write("\n]")
                if tf:
                    tf.close()
        else:
            run_fixed_route_and_save(
                queries, rag, contract, route_id,
                repeat_id=args.repeat_id, seed=args.seed,
                system_snapshot_id=(args.system_snapshot or ""),
                output_file=str(result_file), error_file=str(error_file),
                trace_file=trace_file, save_trace=save_trace,
                validate_trace=validate_trace, gold_context=gold_context,
                input_hashes=input_hashes,
            )

        print(f"Fixed route {route_id} done. Results: {result_file}")
        if trace_file:
            print(f"Trace: {trace_file}")
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
