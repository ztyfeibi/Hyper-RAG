"""_verify_longcat.py - LongCat-2.0 (SiliconFlow) 可用性验证（冻结设计 10.6 前置）

验证项:
  1. API 可调用（模型名/基本 chat）
  2. temperature=0 + max_tokens=3000 行为
  3. response_format=json_object 结构化 JSON 稳定性
  4. 长上下文（模拟 judge 输入: query + gold context + answer）
  5. 超时/重试策略可控性（记录每次调用耗时）
不打印任何 API key。
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from openai import OpenAI
from my_config import (LLM_BASE_URL_SILICONFLOW, LLM_API_KEY_SILICONFLOW,
                       LLM_MODEL_SILICONFLOW)

client = OpenAI(api_key=LLM_API_KEY_SILICONFLOW,
                base_url=LLM_BASE_URL_SILICONFLOW, timeout=180)


def call(prompt, *, max_tokens=3000, temperature=0.0, json_mode=True):
    kw = {}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL_SILICONFLOW,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens, **kw,
    )
    dt = time.time() - t0
    usage = resp.usage
    return {
        "latency": round(dt, 1),
        "content": resp.choices[0].message.content,
        "finish": resp.choices[0].finish_reason,
        "in_tokens": usage.prompt_tokens if usage else None,
        "out_tokens": usage.completion_tokens if usage else None,
    }


def main():
    print(f"== LongCat 验证 == model={LLM_MODEL_SILICONFLOW}")
    print(f"== base_url={LLM_BASE_URL_SILICONFLOW}")

    # 1. 基本调用 + temperature=0
    r = call('Reply with exactly: "OK"', json_mode=False, max_tokens=20)
    print(f"[1] basic+temp0: ok={r['content'] == 'OK'} latency={r['latency']}s "
          f"finish={r['finish']} in={r['in_tokens']} out={r['out_tokens']}")

    # 2. JSON 模式最小用例
    r = call('Return a JSON object: {"verdict": "pass"}', max_tokens=200)
    ok = False
    try:
        d = json.loads(r["content"])
        ok = d.get("verdict") == "pass"
    except Exception:
        pass
    print(f"[2] json_mode min: ok={ok} latency={r['latency']}s "
          f"content={r['content'][:60]!r}")

    # 3. 完整 judge 契约 JSON（模拟正式 Judge 输出结构）
    judge_prompt = (
        'You are an evidence-based judge. Given the question, evidence, and answer, '
        'evaluate the answer. Return a strict JSON object with exactly these fields: '
        '{"verdict": "pass" or "fail" or "uncertain", '
        '"answer_units": [{"unit_id": "AU1", "status": "supported" or "missing" or "contradicted"}], '
        '"unsupported_claims": [], "critical_error": false, "evidence_sufficient": true} '
        'Do not include any text outside the JSON.\n'
        'Question: What is the role of GABA in epilepsy?\n'
        'Evidence: GABA is the main inhibitory neurotransmitter in the central nervous system. '
        'Reduced GABAergic inhibition is associated with seizure generation.\n'
        'Answer: GABA acts as the primary inhibitory neurotransmitter; its reduced function '
        'contributes to seizure susceptibility.\n'
    )
    r = call(judge_prompt, max_tokens=1000)
    parsed = None
    try:
        parsed = json.loads(r["content"])
    except Exception as e:
        print(f"  [3] raw parse failed: {e!r}")
    ok3 = (isinstance(parsed, dict) and parsed.get("verdict") in ("pass", "fail", "uncertain")
           and isinstance(parsed.get("answer_units"), list)
           and isinstance(parsed.get("critical_error"), bool)
           and isinstance(parsed.get("evidence_sufficient"), bool))
    print(f"[3] judge-schema json: ok={ok3} latency={r['latency']}s "
          f"finish={r['finish']} in={r['in_tokens']} out={r['out_tokens']}")
    if parsed:
        print(f"    parsed verdict={parsed.get('verdict')} AUs={parsed.get('answer_units')}")

    # 4. 长上下文（模拟 judge 输入: gold context ~8K chars + answer ~4K chars）
    gold = " ".join(f"Evidence sentence {i} describing neurology findings."
                    for i in range(300))          # ~8.7K chars
    ans = " ".join(f"Answer detail {i} explaining the condition." for i in range(200))
    long_prompt = (
        'Return a strict JSON object: {"verdict": "pass", "reason_len": 0}. '
        'Question: q1\nEvidence: ' + gold + '\nAnswer: ' + ans
    )
    r = call(long_prompt, max_tokens=500)
    print(f"[4] long-context: latency={r['latency']}s finish={r['finish']} "
          f"in={r['in_tokens']} out={r['out_tokens']} (in_tokens 应含全部证据)")

    # 5. 并发稳定性（3 并发 × 2 次，检查限流/超时）
    import concurrent.futures as cf
    def one(i):
        return call(f'Return JSON: {{"n": {i}}}', max_tokens=100)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(one, i) for i in range(6)]
        results = [f.result() for f in futs]
    dts = [r["latency"] for r in results]
    print(f"[5] concurrent x6: total={time.time()-t0:.1f}s latencies={dts}")

    print("== 验证完成 ==")


if __name__ == "__main__":
    main()
