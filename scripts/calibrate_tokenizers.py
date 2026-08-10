# -*- coding: utf-8 -*-
"""Qwen tokenizer 与 tiktoken 校准脚本 (Step 1.1)。

契约规定：正式 token 预算以 Qwen tokenizer 为准，tiktoken 仅作近似。
本脚本量化两者差异，输出校准系数供预算安全边界使用。

Qwen 权威计数来源（按优先级）：
  1. vLLM /tokenize 端点（LLM_BASE_URL 去掉 /v1 后缀）—— 部署端真实 tokenizer，最权威
  2. 本地 transformers AutoTokenizer（环境变量 QWEN_TOKENIZER_PATH 指定路径/名称）

两个来源都不可用时 **直接失败退出**，绝不静默回退到 tiktoken 假装校准。

样本来源：
  - caches/<data_name>/kv_store_text_chunks.json 真实语料随机抽样
  - 内置合成样本（中文/英文/中英混合/含公式代码）

输出：caches/calibration/tokenizer_calibration.json
  - 每个样本的 qwen_tokens / tiktoken_tokens / ratio
  - 聚合统计：ratio 的 mean/median/p95/max
  - 建议的 safety_factor（用 tiktoken 预算时应乘的收缩系数）

用法（先激活 conda hyperrag 环境）：
  python scripts/calibrate_tokenizers.py
  python scripts/calibrate_tokenizers.py --data-name neurology_chunk1000 --num-samples 50
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import my_config  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "caches" / "calibration" / "tokenizer_calibration.json"

# 代码库中 context_budget.py / utils.py 实际使用的 tiktoken 模型名
TIKTOKEN_MODEL_NAME = "gpt-4o-mini"

SYNTHETIC_SAMPLES = {
    "synthetic_english": (
        "Ischemic stroke occurs when a blood vessel supplying the brain is "
        "obstructed, most commonly by a thrombus or embolus, leading to focal "
        "neurological deficits whose pattern reflects the affected vascular territory."
    ),
    "synthetic_chinese": (
        "缺血性脑卒中是由于脑部供血动脉阻塞导致的局灶性神经功能缺损，"
        "常见病因包括动脉粥样硬化性血栓形成、心源性栓塞以及小血管闭塞，"
        "临床表现取决于受累血管的供血区域。"
    ),
    "synthetic_mixed": (
        "患者 MRI 显示 left MCA territory 急性梗死，NIHSS 评分 12 分，"
        "在 4.5h 时间窗内给予 rt-PA 静脉溶栓治疗（0.9 mg/kg, max 90 mg）。"
    ),
    "synthetic_structured": (
        '{"entity_name": "EPILEPSY", "entity_type": "disease", "description": '
        '"A chronic neurological disorder characterized by recurrent unprovoked '
        'seizures due to abnormal synchronized neuronal discharges."}'
    ),
}


class QwenCounterVLLM:
    """通过 vLLM /tokenize 端点计数（部署端真实 tokenizer）。"""

    source_name = "vllm_tokenize_endpoint"

    def __init__(self, base_url: str, model: str, timeout: float = 15.0):
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3].rstrip("/")
        self.url = root + "/tokenize"
        self.model = model
        self.timeout = timeout

    def probe(self) -> bool:
        try:
            return self.count("hello") > 0
        except Exception:  # noqa: BLE001
            return False

    def count(self, text: str) -> int:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return int(body["count"])


class QwenCounterLocal:
    """通过本地 transformers AutoTokenizer 计数。"""

    source_name = "local_transformers"

    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer  # 延迟导入，未安装时由调用方处理

        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def resolve_qwen_counter():
    """按优先级解析 Qwen 权威计数器。全部失败则返回 None（由 main 硬失败）。"""
    counter = QwenCounterVLLM(my_config.LLM_BASE_URL, my_config.LLM_MODEL)
    if counter.probe():
        print(f"[qwen] 使用 vLLM /tokenize 端点: {counter.url} (model={counter.model})")
        return counter
    print(f"[qwen] vLLM /tokenize 不可达: {counter.url}")

    import os
    tok_path = os.environ.get("QWEN_TOKENIZER_PATH", "").strip()
    if tok_path:
        try:
            counter = QwenCounterLocal(tok_path)
            print(f"[qwen] 使用本地 transformers tokenizer: {tok_path}")
            return counter
        except Exception as e:  # noqa: BLE001
            print(f"[qwen] 本地 tokenizer 加载失败: {type(e).__name__}: {e}")
    else:
        print("[qwen] 未设置 QWEN_TOKENIZER_PATH，跳过本地 tokenizer")
    return None


def load_corpus_samples(data_name: str, num_samples: int, seed: int):
    """从真实语料 kv_store_text_chunks.json 随机抽样。"""
    path = REPO_ROOT / "caches" / data_name / "kv_store_text_chunks.json"
    if not path.exists():
        print(f"[corpus] 未找到 {path}，仅使用合成样本")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(data.keys())
    rng = random.Random(seed)
    picked = rng.sample(keys, min(num_samples, len(keys)))
    samples = {}
    for k in picked:
        content = data[k].get("content", "")
        if content:
            samples[f"corpus:{k[:20]}"] = content
    print(f"[corpus] 从 {data_name} 抽样 {len(samples)} 条真实 chunk")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Qwen vs tiktoken 校准")
    parser.add_argument("--data-name", default="neurology_chunk1000")
    parser.add_argument("--num-samples", type=int, default=30, help="真实语料抽样数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    qwen = resolve_qwen_counter()
    if qwen is None:
        print("[FATAL] 无法获得 Qwen tokenizer 计数来源。")
        print("        请确认 vLLM 服务可达，或设置 QWEN_TOKENIZER_PATH 指向本地 Qwen tokenizer。")
        print("        依据契约，禁止静默回退 tiktoken —— 校准失败即退出。")
        sys.exit(1)

    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(TIKTOKEN_MODEL_NAME)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    samples = dict(SYNTHETIC_SAMPLES)
    samples.update(load_corpus_samples(args.data_name, args.num_samples, args.seed))

    rows = []
    for name, text in samples.items():
        q = qwen.count(text)
        t = len(enc.encode(text))
        if t == 0:
            continue
        rows.append({
            "sample": name,
            "chars": len(text),
            "qwen_tokens": q,
            "tiktoken_tokens": t,
            "ratio_qwen_over_tiktoken": round(q / t, 4),
        })

    if not rows:
        print("[FATAL] 无有效样本")
        sys.exit(1)

    ratios = [r["ratio_qwen_over_tiktoken"] for r in rows]
    ratios_sorted = sorted(ratios)
    p95 = ratios_sorted[min(len(ratios_sorted) - 1, int(round(0.95 * (len(ratios_sorted) - 1))))]
    max_ratio = max(ratios)
    # safety_factor: 用 tiktoken 做预算时应乘的收缩系数，保证 Qwen 实际计数不超预算
    # 取 p95 与 1.0 的较大值的倒数（ratio<1 时无需收缩）
    safety_factor = round(1.0 / max(p95, 1.0), 4)

    report = {
        "calibration_version": "tokenizer-calibration-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "qwen_source": qwen.source_name,
        "qwen_model": my_config.LLM_MODEL,
        "tiktoken_model_name": TIKTOKEN_MODEL_NAME,
        "tiktoken_encoding": enc.name,
        "num_samples": len(rows),
        "corpus_data_name": args.data_name,
        "seed": args.seed,
        "ratio_stats": {
            "mean": round(statistics.mean(ratios), 4),
            "median": round(statistics.median(ratios), 4),
            "stdev": round(statistics.stdev(ratios), 4) if len(ratios) > 1 else 0.0,
            "min": round(min(ratios), 4),
            "p95": round(p95, 4),
            "max": round(max_ratio, 4),
        },
        "recommended_safety_factor": safety_factor,
        "note": ("ratio = qwen_tokens / tiktoken_tokens; "
                 "用 tiktoken 计数 x budget 时，应确保 budget * p95 <= Qwen hard cap，"
                 "或等价地将 tiktoken 预算乘以 recommended_safety_factor。"),
        "samples": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[stats] samples={len(rows)} "
          f"mean={report['ratio_stats']['mean']} median={report['ratio_stats']['median']} "
          f"p95={report['ratio_stats']['p95']} max={report['ratio_stats']['max']}")
    print(f"[stats] recommended_safety_factor={safety_factor}")
    print(f"[report] 写入 {args.output}")


if __name__ == "__main__":
    main()
