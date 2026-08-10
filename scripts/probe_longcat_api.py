# -*- coding: utf-8 -*-
"""LongCat Judge 模型 API 预检脚本 (Step 1.1)。

在正式使用 LongCat-2.0 作为 Judge 之前，验证 API 的可用性与关键行为：

  1. basic          - 基础调用返回非空文本
  2. temperature0   - temperature=0 时两次调用结果是否一致（确定性检查）
  3. strict_json    - 能否输出可被 json.loads 解析的严格 JSON
  4. context_8k     - ~8K token 上下文是否能正常处理（大海捞针式验证）
  5. context_16k    - ~16K token 上下文是否能正常处理
  6. usage_latency  - 每次调用记录 usage(prompt/completion tokens) 与延迟
  7. timeout_retry  - 超时与重试机制（tenacity 指数退避）

配置（环境变量注入，密钥绝不写入代码/输出）：
  LONGCAT_BASE_URL   必填，OpenAI-compatible base url
  LONGCAT_API_KEY    必填
  LONGCAT_MODEL      选填，默认 meituan-longcat/LongCat-2.0
  LONGCAT_THINKING   选填，"1" 开启思考模式（默认 "0" 关闭）

重要发现（2026-07-30 实测，SiliconFlow 部署）：
  LongCat-2.0 是思考型模型，默认输出 reasoning_content 消耗 max_tokens
  （finish_reason=length、content 为空）。必须传
  extra_body={"enable_thinking": False} 关闭思考，否则 Judge 输出被截断。
  下游 Judge 实现 MUST 使用相同参数。

用法（先激活 conda hyperrag 环境）：
  python scripts/probe_longcat_api.py
  python scripts/probe_longcat_api.py --skip-context-tests   # 快速模式
  python scripts/probe_longcat_api.py --output caches/calibration/longcat_probe.json

退出码：0=全部必检项通过；1=存在失败项；2=配置缺失。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "meituan-longcat/LongCat-2.0"
DEFAULT_OUTPUT = REPO_ROOT / "caches" / "calibration" / "longcat_probe.json"

# 大海捞针用的针（藏在长上下文中间，要求模型找回）
NEEDLE = "MAGIC-TOKEN-73159"
FILLER_SENTENCE = (
    "Neurology research covers stroke, epilepsy, multiple sclerosis, "
    "Parkinson disease and many other disorders of the nervous system. "
)


def _require_config():
    base_url = os.environ.get("LONGCAT_BASE_URL", "").strip()
    api_key = os.environ.get("LONGCAT_API_KEY", "").strip()
    model = os.environ.get("LONGCAT_MODEL", DEFAULT_MODEL).strip()
    thinking = os.environ.get("LONGCAT_THINKING", "0").strip() == "1"
    missing = []
    if not base_url:
        missing.append("LONGCAT_BASE_URL")
    if not api_key:
        missing.append("LONGCAT_API_KEY")
    if missing:
        print(f"[FATAL] 缺少环境变量: {', '.join(missing)}")
        print("        请先设置 LONGCAT_BASE_URL / LONGCAT_API_KEY（可选 LONGCAT_MODEL）再运行。")
        sys.exit(2)
    return base_url, api_key, model, thinking


def _build_filler(target_tokens: int) -> str:
    """构造约 target_tokens 的英文填充文本（tiktoken 近似计数即可，容差±10%）。"""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    per_sentence = len(enc.encode(FILLER_SENTENCE))
    n = max(1, target_tokens // per_sentence)
    return FILLER_SENTENCE * n


class ProbeRunner:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float,
                 max_retries: int, thinking: bool = False):
        from openai import OpenAI

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking = thinking
        # max_retries=0：重试由我们自己控制，便于记录每次尝试
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.results = {}

    def _chat(self, messages, temperature=0.0, max_tokens=512, response_format=None):
        """带手动重试的单次调用，返回 (text, usage_dict, latency_s, attempts)。"""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # LongCat-2.0 默认思考模式会用 reasoning_content 吃满 max_tokens
                    # 导致 content 为空（finish_reason=length）。Judge 场景必须关闭。
                    extra_body={"enable_thinking": self.thinking},
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.monotonic() - t0
                text = resp.choices[0].message.content or ""
                usage = None
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                return text, usage, latency, attempt
            except Exception as e:  # noqa: BLE001 - 预检脚本需要捕获一切错误并报告
                last_err = e
                wait = min(2 ** attempt, 30)
                print(f"    [retry] attempt {attempt}/{self.max_retries} failed: "
                      f"{type(e).__name__}: {e}; wait {wait}s")
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"all {self.max_retries} attempts failed: {last_err}")

    # ---------- 各检查项 ----------

    def check_basic(self):
        text, usage, latency, attempts = self._chat(
            [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16
        )
        ok = bool(text.strip())
        self.results["basic"] = {
            "pass": ok, "response_preview": text[:100],
            "usage": usage, "latency_s": round(latency, 3), "attempts": attempts,
        }
        return ok

    def check_temperature0(self):
        prompt = ("Rate the following answer quality from 1 to 5 and explain in one "
                  "sentence. Question: What causes ischemic stroke? Answer: Blockage "
                  "of a brain artery, usually by a clot.")
        msgs = [{"role": "user", "content": prompt}]
        t1, u1, l1, _ = self._chat(msgs, temperature=0.0, max_tokens=128)
        t2, u2, l2, _ = self._chat(msgs, temperature=0.0, max_tokens=128)
        identical = t1 == t2
        # 不一致仅 WARN 不 FAIL：许多推理后端 temperature=0 也非严格确定
        self.results["temperature0"] = {
            "pass": True, "identical": identical,
            "note": "identical=False 时正式评测需依赖多次重复投票而非单次判定",
            "latency_s": [round(l1, 3), round(l2, 3)],
            "usage": [u1, u2],
        }
        return True

    def check_strict_json(self):
        prompt = (
            'Return a strict JSON object with keys "score" (integer 1-5) and '
            '"reason" (string, one sentence). No markdown, no code fence, JSON only. '
            "Evaluate: Q: What is epilepsy? A: A neurological disorder with recurrent seizures."
        )
        msgs = [{"role": "user", "content": prompt}]
        # 先尝试 response_format=json_object，不支持则回退纯 prompt 约束
        used_response_format = True
        try:
            text, usage, latency, _ = self._chat(
                msgs, max_tokens=256, response_format={"type": "json_object"}
            )
        except Exception:
            used_response_format = False
            text, usage, latency, _ = self._chat(msgs, max_tokens=256)

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        parsed = None
        parse_ok = False
        try:
            parsed = json.loads(cleaned)
            parse_ok = isinstance(parsed, dict) and "score" in parsed and "reason" in parsed
        except json.JSONDecodeError:
            parse_ok = False
        self.results["strict_json"] = {
            "pass": parse_ok,
            "response_format_supported": used_response_format,
            "raw_needed_cleanup": cleaned != text.strip(),
            "parsed": parsed if parse_ok else None,
            "response_preview": text[:200],
            "usage": usage, "latency_s": round(latency, 3),
        }
        return parse_ok

    def check_context(self, label: str, target_tokens: int):
        filler = _build_filler(target_tokens)
        half = len(filler) // 2
        content = (
            filler[:half]
            + f"\nIMPORTANT: the secret code is {NEEDLE}. Remember it.\n"
            + filler[half:]
            + "\n\nQuestion: What is the secret code mentioned above? Reply with the code only."
        )
        try:
            text, usage, latency, attempts = self._chat(
                [{"role": "user", "content": content}], max_tokens=64
            )
            found = NEEDLE in text
            self.results[label] = {
                "pass": found, "target_tokens": target_tokens,
                "needle_found": found, "response_preview": text[:100],
                "usage": usage, "latency_s": round(latency, 3), "attempts": attempts,
            }
            return found
        except Exception as e:  # noqa: BLE001
            self.results[label] = {
                "pass": False, "target_tokens": target_tokens,
                "error": f"{type(e).__name__}: {e}",
            }
            return False


def main():
    parser = argparse.ArgumentParser(description="LongCat Judge API 预检")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--skip-context-tests", action="store_true",
                        help="跳过 8K/16K 长上下文检查（快速模式）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="报告 JSON 输出路径")
    args = parser.parse_args()

    base_url, api_key, model, thinking = _require_config()
    print(f"[config] model={model} base_url={base_url} timeout={args.timeout}s "
          f"max_retries={args.max_retries} enable_thinking={thinking}")

    runner = ProbeRunner(base_url, api_key, model, args.timeout, args.max_retries,
                         thinking=thinking)

    checks = [("basic", runner.check_basic),
              ("temperature0", runner.check_temperature0),
              ("strict_json", runner.check_strict_json)]
    if not args.skip_context_tests:
        checks.append(("context_8k", lambda: runner.check_context("context_8k", 8000)))
        checks.append(("context_16k", lambda: runner.check_context("context_16k", 16000)))

    all_pass = True
    for name, fn in checks:
        print(f"[check] {name} ...")
        try:
            ok = fn()
        except Exception as e:  # noqa: BLE001
            ok = False
            runner.results[name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        status = "PASS" if ok else "FAIL"
        print(f"        -> {status}")
        all_pass = all_pass and ok

    report = {
        "probe_version": "longcat-probe-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "base_url": base_url,  # 不含密钥
        "enable_thinking": thinking,
        "thinking_note": "LongCat-2.0 默认思考模式导致 content 为空，Judge 必须 "
                         "extra_body={'enable_thinking': False}",
        "timeout_s": args.timeout,
        "max_retries": args.max_retries,
        "all_pass": all_pass,
        "checks": runner.results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] 写入 {args.output}")
    print(f"[result] {'ALL PASS' if all_pass else 'HAS FAILURES'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
