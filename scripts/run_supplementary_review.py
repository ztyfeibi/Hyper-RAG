"""补充盲审执行（Step 2：对 set_D 155 条 + set_E 12 条跑 AI 审核）。

流程：
1. 读取 blind_review_supplementary/ 下 Step 1 产物：
   - REVIEW_GUIDE.md（system prompt，判定规则）
   - set_D_unreviewed_155.md / set_E_recheck_12.md（审核材料，按 ``### SR-xxx`` 分节）
   - verdicts_template_D/E.jsonl（预填 claims 与 review_id 清单）
2. 逐条调 LongCat（SiliconFlow，temperature=0.0，json_object，指数退避重试，
   与 judge_longcat 同一调用模式），要求输出 JSON 判定。
3. 校验每条输出（review_id 匹配 / au_status 覆盖全部 AU / 枚举值合法 /
   预填 claims 必须全部有 status），不合格原地重试（受 --parse-retry 限制）。
4. 增量写 verdicts_ai_supplementary_D.jsonl / verdicts_ai_recheck_E.jsonl
   （断点续跑：已存在的 review_id 跳过），最后写 review_metadata.json。

材料即 md 分节文本（审核者所见即模型所得），不重新渲染，避免与 Step 1 漂移。

用法（由用户在终端执行，非短任务）：
    python scripts/run_supplementary_review.py --set both            # 全量 155+12
    python scripts/run_supplementary_review.py --set D --limit 2     # 冒烟
    python scripts/run_supplementary_review.py --set E               # 仅复核 12 条
中断后重跑同一命令即可续跑（按 review_id 去重）。
"""

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from my_config import (LLM_API_KEY_SILICONFLOW,  # noqa: E402
                       LLM_BASE_URL_SILICONFLOW, LLM_MODEL_SILICONFLOW)

REPEAT, SEED = 0, 42
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import build_supplementary_review as bsr  # noqa: E402  (复用 OUT_DIR / SUP_DIR 常量)

SUP_DIR = bsr.SUP_DIR

TEMPERATURE = 0.0
MAX_TOKENS = 6000
API_MAX_ATTEMPTS = 5
API_BACKOFF_BASE = 1.5
DEFAULT_PARSE_RETRY = 3

AU_STATUS_OPTIONS = {"supported", "missing", "contradicted"}
VERDICT_OPTIONS = {"pass", "fail", "uncertain"}
FATALITY_OPTIONS = {"none", "harmless", "fatal", "unresolved"}
CLAIM_STATUS_OPTIONS = {"supported_by_source", "unsupported_noncritical",
                        "contradicted", "unverifiable"}

SET_MD = {"D": SUP_DIR / "set_D_unreviewed_155.md",
          "E": SUP_DIR / "set_E_recheck_12.md"}
SET_TEMPLATE = {"D": SUP_DIR / "verdicts_template_D.jsonl",
                "E": SUP_DIR / "verdicts_template_E.jsonl"}
SET_OUTPUT = {"D": SUP_DIR / "verdicts_ai_supplementary_D.jsonl",
              "E": SUP_DIR / "verdicts_ai_recheck_E.jsonl"}

OUTPUT_INSTRUCTION = """\
请对上面材料给出 JSON 判定（只输出 JSON，不输出其他文字）：
{{
  "review_id": "{rid}",
  "au_status": {{"AU1": "supported|missing|contradicted", ...}},
  "verdict": "pass|fail|uncertain",
  "unsupported_fatality": "none|harmless|fatal|unresolved",
  "claim_status_by_id": {{"{c1}": "supported_by_source|unsupported_noncritical|contradicted|unverifiable", ...}},
  "additional_claims": [{{"claim": "<候选答案中其他超出证据的陈述原文>", "status": "supported_by_source|unsupported_noncritical|contradicted|unverifiable"}}, ...],
  "notes": "<可选备注，无则空字符串>"
}}
要求：
- au_status 必须覆盖材料中列出的**全部** Answer units（含 optional），键用 AU 编号。
- claim_status_by_id 的键用上面"预填待判 claims"给出的编号（{c1}、{c2}…），**每个编号都必须出现**，
  取值只能是四种 status 之一；不要改写 claim 文本，脚本会按编号自动对应原文。
- 若发现候选答案中还有超出证据的其他陈述，放进 additional_claims（文本尽量原样摘录）。
- 若候选答案没有超出证据的陈述，additional_claims 为空列表，unsupported_fatality 为 none。
- 证据不足判 uncertain，不得强行改成 fail。"""


# ---------------------------------------------------------------------------
# 材料加载（md 分节 = 审核者所见）
# ---------------------------------------------------------------------------

def split_md_sections(md_text: str) -> dict:
    """'### SR-xxx' 分节 -> {review_id: section_text}（保留头行，去掉尾部分隔线）。

    头行必须保留：模型需要从材料中读到 review_id 并在 JSON 中原样返回。
    """
    sections = {}
    parts = re.split(r"(?m)^### (SR-[DE]-\d{3})\s*$", md_text)
    for i in range(1, len(parts), 2):
        rid, body = parts[i], parts[i + 1]
        body = body.strip()
        # 去掉节尾 '---' 分隔线
        body = re.sub(r"\n---\s*$", "", body).strip()
        sections[rid] = f"### {rid}\n{body}"
    return sections


def load_templates(set_name: str) -> list:
    return [json.loads(l) for l in open(SET_TEMPLATE[set_name], encoding="utf-8")]


def load_existing(set_name: str) -> dict:
    f = SET_OUTPUT[set_name]
    if not f.exists():
        return {}
    out = {}
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["review_id"]] = r
    return out


def extract_au_ids(section_text: str) -> list:
    """从材料 Answer units 区块提取 AU 编号列表（保持顺序）。"""
    return re.findall(r"(?m)^- (AU\d+) \((?:required|optional)\)", section_text)


# ---------------------------------------------------------------------------
# LLM 调用（与 judge_longcat.llm_call 同模式）
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=LLM_API_KEY_SILICONFLOW,
                         base_url=LLM_BASE_URL_SILICONFLOW, timeout=300)
    return _client


def llm_call(prompt: str, system_prompt: str) -> str:
    last_err = None
    for attempt in range(1, API_MAX_ATTEMPTS + 1):
        try:
            call_prompt = prompt
            if attempt > 1:
                call_prompt = prompt + f"\n<!-- review-retry:{uuid.uuid4().hex[:8]} -->"
            resp = get_client().chat.completions.create(
                model=LLM_MODEL_SILICONFLOW,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": call_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError(f"空响应 (attempt {attempt})")
            return content
        except Exception as e:  # noqa: BLE001 - 网络/限流/超时/空响应统一重试
            last_err = e
            if attempt < API_MAX_ATTEMPTS:
                time.sleep(API_BACKOFF_BASE ** (attempt - 1))
    raise RuntimeError(f"API 调用失败: {last_err!r}")


# ---------------------------------------------------------------------------
# 输出校验
# ---------------------------------------------------------------------------

def validate_record(rec: dict, template: dict, au_ids: list,
                    claim_ids: list) -> list:
    """返回错误列表；空列表 = 合法。

    预填 claims 按 claim_id（C1..Cn）回填校验，不要求模型抄写原文；
    归一化由 normalize_record 完成。
    """
    errs = []
    if rec.get("review_id") != template["review_id"]:
        errs.append(f"review_id 不匹配: {rec.get('review_id')!r} != {template['review_id']!r}")
    au_status = rec.get("au_status")
    if not isinstance(au_status, dict):
        errs.append("au_status 不是对象")
    else:
        missing_aus = [a for a in au_ids if a not in au_status]
        extra_aus = [a for a in au_status if a not in au_ids]
        bad_vals = [f"{k}={v}" for k, v in au_status.items()
                    if v not in AU_STATUS_OPTIONS]
        if missing_aus:
            errs.append(f"au_status 缺 AU: {missing_aus}")
        if extra_aus:
            errs.append(f"au_status 多出 AU: {extra_aus}")
        if bad_vals:
            errs.append(f"au_status 非法取值: {bad_vals}")
    if rec.get("verdict") not in VERDICT_OPTIONS:
        errs.append(f"verdict 非法: {rec.get('verdict')!r}")
    if rec.get("unsupported_fatality") not in FATALITY_OPTIONS:
        errs.append(f"unsupported_fatality 非法: {rec.get('unsupported_fatality')!r}")
    by_id = rec.get("claim_status_by_id")
    if not isinstance(by_id, dict):
        errs.append("claim_status_by_id 不是对象")
    else:
        missing_ids = [c for c in claim_ids if c not in by_id]
        extra_ids = [c for c in by_id if c not in claim_ids]
        bad_status = [f"{k}={v}" for k, v in by_id.items()
                      if v not in CLAIM_STATUS_OPTIONS]
        if missing_ids:
            errs.append(f"claim_status_by_id 缺编号: {missing_ids}")
        if extra_ids:
            errs.append(f"claim_status_by_id 多出编号: {extra_ids}")
        if bad_status:
            errs.append(f"claim status 非法: {bad_status}")
    adds = rec.get("additional_claims", [])
    if not isinstance(adds, list):
        errs.append("additional_claims 不是列表")
    else:
        for c in adds:
            if not isinstance(c, dict) or not c.get("claim"):
                errs.append(f"additional_claim 结构非法: {c!r}")
            elif c.get("status") not in CLAIM_STATUS_OPTIONS:
                errs.append(f"additional_claim status 非法: {c.get('status')!r}")
    return errs


def normalize_record(rec: dict, template: dict) -> dict:
    """按编号把预填 claim 原文与模型 status 合并成统一的 unsupported_claims。

    输出记录结构与旧格式（unsupported_claims 列表）完全兼容，Step 3 合并无感知。
    """
    prefilled = template.get("unsupported_claims") or []
    by_id = rec.get("claim_status_by_id") or {}
    claims = []
    for i, pc in enumerate(prefilled, 1):
        cid = f"C{i}"
        claims.append({"claim": pc["claim"], "status": by_id[cid],
                       "claim_id": cid, "prefilled": True})
    for c in rec.get("additional_claims") or []:
        claims.append({"claim": c["claim"], "status": c["status"],
                       "claim_id": None, "prefilled": False})
    out = {
        "review_id": rec["review_id"],
        "au_status": rec["au_status"],
        "verdict": rec["verdict"],
        "unsupported_fatality": rec["unsupported_fatality"],
        "unsupported_claims": claims,
        "notes": rec.get("notes", ""),
        "response_schema": "claim_id_v2",
    }
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_set(set_name: str, limit=None, parse_retry=DEFAULT_PARSE_RETRY,
            dry_run=False):
    guide = (SUP_DIR / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
    sections = split_md_sections((SET_MD[set_name]).read_text(encoding="utf-8"))
    templates = load_templates(set_name)
    existing = load_existing(set_name)

    todo = [t for t in templates if t["review_id"] not in existing]
    if limit is not None:
        todo = todo[:limit]
    print(f"[set {set_name}] 模板 {len(templates)} 条, 已完成 {len(existing)}, "
          f"待跑 {len(todo)}" + (f"（limit={limit}）" if limit is not None else ""))
    if dry_run:
        return {"done": 0, "skipped": len(existing), "todo": len(todo)}

    stats = {"api_attempts": 0, "parse_retries": 0, "done": 0,
             "skipped": len(existing)}
    t0 = time.time()
    with open(SET_OUTPUT[set_name], "a", encoding="utf-8") as fout:
        for i, tpl in enumerate(todo, 1):
            rid = tpl["review_id"]
            section = sections[rid]
            au_ids = extract_au_ids(section)
            prefilled = tpl.get("unsupported_claims") or []
            claim_ids = [f"C{k}" for k in range(1, len(prefilled) + 1)]
            lines = [f"{section}\n\n---\n\n预填待判 claims（编号供回填，status 必须逐条给出）："]
            for k, pc in enumerate(prefilled, 1):
                lines.append(f"C{k}: {pc['claim']}")
            prompt = "\n".join(lines) + "\n\n---\n\n" + OUTPUT_INSTRUCTION.format(
                rid=rid, c1=claim_ids[0] if claim_ids else "C1",
                c2=claim_ids[1] if len(claim_ids) > 1 else "C2")

            rec, errs = None, []
            for attempt in range(1, parse_retry + 1):
                stats["api_attempts"] += 1
                raw = llm_call(prompt, guide)
                try:
                    cand = json.loads(raw)
                except json.JSONDecodeError as e:
                    errs = [f"JSON 解析失败: {e}"]
                    cand = None
                if cand is not None:
                    errs = validate_record(cand, tpl, au_ids, claim_ids)
                    if not errs:
                        rec = normalize_record(cand, tpl)
                        break
                stats["parse_retries"] += 1
                print(f"  [{rid}] 尝试 {attempt} 不合格: {errs[:2]}", flush=True)
            if rec is None:
                raise SystemExit(f"[{rid}] {parse_retry} 次尝试后仍不合格: {errs}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            stats["done"] += 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[set {set_name}] {i}/{len(todo)} 完成 "
                      f"({el:.0f}s, 均 {el / i:.1f}s/条)", flush=True)
    return stats


def write_metadata(all_stats: dict, args) -> Path:
    provider = ("SiliconFlow" if "siliconflow" in LLM_BASE_URL_SILICONFLOW
                else "local-vllm")
    meta = {
        "annotation_type": "supplementary_ai_review",
        "review_model": LLM_MODEL_SILICONFLOW,
        "review_provider": provider,
        "review_base_url": LLM_BASE_URL_SILICONFLOW,
        "review_temperature": TEMPERATURE,
        "review_max_tokens": MAX_TOKENS,
        "parse_retry": args.parse_retry,
        "sets_run": args.set,
        "limit": args.limit,
        "prompt_guide_sha256": sha256_text(
            (SUP_DIR / "REVIEW_GUIDE.md").read_text(encoding="utf-8")),
        "material_md_sha256": {
            s: sha256_text(SET_MD[s].read_text(encoding="utf-8"))
            for s in ("D", "E")
        },
        "template_sha256": {
            s: sha256_text(SET_TEMPLATE[s].read_text(encoding="utf-8"))
            for s in ("D", "E")
        },
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "stats": all_stats,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = SUP_DIR / "review_metadata.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="补充盲审执行（set_D/set_E AI 审核）")
    ap.add_argument("--set", default="both", choices=["D", "E", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="每个 set 最多处理多少条待跑记录（冒烟用）")
    ap.add_argument("--parse-retry", type=int, default=DEFAULT_PARSE_RETRY,
                    help="单条校验不合格时的最大尝试次数")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计待跑数量，不调 API")
    args = ap.parse_args()

    sets = ["D", "E"] if args.set == "both" else [args.set]
    all_stats = {}
    for s in sets:
        all_stats[s] = run_set(s, limit=args.limit,
                               parse_retry=args.parse_retry,
                               dry_run=args.dry_run)
    meta_path = write_metadata(all_stats, args)
    print(json.dumps(all_stats, ensure_ascii=False, indent=1))
    print(f"review_metadata.json -> {meta_path}")
    for s in sets:
        n = len(load_existing(s))
        print(f"set {s}: 输出文件现有 {n} 条（{SET_OUTPUT[s].name}）")


if __name__ == "__main__":
    main()
