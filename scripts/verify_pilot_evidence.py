# -*- coding: utf-8 -*-
"""Step 2.2 —— 证据核验与 Gold 构建。

把 80 条 ec-v2 候选转换为经过验证的：精确 evidence spans / 1-5 个 answer units /
evidence groups 与基础 qrels / 必要实体及超边 / Gold answer / 明确的接受、拒绝和
备选替换记录。**本阶段不生成问题文本、不运行 Router、不运行固定路径实验。**

四阶段（均为确定性、可重跑、可 resume）：
    prepare   回读 source chunk 原文 + 实体/超边 provenance，保存输入/prompt/模型/
              代码/候选池哈希到 step2_2/。
    draft     用 qwen-27b-int4 (t=0.1, seed=42) 提取 answer units / 原文 span /
              evidence groups / gold answer 草稿（独立 LLM cache，支持 --resume）。
    review    用独立复核模型（默认 LongCat-2.0，可配置）逐 answer unit 判定
              supported/partially_supported/contradicted/unsupported，给 accept/revise/
              reject；复核模型不改原文 span。
    finalize  执行严格机械校验 + 容量判定；拒收者从同结构 reserve 按固定顺序递补；
              冻结 80 条，生成 5 个产物文件。

用法：
    python scripts/verify_pilot_evidence.py --phase prepare --data-name neurology_chunk1000
    python scripts/verify_pilot_evidence.py --phase draft --resume
    python scripts/verify_pilot_evidence.py --phase review --resume [--allow-fallback-reviewer]
    python scripts/verify_pilot_evidence.py --phase finalize --promote-reserve
    python -m pytest tests/test_evidence_verification.py -q
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyperdb import HypergraphDB  # noqa: E402
from hyperrag.evidence_verification import (  # noqa: E402
    SCRIPT_VERSION, BOUND_CANDIDATE_POOL_HASH, SOURCE_TOKENIZER_MODEL,
    DRAFT_MODEL, DRAFT_BASE_URL, DRAFT_API_KEY, DRAFT_TEMPERATURE, DRAFT_SEED,
    DRAFT_TOP_P, DRAFT_MAX_TOKENS, REVIEW_MODEL_DEFAULT, REVIEW_BASE_URL,
    REVIEW_API_KEY, REVIEW_MAX_TOKENS, P4_SOURCE_CAP_QWEN,
    STRUCTURE_QUOTA, GRAPH_FIRST_STRUCTURES, sha256_text, sha256_file,
    split_source_ids, classify_language, count_qwen_tokens, locate_span,
    validate_span, spans_traceable, hyperedge_supported_by_text,
    make_span, make_answer_unit, make_evidence_group, make_review_verdict,
    make_qrels_entry, check_forbidden_fields,
)
from hyperrag.llm import openai_complete_if_cache  # noqa: E402
from hyperrag.storage import JsonKVStorage  # noqa: E402

POOL_DIR = REPO_ROOT / "caches" / "neurology_chunk1000" / "question_set_v2" / "pilot_v1"
STEP2_2_DIR = POOL_DIR / "step2_2"

# --------------------------------------------------------------------------
# Prompt 模板（冻结；prepare 阶段记录其哈希）
# --------------------------------------------------------------------------
DRAFT_SYSTEM_PROMPT = """\
You are a clinical-evidence extraction engine for a neurology textbook QA benchmark.
Given the verbatim SOURCE CHUNKS below, extract the factual content these chunks \
verifiably support, and ground every claim in exact quotations.

Strict rules:
1. Cite ONLY the provided SOURCE CHUNKS. Never invent facts or use external knowledge.
2. Evidence spans must be EXACT verbatim substrings copied from a chunk (same wording, \
same whitespace). Each span is implicitly id "sp1", "sp2", ... in list order. \
Prefer COMPLETE sentences or self-contained clauses; avoid dangling pronouns like \
"these"/"it" without a clear antecedent within the span.
3. Produce 1-5 atomic answer units. An atomic answer unit is ONE self-contained \
verifiable factual statement (one entity-relation-attribute triple, or one precise \
numeric/diagnostic claim). Do not combine multiple facts into one unit.
4. Each answer unit must be backed by at least one evidence group referencing span ids.
5. The gold_answer is a coherent reference answer that states ONLY facts grounded in the \
spans, and maps each stated fact back to an answer unit id like "[au1]". Keep it concise \
(under ~120 words).
6. Output ONLY a single JSON object (no prose, no markdown fences).

JSON schema:
{
  "answer_units": [ {"unit_id":"au1","statement":"..."} ],
  "spans": [ {"chunk_id":"chunk-...","text":"verbatim excerpt"} ],
  "evidence_groups": [ {"group_id":"eg1","answer_unit_id":"au1","span_ids":["sp1"],"rationale":"..."} ],
  "gold_answer": "coherent answer text mapping facts to [auN]",
  "graph_support": [ {"edge_key":"...","supported":true,"note":"..."} ]
}
The "graph_support" field is REQUIRED only when the candidate is graph-first; otherwise omit it.
"""

REVIEW_SYSTEM_PROMPT = """\
You are an independent evidence reviewer for a neurology QA benchmark. You are given \
source excerpts and a candidate ANSWER UNIT (an atomic factual claim) together with its \
supporting verbatim quotations. Judge whether the claim is supported by the source.

You MUST NOT modify the source quotations or the claim wording. You only assess.
Output ONLY a JSON object (no prose, no markdown fences):

{
  "judgment": "supported" | "partially_supported" | "contradicted" | "unsupported",
  "action": "accept" | "revise" | "reject",
  "note": "short explanation citing the relevant excerpt",
  "revision_suggestion": null or "how the claim could be tightened to match the source"
}

Rules:
- supported -> action "accept"
- partially_supported (claim overstates / understates what source shows) -> "revise"
- contradicted (claim conflicts with source) -> "reject"
- unsupported (claim not grounded in the given source at all) -> "reject"
"""


# --------------------------------------------------------------------------
# 加载工具
# --------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_candidate_pool() -> Tuple[List[dict], List[dict], str]:
    """Return (primary, reserve, primary_file_hash)."""
    primary_path = POOL_DIR / "evidence_candidates.jsonl"
    reserve_path = POOL_DIR / "evidence_candidates_reserve.jsonl"
    primary = _load_jsonl(primary_path)
    reserve = _load_jsonl(reserve_path) if reserve_path.exists() else []
    file_hash = sha256_file(str(primary_path))
    return primary, reserve, file_hash


def select_candidates(primary: Sequence[dict], reserve: Sequence[dict],
                      scope: str) -> List[dict]:
    """Select candidates for prepare/draft/review by explicit scope."""
    if scope == "primary":
        return list(primary)
    if scope == "reserve":
        return list(reserve)
    if scope == "all":
        return list(primary) + list(reserve)
    raise ValueError(f"unknown candidate scope: {scope}")


def _candidate_ids(candidates: Sequence[dict]) -> List[str]:
    return [c["candidate_id"] for c in candidates]


def _json_hash(value) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _valid_existing_draft(path: Path, cand_id: str, pkg: dict) -> bool:
    if not path.exists():
        return False
    try:
        existing = _read_json(path)
    except Exception:  # noqa: BLE001
        return False
    prompt_hash = sha256_text(_build_draft_user_prompt(pkg))
    return (
        existing.get("candidate_id") == cand_id
        and existing.get("script_version") == SCRIPT_VERSION
        and existing.get("model") == DRAFT_MODEL
        and existing.get("prompt_user_hash") == prompt_hash
    )


def _valid_existing_review(path: Path, cand_id: str, parsed: dict,
                           review_model: str) -> bool:
    if not path.exists():
        return False
    try:
        existing = _read_json(path)
    except Exception:  # noqa: BLE001
        return False
    return (
        existing.get("candidate_id") == cand_id
        and existing.get("script_version") == SCRIPT_VERSION
        and existing.get("review_model") == review_model
        and existing.get("draft_parsed_hash") == _json_hash(parsed)
    )


def _candidate_missing_record(cand: dict, reason: str) -> "collections.OrderedDict":
    enriched = collections.OrderedDict()
    enriched["candidate_id"] = cand["candidate_id"]
    enriched["intended_structure"] = cand["intended_structure"]
    enriched["entry_type"] = cand.get("entry_type")
    enriched["source_chunk_ids"] = cand.get("source_chunk_ids", [])
    enriched["decision"] = "unprocessed"
    enriched["unprocessed_reasons"] = [reason]
    return enriched

def load_chunks(data_name: str) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, int]]:
    path = REPO_ROOT / "caches" / data_name / "kv_store_text_chunks.json"
    raw = json.load(open(path, "r", encoding="utf-8"))
    texts: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    tiktok: Dict[str, int] = {}
    for cid, rec in raw.items():
        content = rec.get("content") or ""
        texts[cid] = content
        hashes[cid] = sha256_text(content)
        tiktok[cid] = rec.get("tokens") or 0
    return texts, hashes, tiktok


def load_hypergraph(data_name: str) -> HypergraphDB:
    path = REPO_ROOT / "caches" / data_name / "hypergraph_chunk_entity_relation.hgdb"
    hg = HypergraphDB()
    hg.load(str(path))
    return hg


# --------------------------------------------------------------------------
# Phase: prepare
# --------------------------------------------------------------------------
def phase_prepare(data_name: str, candidate_scope: str) -> None:
    STEP2_2_DIR.mkdir(parents=True, exist_ok=True)
    primary, reserve, pool_hash = load_candidate_pool()
    selected = select_candidates(primary, reserve, candidate_scope)

    # 绑定哈希校验（fail-fast）
    if pool_hash != BOUND_CANDIDATE_POOL_HASH:
        raise SystemExit(
            f"[prepare] 候选池哈希不匹配！\n  实际={pool_hash}\n  绑定="
            f"{BOUND_CANDIDATE_POOL_HASH}\n  说明：候选池已被改动，"
            f"请确认使用的是 ec-v2 版本且未被修改。"
        )
    if len(primary) != 80:
        raise SystemExit(f"[prepare] 主候选数={len(primary)}，期望 80")

    chunk_texts, chunk_hashes, chunk_tik = load_chunks(data_name)
    hg = load_hypergraph(data_name)

    # 机械核验输入包（每主候选一个文件）
    for cand in selected:
        cid = cand["candidate_id"]
        src_ids = cand["source_chunk_ids"]
        # 实体 provenance
        entities = []
        for eid in cand.get("seed_entity_ids", []):
            vd = hg.v(eid) or {}
            entities.append({
                "entity_id": eid,
                "entity_type": vd.get("entity_type"),
                "description": vd.get("description"),
                "source_ids": split_source_ids(vd.get("source_id")),
            })
        # 超边 provenance（回查 hgdb 真实记录）
        hyperedges = []
        for he in cand.get("hyperedges", []):
            ents = tuple(he.get("entity_ids") or [])
            ed = hg.e(ents) or {}
            hyperedges.append({
                "edge_key": he.get("edge_key"),
                "entity_ids": list(ents),
                "edge_type": ed.get("edge_type") or he.get("edge_type"),
                "description": ed.get("description"),
                "source_ids": split_source_ids(ed.get("source_id")),
                "weight": ed.get("weight"),
            })
        pkg = collections.OrderedDict()
        pkg["candidate_id"] = cid
        pkg["intended_structure"] = cand["intended_structure"]
        pkg["entry_type"] = cand["entry_type"]
        pkg["source_chunks"] = [
            {
                "chunk_id": s,
                "content_hash": chunk_hashes.get(s),
                "content": chunk_texts.get(s, ""),
                "tiktoken_tokens": chunk_tik.get(s, 0),
            }
            for s in src_ids
        ]
        pkg["entities"] = entities
        pkg["hyperedges"] = hyperedges
        # 禁止字段校验
        bad = check_forbidden_fields(pkg)
        if bad:
            raise SystemExit(f"[prepare] 候选 {cid} 含禁止字段: {bad}")
        with open(STEP2_2_DIR / f"mech_input_{cid}.json", "w", encoding="utf-8") as f:
            json.dump(pkg, f, ensure_ascii=False, indent=2)

    # 冻结 prompt 模板 + 模型/代码哈希
    (STEP2_2_DIR / "draft_prompt_template.txt").write_text(
        DRAFT_SYSTEM_PROMPT, encoding="utf-8"
    )
    (STEP2_2_DIR / "review_prompt_template.txt").write_text(
        REVIEW_SYSTEM_PROMPT, encoding="utf-8"
    )
    code_hashes = {
        "hyperrag/evidence_verification.py": sha256_file(
            str(REPO_ROOT / "hyperrag" / "evidence_verification.py")
        ),
        "scripts/verify_pilot_evidence.py": sha256_file(
            str(REPO_ROOT / "scripts" / "verify_pilot_evidence.py")
        ),
    }
    manifest = collections.OrderedDict()
    manifest["script_version"] = SCRIPT_VERSION
    manifest["phase"] = "prepare"
    manifest["candidate_scope"] = candidate_scope
    manifest["candidate_pool_hash_bound"] = BOUND_CANDIDATE_POOL_HASH
    manifest["candidate_pool_hash_actual"] = pool_hash
    manifest["candidate_pool_hash_match"] = (pool_hash == BOUND_CANDIDATE_POOL_HASH)
    manifest["source_tokenizer_model"] = SOURCE_TOKENIZER_MODEL
    manifest["candidate_count"] = len(selected)
    manifest["primary_count"] = len(primary)
    manifest["reserve_count"] = len(reserve)
    manifest["candidate_ids"] = _candidate_ids(selected)
    manifest["structure_quota"] = dict(STRUCTURE_QUOTA)
    manifest["draft_model"] = DRAFT_MODEL
    manifest["draft_base_url"] = DRAFT_BASE_URL
    manifest["draft_temperature"] = DRAFT_TEMPERATURE
    manifest["draft_seed"] = DRAFT_SEED
    manifest["draft_top_p"] = DRAFT_TOP_P
    manifest["draft_max_tokens"] = DRAFT_MAX_TOKENS
    manifest["review_model_default"] = REVIEW_MODEL_DEFAULT
    manifest["review_base_url"] = REVIEW_BASE_URL
    manifest["p4_source_cap_qwen"] = P4_SOURCE_CAP_QWEN
    manifest["prompt_template_hashes"] = {
        "draft": sha256_text(DRAFT_SYSTEM_PROMPT),
        "review": sha256_text(REVIEW_SYSTEM_PROMPT),
    }
    manifest["code_hashes"] = code_hashes
    manifest["input_file_hashes"] = {
        "evidence_candidates.jsonl": pool_hash,
        "evidence_candidates_reserve.jsonl": sha256_file(
            str(POOL_DIR / "evidence_candidates_reserve.jsonl")
        ),
        "kv_store_text_chunks.json": sha256_file(
            str(REPO_ROOT / "caches" / data_name / "kv_store_text_chunks.json")
        ),
        "hypergraph_chunk_entity_relation.hgdb": sha256_file(
            str(REPO_ROOT / "caches" / data_name
                / "hypergraph_chunk_entity_relation.hgdb")
        ),
    }
    with open(STEP2_2_DIR / "prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[prepare] completed {len(selected)} mech_input files + prepare_manifest.json (scope={candidate_scope})")
    print(f"[prepare] 候选池哈希匹配: {manifest['candidate_pool_hash_match']}")


# --------------------------------------------------------------------------
# Phase: draft
# --------------------------------------------------------------------------
def _build_draft_user_prompt(pkg: dict) -> str:
    lines = []
    for sc in pkg["source_chunks"]:
        lines.append(f"=== SOURCE CHUNK id={sc['chunk_id']} ===")
        lines.append(sc["content"])
        lines.append("")
    if pkg["intended_structure"] in GRAPH_FIRST_STRUCTURES:
        lines.append("=== GRAPH-FIRST HYPEREDGES (must be supported by source) ===")
        for he in pkg["hyperedges"]:
            lines.append(
                f"- edge_key={he['edge_key']} edge_type={he['edge_type']} "
                f"description={he.get('description')}"
            )
        lines.append("")
    lines.append("Extract evidence per the system instructions. Output ONLY JSON.")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


_DRAFT_REPAIR_SUFFIX = (
    "\n\nYour previous output was truncated or not valid JSON. Re-output ONLY the "
    "complete JSON object now, strictly following the schema, keeping gold_answer "
    "concise. Do not add any prose."
)


async def _call_draft(user_prompt: str, hashing_kv, sem) -> str:
    async with sem:
        return await openai_complete_if_cache(
            DRAFT_MODEL, user_prompt,
            system_prompt=DRAFT_SYSTEM_PROMPT,
            base_url=DRAFT_BASE_URL, api_key=DRAFT_API_KEY,
            hashing_kv=hashing_kv,
            temperature=DRAFT_TEMPERATURE, seed=DRAFT_SEED,
            top_p=DRAFT_TOP_P, max_tokens=DRAFT_MAX_TOKENS,
        )


def _try_parse(raw: str):
    try:
        return json.loads(_strip_fences(raw)), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _looks_truncated(raw: str) -> bool:
    s = raw.rstrip()
    if not s:
        return True
    # 合法 JSON 应以 } 或 ] 结尾（去除尾部空白/反引号）
    return s[-1] not in ("}", "]")


async def _draft_one(cand_id: str, pkg: dict, hashing_kv, sem) -> dict:
    user_prompt = _build_draft_user_prompt(pkg)
    raw = await _call_draft(user_prompt, hashing_kv, sem)
    parsed, parse_error = _try_parse(raw)
    # 截断/解析失败：追加 repair 指令重试一次（同参数，确定性可复现）
    if parsed is None and _looks_truncated(raw):
        raw2 = await _call_draft(user_prompt + _DRAFT_REPAIR_SUFFIX, hashing_kv, sem)
        parsed2, parse_error2 = _try_parse(raw2)
        if parsed2 is not None:
            raw, parsed, parse_error = raw2, parsed2, None
        else:
            parse_error = parse_error2 or parse_error
    out = collections.OrderedDict()
    out["candidate_id"] = cand_id
    out["script_version"] = SCRIPT_VERSION
    out["model"] = DRAFT_MODEL
    out["temperature"] = DRAFT_TEMPERATURE
    out["seed"] = DRAFT_SEED
    out["top_p"] = DRAFT_TOP_P
    out["max_tokens"] = DRAFT_MAX_TOKENS
    out["prompt_user"] = user_prompt
    out["prompt_user_hash"] = sha256_text(user_prompt)
    out["raw_response"] = raw
    out["parsed"] = parsed
    out["parse_error"] = parse_error
    out["draft_status"] = "ok" if parsed is not None else "parse_failed"
    return out


async def phase_draft(data_name: str, resume: bool, candidate_scope: str) -> None:
    STEP2_2_DIR.mkdir(parents=True, exist_ok=True)
    primary, reserve, _h = load_candidate_pool()
    selected = select_candidates(primary, reserve, candidate_scope)
    cache_kv = JsonKVStorage(
        namespace="step2_2_draft_llm_cache",
        global_config={"working_dir": str(STEP2_2_DIR)},
    )
    sem = asyncio.Semaphore(4)
    tasks = []
    for cand in selected:
        cid = cand["candidate_id"]
        out_path = STEP2_2_DIR / f"draft_raw_{cid}.json"
        pkg_path = STEP2_2_DIR / f"mech_input_{cid}.json"
        if not pkg_path.exists():
            raise SystemExit(f"[draft] mech_input missing for {cid}; run prepare --candidate-scope {candidate_scope} first")
        pkg = _read_json(pkg_path)
        if resume and _valid_existing_draft(out_path, cid, pkg):
            continue
        tasks.append(_draft_one(cid, pkg, cache_kv, sem))
    if not tasks:
        print("[draft] 无待处理候选（resume 已全部完成）")
        return
    results = await asyncio.gather(*tasks)
    for res in results:
        with open(STEP2_2_DIR / f"draft_raw_{res['candidate_id']}.json",
                  "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in results if r["draft_status"] == "ok")
    print(f"[draft] 完成 {len(results)} 条（ok={ok}, parse_failed={len(results)-ok}）；"
          f"cache 落盘于 step2_2/kv_store_step2_2_draft_llm_cache.json")


# --------------------------------------------------------------------------
# Phase: review
# --------------------------------------------------------------------------
def _build_review_user_prompt(pkg: dict, au: dict, spans: List[dict]) -> str:
    lines = ["=== SOURCE EXCERPTS ==="]
    for sp in spans:
        lines.append(f"[{sp['span_id']}] (chunk {sp['chunk_id']}): {sp['text']}")
    lines.append("")
    lines.append(f"=== ANSWER UNIT {au['unit_id']} ===")
    lines.append(au["statement"])
    lines.append("")
    lines.append("Judge per system instructions. Output ONLY JSON.")
    return "\n".join(lines)


async def _review_one(cand_id: str, parsed: dict, pkg: dict,
                      review_model: str, review_base_url: str,
                      review_api_key: str, hashing_kv, sem) -> dict:
    # 组装 span 列表（带 spN id）
    spans = []
    for i, sp in enumerate(parsed.get("spans", []), start=1):
        spans.append({"span_id": f"sp{i}", "chunk_id": sp.get("chunk_id"),
                      "text": sp.get("text")})
    verdicts = []
    async with sem:
        for au in parsed.get("answer_units", []):
            au_spans = [s for s in spans if _au_uses_span(s, parsed, au)]
            user_prompt = _build_review_user_prompt(pkg, au, au_spans)
            raw = await openai_complete_if_cache(
                review_model, user_prompt,
                system_prompt=REVIEW_SYSTEM_PROMPT,
                base_url=review_base_url, api_key=review_api_key,
                hashing_kv=hashing_kv,
                temperature=0.0, seed=DRAFT_SEED, top_p=1.0,
                max_tokens=REVIEW_MAX_TOKENS,
            )
            pverdict = None
            perr = None
            try:
                pverdict = json.loads(_strip_fences(raw))
            except Exception as e:  # noqa: BLE001
                perr = f"{type(e).__name__}: {e}"
            verdicts.append({
                "answer_unit_id": au["unit_id"],
                "raw_response": raw,
                "parsed": pverdict,
                "parse_error": perr,
            })
    out = collections.OrderedDict()
    out["candidate_id"] = cand_id
    out["script_version"] = SCRIPT_VERSION
    out["review_model"] = review_model
    out["draft_parsed_hash"] = _json_hash(parsed)
    out["verdicts"] = verdicts
    return out


def _au_uses_span(span: dict, parsed: dict, au: dict) -> bool:
    # 若该 answer unit 的 evidence_group 引用了此 span 的 id，则纳入复核上下文
    au_id = au["unit_id"]
    for eg in parsed.get("evidence_groups", []):
        if eg.get("answer_unit_id") == au_id and span["span_id"] in eg.get("span_ids", []):
            return True
    return False


async def phase_review(data_name: str, resume: bool,
                       review_model: str, review_base_url: str,
                       allow_fallback: bool, candidate_scope: str) -> None:
    STEP2_2_DIR.mkdir(parents=True, exist_ok=True)
    primary, reserve, _h = load_candidate_pool()
    selected = select_candidates(primary, reserve, candidate_scope)
    # 默认指向真实 LongCat（SiliconFlow）；允许 --review-base-url 覆盖。
    if not review_base_url:
        review_base_url = REVIEW_BASE_URL
        review_api_key = REVIEW_API_KEY
    else:
        review_api_key = REVIEW_API_KEY if "siliconflow" in review_base_url else "EMPTY"
    fallback_flag = False
    # 连通性自检：默认 LongCat 不可达且授权时退回 qwen（显式标注）。
    if allow_fallback:
        try:
            await openai_complete_if_cache(
                review_model, "ping", system_prompt="reply ok",
                base_url=review_base_url, api_key=review_api_key,
                temperature=0.0, max_tokens=4,
            )
        except Exception as e:  # noqa: BLE001
            review_model = DRAFT_MODEL
            review_base_url = DRAFT_BASE_URL
            review_api_key = DRAFT_API_KEY
            fallback_flag = True
            print(f"[review] 默认复核模型 {REVIEW_MODEL_DEFAULT} 不可达 "
                  f"({type(e).__name__})，按 --allow-fallback-reviewer 退回 "
                  f"{review_model}（已显式标注 review_model_fallback=True）。")
    cache_kv = JsonKVStorage(
        namespace="step2_2_review_llm_cache",
        global_config={"working_dir": str(STEP2_2_DIR)},
    )
    sem = asyncio.Semaphore(4)
    tasks = []
    for cand in selected:
        cid = cand["candidate_id"]
        out_path = STEP2_2_DIR / f"review_raw_{cid}.json"
        draft_path = STEP2_2_DIR / f"draft_raw_{cid}.json"
        if not draft_path.exists():
            print(f"[review] 跳过 {cid}：无 draft 结果")
            continue
        draft = _read_json(draft_path)
        if draft.get("parsed") is None:
            print(f"[review] 跳过 {cid}：draft 解析失败")
            continue
        pkg = _read_json(STEP2_2_DIR / f"mech_input_{cid}.json")
        if resume and _valid_existing_review(out_path, cid, draft["parsed"], review_model):
            continue
        tasks.append(_review_one(cid, draft["parsed"], pkg, review_model,
                                 review_base_url, review_api_key, cache_kv, sem))
    if not tasks:
        print("[review] 无待处理候选（resume 已全部完成）")
        return
    results = await asyncio.gather(*tasks)
    for res in results:
        res["review_model_fallback"] = fallback_flag
        with open(STEP2_2_DIR / f"review_raw_{res['candidate_id']}.json",
                  "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[review] 完成 {len(results)} 条；fallback={fallback_flag}")


# --------------------------------------------------------------------------
# Phase: finalize
# --------------------------------------------------------------------------
def _assemble_spans_with_bounds(parsed: dict, chunk_texts: Dict[str, str],
                                chunk_hashes: Dict[str, str]):
    """给 draft 的 span 补上 char_start/char_end 并做精确子串校验。"""
    out_spans = []
    problems = []
    for i, sp in enumerate(parsed.get("spans", []), start=1):
        cid = sp.get("chunk_id")
        text = sp.get("text") or ""
        rec = collections.OrderedDict()
        rec["span_id"] = f"sp{i}"
        rec["chunk_id"] = cid
        rec["text"] = text
        rec["chunk_content_hash"] = chunk_hashes.get(cid, "")
        if cid in chunk_texts:
            loc = locate_span(chunk_texts[cid], text)
            if loc is None:
                rec["char_start"] = None
                rec["char_end"] = None
                problems.append(f"sp{i}@{cid}: not_verbatim")
            else:
                rec["char_start"], rec["char_end"] = loc
        else:
            rec["char_start"] = None
            rec["char_end"] = None
            problems.append(f"sp{i}@{cid}: chunk_missing")
        out_spans.append(rec)
    return out_spans, problems


def _qrels_from_answer_units(answer_units: Sequence[dict],
                              evidence_groups: Sequence[dict],
                              spans: Sequence[dict]) -> List[dict]:
    span_by_id = {s["span_id"]: s for s in spans if s.get("span_id")}
    groups_by_au: Dict[str, List[dict]] = collections.defaultdict(list)
    for eg in evidence_groups:
        groups_by_au[eg.get("answer_unit_id")].append(eg)

    qrels = []
    for au in answer_units:
        span_ids = []
        chunk_ids = []
        for eg in groups_by_au.get(au["unit_id"], []):
            for sid in eg.get("span_ids", []):
                if sid in span_by_id:
                    span_ids.append(sid)
                    cid = span_by_id[sid].get("chunk_id")
                    if cid:
                        chunk_ids.append(cid)
        qrels.append(make_qrels_entry(au["unit_id"], chunk_ids, span_ids))
    return qrels


def _evidence_group_problems(answer_units: Sequence[dict],
                             evidence_groups: Sequence[dict],
                             spans: Sequence[dict]) -> List[str]:
    problems: List[str] = []
    au_ids = {au["unit_id"] for au in answer_units}
    span_ids = {s["span_id"] for s in spans if s.get("span_id")}
    eg_au = {eg.get("answer_unit_id") for eg in evidence_groups}
    missing_eg = au_ids - eg_au
    if missing_eg:
        problems.append(f"answer_units missing evidence group: {sorted(missing_eg)}")
    for eg in evidence_groups:
        if eg.get("answer_unit_id") not in au_ids:
            problems.append(f"evidence_group {eg.get('group_id')} references unknown answer_unit")
        missing_spans = sorted(set(eg.get("span_ids", [])) - span_ids)
        if missing_spans:
            problems.append(f"evidence_group {eg.get('group_id')} references missing spans: {missing_spans}")
    return problems


def _validate_candidate(cand: dict, pkg: dict, parsed: dict,
                        chunk_texts: Dict[str, str], chunk_hashes: Dict[str, str],
                        verdicts_by_cid: Dict[str, dict]) -> Tuple[str, dict, List[str]]:
    """Mechanical validation + capacity check + review verdict aggregation."""
    reasons: List[str] = []
    structures = cand["intended_structure"]

    spans, _span_problems = _assemble_spans_with_bounds(
        parsed, chunk_texts, chunk_hashes
    )
    all_ok, trace_problems = spans_traceable(spans, chunk_texts, chunk_hashes)
    if not all_ok:
        reasons.extend(trace_problems)

    answer_units = [
        make_answer_unit(au["unit_id"], au["statement"])
        for au in parsed.get("answer_units", [])
    ]
    evidence_groups = [
        make_evidence_group(eg["group_id"], eg["answer_unit_id"],
                            eg.get("span_ids", []), eg.get("rationale", ""))
        for eg in parsed.get("evidence_groups", [])
    ]
    eg_problems = _evidence_group_problems(answer_units, evidence_groups, spans)
    reasons.extend(eg_problems)

    graph_support_ok = True
    graph_diagnostics = []
    if structures in GRAPH_FIRST_STRUCTURES:
        src_texts = [sc["content"] for sc in pkg["source_chunks"]]
        src_ids = [sc["chunk_id"] for sc in pkg["source_chunks"]]
        for he in pkg["hyperedges"]:
            ok, why = hyperedge_supported_by_text(he, src_texts, src_ids)
            diag = collections.OrderedDict()
            diag["edge_key"] = he.get("edge_key")
            diag["supported"] = ok
            diag["reason"] = why
            diag["source_ids"] = he.get("source_ids", [])
            graph_diagnostics.append(diag)
            if not ok:
                graph_support_ok = False
                reasons.append(f"hyperedge {he.get('edge_key')} provenance unsupported: {why}")

    raw_source_text = "\n".join(sc["content"] for sc in pkg["source_chunks"])
    gold_span_text = "\n".join(s["text"] for s in spans if s["text"])
    raw_source_qwen = count_qwen_tokens(raw_source_text)
    gold_span_qwen = count_qwen_tokens(gold_span_text)

    p4_raw_fit = raw_source_qwen <= P4_SOURCE_CAP_QWEN
    p4_gold_fit = gold_span_qwen <= P4_SOURCE_CAP_QWEN
    if not p4_gold_fit:
        reasons.append(f"gold_span_qwen_tokens={gold_span_qwen} > P4 cap {P4_SOURCE_CAP_QWEN}")
    if not p4_raw_fit:
        reasons.append("raw_source_qwen_tokens > P4 cap")

    rv = verdicts_by_cid.get(cand["candidate_id"])
    judgments = []
    if rv is not None:
        for v in rv.get("verdicts", []):
            p = v.get("parsed") or {}
            judgments.append(p.get("judgment"))
    has_contradicted = "contradicted" in judgments
    has_unsupported = "unsupported" in judgments
    has_partial = "partially_supported" in judgments

    if not p4_gold_fit or not graph_support_ok or not all_ok or eg_problems or has_contradicted:
        decision = "reject"
    elif has_unsupported or has_partial:
        decision = "revise"
    else:
        decision = "accept"

    enriched = collections.OrderedDict()
    enriched["candidate_id"] = cand["candidate_id"]
    enriched["intended_structure"] = structures
    enriched["entry_type"] = cand["entry_type"]
    enriched["source_chunk_ids"] = cand["source_chunk_ids"]
    enriched["seed_entity_ids"] = cand.get("seed_entity_ids", [])
    enriched["hyperedges"] = cand.get("hyperedges", [])
    enriched["answer_units"] = answer_units
    enriched["spans"] = spans
    enriched["evidence_groups"] = evidence_groups
    for eg in enriched["evidence_groups"]:
        for au in enriched["answer_units"]:
            if au["unit_id"] == eg["answer_unit_id"]:
                au["evidence_group_ids"].append(eg["group_id"])
    enriched["qrels"] = _qrels_from_answer_units(
        enriched["answer_units"], enriched["evidence_groups"], spans
    )
    enriched["gold_answer"] = parsed.get("gold_answer")
    enriched["graph_support"] = parsed.get("graph_support")
    enriched["graph_grounding_diagnostics"] = graph_diagnostics
    enriched["capacity"] = collections.OrderedDict([
        ("raw_source_qwen_tokens", raw_source_qwen),
        ("gold_span_qwen_tokens", gold_span_qwen),
        ("p4_raw_chunk_fit", p4_raw_fit),
        ("p4_gold_span_fit", p4_gold_fit),
        ("retrieval_capacity_risk", (not p4_raw_fit)),
    ])
    enriched["review_verdicts"] = [
        make_review_verdict(
            v["answer_unit_id"],
            (v.get("parsed") or {}).get("judgment", "unsupported"),
            (v.get("parsed") or {}).get("action", "reject"),
            (v.get("parsed") or {}).get("note", ""),
            (v.get("parsed") or {}).get("revision_suggestion"),
        )
        for v in (rv.get("verdicts", []) if rv else [])
    ]
    enriched["decision"] = decision
    enriched["rejection_reasons"] = reasons
    enriched["n_answer_units"] = len(enriched["answer_units"])
    bad = check_forbidden_fields(enriched)
    if bad:
        reasons.append(f"forbidden_fields:{bad}")
        enriched["decision"] = "reject"
    return enriched["decision"], enriched, reasons

def _load_review_index(candidates: Sequence[dict]) -> Dict[str, dict]:
    verdicts_by_cid = {}
    for cand in candidates:
        rp = STEP2_2_DIR / f"review_raw_{cand['candidate_id']}.json"
        if rp.exists():
            verdicts_by_cid[cand["candidate_id"]] = _read_json(rp)
    return verdicts_by_cid


def _evaluate_candidate_record(cand: dict, chunk_texts: Dict[str, str],
                               chunk_hashes: Dict[str, str],
                               verdicts_by_cid: Dict[str, dict]) -> Tuple[str, dict]:
    cid = cand["candidate_id"]
    dp = STEP2_2_DIR / f"draft_raw_{cid}.json"
    mp = STEP2_2_DIR / f"mech_input_{cid}.json"
    rp = STEP2_2_DIR / f"review_raw_{cid}.json"
    if not mp.exists():
        rec = _candidate_missing_record(cand, "mech_input_missing")
        return "unprocessed", rec
    if not dp.exists():
        rec = _candidate_missing_record(cand, "draft_missing")
        return "unprocessed", rec
    draft = _read_json(dp)
    if draft.get("parsed") is None:
        rec = _candidate_missing_record(cand, "draft_parse_failed")
        rec["parse_error"] = draft.get("parse_error")
        return "unprocessed", rec
    if not rp.exists() or cid not in verdicts_by_cid:
        rec = _candidate_missing_record(cand, "review_missing")
        return "unprocessed", rec
    pkg = _read_json(mp)
    dec, enriched, _reasons = _validate_candidate(
        cand, pkg, draft["parsed"], chunk_texts, chunk_hashes, verdicts_by_cid
    )
    return dec, enriched


def _write_finalize_outputs(verified: List[dict], rejected: List[dict],
                            human_queue: List[dict], unprocessed: List[dict],
                            primary: Sequence[dict], reserve: Sequence[dict],
                            decisions_primary: Dict[str, str],
                            decisions_reserve: Dict[str, str]) -> None:
    _write_jsonl(STEP2_2_DIR / "verified_evidence.jsonl", verified)
    _write_jsonl(STEP2_2_DIR / "rejected_evidence.jsonl", rejected)
    _write_jsonl(STEP2_2_DIR / "human_review_queue.jsonl", human_queue)
    _write_jsonl(STEP2_2_DIR / "unprocessed_candidates.jsonl", unprocessed)
    _write_jsonl(
        STEP2_2_DIR / "reserve_unprocessed.jsonl",
        [u for u in unprocessed if u["candidate_id"] in {r["candidate_id"] for r in reserve}],
    )

    from collections import Counter as _C
    verified_quota = dict(_C(v["intended_structure"] for v in verified))
    quota_met = all(verified_quota.get(k, 0) == v for k, v in STRUCTURE_QUOTA.items())
    manifest = collections.OrderedDict()
    manifest["script_version"] = SCRIPT_VERSION
    manifest["phase"] = "finalize"
    manifest["candidate_pool_hash_bound"] = BOUND_CANDIDATE_POOL_HASH
    manifest["verified_count"] = len(verified)
    manifest["rejected_count"] = len(rejected)
    manifest["human_review_count"] = len(human_queue)
    manifest["unprocessed_count"] = len(unprocessed)
    manifest["verified_quota"] = verified_quota
    manifest["quota_met"] = quota_met
    manifest["review_model_default"] = REVIEW_MODEL_DEFAULT
    manifest["p4_source_cap_qwen"] = P4_SOURCE_CAP_QWEN
    manifest["output_file_hashes"] = {
        "verified_evidence.jsonl": sha256_file(str(STEP2_2_DIR / "verified_evidence.jsonl")),
        "rejected_evidence.jsonl": sha256_file(str(STEP2_2_DIR / "rejected_evidence.jsonl")),
        "human_review_queue.jsonl": sha256_file(str(STEP2_2_DIR / "human_review_queue.jsonl")),
        "unprocessed_candidates.jsonl": sha256_file(str(STEP2_2_DIR / "unprocessed_candidates.jsonl")),
        "reserve_unprocessed.jsonl": sha256_file(str(STEP2_2_DIR / "reserve_unprocessed.jsonl")),
    }
    with open(STEP2_2_DIR / "verification_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = collections.OrderedDict()
    report["total_primary"] = len(primary)
    report["total_reserve"] = len(reserve)
    report["decisions_primary"] = dict(_C(decisions_primary.values()))
    report["decisions_reserve_evaluated"] = dict(_C(decisions_reserve.values()))
    report["verified_count"] = len(verified)
    report["verified_quota"] = verified_quota
    report["rejected_count"] = len(rejected)
    report["human_review_count"] = len(human_queue)
    report["unprocessed_count"] = len(unprocessed)
    report["reserve_unprocessed_count"] = sum(
        1 for u in unprocessed if u["candidate_id"] in {r["candidate_id"] for r in reserve}
    )
    report["rejection_reason_distribution"] = dict(_C(
        r for e in rejected + human_queue for r in e.get("rejection_reasons", [])
    ))
    report["unprocessed_reason_distribution"] = dict(_C(
        r for e in unprocessed for r in e.get("unprocessed_reasons", [])
    ))
    report["quota_target"] = dict(STRUCTURE_QUOTA)
    report["quota_met"] = quota_met
    with open(STEP2_2_DIR / "verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def phase_finalize(data_name: str, promote_reserve: bool) -> None:
    STEP2_2_DIR.mkdir(parents=True, exist_ok=True)
    primary, reserve, _h = load_candidate_pool()
    chunk_texts, chunk_hashes, _tik = load_chunks(data_name)
    all_candidates = list(primary) + list(reserve)
    verdicts_by_cid = _load_review_index(all_candidates)

    verified: List[dict] = []
    rejected: List[dict] = []
    human_queue: List[dict] = []
    unprocessed: List[dict] = []
    used_chunks = set()
    quota_filled = collections.Counter()
    decisions_primary: Dict[str, str] = {}
    decisions_reserve: Dict[str, str] = {}

    def try_accept(enriched) -> bool:
        struct = enriched["intended_structure"]
        if enriched["decision"] != "accept":
            return False
        if quota_filled[struct] >= STRUCTURE_QUOTA[struct]:
            enriched["rejection_reasons"].append("quota_full")
            return False
        if used_chunks & set(enriched.get("source_chunk_ids", [])):
            enriched["rejection_reasons"].append("evidence_cluster_collision")
            return False
        verified.append(enriched)
        used_chunks.update(enriched.get("source_chunk_ids", []))
        quota_filled[struct] += 1
        return True

    for cand in primary:
        dec, enriched = _evaluate_candidate_record(cand, chunk_texts, chunk_hashes, verdicts_by_cid)
        decisions_primary[cand["candidate_id"]] = dec
        if dec == "accept":
            if not try_accept(enriched):
                rejected.append(enriched)
        elif dec == "revise":
            human_queue.append(enriched)
        elif dec == "unprocessed":
            unprocessed.append(enriched)
        else:
            rejected.append(enriched)

    if promote_reserve:
        reserve_by_struct = collections.defaultdict(list)
        for rc in reserve:
            reserve_by_struct[rc["intended_structure"]].append(rc)
        for struct, quota in STRUCTURE_QUOTA.items():
            if quota_filled[struct] >= quota:
                continue
            for rc in reserve_by_struct.get(struct, []):
                if quota_filled[struct] >= quota:
                    break
                rcid = rc["candidate_id"]
                dec, enriched = _evaluate_candidate_record(rc, chunk_texts, chunk_hashes, verdicts_by_cid)
                decisions_reserve[rcid] = dec
                if dec == "accept":
                    if not try_accept(enriched):
                        rejected.append(enriched)
                elif dec == "revise":
                    human_queue.append(enriched)
                elif dec == "unprocessed":
                    unprocessed.append(enriched)
                else:
                    rejected.append(enriched)

    _write_finalize_outputs(
        verified, rejected, human_queue, unprocessed,
        primary, reserve, decisions_primary, decisions_reserve,
    )
    report = _read_json(STEP2_2_DIR / "verification_report.json")
    print(f"[finalize] verified={len(verified)} rejected={len(rejected)} "
          f"human_review={len(human_queue)} unprocessed={len(unprocessed)} "
          f"quota_met={report['quota_met']}")
    print(f"[finalize] verified_quota={report['verified_quota']}")


def phase_apply_adjudication(data_name: str, adjudication_file: Path) -> None:
    del data_name  # finalized records already carry the evidence payload.
    if not adjudication_file.exists():
        raise SystemExit(f"[apply-adjudication] missing file: {adjudication_file}")
    primary, reserve, _h = load_candidate_pool()
    verified = _load_jsonl(STEP2_2_DIR / "verified_evidence.jsonl") if (STEP2_2_DIR / "verified_evidence.jsonl").exists() else []
    rejected = _load_jsonl(STEP2_2_DIR / "rejected_evidence.jsonl") if (STEP2_2_DIR / "rejected_evidence.jsonl").exists() else []
    human_queue = _load_jsonl(STEP2_2_DIR / "human_review_queue.jsonl") if (STEP2_2_DIR / "human_review_queue.jsonl").exists() else []
    unprocessed = _load_jsonl(STEP2_2_DIR / "unprocessed_candidates.jsonl") if (STEP2_2_DIR / "unprocessed_candidates.jsonl").exists() else []
    adjudications = {r["candidate_id"]: r for r in _load_jsonl(adjudication_file)}

    quota_filled = collections.Counter(v["intended_structure"] for v in verified)
    used_chunks = {cid for v in verified for cid in v.get("source_chunk_ids", [])}
    remaining_human = []
    applied = []

    for item in human_queue:
        cid = item["candidate_id"]
        adj = adjudications.get(cid)
        if not adj:
            remaining_human.append(item)
            continue
        decision = adj.get("decision")
        item["adjudication"] = {
            "decision": decision,
            "reviewer": adj.get("reviewer"),
            "note": adj.get("note", ""),
        }
        updates = adj.get("updates") or {}
        for key in ("answer_units", "spans", "evidence_groups", "qrels", "gold_answer"):
            if key in updates:
                item[key] = updates[key]
        if decision == "accept":
            struct = item["intended_structure"]
            if quota_filled[struct] >= STRUCTURE_QUOTA[struct]:
                item["decision"] = "reject"
                item.setdefault("rejection_reasons", []).append("adjudication_quota_full")
                rejected.append(item)
            elif used_chunks & set(item.get("source_chunk_ids", [])):
                item["decision"] = "reject"
                item.setdefault("rejection_reasons", []).append("adjudication_evidence_cluster_collision")
                rejected.append(item)
            else:
                item["decision"] = "accept"
                item["rejection_reasons"] = []
                verified.append(item)
                used_chunks.update(item.get("source_chunk_ids", []))
                quota_filled[struct] += 1
        elif decision == "reject":
            item["decision"] = "reject"
            item.setdefault("rejection_reasons", []).append("adjudication_reject")
            rejected.append(item)
        elif decision == "revise":
            item["decision"] = "revise"
            remaining_human.append(item)
        else:
            item.setdefault("rejection_reasons", []).append("invalid_adjudication_decision")
            remaining_human.append(item)
        applied.append(cid)

    _write_finalize_outputs(
        verified, rejected, remaining_human, unprocessed,
        primary, reserve, {}, {},
    )
    report = collections.OrderedDict()
    report["applied_count"] = len(applied)
    report["applied_candidate_ids"] = sorted(applied)
    report["remaining_human_review_count"] = len(remaining_human)
    with open(STEP2_2_DIR / "adjudication_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[apply-adjudication] applied={len(applied)} remaining_human={len(remaining_human)}")


def _apply_replacement_updates(item: dict, updates: dict) -> dict:
    """Apply a small, auditable patch to one enriched evidence record."""
    updated = copy.deepcopy(item)
    for key in ("answer_units", "spans", "evidence_groups", "gold_answer"):
        if key in updates:
            updated[key] = copy.deepcopy(updates[key])

    span_by_id = {span.get("span_id"): span for span in updated.get("spans", [])}
    for span_id, patch in (updates.get("span_updates") or {}).items():
        if span_id not in span_by_id:
            raise ValueError(f"unknown span_id in replacement update: {span_id}")
        unknown = set(patch) - {"text", "char_start", "char_end"}
        if unknown:
            raise ValueError(f"unsupported span update fields for {span_id}: {sorted(unknown)}")
        span_by_id[span_id].update(patch)

    au_by_id = {au.get("unit_id"): au for au in updated.get("answer_units", [])}
    for unit_id, patch in (updates.get("answer_unit_updates") or {}).items():
        if unit_id not in au_by_id:
            raise ValueError(f"unknown answer unit in replacement update: {unit_id}")
        if isinstance(patch, str):
            patch = {"statement": patch}
        unknown = set(patch) - {"statement"}
        if unknown:
            raise ValueError(f"unsupported answer-unit fields for {unit_id}: {sorted(unknown)}")
        au_by_id[unit_id].update(patch)
    return updated


def _refresh_replacement_record(item: dict, chunk_texts: Dict[str, str],
                                chunk_hashes: Dict[str, str]) -> List[str]:
    """Recompute all derived evidence fields and return blocking problems."""
    problems: List[str] = []
    answer_units = item.get("answer_units") or []
    spans = item.get("spans") or []
    evidence_groups = item.get("evidence_groups") or []

    if not 1 <= len(answer_units) <= 5:
        problems.append(f"answer_unit_count={len(answer_units)} outside [1, 5]")
    if len({au.get("unit_id") for au in answer_units}) != len(answer_units):
        problems.append("duplicate answer unit ids")
    if len({span.get("span_id") for span in spans}) != len(spans):
        problems.append("duplicate span ids")

    traceable, trace_problems = spans_traceable(spans, chunk_texts, chunk_hashes)
    if not traceable:
        problems.extend(trace_problems)
    problems.extend(_evidence_group_problems(answer_units, evidence_groups, spans))

    source_ids = item.get("source_chunk_ids") or []
    missing_sources = sorted(set(source_ids) - set(chunk_texts))
    if missing_sources:
        problems.append(f"source chunks missing: {missing_sources}")

    gold_answer = item.get("gold_answer") or ""
    for au in answer_units:
        unit_id = au.get("unit_id")
        if unit_id and not re.search(rf"\[{re.escape(unit_id)}\]", gold_answer, re.IGNORECASE):
            problems.append(f"gold_answer missing [{unit_id}]")

    item["qrels"] = _qrels_from_answer_units(answer_units, evidence_groups, spans)
    if not missing_sources:
        raw_source = "\n".join(chunk_texts[cid] for cid in source_ids)
        gold_spans = "\n".join(span.get("text", "") for span in spans)
        raw_tokens = count_qwen_tokens(raw_source)
        span_tokens = count_qwen_tokens(gold_spans)
        item["capacity"] = collections.OrderedDict([
            ("raw_source_qwen_tokens", raw_tokens),
            ("gold_span_qwen_tokens", span_tokens),
            ("p4_raw_chunk_fit", raw_tokens <= P4_SOURCE_CAP_QWEN),
            ("p4_gold_span_fit", span_tokens <= P4_SOURCE_CAP_QWEN),
            ("retrieval_capacity_risk", raw_tokens > P4_SOURCE_CAP_QWEN),
        ])
        if span_tokens > P4_SOURCE_CAP_QWEN:
            problems.append(
                f"gold_span_qwen_tokens={span_tokens} > P4 cap {P4_SOURCE_CAP_QWEN}"
            )

    item["n_answer_units"] = len(answer_units)
    forbidden = check_forbidden_fields(item)
    if forbidden:
        problems.append(f"forbidden_fields:{forbidden}")
    return problems


def _index_unique(rows: Sequence[dict], label: str) -> Dict[str, dict]:
    indexed: Dict[str, dict] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            raise ValueError(f"{label} contains a row without candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"duplicate candidate_id in {label}: {candidate_id}")
        indexed[candidate_id] = row
    return indexed


def phase_apply_replacement(data_name: str, replacement_file: Path) -> None:
    """Replace verified evidence records using a fully revalidated JSONL plan."""
    if not replacement_file.exists():
        raise SystemExit(f"[apply-replacement] missing file: {replacement_file}")

    plans = _load_jsonl(replacement_file)
    if not plans:
        raise SystemExit("[apply-replacement] replacement plan is empty")
    _index_unique(plans, "replacement plan")

    primary, reserve, _h = load_candidate_pool()
    verified_path = STEP2_2_DIR / "verified_evidence.jsonl"
    if not verified_path.exists():
        raise SystemExit(f"[apply-replacement] missing file: {verified_path}")
    verified = _load_jsonl(verified_path)
    rejected = _load_jsonl(STEP2_2_DIR / "rejected_evidence.jsonl") if (STEP2_2_DIR / "rejected_evidence.jsonl").exists() else []
    human_queue = _load_jsonl(STEP2_2_DIR / "human_review_queue.jsonl") if (STEP2_2_DIR / "human_review_queue.jsonl").exists() else []
    unprocessed = _load_jsonl(STEP2_2_DIR / "unprocessed_candidates.jsonl") if (STEP2_2_DIR / "unprocessed_candidates.jsonl").exists() else []
    superseded_path = STEP2_2_DIR / "superseded_evidence.jsonl"
    superseded = _load_jsonl(superseded_path) if superseded_path.exists() else []

    verified_by_id = _index_unique(verified, "verified evidence")
    rejected_by_id = _index_unique(rejected, "rejected evidence")
    human_by_id = _index_unique(human_queue, "human review queue")
    source_by_id = dict(rejected_by_id)
    for candidate_id, item in human_by_id.items():
        if candidate_id in source_by_id:
            raise SystemExit(
                f"[apply-replacement] candidate appears in rejected and human queue: {candidate_id}"
            )
        source_by_id[candidate_id] = item

    old_ids = []
    new_ids = []
    for plan in plans:
        if plan.get("decision") != "accept":
            raise SystemExit(
                f"[apply-replacement] {plan.get('candidate_id')}: decision must be accept"
            )
        new_id = plan["candidate_id"]
        old_id = plan.get("replaces_candidate_id")
        if not old_id:
            raise SystemExit(f"[apply-replacement] {new_id}: replaces_candidate_id is required")
        if old_id not in verified_by_id:
            raise SystemExit(f"[apply-replacement] verified candidate not found: {old_id}")
        if new_id not in source_by_id:
            raise SystemExit(f"[apply-replacement] replacement candidate not found: {new_id}")
        old_ids.append(old_id)
        new_ids.append(new_id)
    if len(set(old_ids)) != len(old_ids):
        raise SystemExit("[apply-replacement] duplicate replaces_candidate_id")
    if set(new_ids) & set(verified_by_id):
        raise SystemExit("[apply-replacement] replacement candidate is already verified")

    chunk_texts, chunk_hashes, _tik = load_chunks(data_name)
    remaining = [copy.deepcopy(item) for item in verified if item["candidate_id"] not in set(old_ids)]
    used_chunks = {cid for item in remaining for cid in item.get("source_chunk_ids", [])}
    replacement_by_old: Dict[str, dict] = {}
    replacement_summary = []

    for plan in plans:
        new_id = plan["candidate_id"]
        old_id = plan["replaces_candidate_id"]
        old_item = verified_by_id[old_id]
        new_item = _apply_replacement_updates(source_by_id[new_id], plan.get("updates") or {})
        if new_item.get("intended_structure") != old_item.get("intended_structure"):
            raise SystemExit(
                f"[apply-replacement] structure mismatch: {old_id} -> {new_id}"
            )
        new_item["adjudication"] = collections.OrderedDict([
            ("decision", "accept"),
            ("reviewer", plan.get("reviewer")),
            ("note", plan.get("note", "")),
            ("replaces_candidate_id", old_id),
        ])
        new_item["decision"] = "accept"
        new_item["rejection_reasons"] = []
        problems = _refresh_replacement_record(new_item, chunk_texts, chunk_hashes)
        if problems:
            raise SystemExit(
                f"[apply-replacement] {new_id} failed validation: " + "; ".join(problems)
            )
        collision = sorted(used_chunks & set(new_item.get("source_chunk_ids", [])))
        if collision:
            raise SystemExit(
                f"[apply-replacement] {new_id} evidence cluster collision: {collision}"
            )
        used_chunks.update(new_item.get("source_chunk_ids", []))
        replacement_by_old[old_id] = new_item
        replacement_summary.append(collections.OrderedDict([
            ("replaces_candidate_id", old_id),
            ("candidate_id", new_id),
            ("intended_structure", new_item["intended_structure"]),
            ("source_chunk_ids", new_item.get("source_chunk_ids", [])),
        ]))

    final_verified = [
        replacement_by_old.get(item["candidate_id"], copy.deepcopy(item))
        for item in verified
    ]
    final_ids = [item["candidate_id"] for item in final_verified]
    if len(final_ids) != len(set(final_ids)):
        raise SystemExit("[apply-replacement] duplicate candidate IDs after replacement")
    quota = collections.Counter(item["intended_structure"] for item in final_verified)
    if len(final_verified) != sum(STRUCTURE_QUOTA.values()) or any(
        quota.get(structure, 0) != expected
        for structure, expected in STRUCTURE_QUOTA.items()
    ):
        raise SystemExit(
            f"[apply-replacement] final quota mismatch: {dict(quota)}"
        )

    new_id_set = set(new_ids)
    remaining_rejected = [item for item in rejected if item["candidate_id"] not in new_id_set]
    remaining_human = [item for item in human_queue if item["candidate_id"] not in new_id_set]
    for old_id in old_ids:
        old_item = copy.deepcopy(verified_by_id[old_id])
        old_item["superseded_by"] = replacement_by_old[old_id]["candidate_id"]
        old_item["superseded_reason"] = "manual_question_quality_replacement"
        superseded.append(old_item)

    _write_finalize_outputs(
        final_verified, remaining_rejected, remaining_human, unprocessed,
        primary, reserve, {}, {},
    )
    _write_jsonl(superseded_path, superseded)

    manifest_path = STEP2_2_DIR / "verification_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["phase"] = "apply-replacement"
    manifest["replacement_plan_hash"] = sha256_file(str(replacement_file))
    manifest["replacements"] = replacement_summary
    manifest["output_file_hashes"]["superseded_evidence.jsonl"] = sha256_file(
        str(superseded_path)
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    report = collections.OrderedDict()
    report["replacement_count"] = len(replacement_summary)
    report["replacements"] = replacement_summary
    report["verified_count"] = len(final_verified)
    report["verified_quota"] = dict(quota)
    report["quota_met"] = True
    report["replacement_plan_hash"] = sha256_file(str(replacement_file))
    with open(STEP2_2_DIR / "replacement_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"[apply-replacement] replacements={len(replacement_summary)} "
        f"verified={len(final_verified)} quota_met=True"
    )

def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2.2 证据核验与 Gold 构建")
    ap.add_argument("--phase", required=True,
                    choices=["prepare", "draft", "review", "finalize",
                             "apply-adjudication", "apply-replacement"])
    ap.add_argument("--data-name", default="neurology_chunk1000")
    ap.add_argument("--resume", action="store_true",
                    help="draft/review 阶段跳过已存在的产物")
    ap.add_argument("--promote-reserve", action="store_true",
                    help="finalize 阶段从 reserve 递补以保持配额")
    ap.add_argument("--candidate-scope", choices=["primary", "reserve", "all"],
                    default="primary",
                    help="candidate scope for prepare/draft/review")
    ap.add_argument("--adjudication-file", default=str(STEP2_2_DIR / "human_adjudications.jsonl"),
                    help="JSONL decisions for apply-adjudication")
    ap.add_argument("--replacement-file", default=str(STEP2_2_DIR / "evidence_replacements.jsonl"),
                    help="JSONL replacement plan for apply-replacement")
    ap.add_argument("--review-model", default=REVIEW_MODEL_DEFAULT)
    ap.add_argument("--review-base-url", default=os.environ.get("REVIEW_BASE_URL", ""))
    ap.add_argument("--allow-fallback-reviewer", action="store_true",
                    help="默认复核模型不可达时退回 qwen-27b-int4（显式标注）")
    args = ap.parse_args()

    if args.phase == "prepare":
        phase_prepare(args.data_name, args.candidate_scope)
    elif args.phase == "draft":
        asyncio.run(phase_draft(args.data_name, args.resume, args.candidate_scope))
    elif args.phase == "review":
        asyncio.run(phase_review(
            args.data_name, args.resume, args.review_model,
            args.review_base_url, args.allow_fallback_reviewer, args.candidate_scope))
    elif args.phase == "finalize":
        phase_finalize(args.data_name, args.promote_reserve)
    elif args.phase == "apply-adjudication":
        phase_apply_adjudication(args.data_name, Path(args.adjudication_file))
    elif args.phase == "apply-replacement":
        phase_apply_replacement(args.data_name, Path(args.replacement_file))


if __name__ == "__main__":
    main()
