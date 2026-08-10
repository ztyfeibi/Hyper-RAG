#!/usr/bin/env python
"""Step 2.3: Generate English pilot questions from verified evidence.

Workflow
--------
  seed   – offline conversion: verified_evidence.jsonl → question_seed_items.jsonl
  draft  – Qwen generates an open-ended English question per seed
  review – LongCat audits each generated question
  finalize – merge approved questions → questions_v2.jsonl + validate

Usage
-----
  python scripts/generate_pilot_questions.py --phase seed   --data-name neurology_chunk1000
  python scripts/generate_pilot_questions.py --phase draft  --data-name neurology_chunk1000 --resume
  python scripts/generate_pilot_questions.py --phase review --data-name neurology_chunk1000 --resume
  python scripts/generate_pilot_questions.py --phase finalize --data-name neurology_chunk1000
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

CACHES_ROOT = REPO_ROOT / "caches"

# Qwen (draft model) — vLLM local
QWEN_BASE_URL = "http://10.65.1.110:8002/v1"
QWEN_MODEL = "qwen-27b-int4"
QWEN_API_KEY = "EMPTY"

# LongCat (review model) — via SiliconFlow cloud API
# Credential resolution: env → my_config → fallback
REVIEW_MODEL_DEFAULT = "meituan-longcat/LongCat-2.0"
REVIEW_BASE_URL_DEFAULT = "https://api.siliconflow.cn/v1"
REVIEW_API_KEY_DEFAULT = ""
REVIEW_MAX_TOKENS = 2048
REVIEW_MAX_ATTEMPTS = 4
DRAFT_MAX_TOKENS = 512
DRAFT_TEMPERATURE = 0.1

# Concurrency
DRAFT_CONCURRENCY = 4
REVIEW_CONCURRENCY = 1

SCRIPT_VERSION = "question-gen-v1"


# ---------------------------------------------------------------------------
# Config resolution (mirrors evidence_verification._resolve)
# ---------------------------------------------------------------------------
def _load_my_config():
    try:
        import my_config
        return my_config
    except Exception:
        return None


def _resolve(env_name: str, cfg_attr: str, default: str) -> str:
    env = os.environ.get(env_name)
    if env:
        return env
    cfg = _load_my_config()
    if cfg is not None:
        val = getattr(cfg, cfg_attr, "")
        if val:
            return val
    return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _json_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strip_fences(text: str) -> str:
    """Extract JSON from a markdown-fenced code block."""
    text = text.strip()
    # Remove thinking/reasoning blocks
    text = re.sub(r"<\|start_header\|>.*?<\|end_header\|>", "", text, flags=re.DOTALL)
    # Try ```json ... ``` first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _map_candidate_to_question_id(candidate_id: str) -> str:
    """ec-v2-NNNN → qv2-NNNN."""
    m = re.match(r"ec-v2-(\d+)", candidate_id)
    if m:
        return f"qv2-{m.group(1)}"
    # fallback: replace ec-v2 prefix
    return candidate_id.replace("ec-v2-", "qv2-")


def _clean_gold_answer(raw: str) -> str:
    """Remove trailing [auN] markers from gold answer."""
    return re.sub(r"\s*\[au\d+\]", "", raw).strip()


# ---------------------------------------------------------------------------
# Seed phase: offline conversion
# ---------------------------------------------------------------------------
def _build_seed_item(ev: dict) -> dict:
    """Convert one verified_evidence record into a question seed item.

    Maps the internal evidence format to the question_item_v2 schema skeleton
    with question=null (placeholder).
    """
    qid = _map_candidate_to_question_id(ev["candidate_id"])

    # answer_units: au1 → AU1, statement → claim
    answer_units = []
    for au in ev.get("answer_units", []):
        au_number_str = au["unit_id"].replace("au", "")
        answer_units.append({
            "unit_id": f"AU{au_number_str}",
            "claim": au["statement"],
            "required": True,
        })

    # evidence_requirements: map from evidence_groups
    evidence_requirements = []
    for eg in ev.get("evidence_groups", []):
        au_id_raw = eg.get("answer_unit_id", "")
        au_number_str = au_id_raw.replace("au", "")
        req_id = f"ER{au_number_str}"
        # Collect chunk_ids from the spans referenced by this group
        chunk_ids = []
        for sp_id in eg.get("span_ids", []):
            for sp in ev.get("spans", []):
                if sp.get("span_id") == sp_id:
                    cid = sp["chunk_id"]
                    if cid not in chunk_ids:
                        chunk_ids.append(cid)
        evidence_requirements.append({
            "requirement_id": req_id,
            "answer_unit_ids": [f"AU{au_number_str}"],
            "alternative_chunk_ids": chunk_ids,
        })

    # evidence_spans: map from per-AU spans
    evidence_spans = []
    for au in ev.get("answer_units", []):
        au_id_raw = au["unit_id"]
        au_number_str = au_id_raw.replace("au", "")
        au_spans = []
        eg_list = [eg for eg in ev.get("evidence_groups", [])
                   if eg.get("answer_unit_id") == au_id_raw]
        seen_span_ids = set()
        for eg in eg_list:
            for sp_id in eg.get("span_ids", []):
                if sp_id in seen_span_ids:
                    continue
                seen_span_ids.add(sp_id)
                for sp in ev.get("spans", []):
                    if sp.get("span_id") == sp_id:
                        au_spans.append({
                            "chunk_id": sp["chunk_id"],
                            "start_char": sp["char_start"],
                            "end_char": sp["char_end"],
                            "quote": sp["text"],
                        })
        evidence_spans.append({
            "unit_id": f"AU{au_number_str}",
            "evidence_spans": au_spans,
        })

    # required_subgraph
    required_vertices = list(ev.get("seed_entity_ids", []))
    required_hyperedges = []
    for i, he in enumerate(ev.get("hyperedges", []), start=1):
        required_hyperedges.append({
            "hyperedge_id": f"he-{i}",
            "entity_set": list(he.get("entity_ids", [])),
            "source_chunk_ids": list(he.get("source_chunk_ids", [])),
        })
    au_links = {}
    for au in ev.get("answer_units", []):
        n = au["unit_id"].replace("au", "")
        au_links[f"AU{n}"] = [f"he-{i}" for i in range(1, len(required_hyperedges) + 1)]

    required_subgraph = {
        "required_vertices": required_vertices,
        "required_hyperedges": required_hyperedges,
        "answer_unit_links": au_links,
        "topology_metrics": {},
    }

    # Clean gold answer
    gold_answer = _clean_gold_answer(ev.get("gold_answer", ""))

    seed = {
        "question_id": qid,
        "source_candidate_id": ev["candidate_id"],
        "language": "en",
        "question": None,  # placeholder, filled by draft phase
        "gold_answer": gold_answer,
        "answer_units": answer_units,
        "evidence_requirements": evidence_requirements,
        "evidence_spans": evidence_spans,
        "required_subgraph": required_subgraph,
        "intended_structure": ev.get("intended_structure"),
        "verified_structure": ev.get("intended_structure"),
        # carry source chunk IDs for draft prompt context
        "_source_chunks": ev.get("source_chunk_ids", []),
        "_n_answer_units": len(answer_units),
    }
    return seed


def phase_seed(data_name: str) -> None:
    """Offline conversion: verified_evidence → question_seed_items."""
    step2_2_dir = CACHES_ROOT / data_name / "question_set_v2" / "pilot_v1" / "step2_2"
    verified_path = step2_2_dir / "verified_evidence.jsonl"
    if not verified_path.exists():
        raise SystemExit(f"verified_evidence.jsonl not found: {verified_path}")

    out_dir = CACHES_ROOT / data_name / "question_set_v2" / "pilot_v1" / "question_generation"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _read_jsonl(verified_path)
    seeds = []
    for ev in items:
        seed = _build_seed_item(ev)
        seeds.append(seed)

    seed_path = out_dir / "question_seed_items.jsonl"
    _write_jsonl(seed_path, seeds)

    # Summary stats
    struct_counts: Dict[str, int] = collections.Counter()
    for s in seeds:
        struct_counts[s["intended_structure"]] += 1

    print(f"[seed] 生成 {len(seeds)} 条种子问题 → {seed_path}")
    for struct, count in struct_counts.most_common():
        print(f"  {struct}: {count}")

    # manifest
    manifest = {
        "script_version": SCRIPT_VERSION,
        "phase": "seed",
        "source": str(verified_path),
        "n_seeds": len(seeds),
        "structure_distribution": dict(struct_counts),
        "output_file": str(seed_path),
        "output_sha256": _json_hash(seeds),
    }
    _write_json(out_dir / "seed_manifest.json", manifest)


# ---------------------------------------------------------------------------
# Draft phase: Qwen generates English questions
# ---------------------------------------------------------------------------
DRAFT_SYSTEM_PROMPT = (
    "You are a neuroscience exam question writer. "
    "Your task is to create an open-ended English question based ONLY on "
    "the provided answer points and supporting evidence. "
    "Follow these rules STRICTLY:\n"
    "- Output ONLY a single JSON object with key \"question\".\n"
    "- The question MUST be in English.\n"
    "- The question MUST be open-ended (NOT multiple choice, NOT yes/no).\n"
    "- The question MUST NOT directly reveal any answer details.\n"
    "- The question MUST be answerable from the provided evidence only.\n"
    "- Do NOT ask for lists of everything — focus on specific connections.\n"
    "- Do NOT create \"review\" or \"overview\" type questions.\n"
    "- Keep it concise: one or two sentences.\n"
    "- The question MUST NOT require external knowledge beyond the evidence."
)


def _build_draft_user_prompt(seed: dict) -> str:
    lines = ["=== ANSWER POINTS (to be covered by the question) ==="]
    for au in seed["answer_units"]:
        lines.append(f"[{au['unit_id']}] {au['claim']}")
    lines.append("")
    lines.append(f"=== GOLD ANSWER ===")
    lines.append(seed["gold_answer"])
    lines.append("")
    lines.append(f"=== INTENDED STRUCTURE ===")
    lines.append(seed["intended_structure"])
    lines.append("")
    lines.append("Generate an open-ended English question. Output ONLY JSON: {\"question\": \"...\"}")
    return "\n".join(lines)


async def _draft_one(qid: str, seed: dict, hashing_kv, sem) -> dict:
    from hyperrag.llm import openai_complete_if_cache

    async with sem:
        user_prompt = _build_draft_user_prompt(seed)
        raw = await openai_complete_if_cache(
            QWEN_MODEL, user_prompt,
            system_prompt=DRAFT_SYSTEM_PROMPT,
            base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY,
            hashing_kv=hashing_kv,
            temperature=DRAFT_TEMPERATURE, max_tokens=DRAFT_MAX_TOKENS,
        )
    question = None
    parse_error = None
    try:
        parsed = json.loads(_strip_fences(raw))
        question = parsed.get("question", "").strip()
        if not question:
            parse_error = "empty question field"
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
    return {
        "question_id": qid,
        "script_version": SCRIPT_VERSION,
        "draft_model": QWEN_MODEL,
        "raw_response": raw,
        "question": question,
        "parse_error": parse_error,
        "seed_hash": _json_hash(seed),
    }


async def phase_draft(data_name: str, resume: bool) -> None:
    from hyperrag.storage import JsonKVStorage

    out_dir = CACHES_ROOT / data_name / "question_set_v2" / "pilot_v1" / "question_generation"
    seed_path = out_dir / "question_seed_items.jsonl"
    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}. Run --phase seed first.")

    seeds = _read_jsonl(seed_path)
    draft_raw_dir = out_dir / "draft_raw"
    draft_raw_dir.mkdir(parents=True, exist_ok=True)

    cache_kv = JsonKVStorage(
        namespace="question_gen_draft_llm_cache",
        global_config={"working_dir": str(out_dir)},
    )
    sem = asyncio.Semaphore(DRAFT_CONCURRENCY)
    tasks = []
    skipped = 0
    for seed in seeds:
        qid = seed["question_id"]
        out_path = draft_raw_dir / f"{qid}.json"
        if resume and out_path.exists():
            existing = _read_json(out_path)
            if existing.get("question") and not existing.get("parse_error"):
                skipped += 1
                continue
        tasks.append(_draft_one(qid, seed, cache_kv, sem))

    if not tasks:
        print(f"[draft] 无待处理种子（resume 已全部完成，跳过 {skipped} 条）")
        return

    print(f"[draft] 开始生成 {len(tasks)} 条问题（跳过 {skipped} 条已完成）...")
    results = await asyncio.gather(*tasks)
    for res in results:
        _write_json(draft_raw_dir / f"{res['question_id']}.json", res)

    ok = sum(1 for r in results if r["question"] and not r["parse_error"])
    fail = len(results) - ok
    print(f"[draft] 完成 {len(results)} 条（ok={ok}, parse_failed={fail}）")


# ---------------------------------------------------------------------------
# Review phase: LongCat audits generated questions
# ---------------------------------------------------------------------------
REVIEW_SYSTEM_PROMPT = (
    "You are a rigorous exam question quality auditor. "
    "Your task is to evaluate whether a generated English question meets ALL criteria. "
    "You must output ONLY a single JSON object with keys: "
    "\"verdict\" (pass/fail/needs_revision), "
    "\"checks\" (object with boolean fields), "
    "\"issues\" (list of strings), "
    "\"revised_question\" (string, or null if pass).\n\n"
    "Required checks:\n"
    "- is_english: question is in English\n"
    "- is_open_ended: NOT multiple choice, NOT yes/no, NOT fill-in-the-blank\n"
    "- answerable_from_evidence: question can be fully answered from the provided answer points\n"
    "- no_answer_leak: question does NOT directly reveal any answer detail\n"
    "- no_external_knowledge: question does NOT require external knowledge beyond evidence\n"
    "- not_overly_broad: question is specific enough, not a broad review/overview\n"
    "- not_choice_like: question is NOT structured like a multiple choice\n\n"
    "If verdict is \"fail\" or \"needs_revision\", provide a corrected version in revised_question."
)


def _build_review_user_prompt(seed: dict, question: str) -> str:
    lines = ["=== ANSWER POINTS (ground truth) ==="]
    for au in seed["answer_units"]:
        lines.append(f"[{au['unit_id']}] {au['claim']}")
    lines.append("")
    lines.append(f"=== GOLD ANSWER ===")
    lines.append(seed["gold_answer"])
    lines.append("")
    lines.append(f"=== GENERATED QUESTION TO AUDIT ===")
    lines.append(question)
    lines.append("")
    lines.append("Audit the question per system instructions. Output ONLY JSON.")
    return "\n".join(lines)


async def _review_one(qid: str, seed: dict, question: str,
                      review_model: str, review_base_url: str,
                      review_api_key: str, hashing_kv, sem) -> dict:
    from hyperrag.llm import openai_complete_if_cache

    user_prompt = _build_review_user_prompt(seed, question)
    raw = ""
    parsed = None
    parse_error = None
    attempts_used = 0

    async with sem:
        for attempt in range(1, REVIEW_MAX_ATTEMPTS + 1):
            attempts_used = attempt
            raw = await openai_complete_if_cache(
                review_model, user_prompt,
                system_prompt=REVIEW_SYSTEM_PROMPT,
                base_url=review_base_url, api_key=review_api_key,
                # Do not cache review calls: empty/truncated provider responses
                # must not poison retry runs.
                hashing_kv=None,
                temperature=0.0, max_tokens=REVIEW_MAX_TOKENS,
            )
            try:
                parsed = json.loads(_strip_fences(raw))
                parse_error = None
                break
            except Exception as e:
                parsed = None
                parse_error = f"{type(e).__name__}: {e}"

    return {
        "question_id": qid,
        "script_version": SCRIPT_VERSION,
        "review_model": review_model,
        "raw_response": raw,
        "parsed": parsed,
        "parse_error": parse_error,
        "review_attempts": attempts_used,
        "seed_hash": _json_hash(seed),
        "question_audited": question,
    }


async def phase_review(data_name: str, resume: bool,
                       review_model: str = "", review_base_url: str = "",
                       allow_fallback: bool = False) -> None:
    from hyperrag.storage import JsonKVStorage

    out_dir = CACHES_ROOT / data_name / "question_set_v2" / "pilot_v1" / "question_generation"
    seed_path = out_dir / "question_seed_items.jsonl"
    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}. Run --phase seed first.")

    review_model = review_model or REVIEW_MODEL_DEFAULT
    review_base_url = review_base_url or REVIEW_BASE_URL_DEFAULT
    review_api_key = _resolve(
        "SILICONFLOW_API_KEY", "LLM_API_KEY_SILICONFLOW", REVIEW_API_KEY_DEFAULT
    )

    seeds = _read_jsonl(seed_path)
    draft_raw_dir = out_dir / "draft_raw"
    review_raw_dir = out_dir / "review_raw"
    review_raw_dir.mkdir(parents=True, exist_ok=True)

    # Build seed lookup
    seed_by_qid = {s["question_id"]: s for s in seeds}

    cache_kv = JsonKVStorage(
        namespace="question_gen_review_llm_cache",
        global_config={"working_dir": str(out_dir)},
    )
    sem = asyncio.Semaphore(REVIEW_CONCURRENCY)
    tasks = []
    skipped = 0
    for seed in seeds:
        qid = seed["question_id"]
        draft_path = draft_raw_dir / f"{qid}.json"
        if not draft_path.exists():
            print(f"[review] WARNING: draft_raw missing for {qid}, skipping")
            continue
        draft = _read_json(draft_path)
        question = draft.get("question")
        if not question:
            print(f"[review] WARNING: draft has no question for {qid}, skipping")
            continue

        out_path = review_raw_dir / f"{qid}.json"
        if resume and out_path.exists():
            existing = _read_json(out_path)
            if existing.get("parsed") and not existing.get("parse_error"):
                skipped += 1
                continue
        tasks.append(_review_one(
            qid, seed, question,
            review_model, review_base_url, review_api_key,
            cache_kv, sem,
        ))

    if not tasks:
        print(f"[review] 无待审核问题（resume 已全部完成，跳过 {skipped} 条）")
        return

    print(f"[review] 开始审核 {len(tasks)} 条问题（跳过 {skipped} 条已完成）...")
    results = await asyncio.gather(*tasks)
    for res in results:
        _write_json(review_raw_dir / f"{res['question_id']}.json", res)

    pass_count = 0
    fail_count = 0
    needs_rev = 0
    parse_err = 0
    for res in results:
        if res.get("parse_error"):
            parse_err += 1
        elif res.get("parsed"):
            v = res["parsed"].get("verdict", "unknown")
            if v == "pass":
                pass_count += 1
            elif v == "fail":
                fail_count += 1
            elif v == "needs_revision":
                needs_rev += 1
    print(f"[review] 完成 {len(results)} 条 "
          f"(pass={pass_count}, needs_revision={needs_rev}, fail={fail_count}, parse_error={parse_err})")

    # Report issues for non-pass
    for res in results:
        if res.get("parsed"):
            v = res["parsed"].get("verdict", "unknown")
            if v != "pass":
                issues = res["parsed"].get("issues", [])
                if issues:
                    print(f"  {res['question_id']}: {v} — {'; '.join(issues[:3])}")
                else:
                    print(f"  {res['question_id']}: {v}")


# ---------------------------------------------------------------------------
# Finalize phase: output questions_v2.jsonl + validate
# ---------------------------------------------------------------------------
def phase_finalize(data_name: str) -> None:
    out_dir = CACHES_ROOT / data_name / "question_set_v2" / "pilot_v1" / "question_generation"
    seed_path = out_dir / "question_seed_items.jsonl"
    draft_raw_dir = out_dir / "draft_raw"
    review_raw_dir = out_dir / "review_raw"

    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}")

    seeds = _read_jsonl(seed_path)
    seed_by_qid = {s["question_id"]: s for s in seeds}

    questions = []
    issues = []
    for seed in seeds:
        qid = seed["question_id"]
        draft_path = draft_raw_dir / f"{qid}.json"
        review_path = review_raw_dir / f"{qid}.json"

        question_text = None
        # Strict gate: every finalized question must have valid draft and review.
        if not draft_path.exists():
            issues.append(f"{qid}: draft_raw missing")
            continue
        draft = _read_json(draft_path)
        if draft.get("parse_error") or not draft.get("question"):
            issues.append(f"{qid}: draft invalid ({draft.get('parse_error') or 'empty question'})")
            continue

        if not review_path.exists():
            issues.append(f"{qid}: review_raw missing")
            continue
        review = _read_json(review_path)
        if review.get("parse_error") or not review.get("parsed"):
            issues.append(f"{qid}: review invalid ({review.get('parse_error') or 'missing parsed review'})")
            continue

        parsed = review["parsed"]
        verdict = parsed.get("verdict", "unknown")
        if verdict == "pass":
            question_text = review.get("question_audited") or draft.get("question", "")
        elif verdict == "needs_revision" and parsed.get("revised_question"):
            question_text = parsed["revised_question"]
        else:
            issues.append(f"{qid}: review verdict {verdict} without acceptable revised question")
            continue

        if not question_text or not question_text.strip():
            issues.append(f"{qid}: empty question")
            continue

        # Build final question item
        item = collections.OrderedDict()
        item["question_id"] = qid
        item["language"] = "en"
        item["question"] = question_text.strip()
        item["gold_answer"] = seed["gold_answer"]
        item["answer_units"] = seed["answer_units"]
        item["evidence_requirements"] = seed["evidence_requirements"]
        item["evidence_spans"] = seed["evidence_spans"]

        # optional fields
        rsg = seed.get("required_subgraph")
        if rsg and (rsg.get("required_vertices") or rsg.get("required_hyperedges")):
            item["required_subgraph"] = rsg
        item["intended_structure"] = seed.get("intended_structure")
        item["verified_structure"] = seed.get("verified_structure")

        questions.append(item)

    # Write output
    final_path = out_dir / "questions_v2.jsonl"
    _write_jsonl(final_path, questions)

    # Structure distribution
    struct_counts: Dict[str, int] = collections.Counter()
    for q in questions:
        struct_counts[q.get("intended_structure", "unknown")] += 1

    quota_target = {
        "single_fact": 20,
        "single_high_arity": 15,
        "multi_edge_chain": 20,
        "multi_branch": 15,
        "similar_subgraph_disambiguation": 10,
    }
    expected_total = sum(quota_target.values())
    if len(questions) != expected_total:
        issues.append(f"finalized question count {len(questions)} != expected {expected_total}")
    for struct, expected in quota_target.items():
        actual = struct_counts.get(struct, 0)
        if actual != expected:
            issues.append(f"{struct}: finalized count {actual} != expected {expected}")

    print(f"[finalize] 输出 {len(questions)} 条问题 → {final_path}")
    for struct, count in struct_counts.most_common():
        print(f"  {struct}: {count}")

    if issues:
        print(f"\n[finalize] 警告: {len(issues)} 条问题有问题:")
        for iss in issues[:10]:
            print(f"  {iss}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    # Write manifest
    manifest = {
        "script_version": SCRIPT_VERSION,
        "phase": "finalize",
        "n_questions": len(questions),
        "structure_distribution": dict(struct_counts),
        "quota_target": quota_target,
        "output_file": str(final_path),
        "output_sha256": _json_hash(questions),
        "issues": issues,
    }
    _write_json(out_dir / "questions_v2_manifest.json", manifest)

    # Quick schema check (best-effort)
    try:
        from hyperrag.experiment_schema import validate_question_item
        schema_path = REPO_ROOT / "docs" / "schema" / "question_item_v2.schema.json"
        schema = _read_json(schema_path)
        schema_errors = []
        for i, item in enumerate(questions):
            try:
                validate_question_item(item, schema, "$")
            except Exception as e:
                schema_errors.append(f"[{i}] {item.get('question_id')}: {e}")
        if schema_errors:
            print(f"\n[finalize] 架构校验失败 ({len(schema_errors)} 条):")
            for err in schema_errors[:10]:
                print(f"  {err}")
        else:
            print(f"\n[finalize] 架构校验: 全部通过 ✓")
    except Exception as e:
        print(f"\n[finalize] 架构校验跳过 (schema import failed): {e}")


    if issues:
        raise SystemExit("[finalize] strict gate failed; fix review outputs and rerun finalize.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 2.3: Generate pilot questions from verified evidence")
    parser.add_argument("--phase", required=True,
                        choices=["seed", "draft", "review", "finalize"])
    parser.add_argument("--data-name", default="neurology_chunk1000")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed items")
    parser.add_argument("--review-model", default="",
                        help="Override review model (default: LongCat-2.0)")
    parser.add_argument("--review-base-url", default="",
                        help="Override review API base URL")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Allow review fallback to Qwen if LongCat unreachable")
    args = parser.parse_args()

    phase = args.phase
    data_name = args.data_name

    if phase == "seed":
        phase_seed(data_name)
    elif phase == "draft":
        asyncio.run(phase_draft(data_name, resume=args.resume))
    elif phase == "review":
        asyncio.run(phase_review(
            data_name, resume=args.resume,
            review_model=args.review_model,
            review_base_url=args.review_base_url,
            allow_fallback=args.allow_fallback,
        ))
    elif phase == "finalize":
        phase_finalize(data_name)


if __name__ == "__main__":
    main()
