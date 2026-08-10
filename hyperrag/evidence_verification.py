# -*- coding: utf-8 -*-
"""Step 2.2 —— 证据核验与 Gold 构建：Schema + 机械校验基元。

本模块只负责"纯函数 / 数据契约 / 机械校验"，不发起任何网络或模型调用
（token 计数通过同步 urllib 调本地 vLLM /tokenize，属于确定性本地调用）。
这样单元测试可以不依赖 hyperdb / LLM 即可覆盖全部不变量。

设计原则
--------
1. **禁止字段**：任何产出记录都不得包含 ``difficulty`` / ``route`` / ``question``
   （问题文本、路径标签、难度标签属于 Step 2.3 及之后）。本模块定义
   :data:`FORBIDDEN_RECORD_FIELDS` 供产出时校验。
2. **精确子串**：evidence span 必须是 source chunk 原文的精确子串，且
   ``char_start/char_end`` 可由原文复算，chunk 内容哈希必须匹配。
3. **真实 Qwen token**：用 vLLM ``/tokenize``（model=qwen-27b-int4）统计，
   与建库期的 tiktoken（gpt-4o-mini）明确区分。
4. **确定性**：所有集合先排序再序列化；不写时间戳。
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 冻结常量（Step 2.2 契约）
# --------------------------------------------------------------------------
SCRIPT_VERSION = "evidence-verify-v2"

#: 绑定的候选池哈希（由 build_pilot_evidence_pool.py v2 产出）。
#: prepare 阶段会重算并比对，不一致即 fail-fast。
BOUND_CANDIDATE_POOL_HASH = (
    "f2fa1cb6d2058db0741bea65b1d639f9bc09424b67d45727f60665b912d525dd"
)

#: 候选池声明使用的 source tokenizer（建库期保存的 chunk.tokens 字段单位）。
SOURCE_TOKENIZER_MODEL = "gpt-4o-mini"

#: Draft 抽取模型（生成 Gold 草稿）。
DRAFT_MODEL = "qwen-27b-int4"
DRAFT_BASE_URL = "http://10.65.1.110:8002/v1"
DRAFT_API_KEY = "EMPTY"
DRAFT_TEMPERATURE = 0.1
DRAFT_SEED = 42
DRAFT_TOP_P = 0.95
DRAFT_MAX_TOKENS = 4096

#: Review 模型（规范默认 LongCat-2.0），通过 SiliconFlow 云 API 访问。
#: 凭证 **绝不硬编码**：优先读环境变量，回退到被 .gitignore 排除的 my_config
#: （LLM_API_KEY_SILICONFLOW / LLM_BASE_URL_SILICONFLOW / LLM_MODEL_SILICONFLOW）。
#: 若该端点不可用，verify 脚本在 --allow-fallback-reviewer 下退回 qwen-27b-int4
#: 并显式标注 review_model_fallback=True（不静默替换）。


def _load_my_config() -> "module | None":
    """安全导入本地 gitignore 配置（my_config.py 不入库）。失败返回 None。"""
    try:
        import my_config  # 本地 gitignore 配置，不入库
        return my_config
    except Exception:
        return None


def _resolve(env_name: str, cfg_attr: str, default: str) -> str:
    """配置解析：环境变量优先，回退到 my_config 的指定属性，最后用 default。"""
    env = os.environ.get(env_name)
    if env:
        return env
    cfg = _load_my_config()
    if cfg is not None:
        val = getattr(cfg, cfg_attr, "")
        if val:
            return val
    return default


REVIEW_MODEL_DEFAULT = _resolve(
    "SILICONFLOW_MODEL",
    "LLM_MODEL_SILICONFLOW",
    "meituan-longcat/LongCat-2.0",
)
REVIEW_BASE_URL = _resolve(
    "SILICONFLOW_BASE_URL",
    "LLM_BASE_URL_SILICONFLOW",
    "https://api.siliconflow.cn/v1",
)
REVIEW_API_KEY = _resolve(
    "SILICONFLOW_API_KEY",
    "LLM_API_KEY_SILICONFLOW",
    "",
)
#: LongCat 是推理模型：reasoning 会占用 token 预算，max_tokens 过小会导致
#: 最终 content 为空。review 取 1024 以留足 reasoning + JSON 输出空间。
REVIEW_MAX_TOKENS = 1024

#: 真实 Qwen tokenizer 端点（vLLM /tokenize，注意不是 /v1/tokenize）。
QWEN_TOKENIZE_URL = "http://10.65.1.110:8002/tokenize"
QWEN_TOKENIZE_MODEL = "qwen-27b-int4"

#: P4 路径上下文预算（Qwen token 口径，与 source cap 同单位）。
P4_SOURCE_CAP_QWEN = 4000

#: 结构配额（必须与候选池一致）。
STRUCTURE_QUOTA: "collections.OrderedDict[str, int]" = collections.OrderedDict([
    ("single_fact", 20),
    ("single_high_arity", 15),
    ("multi_edge_chain", 20),
    ("multi_branch", 15),
    ("similar_subgraph_disambiguation", 10),
])

#: 出现即判定为非法的记录字段（难度 / 路径 / 问题文本属于后续阶段）。
FORBIDDEN_RECORD_FIELDS = frozenset({
    "difficulty", "route", "route_label", "question", "question_text",
    "query", "query_text", "answer_route", "path_label",
})

#: graph-first 结构（必要超边必须能被原文支持）。
GRAPH_FIRST_STRUCTURES = frozenset({
    "single_high_arity", "multi_edge_chain", "multi_branch",
    "similar_subgraph_disambiguation",
})

# --------------------------------------------------------------------------
# 哈希 / 文本工具
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def split_source_ids(s: Optional[str]) -> List[str]:
    """把 'chunk-a<SEP>chunk-b' 拆成排序去重列表。"""
    if not s:
        return []
    return sorted(set(part for part in s.split("<SEP>") if part))


_CJK_RE = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")


def classify_language(texts: Sequence[str]) -> str:
    """按字符比例判定 zh / en / mixed（客观，非难度标签）。"""
    cjk = 0
    total = 0
    for t in texts:
        if not t:
            continue
        for ch in t:
            if ch.isspace() or not ch.isalpha():
                continue
            total += 1
            if _CJK_RE.match(ch):
                cjk += 1
    if total == 0:
        return "en"
    ratio = cjk / total
    if ratio >= 0.5:
        return "zh"
    if ratio > 0.0:
        return "mixed"
    return "en"


# --------------------------------------------------------------------------
# 真实 Qwen tokenizer 计数（本地 /tokenize，确定性可复算）
# --------------------------------------------------------------------------
def count_qwen_tokens(text: str, url: str = QWEN_TOKENIZE_URL,
                      model: str = QWEN_TOKENIZE_MODEL) -> int:
    """用 vLLM /tokenize（qwen-27b-int4）统计真实 Qwen token 数。

    返回响应里的 ``count`` 字段。网络异常会原样上浮，不静默造假。
    """
    if text is None:
        text = ""
    body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    if "count" in d:
        return int(d["count"])
    toks = d.get("tokens") or []
    return len(toks)


# --------------------------------------------------------------------------
# 精确子串 / 边界复算
# --------------------------------------------------------------------------
def locate_span(chunk_text: str, excerpt: str) -> Optional[Tuple[int, int]]:
    """在原文中定位摘录的精确起止；找不到返回 None（非逐字子串）。"""
    if not excerpt:
        return None
    idx = chunk_text.find(excerpt)
    if idx == -1:
        return None
    return idx, idx + len(excerpt)


def validate_span(chunk_text: str, chunk_hash: str, span: dict) -> Tuple[bool, str]:
    """校验单条 span：边界可复算 + 文本一致 + chunk 哈希匹配。

    返回 (ok, reason)。reason ∈ {ok, missing_bounds, bounds_out_of_range,
    text_mismatch, chunk_hash_mismatch}。
    """
    if span.get("chunk_content_hash") and span["chunk_content_hash"] != chunk_hash:
        return False, "chunk_hash_mismatch"
    s, e = span.get("char_start"), span.get("char_end")
    if s is None or e is None:
        return False, "missing_bounds"
    if not (0 <= s <= e <= len(chunk_text)):
        return False, "bounds_out_of_range"
    extracted = chunk_text[s:e]
    if extracted != span.get("text"):
        return False, "text_mismatch"
    return True, "ok"


def spans_traceable(spans: Sequence[dict], chunk_texts: Dict[str, str],
                    chunk_hashes: Dict[str, str]) -> Tuple[bool, List[str]]:
    """批量校验 span 可逐字回溯。返回 (all_ok, 失败原因列表)。"""
    problems: List[str] = []
    for sp in spans:
        cid = sp.get("chunk_id")
        if cid not in chunk_texts:
            problems.append(f"span chunk_id {cid} 不在可用原文中")
            continue
        ok, reason = validate_span(
            chunk_texts[cid], chunk_hashes.get(cid, ""), sp
        )
        if not ok:
            problems.append(f"span@{cid}:{reason}")
    return (len(problems) == 0), problems


# --------------------------------------------------------------------------
# graph-first 超边 provenance 校验
# --------------------------------------------------------------------------
def hyperedge_supported_by_text(hyperedge: dict,
                                source_texts: Sequence[str],
                                source_chunk_ids: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    """Return whether graph-first evidence has source provenance.

    Lexical entity matches are useful diagnostics, but they are not a safe hard
    gate for this project: graph entity labels may be Chinese while the frozen
    source chunks are English. A hyperedge is therefore accepted when its stored
    provenance overlaps the candidate source chunks, even if entity names do not
    appear verbatim in the source text. Missing or mismatched provenance remains
    blocking.
    """
    ents = hyperedge.get("entity_ids") or []
    if len(ents) < 2:
        return False, "arity_too_small"

    candidate_sources = set(source_chunk_ids or [])
    edge_sources = set(hyperedge.get("source_ids") or [])
    provenance_hits = sorted(candidate_sources & edge_sources)

    low = [str(e).lower() for e in ents]
    lexical_hits = 0
    for txt in source_texts:
        tl = txt.lower()
        lexical_hits = max(lexical_hits, sum(1 for e in low if e and e in tl))

    if provenance_hits:
        if lexical_hits >= 2:
            return True, f"source_provenance_overlap;matched_{lexical_hits}_entities"
        return True, "source_provenance_overlap;weak_lexical_grounding"
    if lexical_hits >= 2:
        return True, f"matched_{lexical_hits}_entities;missing_source_provenance"
    if edge_sources:
        return False, "source_provenance_mismatch"
    return False, "missing_hyperedge_provenance"


# --------------------------------------------------------------------------
# ?????OrderedDict?????????????
# --------------------------------------------------------------------------
def make_span(chunk_id: str, text: str, char_start: Optional[int],
              char_end: Optional[int], chunk_content_hash: str) -> "collections.OrderedDict":
    od = collections.OrderedDict()
    od["chunk_id"] = chunk_id
    od["char_start"] = char_start
    od["char_end"] = char_end
    od["text"] = text
    od["chunk_content_hash"] = chunk_content_hash
    return od


def make_answer_unit(unit_id: str, statement: str) -> "collections.OrderedDict":
    od = collections.OrderedDict()
    od["unit_id"] = unit_id
    od["statement"] = statement
    od["evidence_group_ids"] = []
    od["qwen_tokens"] = None  # 由 finalize 按 statement 计
    return od


def make_evidence_group(group_id: str, answer_unit_id: str,
                        span_ids: Sequence[str],
                        rationale: str) -> "collections.OrderedDict":
    od = collections.OrderedDict()
    od["group_id"] = group_id
    od["answer_unit_id"] = answer_unit_id
    od["span_ids"] = sorted(set(span_ids))
    od["rationale"] = rationale
    return od


def make_review_verdict(answer_unit_id: str, judgment: str,
                        action: str, note: str,
                        revision_suggestion: Optional[str] = None) -> "collections.OrderedDict":
    od = collections.OrderedDict()
    od["answer_unit_id"] = answer_unit_id
    # supported | partially_supported | contradicted | unsupported
    od["judgment"] = judgment
    # accept | revise | reject
    od["action"] = action
    od["note"] = note
    od["revision_suggestion"] = revision_suggestion
    return od


def make_qrels_entry(answer_unit_id: str, chunk_ids: Sequence[str],
                     span_ids: Sequence[str]) -> "collections.OrderedDict":
    od = collections.OrderedDict()
    od["answer_unit_id"] = answer_unit_id
    od["relevant_chunk_ids"] = sorted(set(chunk_ids))
    od["relevant_span_ids"] = sorted(set(span_ids))
    return od


def check_forbidden_fields(record: dict, path: str = "") -> List[str]:
    """递归检查记录中是否出现禁止字段。返回违规路径列表。"""
    bad: List[str] = []
    if isinstance(record, dict):
        for k, v in record.items():
            full = f"{path}.{k}" if path else k
            if k in FORBIDDEN_RECORD_FIELDS:
                bad.append(full)
            bad.extend(check_forbidden_fields(v, full))
    elif isinstance(record, list):
        for i, v in enumerate(record):
            bad.extend(check_forbidden_fields(v, f"{path}[{i}]"))
    return bad
