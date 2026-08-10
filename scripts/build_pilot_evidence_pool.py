"""Step 2.1 —— 构建候选证据池（evidence-first，不生成问题 / 不写 Gold answer）。

本脚本从已建好的 HyperRAG 索引中**确定性**抽取 80 个证据优先（evidence-first）
候选。每个候选只描述"有哪些证据、证据构成什么结构"，不含任何问题文本、
答案文本或主观难度标签——那些属于 Step 2.2 之后的阶段。

设计要点
--------
1. **intended_structure 而非 verified_structure**
   本阶段的结构标签仅表示"按图拓扑判定它*应当*是这种结构"。是否真的只能
   靠这些证据回答，要等下一步证据核验（``status`` 固定为
   ``pending_evidence_verification``）。

2. **确定性**
   - 所有来自 ``set`` 的集合（``all_v`` / ``all_e`` / ``nbr_e_of_v``）在使用
     前一律排序。
   - 抽样顺序 = ``sha256(f"{seed}|{signature}")`` 的字典序，即"以 seed 为
     盐的确定性伪随机排序"，不依赖 Python 内置 ``hash()``（其带 PYTHONHASHSEED
     随机化）也不依赖 ``random`` 模块的实现版本。
   - 输出文件中不写入时间戳，保证 **同 seed 重跑得到逐字节相同的文件**。

3. **证据簇去重**
   已被选中的候选所使用的 source chunk 会被占用；后续候选只要与之共享任一
   source chunk 就跳过（``evidence_cluster_collision``）。这保证 80 条候选的
   证据互不重叠，避免"同一段原文换个问法"。

用法::

    python scripts/build_pilot_evidence_pool.py --data-name neurology_chunk1000 \
        --system-snapshot 3518fe4a... --seed 42 --candidate-count 80
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyperdb import HypergraphDB  # noqa: E402
from hyperrag.system_snapshot import compute_file_hash  # noqa: E402

# --------------------------------------------------------------------------
# 冻结常量
# --------------------------------------------------------------------------
SCRIPT_VERSION = "evidence-pool-v2"
CANDIDATE_ID_PREFIX = "ec-v2"
GRAPH_FIELD_SEP = "<SEP>"

DEFAULT_DATA_NAME = "neurology_chunk1000"
DEFAULT_SNAPSHOT_ID = (
    "3518fe4a630e4459372409b55b20f0d6d753ac6313637b4617d3855f3e0dadfa"
)
DEFAULT_SEED = 42
DEFAULT_CANDIDATE_COUNT = 80

#: 结构配额（总和必须等于 candidate_count）。顺序即候选编号顺序。
STRUCTURE_QUOTA: "collections.OrderedDict[str, int]" = collections.OrderedDict([
    ("single_fact", 20),
    ("single_high_arity", 15),
    ("multi_edge_chain", 20),
    ("multi_branch", 15),
    ("similar_subgraph_disambiguation", 10),
])

#: 各 motif 的可执行判定说明（原样写入 manifest，作为本批次的判定契约）。
MOTIF_DEFINITIONS = {
    "single_fact": (
        "single_fact 是 source-first：候选资格与排序只依赖合格 chunk，不依赖图实体"
        "抽取（避免 23.9% 的有效 chunk 因无合格实体被排除）。每个合格 chunk 都是"
        "一个候选单元；仅出自该 chunk 的单源实体（描述自足）作为可选锚点——有则附上、"
        "无也不排除该 chunk。签名只含 chunk_id。Step 2.2 提取原子事实时可补充实体。"
    ),
    "single_high_arity": (
        "仅需一条超边：arity >= 3，描述非空，实体集合完整，全部 source_id 可解析"
        "且指向合格 chunk。"
    ),
    "multi_edge_chain": (
        "至少两条超边通过共享实体连续相连（恰好共享 1 个实体，形成 A -> X -> B 的"
        "二跳路径），且两条边合计覆盖 >= 2 个不同的有效证据来源。"
    ),
    "multi_branch": (
        "同一核心实体连接 >= MIN_BRANCHES 条超边，分支之间除核心实体外实体集合"
        "两两不相交，且分支来自 >= MIN_BRANCHES 个不同证据来源（聚合 / 比较）。"
    ),
    "similar_subgraph_disambiguation": (
        "两条同 edge_type 的超边共享 >= 2 个实体、各自又有独有实体，Jaccard 落在"
        "[JACCARD_LOW, JACCARD_HIGH) 区间，且来自不同证据来源——结构相似，必须靠"
        "区分性证据才能定位目标。"
    ),
}

# --- 硬过滤阈值 ---
MIN_CHUNK_TOKENS = 200        # 低于此值视为严重截断
MIN_CHUNK_CHARS = 300         # 低于此值内容不足以支撑事实
MIN_CLEAN_CHAR_RATIO = 0.85   # 可读字符占比，低于此值视为乱码
MIN_UNIQUE_WORD_RATIO = 0.15  # 唯一词占比，低于此值视为高度重复
MIN_DESC_CHARS = 20           # 实体 / 超边描述最短长度
MIN_FACT_DESC_CHARS = 40      # single_fact 对实体描述的额外要求
MIN_HIGH_ARITY = 3            # 高元超边阈值
MIN_BRANCHES = 3              # multi_branch 的分支数下限
MAX_PAIR_DEGREE = 10          # 参与配对枚举的共享实体最大度（防组合爆炸）
MAX_PAIRS_PER_ENTITY = 3      # 每个共享实体每类最多保留的配对数
JACCARD_LOW = 0.30
JACCARD_HIGH = 0.80
DEFAULT_RESERVE_MULTIPLIER = 3  # 每种结构的备选数量 = 配额 × 该系数

_CLEAN_EXTRA_CHARS = set(
    ".,;:!?()[]{}-–—'\"/%°+=<>*&#@$~^_|\\`"
    "。，、；：！？（）【】《》“”‘’…·"
)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seeded_order_key(seed: int, signature: str) -> str:
    """以 seed 为盐的确定性伪随机排序键。

    不使用内置 ``hash()``（受 PYTHONHASHSEED 影响）或 ``random``（实现可能随
    版本变化），保证跨机器 / 跨 Python 版本得到同一顺序。
    """
    return hashlib.sha256(f"{seed}|{signature}".encode("utf-8")).hexdigest()


def split_source_ids(raw: object) -> List[str]:
    """拆分 ``source_id`` 字段（多来源以 <SEP> 连接），排序去重。"""
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(GRAPH_FIELD_SEP)]
    return sorted({p for p in parts if p})


def clean_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(
        1 for ch in text
        if ch.isalnum() or ch.isspace() or ch in _CLEAN_EXTRA_CHARS
    )
    return good / len(text)


def unique_word_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def edge_key(entities: Sequence[str]) -> str:
    """超边的稳定字符串标识：实体名排序后以 <SEP> 连接。"""
    return GRAPH_FIELD_SEP.join(sorted(entities))


def cjk_ratio(text: str) -> float:
    """中日韩表意字符占比，用于统计证据描述的语言分布。

    本语料的实体 / 超边描述由 LLM 生成，中英混杂；下一步出题需要知道每条
    候选的证据是中文还是英文，因此在报告里统计（不影响候选内容）。
    """
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    return cjk / max(1, letters)


def classify_language(texts: Sequence[str]) -> str:
    joined = " ".join(t for t in texts if t)
    r = cjk_ratio(joined)
    if r >= 0.7:
        return "zh"
    if r <= 0.05:
        return "en"
    return "mixed"


# --------------------------------------------------------------------------
# 硬过滤：chunk
# --------------------------------------------------------------------------
def filter_chunks(chunks: Dict[str, dict], rejections: collections.Counter) -> Dict[str, dict]:
    """返回通过质量过滤的 chunk 子集，并累计拒绝原因。"""
    content_first_seen: Dict[str, str] = {}
    valid: Dict[str, dict] = {}
    for cid in sorted(chunks):
        rec = chunks[cid]
        content = (rec or {}).get("content") or ""
        if not content.strip():
            rejections["chunk_empty_content"] += 1
            continue
        if int(rec.get("tokens") or 0) < MIN_CHUNK_TOKENS:
            rejections["chunk_severely_truncated"] += 1
            continue
        if len(content) < MIN_CHUNK_CHARS:
            rejections["chunk_too_short"] += 1
            continue
        if "\ufffd" in content or clean_char_ratio(content) < MIN_CLEAN_CHAR_RATIO:
            rejections["chunk_garbled"] += 1
            continue
        if unique_word_ratio(content) < MIN_UNIQUE_WORD_RATIO:
            rejections["chunk_repetitive"] += 1
            continue
        chash = sha256_text(content)
        if chash in content_first_seen:
            rejections["chunk_duplicate_content"] += 1
            continue
        content_first_seen[chash] = cid
        valid[cid] = rec
    return valid


# --------------------------------------------------------------------------
# 硬过滤：实体 / 超边
# --------------------------------------------------------------------------
def filter_entities(
    hg: HypergraphDB,
    valid_chunks: Dict[str, dict],
    all_chunk_ids: Iterable[str],
    rejections: collections.Counter,
) -> Dict[str, dict]:
    """返回 {entity_name: {"data":..., "sources": [...]}}。"""
    known = set(all_chunk_ids)
    out: Dict[str, dict] = {}
    for name in sorted(hg.all_v):
        data = hg.v(name) or {}
        desc = (data.get("description") or "").strip()
        if not str(name).strip():
            rejections["entity_empty_name"] += 1
            continue
        if len(desc) < MIN_DESC_CHARS:
            rejections["entity_description_too_short"] += 1
            continue
        sources = split_source_ids(data.get("source_id"))
        if not sources:
            rejections["entity_no_source"] += 1
            continue
        if any(s not in known for s in sources):
            rejections["entity_source_unresolvable"] += 1
            continue
        usable = [s for s in sources if s in valid_chunks]
        if len(usable) != len(sources):
            rejections["entity_source_chunk_rejected"] += 1
            continue
        out[name] = {"data": data, "sources": sources, "description": desc}
    return out


def filter_hyperedges(
    hg: HypergraphDB,
    valid_chunks: Dict[str, dict],
    all_chunk_ids: Iterable[str],
    valid_entities: Dict[str, dict],
    rejections: collections.Counter,
) -> "collections.OrderedDict[Tuple[str, ...], dict]":
    """返回 {sorted_entity_tuple: {...}}，按实体元组排序。"""
    known = set(all_chunk_ids)
    out: "collections.OrderedDict[Tuple[str, ...], dict]" = collections.OrderedDict()
    for raw in sorted(hg.all_e, key=lambda t: tuple(sorted(t))):
        ents = tuple(sorted(raw))
        data = hg.e(raw) or {}
        desc = (data.get("description") or "").strip()
        if len(ents) < 2:
            rejections["edge_arity_below_2"] += 1
            continue
        if len(desc) < MIN_DESC_CHARS:
            rejections["edge_description_too_short"] += 1
            continue
        if any(not str(e).strip() for e in ents):
            rejections["edge_empty_entity"] += 1
            continue
        if any(e not in valid_entities for e in ents):
            rejections["edge_entity_rejected"] += 1
            continue
        sources = split_source_ids(data.get("source_id"))
        if not sources:
            rejections["edge_no_source"] += 1
            continue
        if any(s not in known for s in sources):
            rejections["edge_source_unresolvable"] += 1
            continue
        if any(s not in valid_chunks for s in sources):
            rejections["edge_source_chunk_rejected"] += 1
            continue
        out[ents] = {
            "entities": ents,
            "data": data,
            "description": desc,
            "sources": sources,
            "edge_type": data.get("edge_type") or "OTHER",
            "weight": float(data.get("weight") or 0.0),
            "keywords": data.get("keywords") or "",
        }
    return out


# --------------------------------------------------------------------------
# motif 候选构造
# --------------------------------------------------------------------------
def _edge_payload(rec: dict) -> dict:
    return {
        "edge_key": edge_key(rec["entities"]),
        "entity_ids": list(rec["entities"]),
        "arity": len(rec["entities"]),
        "edge_type": rec["edge_type"],
        "weight": rec["weight"],
        "source_chunk_ids": list(rec["sources"]),
        "description_hash": sha256_text(rec["description"]),
    }


def _mk(
    structure: str,
    entry_type: str,
    seed_chunk_ids: Sequence[str],
    seed_entity_ids: Sequence[str],
    edge_recs: Sequence[dict],
    source_chunk_ids: Sequence[str],
    topology: dict,
    signature: str,
    chunk_texts: "Dict[str, str]" = None,
    chunk_tokens: "Dict[str, int]" = None,
) -> dict:
    topology = dict(topology)
    # 客观语言标注（zh / en / mixed）：基于 source chunk 原文统计（P2 修正——
    # 之前误用实体/超边的 LLM 描述，会让图抽取语言污染证据语言标注）。
    # 这不是主观难度标签。
    chunk_texts = chunk_texts or {}
    chunk_tokens = chunk_tokens or {}
    src_texts = [chunk_texts[c] for c in source_chunk_ids if c in chunk_texts]
    topology["evidence_language"] = classify_language(src_texts)
    # 客观 token 计数：源 chunk 的 tokens 字段求和。
    # 注意：该字段是 Step_1 建库时的 tiktoken 计数（非 Qwen token），仅供
    # Step 2.2 做粗略参照；精确的 evidence-span 容量判定（P4 source cap=4000
    # 指 Qwen token）须改用真实 Qwen tokenizer 计数。
    topology["source_tiktoken_tokens"] = sum(
        chunk_tokens.get(c, 0) for c in source_chunk_ids
    )
    topology["source_token_unit"] = "tiktoken"
    return {
        "intended_structure": structure,
        "entry_type": entry_type,
        "seed_chunk_ids": sorted(set(seed_chunk_ids)),
        "seed_entity_ids": sorted(set(seed_entity_ids)),
        "hyperedges": [_edge_payload(r) for r in edge_recs],
        "source_chunk_ids": sorted(set(source_chunk_ids)),
        "topology_metrics": topology,
        "signature": signature,
    }


def build_single_fact(
    valid_entities: Dict[str, dict],
    chunk_texts: Dict[str, str],
    degree: Dict[str, int],
    chunk_tokens: Dict[str, int] = None,
) -> List[dict]:
    """source-first（P1 修正 v2）：候选资格与排序只依赖合格 chunk，不依赖图实体。

    冻结设计要求"单 chunk/单实体控制题从 source 出发"——源证据是 chunk 本身。
    因此每个合格 chunk 都成为一个候选单元（无论图是否抽到实体）。仅出自该 chunk
    的单源实体（描述自足）作为**可选**锚点：有则附上、无也不排除该 chunk。
    签名只含 chunk_id，保证图抽取覆盖率不影响候选池（544/2277 不再被排除）。
    Step 2.2 提取原子事实时可再补充实体。
    """
    # chunk -> 仅出自该 chunk 的合格实体（描述自足），作为可选锚点
    chunk_to_entities: Dict[str, List[str]] = collections.defaultdict(list)
    for name in sorted(valid_entities):
        rec = valid_entities[name]
        if len(rec["sources"]) != 1:
            continue
        if len(rec["description"]) < MIN_FACT_DESC_CHARS:
            continue
        chunk_id = rec["sources"][0]
        if chunk_id not in chunk_texts:
            continue
        chunk_to_entities[chunk_id].append(name)
    out = []
    for chunk_id in sorted(chunk_texts):
        # 可选实体锚点：优先描述最长者，长度相同按名字排序（确定性）
        anchored = chunk_to_entities.get(chunk_id) or []
        anchored.sort(key=lambda n: (-len(valid_entities[n]["description"]), n))
        anchor = anchored[0] if anchored else None
        seed_entity_ids = [anchor] if anchor else []
        rec = valid_entities[anchor] if anchor else None
        out.append(_mk(
            "single_fact", "source_first",
            seed_chunk_ids=[chunk_id],
            seed_entity_ids=seed_entity_ids,
            edge_recs=[],
            source_chunk_ids=[chunk_id],
            topology={
                "n_hyperedges": 0,
                "n_entities": len(seed_entity_ids),
                "n_source_chunks": 1,
                "has_entity_anchor": anchor is not None,
                "n_anchored_entities_in_chunk": len(anchored),
                "entity_degree": degree.get(anchor, 0) if anchor else 0,
                "entity_type": (rec["data"].get("entity_type") or "UNKNOWN") if rec else "NONE",
                "description_chars": len(rec["description"]) if rec else 0,
            },
            signature=f"single_fact|{chunk_id}",
            chunk_texts=chunk_texts,
            chunk_tokens=chunk_tokens or {},
        ))
    return out


def build_single_high_arity(
    edges: "collections.OrderedDict[Tuple[str, ...], dict]",
    chunk_texts: Dict[str, str],
    chunk_tokens: Dict[str, int] = None,
) -> List[dict]:
    out = []
    for ents, rec in edges.items():
        if len(ents) < MIN_HIGH_ARITY:
            continue
        out.append(_mk(
            "single_high_arity", "graph_first",
            seed_chunk_ids=list(rec["sources"]),
            seed_entity_ids=list(ents),
            edge_recs=[rec],
            source_chunk_ids=list(rec["sources"]),
            topology={
                "n_hyperedges": 1,
                "n_entities": len(ents),
                "n_source_chunks": len(rec["sources"]),
                "arity": len(ents),
                "edge_type": rec["edge_type"],
                "weight": rec["weight"],
            },
            signature=f"single_high_arity|{edge_key(ents)}",
            chunk_texts=chunk_texts,
            chunk_tokens=chunk_tokens or {},
        ))
    return out


def build_pairwise(
    edges: "collections.OrderedDict[Tuple[str, ...], dict]",
    entity_to_edges: Dict[str, List[Tuple[str, ...]]],
    seed: int,
    chunk_texts: Dict[str, str],
    chunk_tokens: Dict[str, int] = None,
) -> Tuple[List[dict], List[dict]]:
    """一次遍历同时产出 multi_edge_chain 与 similar_subgraph_disambiguation。

    两类候选都来自"共享实体的超边二元组"，区别在共享实体个数与结构相似度：
    - 恰好共享 1 个实体 -> 二跳链（chain）
    - 共享 >= 2 个实体 + 同 edge_type + Jaccard 落区间 -> 相似子图消歧
    """
    chains: List[dict] = []
    disambig: List[dict] = []
    for hub in sorted(entity_to_edges):
        incident = entity_to_edges[hub]
        deg = len(incident)
        if deg < 2 or deg > MAX_PAIR_DEGREE:
            continue
        local_chain: List[Tuple[str, dict]] = []
        local_disa: List[Tuple[str, dict]] = []
        for a, b in combinations(incident, 2):
            ra, rb = edges[a], edges[b]
            sa, sb = set(a), set(b)
            shared = sa & sb
            src_union = sorted(set(ra["sources"]) | set(rb["sources"]))
            if len(src_union) < 2:
                continue  # 必须涉及至少两个有效证据来源
            pair_sig_body = f"{edge_key(a)}||{edge_key(b)}||{hub}"

            if len(shared) == 1:
                # A -> hub -> B 的二跳链：两端各自有独有实体
                if not (sa - shared) or not (sb - shared):
                    continue
                sig = f"multi_edge_chain|{pair_sig_body}"
                local_chain.append((seeded_order_key(seed, sig), _mk(
                    "multi_edge_chain", "graph_first",
                    seed_chunk_ids=src_union,
                    seed_entity_ids=sorted(sa | sb),
                    edge_recs=[ra, rb],
                    source_chunk_ids=src_union,
                    topology={
                        "n_hyperedges": 2,
                        "n_entities": len(sa | sb),
                        "n_source_chunks": len(src_union),
                        "shared_entity_ids": sorted(shared),
                        "shared_entity_count": 1,
                        "arity_list": [len(a), len(b)],
                        "edge_types": sorted({ra["edge_type"], rb["edge_type"]}),
                        "chain_length": 2,
                    },
                    signature=sig,
                    chunk_texts=chunk_texts,
                    chunk_tokens=chunk_tokens or {},
                )))
            elif len(shared) >= 2 and ra["edge_type"] == rb["edge_type"]:
                union = sa | sb
                jac = len(shared) / len(union)
                if not (JACCARD_LOW <= jac < JACCARD_HIGH):
                    continue
                if not (sa - shared) or not (sb - shared):
                    continue
                if set(ra["sources"]) == set(rb["sources"]):
                    continue  # 必须来自不同证据来源，才存在"需要区分"的问题
                sig = f"similar_subgraph|{pair_sig_body}"
                local_disa.append((seeded_order_key(seed, sig), _mk(
                    "similar_subgraph_disambiguation", "graph_first",
                    seed_chunk_ids=src_union,
                    seed_entity_ids=sorted(union),
                    edge_recs=[ra, rb],
                    source_chunk_ids=src_union,
                    topology={
                        "n_hyperedges": 2,
                        "n_entities": len(union),
                        "n_source_chunks": len(src_union),
                        "shared_entity_ids": sorted(shared),
                        "shared_entity_count": len(shared),
                        "distinctive_entity_ids": {
                            edge_key(a): sorted(sa - shared),
                            edge_key(b): sorted(sb - shared),
                        },
                        "jaccard": round(jac, 4),
                        "edge_type": ra["edge_type"],
                        "arity_list": [len(a), len(b)],
                    },
                    signature=sig,
                    chunk_texts=chunk_texts,
                    chunk_tokens=chunk_tokens or {},
                )))
        local_chain.sort(key=lambda kv: kv[0])
        local_disa.sort(key=lambda kv: kv[0])
        chains.extend(c for _, c in local_chain[:MAX_PAIRS_PER_ENTITY])
        disambig.extend(c for _, c in local_disa[:MAX_PAIRS_PER_ENTITY])
    return chains, disambig


def build_multi_branch(
    edges: "collections.OrderedDict[Tuple[str, ...], dict]",
    entity_to_edges: Dict[str, List[Tuple[str, ...]]],
    seed: int,
    chunk_texts: Dict[str, str],
    chunk_tokens: Dict[str, int] = None,
) -> List[dict]:
    """核心实体的多分支聚合：分支间除核心实体外互不相交，且来源各不相同。"""
    out: List[dict] = []
    for hub in sorted(entity_to_edges):
        incident = entity_to_edges[hub]
        if len(incident) < MIN_BRANCHES:
            continue
        ordered = sorted(
            incident,
            key=lambda t: seeded_order_key(seed, f"branch|{hub}|{edge_key(t)}"),
        )
        picked: List[Tuple[str, ...]] = []
        used_sources: set = set()
        for cand in ordered:
            rec = edges[cand]
            csrc = set(rec["sources"])
            if csrc & used_sources:
                continue  # 分支必须来自不同证据来源
            ok = True
            for prev in picked:
                if (set(cand) & set(prev)) != {hub}:
                    ok = False  # 分支之间除核心实体外不得共享实体
                    break
            if not ok:
                continue
            picked.append(cand)
            used_sources |= csrc
            if len(picked) == MIN_BRANCHES:
                break
        if len(picked) < MIN_BRANCHES:
            continue
        picked = sorted(picked, key=lambda t: edge_key(t))
        recs = [edges[p] for p in picked]
        src = sorted({s for r in recs for s in r["sources"]})
        ents = sorted({e for p in picked for e in p})
        sig = "multi_branch|%s|%s" % (hub, "&&".join(edge_key(p) for p in picked))
        out.append(_mk(
            "multi_branch", "graph_first",
            seed_chunk_ids=src,
            seed_entity_ids=ents,
            edge_recs=recs,
            source_chunk_ids=src,
            topology={
                "n_hyperedges": len(picked),
                "n_entities": len(ents),
                "n_source_chunks": len(src),
                "hub_entity_id": hub,
                "hub_degree": len(incident),
                "n_branches": len(picked),
                "branch_edge_keys": [edge_key(p) for p in picked],
                "arity_list": [len(p) for p in picked],
                "edge_types": sorted({r["edge_type"] for r in recs}),
            },
            signature=sig,
            chunk_texts=chunk_texts,
            chunk_tokens=chunk_tokens or {},
        ))
    return out


# --------------------------------------------------------------------------
# 选择：配额 + 证据簇去重
# --------------------------------------------------------------------------
def select_candidates(
    pool: List[dict],
    quota: int,
    reserve_quota: int,
    seed: int,
    used_chunks: set,
    used_clusters: set,
    rejections: collections.Counter,
) -> Tuple[List[dict], List[dict], int]:
    """按 seed 顺序贪心选出 quota 条主候选 + reserve_quota 条备选。

    主候选与备选彼此之间、以及与其他结构之间都不共享任何 source chunk。
    """
    ordered = sorted(pool, key=lambda c: seeded_order_key(seed, c["signature"]))
    selected: List[dict] = []
    reserve: List[dict] = []
    collisions = 0
    for cand in ordered:
        if len(selected) >= quota and len(reserve) >= reserve_quota:
            break
        chunks = set(cand["source_chunk_ids"])
        cluster = "|".join(cand["source_chunk_ids"])
        if cluster in used_clusters or (chunks & used_chunks):
            collisions += 1
            continue
        used_clusters.add(cluster)
        used_chunks |= chunks
        if len(selected) < quota:
            selected.append(cand)
        else:
            reserve.append(cand)
    if len(selected) < quota:
        rejections[f"quota_unfilled::{pool[0]['intended_structure'] if pool else 'unknown'}"] += (
            quota - len(selected)
        )
    return selected, reserve, collisions


def finalize(cand: dict, cid: str, seed: int, snapshot_id: str,
             chunk_hashes: Dict[str, str], reserve: bool) -> dict:
    """补齐输出字段，字段顺序与 Step 2.1 约定的 schema 一致。"""
    return collections.OrderedDict([
        ("candidate_id", cid),
        ("entry_type", cand["entry_type"]),
        ("intended_structure", cand["intended_structure"]),
        ("seed_chunk_ids", cand["seed_chunk_ids"]),
        ("seed_entity_ids", cand["seed_entity_ids"]),
        ("hyperedges", cand["hyperedges"]),
        ("source_chunk_ids", cand["source_chunk_ids"]),
        ("source_text_hashes", collections.OrderedDict(
            (c, chunk_hashes[c]) for c in cand["source_chunk_ids"]
        )),
        ("topology_metrics", cand["topology_metrics"]),
        ("sampling_seed", seed),
        ("system_snapshot_id", snapshot_id),
        ("status", "pending_evidence_verification"),
        ("rejection_reasons", ["not_selected_as_primary"] if reserve else []),
        ("evidence_signature", sha256_text(cand["signature"])),
        ("pool_role", "reserve" if reserve else "primary"),
    ])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2.1 构建候选证据池（不生成问题/答案）")
    ap.add_argument("--data-name", default=DEFAULT_DATA_NAME)
    ap.add_argument("--system-snapshot", default=DEFAULT_SNAPSHOT_ID)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    ap.add_argument("--reserve-multiplier", type=int, default=DEFAULT_RESERVE_MULTIPLIER)
    ap.add_argument("--output-subdir", default="question_set_v2/pilot_v1")
    args = ap.parse_args()

    if sum(STRUCTURE_QUOTA.values()) != args.candidate_count:
        print(f"[FATAL] 配额之和 {sum(STRUCTURE_QUOTA.values())} != "
              f"candidate_count {args.candidate_count}", file=sys.stderr)
        return 2

    working = REPO_ROOT / "caches" / args.data_name
    chunk_path = working / "kv_store_text_chunks.json"
    hg_path = working / "hypergraph_chunk_entity_relation.hgdb"
    snap_path = working / "snapshots" / f"{args.system_snapshot}.json"
    for p in (chunk_path, hg_path):
        if not p.exists():
            print(f"[FATAL] 输入缺失: {p}", file=sys.stderr)
            return 2
    if not snap_path.exists():
        print(f"[FATAL] 快照缺失: {snap_path}", file=sys.stderr)
        return 2

    out_dir = working / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 加载 chunk KV 与超图 ...")
    chunks: Dict[str, dict] = json.loads(chunk_path.read_text(encoding="utf-8"))
    hg = HypergraphDB()
    hg.load(str(hg_path))
    raw_counts = {
        "chunks": len(chunks),
        "vertices": hg.num_v,
        "hyperedges": hg.num_e,
    }
    print(f"      chunks={raw_counts['chunks']} vertices={raw_counts['vertices']} "
          f"hyperedges={raw_counts['hyperedges']}")

    rejections: collections.Counter = collections.Counter()

    print("[2/6] 硬过滤 chunk / 实体 / 超边 ...")
    valid_chunks = filter_chunks(chunks, rejections)
    chunk_texts = {cid: (rec or {}).get("content") or "" for cid, rec in valid_chunks.items()}
    chunk_tokens = {
        cid: int((rec or {}).get("tokens") or 0) for cid, rec in valid_chunks.items()
    }
    valid_entities = filter_entities(hg, valid_chunks, chunks.keys(), rejections)
    valid_edges = filter_hyperedges(hg, valid_chunks, chunks.keys(), valid_entities, rejections)
    filtered_counts = {
        "chunks": len(valid_chunks),
        "entities": len(valid_entities),
        "hyperedges": len(valid_edges),
    }
    print(f"      valid chunks={filtered_counts['chunks']} "
          f"entities={filtered_counts['entities']} edges={filtered_counts['hyperedges']}")

    # 实体 -> 合格超边（排序，保证确定性）
    entity_to_edges: Dict[str, List[Tuple[str, ...]]] = collections.defaultdict(list)
    for ents in valid_edges:
        for e in ents:
            entity_to_edges[e].append(ents)
    for e in entity_to_edges:
        entity_to_edges[e].sort(key=lambda t: edge_key(t))
    degree = {e: len(v) for e, v in entity_to_edges.items()}

    print("[3/6] 构造 motif 候选池 ...")
    pools: Dict[str, List[dict]] = {}
    pools["single_fact"] = build_single_fact(valid_entities, chunk_texts, degree, chunk_tokens)
    pools["single_high_arity"] = build_single_high_arity(valid_edges, chunk_texts, chunk_tokens)
    chains, disambig = build_pairwise(
        valid_edges, entity_to_edges, args.seed, chunk_texts, chunk_tokens)
    pools["multi_edge_chain"] = chains
    pools["similar_subgraph_disambiguation"] = disambig
    pools["multi_branch"] = build_multi_branch(
        valid_edges, entity_to_edges, args.seed, chunk_texts, chunk_tokens)
    for k in STRUCTURE_QUOTA:
        print(f"      pool[{k}] = {len(pools[k])}")

    print("[4/6] 配额选择 + 证据簇去重 ...")
    used_chunks: set = set()
    used_clusters: set = set()
    collision_stats: Dict[str, int] = {}
    selected_by_struct: "collections.OrderedDict[str, List[dict]]" = collections.OrderedDict()
    reserve_by_struct: "collections.OrderedDict[str, List[dict]]" = collections.OrderedDict()
    for struct, quota in STRUCTURE_QUOTA.items():
        sel, res, coll = select_candidates(
            pools[struct], quota, quota * args.reserve_multiplier,
            args.seed, used_chunks, used_clusters, rejections,
        )
        selected_by_struct[struct] = sel
        reserve_by_struct[struct] = res
        collision_stats[struct] = coll
        print(f"      {struct}: selected={len(sel)}/{quota} reserve={len(res)} "
              f"cluster_collisions={coll}")

    total_selected = sum(len(v) for v in selected_by_struct.values())
    if total_selected != args.candidate_count:
        print(f"[FATAL] 主候选数量 {total_selected} != {args.candidate_count}，"
              f"配额未满，拒绝写出残缺产物。", file=sys.stderr)
        return 3

    print("[5/6] 写出产物 ...")
    chunk_hashes = {
        cid: sha256_text(valid_chunks[cid]["content"]) for cid in used_chunks
    }
    primary_records: List[dict] = []
    idx = 0
    for struct in STRUCTURE_QUOTA:
        for cand in selected_by_struct[struct]:
            idx += 1
            primary_records.append(finalize(
                cand, f"{CANDIDATE_ID_PREFIX}-{idx:04d}", args.seed,
                args.system_snapshot, chunk_hashes, reserve=False))
    reserve_records: List[dict] = []
    ridx = 0
    for struct in STRUCTURE_QUOTA:
        for cand in reserve_by_struct[struct]:
            ridx += 1
            reserve_records.append(finalize(
                cand, f"{CANDIDATE_ID_PREFIX}-r{ridx:04d}", args.seed,
                args.system_snapshot, chunk_hashes, reserve=True))

    def write_jsonl(path: Path, records: List[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cand_file = out_dir / "evidence_candidates.jsonl"
    reserve_file = out_dir / "evidence_candidates_reserve.jsonl"
    write_jsonl(cand_file, primary_records)
    write_jsonl(reserve_file, reserve_records)

    manifest = collections.OrderedDict([
        ("script_version", SCRIPT_VERSION),
        ("stage", "step2.1_evidence_pool"),
        ("data_name", args.data_name),
        ("system_snapshot_id", args.system_snapshot),
        ("sampling_seed", args.seed),
        ("candidate_count", args.candidate_count),
        ("structure_quota", dict(STRUCTURE_QUOTA)),
        ("reserve_multiplier", args.reserve_multiplier),
        ("candidate_id_prefix", CANDIDATE_ID_PREFIX),
        # P2：源 token 字段是 tiktoken 计数（建库时保存），与 P4 source cap
        # 的 Qwen token 不是同一单位；精确容量判定须改用真实 Qwen tokenizer。
        ("source_token_unit", "tiktoken"),
        ("status_of_all_candidates", "pending_evidence_verification"),
        ("motif_definitions", MOTIF_DEFINITIONS),
        ("hard_filter_thresholds", collections.OrderedDict([
            ("MIN_CHUNK_TOKENS", MIN_CHUNK_TOKENS),
            ("MIN_CHUNK_CHARS", MIN_CHUNK_CHARS),
            ("MIN_CLEAN_CHAR_RATIO", MIN_CLEAN_CHAR_RATIO),
            ("MIN_UNIQUE_WORD_RATIO", MIN_UNIQUE_WORD_RATIO),
            ("MIN_DESC_CHARS", MIN_DESC_CHARS),
            ("MIN_FACT_DESC_CHARS", MIN_FACT_DESC_CHARS),
            ("MIN_HIGH_ARITY", MIN_HIGH_ARITY),
            ("MIN_BRANCHES", MIN_BRANCHES),
            ("MAX_PAIR_DEGREE", MAX_PAIR_DEGREE),
            ("MAX_PAIRS_PER_ENTITY", MAX_PAIRS_PER_ENTITY),
            ("JACCARD_LOW", JACCARD_LOW),
            ("JACCARD_HIGH", JACCARD_HIGH),
        ])),
        ("input_files", collections.OrderedDict([
            ("kv_store_text_chunks.json", compute_file_hash(chunk_path)),
            ("hypergraph_chunk_entity_relation.hgdb", compute_file_hash(hg_path)),
            ("snapshot_json", compute_file_hash(snap_path)),
        ])),
        ("output_files", collections.OrderedDict([
            ("evidence_candidates.jsonl", compute_file_hash(cand_file)),
            ("evidence_candidates_reserve.jsonl", compute_file_hash(reserve_file)),
        ])),
        ("determinism", collections.OrderedDict([
            ("order_key", "sha256(f'{seed}|{signature}')"),
            ("set_iteration", "all sets sorted before use"),
            ("timestamp_in_outputs", False),
        ])),
        ("guarantees", [
            "no question text generated",
            "no gold answer generated",
            "no subjective difficulty label",
            "structure labels are intended_structure only (unverified)",
        ]),
    ])
    (out_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = collections.OrderedDict([
        ("script_version", SCRIPT_VERSION),
        ("data_name", args.data_name),
        ("sampling_seed", args.seed),
        ("raw_counts", raw_counts),
        ("filtered_counts", filtered_counts),
        ("filter_pass_rate", collections.OrderedDict([
            ("chunks", round(filtered_counts["chunks"] / max(1, raw_counts["chunks"]), 4)),
            ("entities", round(filtered_counts["entities"] / max(1, raw_counts["vertices"]), 4)),
            ("hyperedges", round(filtered_counts["hyperedges"] / max(1, raw_counts["hyperedges"]), 4)),
        ])),
        ("motif_pool_sizes", collections.OrderedDict(
            (k, len(pools[k])) for k in STRUCTURE_QUOTA)),
        ("selected_per_structure", collections.OrderedDict(
            (k, len(selected_by_struct[k])) for k in STRUCTURE_QUOTA)),
        ("reserve_per_structure", collections.OrderedDict(
            (k, len(reserve_by_struct[k])) for k in STRUCTURE_QUOTA)),
        ("evidence_cluster_collisions_per_structure", collision_stats),
        ("evidence_language_distribution", collections.OrderedDict(
            sorted(collections.Counter(
                r["topology_metrics"]["evidence_language"] for r in primary_records
            ).items()))),
        ("evidence_language_by_structure", collections.OrderedDict(
            (k, collections.OrderedDict(sorted(collections.Counter(
                c["topology_metrics"]["evidence_language"]
                for c in selected_by_struct[k]).items())))
            for k in STRUCTURE_QUOTA)),
        ("rejection_reason_distribution", collections.OrderedDict(
            sorted(rejections.items()))),
        ("total_rejections", sum(rejections.values())),
        ("distinct_source_chunks_used", len(used_chunks)),
        ("primary_candidate_count", len(primary_records)),
        ("reserve_candidate_count", len(reserve_records)),
    ])
    (out_dir / "sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[6/6] 完成。输出目录:", out_dir)
    print(f"      primary={len(primary_records)} reserve={len(reserve_records)} "
          f"distinct_source_chunks={len(used_chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
