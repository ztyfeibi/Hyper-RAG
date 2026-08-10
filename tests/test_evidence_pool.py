"""Step 2.1 证据池的验收回归测试。

两部分：
1. 纯函数单测（始终运行）——确定性排序键、来源解析、质量判定、语言分类。
2. 产物验收（产物缺失时 skip）——配额、ID 唯一性、来源可回溯、无问题/答案泄漏。

产物位于 caches/（已 gitignore），CI 上不存在时自动跳过，不会误报。
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_pilot_evidence_pool import (  # noqa: E402
    STRUCTURE_QUOTA,
    classify_language,
    clean_char_ratio,
    edge_key,
    seeded_order_key,
    sha256_text,
    split_source_ids,
    unique_word_ratio,
)

DATA_NAME = "neurology_chunk1000"
POOL_DIR = REPO_ROOT / "caches" / DATA_NAME / "question_set_v2" / "pilot_v1"
CANDIDATES = POOL_DIR / "evidence_candidates.jsonl"
RESERVE = POOL_DIR / "evidence_candidates_reserve.jsonl"
MANIFEST = POOL_DIR / "sampling_manifest.json"
REPORT = POOL_DIR / "sampling_report.json"

FORBIDDEN_FIELDS = {
    "question", "question_text", "answer", "gold_answer", "gold",
    "difficulty", "difficulty_label", "verified_structure",
}


# ---------------------------------------------------------------------------
# 1. 纯函数：确定性与判定逻辑
# ---------------------------------------------------------------------------
def test_seeded_order_key_is_stable_and_seed_sensitive():
    """排序键必须只依赖 (seed, signature)，且换 seed 就换顺序。"""
    a = seeded_order_key(42, "sig-x")
    assert a == seeded_order_key(42, "sig-x")          # 同输入恒定
    assert a != seeded_order_key(43, "sig-x")          # 换 seed 变化
    assert a != seeded_order_key(42, "sig-y")          # 换签名变化
    assert len(a) == 64


def test_seeded_order_does_not_use_builtin_hash():
    """排序键必须是 sha256，而非受 PYTHONHASHSEED 影响的内置 hash。"""
    assert seeded_order_key(42, "abc") == hashlib.sha256(
        b"42|abc").hexdigest()


def test_edge_key_is_order_independent():
    assert edge_key(["B", "A", "C"]) == edge_key(["C", "A", "B"])


def test_split_source_ids_dedups_and_sorts():
    raw = "chunk-b<SEP>chunk-a<SEP>chunk-b<SEP>"
    assert split_source_ids(raw) == ["chunk-a", "chunk-b"]
    assert split_source_ids("") == []
    assert split_source_ids(None) == []


def test_quality_heuristics():
    assert clean_char_ratio("Normal English text, with punctuation.") > 0.95
    assert clean_char_ratio("\ufffd\ufffd\ufffd\ufffd\ufffd") < 0.85
    assert unique_word_ratio("a b c d e") == 1.0
    assert unique_word_ratio("x " * 50) < 0.15


def test_classify_language():
    assert classify_language(["Corticosteroids can induce psychoses."]) == "en"
    assert classify_language(["皮质类固醇在特定药物组合下可导致长期瘫痪。"]) == "zh"
    assert classify_language(
        ["皮质类固醇导致 prolonged paralysis 的机制 involves multiple factors here"]
    ) == "mixed"


def test_quota_sums_to_eighty():
    assert sum(STRUCTURE_QUOTA.values()) == 80
    assert list(STRUCTURE_QUOTA) == [
        "single_fact", "single_high_arity", "multi_edge_chain",
        "multi_branch", "similar_subgraph_disambiguation",
    ]


# ---------------------------------------------------------------------------
# 2. 产物验收
# ---------------------------------------------------------------------------
pytestmark_artifacts = pytest.mark.skipif(
    not CANDIDATES.exists(), reason="证据池产物不存在（caches/ 未生成）"
)


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def primary():
    if not CANDIDATES.exists():
        pytest.skip("证据池产物不存在")
    return _load_jsonl(CANDIDATES)


@pytest.fixture(scope="module")
def chunks():
    p = REPO_ROOT / "caches" / DATA_NAME / "kv_store_text_chunks.json"
    if not p.exists():
        pytest.skip("chunk KV 不存在")
    return json.loads(p.read_text(encoding="utf-8"))


@pytestmark_artifacts
def test_exactly_eighty_candidates(primary):
    assert len(primary) == 80


@pytestmark_artifacts
def test_structure_quota_is_exact(primary):
    got = Counter(r["intended_structure"] for r in primary)
    assert dict(got) == dict(STRUCTURE_QUOTA)


@pytestmark_artifacts
def test_candidate_ids_unique_and_disjoint_from_reserve(primary):
    ids = [r["candidate_id"] for r in primary]
    assert len(set(ids)) == 80
    if RESERVE.exists():
        reserve_ids = {r["candidate_id"] for r in _load_jsonl(RESERVE)}
        assert not (set(ids) & reserve_ids)


@pytestmark_artifacts
def test_every_source_chunk_resolves_and_hash_matches(primary, chunks):
    for rec in primary:
        assert rec["source_chunk_ids"], rec["candidate_id"]
        for cid in rec["source_chunk_ids"]:
            assert cid in chunks, f"{rec['candidate_id']} 引用了不存在的 chunk {cid}"
            content = chunks[cid]["content"]
            assert rec["source_text_hashes"][cid] == sha256_text(content)


@pytestmark_artifacts
def test_hyperedges_are_traceable(primary):
    """超边必须能用排序后的实体元组回查到原超图。"""
    from hyperdb import HypergraphDB
    hg_path = REPO_ROOT / "caches" / DATA_NAME / "hypergraph_chunk_entity_relation.hgdb"
    if not hg_path.exists():
        pytest.skip("超图文件不存在")
    hg = HypergraphDB()
    hg.load(str(hg_path))
    for rec in primary:
        for he in rec["hyperedges"]:
            assert hg.has_e(tuple(sorted(he["entity_ids"]))), \
                f"{rec['candidate_id']} 的超边无法回查: {he['edge_key']}"
            assert he["arity"] == len(he["entity_ids"])


@pytestmark_artifacts
def test_no_question_answer_or_difficulty_leaked(primary):
    for rec in primary:
        leaked = set(rec) & FORBIDDEN_FIELDS
        assert not leaked, f"{rec['candidate_id']} 泄漏了非法字段 {leaked}"
        assert rec["status"] == "pending_evidence_verification"
        assert "intended_structure" in rec


@pytestmark_artifacts
def test_evidence_clusters_do_not_overlap(primary):
    """80 条主候选的证据 chunk 互不重叠，避免同段原文重复出题。"""
    used = [c for rec in primary for c in rec["source_chunk_ids"]]
    assert len(used) == len(set(used))


@pytestmark_artifacts
def test_motif_structural_invariants(primary):
    """每种 intended_structure 的拓扑必须满足其判定定义。"""
    for rec in primary:
        s = rec["intended_structure"]
        tm = rec["topology_metrics"]
        edges = rec["hyperedges"]
        if s == "single_fact":
            assert len(edges) == 0
            assert len(rec["source_chunk_ids"]) == 1
            # 实体锚点可选：0（无合格单源实体）或 1（有锚点），都不强制
            assert len(rec["seed_entity_ids"]) in (0, 1)
        elif s == "single_high_arity":
            assert len(edges) == 1
            assert edges[0]["arity"] >= 3
        elif s == "multi_edge_chain":
            assert len(edges) == 2
            assert tm["shared_entity_count"] == 1
            assert tm["n_source_chunks"] >= 2
        elif s == "multi_branch":
            assert len(edges) >= 3
            assert tm["n_source_chunks"] >= 3
            assert tm["hub_entity_id"] in rec["seed_entity_ids"]
        elif s == "similar_subgraph_disambiguation":
            assert len(edges) == 2
            assert tm["shared_entity_count"] >= 2
            assert 0.30 <= tm["jaccard"] < 0.80
            assert edges[0]["edge_type"] == edges[1]["edge_type"]
        else:
            pytest.fail(f"未知结构 {s}")


@pytestmark_artifacts
def test_single_fact_is_source_first(primary):
    """P1 回归 v2：single_fact 资格/排序只依赖 source chunk，实体锚点可选。

    核心契约：每个合格 chunk 都应成为候选（池规模 == 合格 chunk 数），
    不能因图抽取覆盖率而被排除。锚点若存在，必须是该 chunk 的单源实体。
    """
    from hyperdb import HypergraphDB
    hg_path = REPO_ROOT / "caches" / DATA_NAME / "hypergraph_chunk_entity_relation.hgdb"
    if not hg_path.exists():
        pytest.skip("超图文件不存在")
    hg = HypergraphDB()
    hg.load(str(hg_path))

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    # 关键不变量：single_fact 池规模必须等于合格 chunk 数
    # （证明图实体抽取不再决定候选资格）。
    assert report["motif_pool_sizes"]["single_fact"] == report["filtered_counts"]["chunks"], \
        "single_fact 池规模应等于合格 chunk 数（P1: 资格只依赖 chunk）"

    seen = 0
    for rec in primary:
        if rec["intended_structure"] != "single_fact":
            continue
        seen += 1
        assert rec["entry_type"] == "source_first"
        assert len(rec["source_chunk_ids"]) == 1
        cid = rec["source_chunk_ids"][0]
        anchors = rec["seed_entity_ids"]
        # 锚点可选：无锚点（空）也合法；有则必须是该 chunk 的单源实体
        if anchors:
            name = anchors[0]
            data = hg.v(name) or {}
            sources = split_source_ids(data.get("source_id"))
            assert sources == [cid], (
                f"{rec['candidate_id']} 的实体 {name} 并非单源锚定到 chunk {cid}，"
                f"而是 {sources}"
            )
    assert seen == STRUCTURE_QUOTA["single_fact"]


@pytestmark_artifacts
def test_source_token_unit_is_tiktoken(primary):
    """P2 回归 v2：source token 字段必须明确标记为 tiktoken，避免与 P4 的 Qwen token 混淆。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_token_unit"] == "tiktoken"
    for rec in primary:
        tm = rec["topology_metrics"]
        assert "source_tiktoken_tokens" in tm, rec["candidate_id"]
        assert tm["source_token_unit"] == "tiktoken"
        val = tm["source_tiktoken_tokens"]
        assert isinstance(val, int) and val >= 0


@pytestmark_artifacts
def test_evidence_language_based_on_source_text(primary, chunks):
    """P2 回归：evidence_language 必须基于 source chunk 原文，而非 LLM 描述。"""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counter = Counter()
    for rec in primary:
        texts = [chunks[c]["content"] for c in rec["source_chunk_ids"] if c in chunks]
        counter[classify_language(texts)] += 1
    assert dict(report["evidence_language_distribution"]) == dict(counter), \
        "evidence_language 未基于源 chunk 原文统计（可能仍混用了 LLM 描述）"


@pytestmark_artifacts
def test_manifest_and_report_are_complete():
    for p in (MANIFEST, REPORT):
        assert p.exists(), p
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sampling_seed"] == 42
    assert manifest["candidate_count"] == 80
    assert manifest["determinism"]["timestamp_in_outputs"] is False
    assert set(manifest["motif_definitions"]) == set(STRUCTURE_QUOTA)
    for key in ("kv_store_text_chunks.json",
                "hypergraph_chunk_entity_relation.hgdb"):
        assert len(manifest["input_files"][key]) == 64

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for key in ("raw_counts", "filtered_counts", "motif_pool_sizes",
                "selected_per_structure", "rejection_reason_distribution"):
        assert key in report, key
    assert report["primary_candidate_count"] == 80
    assert dict(report["selected_per_structure"]) == dict(STRUCTURE_QUOTA)


@pytestmark_artifacts
def test_outputs_contain_no_timestamp():
    """输出不得含时间戳，否则同 seed 重跑无法得到相同文件哈希。"""
    for p in (MANIFEST, REPORT):
        text = p.read_text(encoding="utf-8")
        assert "generated_at" not in text
        assert "timestamp" not in text.replace("timestamp_in_outputs", "")
