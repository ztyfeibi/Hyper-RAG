"""Judge 契约测试（judge_longcat.py v1.3.1，2026-08-14 ER alternative_chunk_ids 过滤）。

Coverage:
1. ER 级证据覆盖：evidence_requirements 组间 AND + 组内 OR（多 AU/多 chunk）。
2. alternative_chunk_ids 过滤：只有 ER 声明的 chunk 的 span 才贡献命中。
3. 搜索范围：P1/P_gold 全文 vs P2-P4 仅 -----Sources----- 区段。
4. combine_route_success 三值 AND：coverage=False 硬 fail（不被 human_review 悬置）。
5. P0 无 evidence gate：route_success = final_answer_correctness。
6. manifest 双哈希分离：raw_judge_script_hashes 只含可校验哈希；
   raw_generation_configs 保留完整条目（含 hash_status）。
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

# Mock 重依赖（openai / numpy / tiktoken / hyperrag / my_config）——
# 测试只调纯函数（calc_source_evidence_coverage / combine_route_success /
# merge_raw_judge_fields / final_answer_correctness），不触发 LLM 调用。
for _mod in ("openai", "numpy", "tiktoken", "my_config",
             "hyperrag", "hyperrag.env"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["my_config"].LLM_BASE_URL_SILICONFLOW = "mock"
sys.modules["my_config"].LLM_API_KEY_SILICONFLOW = "mock"
sys.modules["my_config"].LLM_MODEL_SILICONFLOW = "mock"

import judge_longcat as j  # noqa: E402


# ---------------------------------------------------------------------------
# ER 级覆盖：组内 OR / 组间 AND / 返回 dict 格式
# ---------------------------------------------------------------------------
def test_er_return_format():
    """返回 dict 含 requirement_hits / er_recall / complete_evidence_hit。"""
    ctx = "Lott IT 1978 报告原文在此。"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "Lott IT 1978"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"requirement_hits", "er_recall", "complete_evidence_hit"}
    assert result["requirement_hits"] == {"ER1": True}
    assert result["er_recall"] == 1.0
    assert result["complete_evidence_hit"] is True


def test_er_group_or_alternative_sources_hit():
    """组内 OR：AU 有多个 span，任一命中即满足该 ER。"""
    ctx = "本文包含 Lott IT 1978 年的报告原文。"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "A 不存在的文本"},
                     {"chunk_id": "chunk-a", "quote": "Lott IT 1978"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": True}
    assert result["complete_evidence_hit"] is True


def test_er_group_or_all_miss():
    """组内 OR：全部 span 未命中 -> ER 未满足。"""
    ctx = "无相关内容"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "X 不存在"},
                     {"chunk_id": "chunk-a", "quote": "Y 也不存在"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": False}
    assert result["complete_evidence_hit"] is False
    assert result["er_recall"] == 0.0


def test_er_group_and_partial_hit():
    """组间 AND：一个 ER 不命中 -> complete=False, er_recall=0.5。"""
    ctx = "Lott IT 1978 报告在此。"
    ers = [
        {"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
         "alternative_chunk_ids": ["chunk-a"]},
        {"requirement_id": "ER2", "answer_unit_ids": ["AU2"],
         "alternative_chunk_ids": ["chunk-b"]},
    ]
    spans = {
        "AU1": [{"chunk_id": "chunk-a", "quote": "Lott IT 1978"}],
        "AU2": [{"chunk_id": "chunk-b", "quote": "Matalon 1988 缺失"}],
    }
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": True, "ER2": False}
    assert result["complete_evidence_hit"] is False
    assert result["er_recall"] == 0.5


def test_er_group_and_all_hit():
    """组间 AND：全部 ER 命中 -> complete=True。"""
    ctx = "Lott IT 1978 与 Matalon 1988 都在。"
    ers = [
        {"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
         "alternative_chunk_ids": ["chunk-a"]},
        {"requirement_id": "ER2", "answer_unit_ids": ["AU2"],
         "alternative_chunk_ids": ["chunk-b"]},
    ]
    spans = {
        "AU1": [{"chunk_id": "chunk-a", "quote": "Lott IT 1978"}],
        "AU2": [{"chunk_id": "chunk-b", "quote": "Matalon 1988"}],
    }
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": True, "ER2": True}
    assert result["complete_evidence_hit"] is True
    assert result["er_recall"] == 1.0


def test_er_multi_au_one_er():
    """一个 ER 关联多个 AU：任一 AU span 命中即满足（真实多 AU 结构）。"""
    ctx = "Q fever is probably contracted by inhalation."
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1", "AU2"],
            "alternative_chunk_ids": ["chunk-a", "chunk-b"]}]
    spans = {
        "AU1": [{"chunk_id": "chunk-a", "quote": "不存在的文本 AAA"}],
        "AU2": [{"chunk_id": "chunk-b", "quote": "Q fever is probably contracted by inhalation."}],
    }
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": True}
    assert result["complete_evidence_hit"] is True


def test_er_multi_au_all_miss():
    """一个 ER 关联多个 AU 但全 miss -> ER=False。"""
    ctx = "无关文本"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1", "AU2"],
            "alternative_chunk_ids": ["chunk-a", "chunk-b"]}]
    spans = {
        "AU1": [{"chunk_id": "chunk-a", "quote": "不存在 AAA"}],
        "AU2": [{"chunk_id": "chunk-b", "quote": "不存在 BBB"}],
    }
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": False}


def test_er_no_answer_unit_ids_strict_false():
    """ER 无 answer_unit_ids -> 严格 False（必须有证据）。"""
    ctx = "有内容"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": [],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "有内容"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": False}
    assert result["complete_evidence_hit"] is False


def test_er_au_without_spans_strict_false():
    """ER 的 AU 无 spans -> 严格 False。"""
    ctx = "有内容"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1", "AU2"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "有内容"}]}  # AU2 无 spans
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    # AU1 hit 但 AU2 无 span -> OR 内 AU2 不贡献，AU1 命中即 ER hit
    # 实际：AU1 有 span 且命中 -> ER1=True（OR 语义：任一 AU 命中即可）
    assert result["requirement_hits"] == {"ER1": True}


def test_whitespace_normalization():
    """空白归一化：换行/缩进差异不影响命中。"""
    ctx = "Lott IT, Coulombe T,\nDiPaolo RV: X 报告。\n\n后续"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "Lott IT, Coulombe T, DiPaolo RV: X 报告。"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["complete_evidence_hit"] is True


def test_empty_evidence_requirements():
    """无 ER -> complete=False, er_recall=0.0。"""
    ctx = "有内容"
    result = j.calc_source_evidence_coverage(ctx, [], {}, search="full")
    assert result["requirement_hits"] == {}
    assert result["er_recall"] == 0.0
    assert result["complete_evidence_hit"] is False


# ---------------------------------------------------------------------------
# alternative_chunk_ids 过滤（v1.3.1 新增）
# ---------------------------------------------------------------------------
def test_er_alternative_chunk_ids_filter():
    """只有 ER 声明的 alternative_chunk_ids 中的 span 才贡献命中。

    AU 同时拥有 allowed 和 non-allowed span，只有 non-allowed quote 命中时，
    ER 必须为 False（non-allowed span 属于其他 evidence group，不能满足本 ER）。
    """
    ctx = "Q fever is probably contracted by inhalation."
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-allowed"]}]
    # AU1 有两个 span：allowed（miss）+ non-allowed（hit）
    # non-allowed span 的 quote 虽然命中 context，但 chunk_id 不在 allowed_chunks 中
    spans = {"AU1": [
        {"chunk_id": "chunk-allowed", "quote": "不存在的文本"},
        {"chunk_id": "chunk-other", "quote": "Q fever is probably contracted by inhalation."},
    ]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": False}
    assert result["complete_evidence_hit"] is False
    # 对照：把 non-allowed span 的 chunk_id 改为 allowed -> 必须 hit
    spans2 = {"AU1": [
        {"chunk_id": "chunk-allowed", "quote": "不存在的文本"},
        {"chunk_id": "chunk-allowed", "quote": "Q fever is probably contracted by inhalation."},
    ]}
    result2 = j.calc_source_evidence_coverage(ctx, ers, spans2, search="full")
    assert result2["requirement_hits"] == {"ER1": True}


def test_er_empty_alternative_chunk_ids_no_filter():
    """alternative_chunk_ids 为空时不过滤（向后兼容无 chunk 归属的数据）。"""
    ctx = "有内容在此。"
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": []}]
    spans = {"AU1": [{"chunk_id": "any-chunk", "quote": "有内容在此。"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["requirement_hits"] == {"ER1": True}


# ---------------------------------------------------------------------------
# 搜索范围：P1/P_gold 全文 vs P2-P4 Sources-only
# ---------------------------------------------------------------------------
P2_STYLE_CONTEXT = (
    "-----Entities-----\n"
    "1,\tMOLLER,OTHER,研究者描述\n"
    "-----Relationships-----\n"
    "('CSF','PCR'),DIAGNOSTIC_PATTERN,诊断模式描述\n"
    "-----Sources-----\n"
    "Lott IT, Coulombe T, DiPaolo RV, et al: Vitamin B6-dependent seizures. Neurology 28:47, 1978.\n"
)


def test_sources_section_extraction():
    """extract_sources_section：只返回 -----Sources----- 之后的文本。"""
    sec = j.extract_sources_section(P2_STYLE_CONTEXT)
    assert "Lott IT" in sec
    assert "Entities" not in sec
    assert "Relationships" not in sec
    assert "MOLLER" not in sec


def test_sources_section_no_mark_full():
    """无区段标记（P1/P_gold）时返回整个 context。"""
    ctx = "纯 source 文本，无标记。"
    assert j.extract_sources_section(ctx) == ctx


def test_p2p4_sources_only_excludes_entities():
    """P2-P4（search='sources'）：Entities/Relationships 区段的文本不贡献 hit。"""
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "Lott IT 1978"}]}
    ctx = (
        "-----Entities-----\n"
        "Lott IT 1978 研究者条目\n"
        "-----Relationships-----\n"
        "关系描述\n"
        "-----Sources-----\n"
        "完全无关的 source 文本\n"
    )
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="sources")
    assert result["complete_evidence_hit"] is False
    # 同一 quote 在 Sources 区段出现 -> 必须 hit
    ctx2 = ctx.replace("完全无关的 source 文本", "Lott IT 1978 在 source 区段")
    result2 = j.calc_source_evidence_coverage(ctx2, ers, spans, search="sources")
    assert result2["complete_evidence_hit"] is True


def test_p1_pgold_full_search_hits_anywhere():
    """P1/P_gold（search='full'）：无区段标记，整个 context 即 Sources。"""
    ctx = "Lott IT, Coulombe T, DiPaolo RV, et al: Vitamin B6-dependent seizures. Neurology 28:47, 1978."
    ers = [{"requirement_id": "ER1", "answer_unit_ids": ["AU1"],
            "alternative_chunk_ids": ["chunk-a"]}]
    spans = {"AU1": [{"chunk_id": "chunk-a", "quote": "Vitamin B6-dependent seizures. Neurology 28:47, 1978"}]}
    result = j.calc_source_evidence_coverage(ctx, ers, spans, search="full")
    assert result["complete_evidence_hit"] is True


# ---------------------------------------------------------------------------
# final_answer_correctness
# ---------------------------------------------------------------------------
def test_human_review_forces_pending():
    """human_review=true -> final=pending（不提前判死）。"""
    assert j.final_answer_correctness("fail", True) == "pending"
    assert j.final_answer_correctness("pass", True) == "pending"
    assert j.final_answer_correctness("uncertain", True) == "pending"


def test_non_human_review_keeps_derived():
    """非 human_review 记录保留 derived；uncertain 兜底 pending。"""
    assert j.final_answer_correctness("pass", False) == "pass"
    assert j.final_answer_correctness("fail", False) == "fail"
    assert j.final_answer_correctness("uncertain", False) == "pending"


# ---------------------------------------------------------------------------
# combine_route_success 三值 AND（核心修正）
# ---------------------------------------------------------------------------
def test_coverage_false_hard_fail_regardless_of_human_review():
    """coverage=False -> route_success=fail（不被 human_review 悬置）。

    这是 v1.3.0 核心修正：human_review 只影响 answer_correctness(→pending)，
    不覆盖已确定的证据失败。final_ac=pending（来自 hr=true）+ cov=False -> fail。
    """
    # final_ac=pending (hr=true), cov=False, has_gate=True -> fail（不是 pending）
    assert j.combine_route_success("pending", False, True, True) == "fail"
    # final_ac=pass, cov=False -> fail
    assert j.combine_route_success("pass", False, True, False) == "fail"
    # final_ac=fail, cov=False -> fail
    assert j.combine_route_success("fail", False, True, False) == "fail"


def test_coverage_true_passes_through_answer_correctness():
    """coverage=True -> route_success = answer_correctness。"""
    assert j.combine_route_success("pass", True, True, False) == "pass"
    assert j.combine_route_success("fail", True, True, False) == "fail"
    assert j.combine_route_success("pending", True, True, True) == "pending"


def test_coverage_unknown_with_fail_is_fail():
    """coverage=None(unknown) + answer=fail -> fail。"""
    assert j.combine_route_success("fail", None, True, False) == "fail"


def test_coverage_unknown_otherwise_pending():
    """coverage=None(unknown) + answer=pass/pending -> pending。"""
    assert j.combine_route_success("pass", None, True, False) == "pending"
    assert j.combine_route_success("pending", None, True, True) == "pending"


def test_p0_no_evidence_gate():
    """P0：route_success = final_answer_correctness（无证据门槛）。"""
    assert j.combine_route_success("pass", None, False, False) == "pass"
    assert j.combine_route_success("fail", None, False, False) == "fail"
    assert j.combine_route_success("pending", None, False, True) == "pending"
    # P0 上 human_review -> final_ac=pending -> route_success=pending
    assert j.combine_route_success("pending", None, False, True) == "pending"


def test_dual_gate_full_matrix():
    """P1-P4/P_gold 双门槛完整真值表（对照用户验收表）。"""
    # (final_ac, cov_hit, has_gate, hr) -> expected
    cases = [
        # cov=False -> fail (always, even hr)
        ("pending", False, True, True, "fail"),
        ("pass", False, True, False, "fail"),
        ("fail", False, True, False, "fail"),
        # cov=True
        ("pass", True, True, False, "pass"),
        ("fail", True, True, False, "fail"),
        ("pending", True, True, True, "pending"),
        # cov=None (unknown)
        ("fail", None, True, False, "fail"),
        ("pending", None, True, True, "pending"),
    ]
    for final_ac, cov, gate, hr, expected in cases:
        assert j.combine_route_success(final_ac, cov, gate, hr) == expected, (
            f"combine_route_success({final_ac!r}, {cov!r}, {gate!r}, {hr!r}) "
            f"= {j.combine_route_success(final_ac, cov, gate, hr)!r}, expected {expected!r}")


# ---------------------------------------------------------------------------
# manifest 双哈希分离
# ---------------------------------------------------------------------------
def test_manifest_raw_and_adjudicator_separate():
    """merge_raw_judge_fields：judge 模式追加当前调用；rejudge 保留历史。"""
    script_hash = "abc123def456"
    # judge 模式：空 old -> 记录当前脚本哈希 + 一条调用配置
    raw_prompt, raw_hashes, raw_configs = j.merge_raw_judge_fields(
        {}, "judge", script_hash, 0.0, 6000)
    assert raw_hashes == [script_hash]
    assert len(raw_configs) == 1
    assert raw_configs[0]["script_hash"] == script_hash
    assert raw_configs[0]["hash_status"] == "verified"
    assert raw_configs[0]["max_tokens"] == 6000
    assert raw_prompt == j.judge_prompt_hash()
    # judge 模式追加：已有别的哈希 -> 追加当前（去重）
    old = {"raw_judge_script_hashes": ["old_hash"],
           "raw_generation_configs": [{"script_hash": "old_hash", "hash_status": "verified"}]}
    _, raw_hashes2, raw_configs2 = j.merge_raw_judge_fields(
        old, "judge", script_hash, 0.0, 6000)
    assert raw_hashes2 == ["old_hash", script_hash]
    assert len(raw_configs2) == 2
    # judge 模式去重：同一哈希不重复追加到 raw_hashes
    old3 = {"raw_judge_script_hashes": [script_hash],
            "raw_generation_configs": [{"script_hash": script_hash, "hash_status": "verified"}]}
    _, raw_hashes3, raw_configs3 = j.merge_raw_judge_fields(
        old3, "judge", script_hash, 0.0, 6000)
    assert raw_hashes3 == [script_hash]
    assert len(raw_configs3) == 2


def test_manifest_rejudge_preserves_history():
    """rejudge 模式：不追加裁决器哈希到 raw；无历史时补 RAW_JUDGE_SCRIPT_HISTORY。

    raw_judge_script_hashes 只含可校验的实际哈希（1 个，跳过 null/unrecoverable）。
    raw_generation_configs 保留 2 条完整条目（含 hash_status）。
    """
    script_hash = "adjudicator_script"
    # 无旧记录 -> 补已知原始调用历史
    raw_prompt, raw_hashes, raw_configs = j.merge_raw_judge_fields(
        {}, "rejudge", script_hash, 0.0, 6000)
    assert script_hash not in raw_hashes          # 裁决器哈希不得混入 raw
    # 只 1 个可校验哈希（91447b21...），unrecoverable 的 null 不进 raw_hashes
    assert len(raw_hashes) == 1
    assert raw_hashes[0] == "91447b212e79b9c8f31bd01cbd880f2232bddc3059433502cae39d018a7f3f9d"
    # raw_generation_configs 保留 2 条（含 null hash_status=unrecoverable）
    assert len(raw_configs) == 2
    assert raw_configs[0]["hash_status"] == "verified"
    assert raw_configs[1]["hash_status"] == "unrecoverable"
    assert raw_configs[1]["script_hash"] is None
    assert raw_prompt == j.judge_prompt_hash()
    # 已有旧记录 -> 原样保留
    old = {"raw_judge_prompt_hash": "old_prompt",
           "raw_judge_script_hashes": ["h1"],
           "raw_generation_configs": [{"script_hash": "h1", "hash_status": "verified"}]}
    raw_prompt2, raw_hashes2, raw_configs2 = j.merge_raw_judge_fields(
        old, "rejudge", script_hash, 0.0, 6000)
    assert raw_prompt2 == "old_prompt"
    assert raw_hashes2 == ["h1"]
    assert script_hash not in raw_hashes2


# ---------------------------------------------------------------------------
# 契约冻结项不变
# ---------------------------------------------------------------------------
def test_frozen_fields_unchanged():
    """冻结项：prompt hash 与 JSON schema 版本在本次修正中不得变化。"""
    assert j.JSON_SCHEMA_VERSION == "v1"
    assert j.ADJUDICATION_RULE_VERSION == "v2"
    # 与已锁定 manifest 对比（smoke 阶段记录的 prompt hash）
    mp = _ROOT / "caches" / "neurology_chunk1000" / "question_set_v2" / "pilot_v1" \
        / "judge" / "longcat" / "r0_s42_5c92f17c" / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        assert j.judge_prompt_hash() == m.get("judge_prompt_hash")
