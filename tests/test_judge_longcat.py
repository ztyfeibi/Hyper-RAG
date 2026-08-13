"""Judge 契约测试（judge_longcat.py，2026-08-13 步骤 3 修正验收）。

Coverage:
1. evidence-group 契约：组内 OR（AU 任一 span 命中即满足）、组间 AND（全部 required AU 满足）。
2. 搜索范围：P1/P_gold 全文（无区段标记）；P2-P4 仅 -----Sources----- 区段
   （Entities/Relationships 区段文本不能贡献 source hit）。
3. human_review=true -> final_answer_correctness=pending -> route_success=pending
   （§12.1 line 867：矛盾/critical/unsupported/uncertain 一律人工复核，不由规则判死）。
4. P0 无 evidence gate：route_success = final_answer_correctness。
5. manifest 双哈希分离：raw_judge_*（原始 480 次 API 调用环境）与
   adjudicator_script_hash（离线裁决器）互不冒充。
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import judge_longcat as j  # noqa: E402


# ---------------------------------------------------------------------------
# evidence-group：组内 OR / 组间 AND / 无 span 严格处理
# ---------------------------------------------------------------------------
def test_group_or_alternative_sources_hit():
    """组内 OR：AU 有多个 alternative source，任一命中即满足该组。"""
    ctx = "本文包含 Lott IT 1978 年的报告原文。"
    spans = {"AU1": [{"quote": "A 不存在的文本"}, {"quote": "Lott IT 1978"}]}
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1"}, search="full")
    assert per == {"AU1": True}
    assert all_hit is True


def test_group_or_all_miss():
    """组内 OR：全部 alternative source 都未命中 -> 组未满足。"""
    ctx = "无相关内容"
    spans = {"AU1": [{"quote": "X 不存在"}, {"quote": "Y 也不存在"}]}
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1"}, search="full")
    assert per == {"AU1": False}
    assert all_hit is False


def test_group_and_all_units_required():
    """组间 AND：一个 AU 不命中 -> all_hit=False。"""
    ctx = "Lott IT 1978 报告在此。"
    spans = {
        "AU1": [{"quote": "Lott IT 1978"}],
        "AU2": [{"quote": "Matalon 1988 缺失"}],
    }
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1", "AU2"}, search="full")
    assert per == {"AU1": True, "AU2": False}
    assert all_hit is False


def test_group_and_all_hit():
    """组间 AND：全部 required AU 命中 -> all_hit=True。"""
    ctx = "Lott IT 1978 与 Matalon 1988 都在。"
    spans = {
        "AU1": [{"quote": "Lott IT 1978"}],
        "AU2": [{"quote": "Matalon 1988"}],
    }
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1", "AU2"}, search="full")
    assert per == {"AU1": True, "AU2": True}
    assert all_hit is True


def test_required_unit_without_span_is_strict_fail():
    """无 evidence_spans 的 required AU 视为未满足（严格，required AU 必须有证据）。"""
    ctx = "有内容"
    spans = {"AU1": [{"quote": "有内容"}]}  # AU2 无 spans
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1", "AU2"}, search="full")
    assert per == {"AU1": True, "AU2": False}
    assert all_hit is False


def test_whitespace_normalization():
    """空白归一化：换行/缩进差异不影响命中。"""
    ctx = "Lott IT, Coulombe T,\nDiPaolo RV: X 报告。\n\n后续"
    spans = {"AU1": [{"quote": "Lott IT, Coulombe T, DiPaolo RV: X 报告。"}]}
    per, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1"}, search="full")
    assert all_hit is True


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
    # quote 只出现在 Entities 区段，Sources 区段没有 -> 必须 miss
    ctx = (
        "-----Entities-----\n"
        "Lott IT 1978 研究者条目\n"
        "-----Relationships-----\n"
        "关系描述\n"
        "-----Sources-----\n"
        "完全无关的 source 文本\n"
    )
    spans = {"AU1": [{"quote": "Lott IT 1978"}]}
    _, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1"}, search="sources")
    assert all_hit is False
    # 同一 quote 在 Sources 区段出现 -> 必须 hit
    ctx2 = ctx.replace("完全无关的 source 文本", "Lott IT 1978 在 source 区段")
    _, all_hit2 = j.calc_source_evidence_coverage(ctx2, spans, {"AU1"}, search="sources")
    assert all_hit2 is True


def test_p1_pgold_full_search_hits_anywhere():
    """P1/P_gold（search='full'）：无区段标记，整个 context 即 Sources，正常命中。"""
    ctx = "Lott IT, Coulombe T, DiPaolo RV, et al: Vitamin B6-dependent seizures. Neurology 28:47, 1978."
    spans = {"AU1": [{"quote": "Vitamin B6-dependent seizures. Neurology 28:47, 1978"}]}
    _, all_hit = j.calc_source_evidence_coverage(ctx, spans, {"AU1"}, search="full")
    assert all_hit is True


# ---------------------------------------------------------------------------
# human_review -> final pending
# ---------------------------------------------------------------------------
def test_human_review_forces_pending():
    """human_review=true 且 derived=fail -> final=pending（不提前判死）。"""
    assert j.final_answer_correctness("fail", True) == "pending"
    assert j.final_answer_correctness("pass", True) == "pending"
    assert j.final_answer_correctness("uncertain", True) == "pending"


def test_non_human_review_keeps_derived():
    """非 human_review 记录保留 derived；uncertain 兜底 pending。"""
    assert j.final_answer_correctness("pass", False) == "pass"
    assert j.final_answer_correctness("fail", False) == "fail"
    assert j.final_answer_correctness("uncertain", False) == "pending"


def test_route_success_human_review_pending_ignores_cov():
    """human_review=true -> route_success=pending（即使 cov 已通过也不提前判 pass）。"""
    assert j.combine_route_success("pass", True, True, True) == "pending"
    assert j.combine_route_success("fail", False, True, True) == "pending"


# ---------------------------------------------------------------------------
# P0 无 evidence gate
# ---------------------------------------------------------------------------
def test_p0_no_evidence_gate():
    """P0：route_success = final_answer_correctness（无证据门槛）。"""
    assert j.combine_route_success("pass", False, False, False) == "pass"
    assert j.combine_route_success("fail", False, False, False) == "fail"
    assert j.combine_route_success("pending", False, False, False) == "pending"
    # P0 上 human_review 同样优先 pending
    assert j.combine_route_success("fail", False, False, True) == "pending"


def test_route_success_dual_gate():
    """P1-P4/P_gold 双门槛组合矩阵。"""
    assert j.combine_route_success("pass", True, True, False) == "pass"
    assert j.combine_route_success("pass", False, True, False) == "fail"   # 答案对但证据缺
    assert j.combine_route_success("fail", True, True, False) == "fail"    # 答案错
    assert j.combine_route_success("pending", True, True, False) == "pending"


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
    assert raw_configs[0]["max_tokens"] == 6000
    assert raw_prompt == j.judge_prompt_hash()
    # judge 模式追加：已有别的哈希 -> 追加当前（去重）
    old = {"raw_judge_script_hashes": ["old_hash"],
           "raw_generation_configs": [{"script_hash": "old_hash"}]}
    _, raw_hashes2, raw_configs2 = j.merge_raw_judge_fields(
        old, "judge", script_hash, 0.0, 6000)
    assert raw_hashes2 == ["old_hash", script_hash]
    assert len(raw_configs2) == 2
    # judge 模式去重：同一哈希不重复追加
    old3 = {"raw_judge_script_hashes": [script_hash],
            "raw_generation_configs": [{"script_hash": script_hash}]}
    _, raw_hashes3, raw_configs3 = j.merge_raw_judge_fields(
        old3, "judge", script_hash, 0.0, 6000)
    assert raw_hashes3 == [script_hash]
    assert len(raw_configs3) == 2


def test_manifest_rejudge_preserves_history():
    """rejudge 模式：不追加裁决器哈希到 raw；无历史时补 RAW_JUDGE_SCRIPT_HISTORY。"""
    script_hash = "adjudicator_script"
    # 无旧记录 -> 补已知原始调用历史（91447b21 主跑版 + 6000 补跑版）
    raw_prompt, raw_hashes, raw_configs = j.merge_raw_judge_fields(
        {}, "rejudge", script_hash, 0.0, 6000)
    assert script_hash not in raw_hashes          # 裁决器哈希不得混入 raw
    assert len(raw_hashes) == 2                    # 两个历史 raw 版本
    assert len(raw_configs) == 2
    assert "91447b21" in raw_hashes[0]
    assert raw_prompt == j.judge_prompt_hash()
    # 已有旧记录 -> 原样保留
    old = {"raw_judge_prompt_hash": "old_prompt",
           "raw_judge_script_hashes": ["h1"],
           "raw_generation_configs": [{"script_hash": "h1"}]}
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
