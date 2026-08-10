# -*- coding: utf-8 -*-
"""Step 1.2 验收测试：运行时契约加载器与共享 Schema Owner。

验收核心：运行时解析出的 P0-P4 top-k / 预算与冻结契约 YAML 逐项相等，
且共享校验逻辑与门禁脚本行为一致。
"""

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONTRACT_YAML = REPO_ROOT / "docs" / "question_set_v2_contract.yaml"


def _load_by_path(name, relpath):
    """以文件路径方式加载模块，绕开 hyperrag/__init__ 的重依赖。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ec = _load_by_path("experiment_contract", "hyperrag/experiment_contract.py")
es = _load_by_path("experiment_schema", "hyperrag/experiment_schema.py")


def _raw():
    return yaml.safe_load(CONTRACT_YAML.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 加载与版本
# --------------------------------------------------------------------------- #
def test_load_contract_default():
    c = ec.load_contract()
    assert c.contract_version == "question-set-v2.1-contract-v1"
    assert c.trace_schema_version == "retrieval-trace-v2"
    assert c.success_rule_version
    assert c.metric_version
    assert c.cost_schema_version


def test_version_mismatch_raises(tmp_path):
    raw = _raw()
    raw["contract_version"] = "question-set-v2.1-contract-v999"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        ec.load_contract(p)
        assert False, "expected ContractError"
    except ec.ContractError as e:
        assert "mismatch" in str(e)


def test_missing_version_key_raises(tmp_path):
    raw = _raw()
    del raw["cost_schema_version"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        ec.load_contract(p)
        assert False, "expected ContractError"
    except ec.ContractError as e:
        assert "cost_schema_version" in str(e)


def test_ladder_length_mismatch_raises(tmp_path):
    raw = _raw()
    raw["policy_ladder"]["candidate_budget"]["entity_vdb_top_k"] = [0, 0, 25]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        ec.load_contract(p)
        assert False, "expected ContractError"
    except ec.ContractError as e:
        assert "entity_vdb_top_k" in str(e)


# --------------------------------------------------------------------------- #
# 核心验收：运行时值与 YAML 逐项相等
# --------------------------------------------------------------------------- #
def test_route_values_equal_yaml_itemwise():
    c = ec.load_contract()
    raw = _raw()
    ladder = raw["policy_ladder"]
    fields = {
        "chunk_vdb_top_k": ladder["candidate_budget"]["chunk_vdb_top_k"],
        "entity_vdb_top_k": ladder["candidate_budget"]["entity_vdb_top_k"],
        "relation_vdb_top_k": ladder["candidate_budget"]["relation_vdb_top_k"],
        "entity_description_cap": ladder["pre_merge_caps"]["entity_description_cap"],
        "relation_description_cap": ladder["pre_merge_caps"]["relation_description_cap"],
        "source_text_cap": ladder["pre_merge_caps"]["source_text_cap"],
        "final_context_hard_cap": ladder["post_merge_hard_control"]["final_context_hard_cap"],
    }
    for i, stage in enumerate(ec.POLICY_STAGES):
        route = c.route(stage)
        for fname, vec in fields.items():
            assert getattr(route, fname) == vec[i], \
                f"{stage}.{fname}: runtime={getattr(route, fname)} yaml={vec[i]}"


def test_frozen_expected_values_spot_check():
    """抽查冻结数值（防 YAML 本身被误改而测试仍互证通过）。"""
    c = ec.load_contract()
    p0, p1, p2, p3, p4 = (c.route(s) for s in ["P0", "P1", "P2", "P3", "P4"])
    assert p0.is_llm_only
    assert (p1.chunk_vdb_top_k, p1.source_text_cap, p1.final_context_hard_cap) \
        == (20, 4000, 4000)
    assert (p2.entity_vdb_top_k, p2.relation_vdb_top_k) == (25, 0)
    assert p2.relation_description_cap == 800  # P2 邻接扩展仍生效
    assert (p3.entity_vdb_top_k, p3.relation_vdb_top_k,
            p3.final_context_hard_cap) == (30, 30, 12000)
    assert (p4.entity_vdb_top_k, p4.relation_vdb_top_k,
            p4.final_context_hard_cap) == (50, 50, 15000)


def test_section_ratios_normalized_lowercase():
    c = ec.load_contract()
    assert c.section_allocation_ratios == {
        "sources": 0.55, "relationships": 0.30, "entities": 0.15}
    assert abs(sum(c.section_allocation_ratios.values()) - 1.0) < 1e-9


def test_system_boundary_values():
    c = ec.load_contract()
    sb = c.system_boundary
    assert sb.answer_model == "qwen-27b-int4"
    assert sb.embedding_model == "qwen-8b-embed"
    assert sb.judge_model == "meituan-longcat/LongCat-2.0"
    assert sb.llm_max_length == 24576
    assert sb.temperature == 0.1
    assert sb.max_response_tokens == 3000
    assert sb.router == "disabled"
    assert sb.type_aware_weighting == "disabled"
    assert sb.llm_response_cache == "disabled"


def test_p_gold_and_unknown_route_raise():
    c = ec.load_contract()
    for bad in ["P_gold", "P5", "naive"]:
        try:
            c.route(bad)
            assert False, f"expected ContractError for {bad}"
        except ec.ContractError:
            pass


# --------------------------------------------------------------------------- #
# 共享 Schema Owner 与门禁脚本一致性
# --------------------------------------------------------------------------- #
def test_schema_owner_is_single_source():
    """门禁脚本的校验符号必须是 experiment_schema 的 re-export（同一函数对象）。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import validate_question_set_contract as v
    # 同一逻辑：代码对象一致（file-path 双加载导致对象不同，比较 co_code）
    assert v.validate_schema.__code__.co_code == es.validate_schema.__code__.co_code
    assert v.derive_labels.__code__.co_code == es.derive_labels.__code__.co_code
    assert v.COST_FIELDS == es.COST_FIELDS
    assert v.POLICY_STAGES == es.POLICY_STAGES


def test_stable_success_frozen_stopping_points():
    ss = es.stable_success
    assert ss(3, 3) is True and ss(1, 3) is False and ss(2, 3) is None
    assert ss(4, 5) is True and ss(3, 5) is False
    assert ss(8, 10) is True and ss(7, 10) is False
    assert ss(4, 4) is None  # 非停止点 -> 未决


def test_trace_schema_loadable_and_validates_example():
    schema = es.load_trace_schema()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import validate_question_set_contract as v
    es.validate_trace_record(v.EXAMPLE_TRACE, schema)  # 不抛即通过


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
