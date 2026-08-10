# -*- coding: utf-8 -*-
"""Step 1.7: System Snapshot 单元测试。

覆盖：
1. 相同输入两次构建 -> 相同 snapshot_id（可复现硬验收）。
2. 快照内容字段齐全（契约/schema 版本、routes、models、rules、tokenizer、prompt hash）。
3. 数据文件缺失 -> 哈希为 None 而不崩溃，且仍为确定性 id。
4. 系统状态改变（不同 working_dir 文件哈希）-> snapshot_id 改变（敏感性）。
5. write_snapshot 写出 <snapshot_id>.json 且文件名与 id 一致。
"""
import json
import tempfile
from pathlib import Path

from hyperrag.experiment_contract import load_contract
from hyperrag.system_snapshot import (
    _NON_HASH_FIELDS,
    build_system_snapshot,
    compute_file_hash,
    compute_runtime_source_fingerprint,
    write_snapshot,
)


def _make_dummy_working_dir(tmp: Path) -> Path:
    """创建含若干确定性数据文件的临时 working_dir。"""
    wd = tmp / "caches" / "dummy"
    wd.mkdir(parents=True)
    (wd / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
    (wd / "vdb_chunks.json").write_text("[]", encoding="utf-8")
    (wd / "vdb_entities.json").write_text("[]", encoding="utf-8")
    (wd / "vdb_relationships.json").write_text("[]", encoding="utf-8")
    (wd / "hypergraph_chunk_entity_relation.hgdb").write_bytes(b"\x00\x01\x02")
    return wd


def test_snapshot_id_reproducible():
    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wd1 = _make_dummy_working_dir(td / "a")
        wd2 = _make_dummy_working_dir(td / "b")  # 相同内容，不同路径
        s1 = build_system_snapshot(contract, "dummy", working_dir=wd1)
        s2 = build_system_snapshot(contract, "dummy", working_dir=wd2)
    assert s1["snapshot_id"] == s2["snapshot_id"]
    assert len(s1["snapshot_id"]) == 64  # sha256 hex


def test_snapshot_content_fields():
    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        wd = _make_dummy_working_dir(Path(td))
        snap = build_system_snapshot(contract, "dummy", working_dir=wd)

    # 版本字段
    assert snap["contract_version"] == contract.contract_version
    assert snap["trace_schema_version"] == "retrieval-trace-v2"
    assert snap["success_rule_version"] == "operational-stable-success-v1"
    assert snap["metric_version"] == "evidence-metrics-v1"
    assert snap["cost_schema_version"] == "route-cost-v1"

    # P0-P4 配置齐全且与契约逐项相等
    for stage in ("P0", "P1", "P2", "P3", "P4"):
        rc = contract.route(stage)
        r = snap["routes"][stage]
        assert r["chunk_vdb_top_k"] == rc.chunk_vdb_top_k
        assert r["entity_vdb_top_k"] == rc.entity_vdb_top_k
        assert r["relation_vdb_top_k"] == rc.relation_vdb_top_k
        assert r["final_context_hard_cap"] == rc.final_context_hard_cap
        assert r["is_llm_only"] == rc.is_llm_only

    # 模型标识
    assert snap["models"]["answer_model"] == contract.system_boundary.answer_model
    assert snap["models"]["embedding_model"] == contract.system_boundary.embedding_model
    assert snap["models"]["judge_model"] == contract.system_boundary.judge_model

    # 规则
    assert snap["rules"]["temperature"] == 0.1
    assert snap["rules"]["max_response_tokens"] == 3000
    assert snap["rules"]["llm_response_cache"] == "disabled"
    assert snap["rules"]["default_seed"] == 42

    # prompt hash 存在（64 hex）
    ph = snap["prompt_hashes"]["formal_answer_response"]
    assert isinstance(ph, str) and len(ph) == 64

    # tokenizer / data_hashes / 版本字段存在
    assert "version" in snap["tokenizer"]
    assert set(snap["data_hashes"].keys()) == {
        "dataset_text_chunks", "vdb_chunks", "vdb_entities",
        "vdb_relationships", "hypergraph",
    }
    assert snap["retriever_version"] == "v1"
    assert "score_normalization_version" in snap


def test_missing_data_files_yield_none_hash():
    contract = load_contract()
    # 指向一个不存在的 working_dir：所有文件哈希应为 None 且不抛错
    snap = build_system_snapshot(contract, "nonexistent_dataset",
                                 working_dir=Path("/no/such/dir"))
    assert all(v is None for v in snap["data_hashes"].values())
    # 仍然可复现（全 None 也是确定性状态）
    snap2 = build_system_snapshot(contract, "nonexistent_dataset",
                                  working_dir=Path("/no/such/dir"))
    assert snap["snapshot_id"] == snap2["snapshot_id"]


def test_snapshot_id_sensitive_to_data_state():
    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wd_a = _make_dummy_working_dir(td / "a")
        wd_b = td / "b"
        wd_b.mkdir(parents=True)
        # 不同的 hypergraph 字节 -> 不同数据哈希
        (wd_b / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
        (wd_b / "vdb_chunks.json").write_text("[]", encoding="utf-8")
        (wd_b / "vdb_entities.json").write_text("[]", encoding="utf-8")
        (wd_b / "vdb_relationships.json").write_text("[]", encoding="utf-8")
        (wd_b / "hypergraph_chunk_entity_relation.hgdb").write_bytes(b"\xff\xff\xff")
        s_a = build_system_snapshot(contract, "dummy", working_dir=wd_a)
        s_b = build_system_snapshot(contract, "dummy", working_dir=wd_b)
    assert s_a["snapshot_id"] != s_b["snapshot_id"]


def test_write_snapshot_files_named_by_id():
    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wd = _make_dummy_working_dir(td / "a")
        snap = build_system_snapshot(contract, "dummy", working_dir=wd)
        out_dir = td / "snapshots"
        path = write_snapshot(snap, out_dir)
        assert path.name == f"{snap['snapshot_id']}.json"
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["snapshot_id"] == snap["snapshot_id"]


def test_compute_file_hash_none_when_missing():
    assert compute_file_hash(Path("/no/such/file.json")) is None


def test_runtime_fingerprint_tracks_whitelist_bytes(tmp_path):
    """运行时指纹必须覆盖白名单内未跟踪文件：新增/修改 hyperrag .py 字节都改变指纹。"""
    (tmp_path / "hyperrag").mkdir()
    f = tmp_path / "hyperrag" / "a.py"
    f.write_text("x = 1", encoding="utf-8")
    fp1 = compute_runtime_source_fingerprint(tmp_path)
    assert fp1["runtime_source_file_count"] == 1

    # 修改已有文件字节 -> 指纹变化
    f.write_text("x = 2", encoding="utf-8")
    fp2 = compute_runtime_source_fingerprint(tmp_path)
    assert fp2["runtime_source_fingerprint"] != fp1["runtime_source_fingerprint"]

    # 新增（等价于 git untracked）文件 -> 指纹变化 + 计数增加
    (tmp_path / "hyperrag" / "b.py").write_text("y = 1", encoding="utf-8")
    fp3 = compute_runtime_source_fingerprint(tmp_path)
    assert fp3["runtime_source_file_count"] == 2
    assert fp3["runtime_source_fingerprint"] != fp2["runtime_source_fingerprint"]

    # 相同内容重复计算 -> 确定性
    fp4 = compute_runtime_source_fingerprint(tmp_path)
    assert fp4 == fp3


def test_runtime_fingerprint_ignores_tests_and_verification_scripts(tmp_path):
    """回归：修改 tests/ 或验证脚本 **不得** 改变运行时指纹（白名单语义）。"""
    (tmp_path / "hyperrag").mkdir()
    (tmp_path / "hyperrag" / "a.py").write_text("x = 1", encoding="utf-8")
    baseline = compute_runtime_source_fingerprint(tmp_path)

    # 新增 tests/ 与 scripts/ 内容 -> 指纹不变
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("assert True", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    verify = tmp_path / "scripts" / "verify_trace_completeness.py"
    verify.write_text("print('v1')", encoding="utf-8")
    fp2 = compute_runtime_source_fingerprint(tmp_path)
    assert fp2 == baseline

    # 修改 tests/ 与验证脚本字节 -> 指纹仍不变
    (tmp_path / "tests" / "test_x.py").write_text("assert 1 == 1", encoding="utf-8")
    verify.write_text("print('v2, changed')", encoding="utf-8")
    fp3 = compute_runtime_source_fingerprint(tmp_path)
    assert fp3 == baseline


def test_runtime_fingerprint_tracks_step3_and_contract(tmp_path):
    """回归：修改 Step_3 / 冻结契约 / Trace Schema 必须改变运行时指纹；
    reproduce/ 下其他脚本不在白名单，不影响指纹。"""
    (tmp_path / "hyperrag").mkdir()
    (tmp_path / "hyperrag" / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "reproduce").mkdir()
    step3 = tmp_path / "reproduce" / "Step_3_response_question.py"
    step3.write_text("# v1", encoding="utf-8")
    (tmp_path / "docs" / "schema").mkdir(parents=True)
    contract = tmp_path / "docs" / "question_set_v2_contract.yaml"
    contract.write_text("version: 1", encoding="utf-8")
    schema = tmp_path / "docs" / "schema" / "retrieval_trace_v2.schema.json"
    schema.write_text("{}", encoding="utf-8")

    fp1 = compute_runtime_source_fingerprint(tmp_path)
    assert fp1["runtime_source_file_count"] == 4  # a.py + Step_3 + contract + schema

    # 修改 Step_3 -> 指纹变化
    step3.write_text("# v2", encoding="utf-8")
    fp2 = compute_runtime_source_fingerprint(tmp_path)
    assert fp2["runtime_source_fingerprint"] != fp1["runtime_source_fingerprint"]

    # 修改冻结契约 -> 指纹变化
    contract.write_text("version: 2", encoding="utf-8")
    fp3 = compute_runtime_source_fingerprint(tmp_path)
    assert fp3["runtime_source_fingerprint"] != fp2["runtime_source_fingerprint"]

    # 修改 Trace Schema -> 指纹变化
    schema.write_text('{"v": 2}', encoding="utf-8")
    fp4 = compute_runtime_source_fingerprint(tmp_path)
    assert fp4["runtime_source_fingerprint"] != fp3["runtime_source_fingerprint"]

    # reproduce/ 下白名单外脚本 -> 指纹不变
    (tmp_path / "reproduce" / "Step_1_insert.py").write_text("# aux", encoding="utf-8")
    fp5 = compute_runtime_source_fingerprint(tmp_path)
    assert fp5 == fp4


def test_snapshot_contains_runtime_fingerprint_and_audit_fields():
    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        wd = _make_dummy_working_dir(Path(td) / "a")
        snap = build_system_snapshot(contract, "dummy", working_dir=wd)
    assert len(snap["runtime_source_fingerprint"]) == 64
    assert snap["runtime_source_file_count"] > 0
    # 审计字段存在（真实仓库下验证脚本存在 -> 非 None）
    assert "verification_script_hash" in snap


def test_snapshot_id_excludes_git_and_verification_fields():
    """回归：git 元信息与验证脚本哈希是纯审计字段，不得影响 snapshot_id。
    否则提交一个只改 tests/ 的 commit 就会使已有回答产物被误判失效。"""
    import hashlib as _hashlib

    contract = load_contract()
    with tempfile.TemporaryDirectory() as td:
        wd = _make_dummy_working_dir(Path(td) / "a")
        snap = build_system_snapshot(contract, "dummy", working_dir=wd)

    def _recompute_id(s: dict) -> str:
        body = {k: v for k, v in s.items() if k not in _NON_HASH_FIELDS}
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert _recompute_id(snap) == snap["snapshot_id"]

    # 篡改审计字段 -> snapshot_id 不变
    mutated = dict(snap)
    mutated["git_commit_sha"] = "0" * 40
    mutated["git_dirty"] = not snap["git_dirty"]
    mutated["verification_script_hash"] = "f" * 64
    assert _recompute_id(mutated) == snap["snapshot_id"]

    # 篡改运行时指纹 -> snapshot_id 必须变化
    mutated2 = dict(snap)
    mutated2["runtime_source_fingerprint"] = "e" * 64
    assert _recompute_id(mutated2) != snap["snapshot_id"]
