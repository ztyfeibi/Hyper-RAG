"""盲审生成器测试（build_blind_review_set.py，2026-08-13 步骤 4）。

Coverage:
1. set_A = 100 条（P0-P4 每路径 20），(route, qid) 去重。
2. set_B = 43 条 P_gold pending 全量。
3. set_C = 20 条 P_gold unsupported 阴性对照（n_unsupported=0, human_review=False）。
4. BL-xxx ID 跨三集唯一。
5. verdicts_template.jsonl 含 unsupported_claims 字段（空列表占位）。
6. 材料正文零路径标识泄漏。
"""

import json
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

# Mock 重依赖（同 test_judge_longcat.py）
for _mod in ("openai", "numpy", "tiktoken", "aioboto3", "aiohttp",
             "my_config", "hyperrag", "hyperrag.env"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["my_config"].LLM_BASE_URL_SILICONFLOW = "mock"
sys.modules["my_config"].LLM_API_KEY_SILICONFLOW = "mock"
sys.modules["my_config"].LLM_MODEL_SILICONFLOW = "mock"

import build_blind_review_set as br  # noqa: E402
import judge_longcat as j  # noqa: E402

BLIND_DIR = j.out_dir(0, 42, j.SNAPSHOT_DEFAULT) / "blind_review"


# ---------------------------------------------------------------------------
# 纯函数单测
# ---------------------------------------------------------------------------
def test_sample_neg_controls_selects_unsupported_negative():
    """sample_neg_controls 只选 P_gold + n_unsupported=0 + human_review=False。"""
    recs = [
        {"route": "P_gold", "qid": "q1", "n_unsupported": 0, "human_review": False, "candidate": "a"},
        {"route": "P_gold", "qid": "q2", "n_unsupported": 0, "human_review": False, "candidate": "b"},
        {"route": "P_gold", "qid": "q3", "n_unsupported": 2, "human_review": True, "candidate": "c"},
        {"route": "P1", "qid": "q4", "n_unsupported": 0, "human_review": False, "candidate": "d"},
    ]
    rng = random.Random(42)
    result = br.sample_neg_controls(recs, rng, n=20)
    assert len(result) == 2  # 只有 q1, q2 符合
    qids = {r["qid"] for r in result}
    assert qids == {"q1", "q2"}


def test_sample_neg_controls_respects_n_limit():
    """n 参数限制返回数量。"""
    recs = [
        {"route": "P_gold", "qid": f"q{i}", "n_unsupported": 0, "human_review": False, "candidate": "x"}
        for i in range(30)
    ]
    rng = random.Random(42)
    result = br.sample_neg_controls(recs, rng, n=10)
    assert len(result) == 10


def test_write_verdicts_template_has_unsupported_claims():
    """verdicts_template.jsonl 每行含 unsupported_claims 空列表。"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        tmp = f.name
    try:
        rows = [{"blind_id": "BL-001"}, {"blind_id": "BL-002"}]
        br.write_verdicts_template(Path(tmp), rows)
        lines = open(tmp, encoding="utf-8").readlines()
        assert len(lines) == 2
        for line in lines:
            r = json.loads(line)
            assert "unsupported_claims" in r
            assert r["unsupported_claims"] == []
            assert "unsupported_fatality" in r
            assert "au_status" in r
            assert "verdict" in r
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# 集成测试（验证已生成产物）
# ---------------------------------------------------------------------------
def test_set_a_count_and_routes():
    """set_A = 100 条，P0-P4 各 20。"""
    manifest = json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    set_a = manifest["set_A_main_100"]
    assert len(set_a) == 100
    routes = [r["route"] for r in set_a]
    for route in ("P0", "P1", "P2", "P3", "P4"):
        assert routes.count(route) == 20, f"{route} 应有 20 条"


def test_set_b_pgold_pending():
    """set_B = 43 条 P_gold pending。"""
    manifest = json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    set_b = manifest["set_B_pgold_pending_43"]
    assert len(set_b) == 43
    assert all(r["route"] == "P_gold" for r in set_b)
    assert all(r["human_review"] for r in set_b)


def test_set_c_negative_controls():
    """set_C = 20 条 P_gold unsupported 阴性（n_unsupported=0, human_review=False）。"""
    manifest = json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    set_c = manifest.get("set_C_neg_control_20", [])
    assert len(set_c) == 20
    assert all(r["route"] == "P_gold" for r in set_c)
    assert all(r["n_unsupported"] == 0 for r in set_c)
    assert all(not r["human_review"] for r in set_c)


def test_blind_ids_unique_across_sets():
    """BL-xxx 跨三集唯一。"""
    manifest = json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    all_ids = []
    for name in ("set_A_main_100", "set_B_pgold_pending_43", "set_C_neg_control_20"):
        all_ids.extend(r["blind_id"] for r in manifest[name])
    assert len(all_ids) == len(set(all_ids)), "BL-xxx 有重复"
    assert len(all_ids) == 163


def test_verdicts_template_count_and_fields():
    """verdicts_template.jsonl = 163 条，每行含 unsupported_claims 字段。"""
    lines = (BLIND_DIR / "verdicts_template.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 163
    for line in lines:
        r = json.loads(line)
        assert "unsupported_claims" in r
        assert r["unsupported_claims"] == []
        assert "unsupported_fatality" in r


def test_no_path_leakage_in_materials():
    """三份材料正文零路径标识泄漏。"""
    import re
    for fname in ("set_A_main_100.md", "set_B_pgold_pending_43.md", "set_C_neg_control_20.md"):
        text = (BLIND_DIR / fname).read_text(encoding="utf-8")
        # 真路径标识：P0/P1/.../P4（非 P3xx 医学术语）、P_gold、qv2-xxx、route=
        leaks = re.findall(r"\bP[0-4]\b(?!\d)|P_gold|qv2-\d+|route\s*=", text)
        assert len(leaks) == 0, f"{fname} 路径泄漏: {leaks[:5]}"


def test_set_a_route_qid_dedup():
    """set_A 内 (route, qid) 无重复。"""
    manifest = json.loads((BLIND_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    set_a = manifest["set_A_main_100"]
    keys = [(r["route"], r["qid"]) for r in set_a]
    assert len(keys) == len(set(keys)), "set_A 有 (route, qid) 重复"


def test_review_guide_exists():
    """REVIEW_GUIDE.md 存在且含 unsupported_claims 说明。"""
    guide = (BLIND_DIR / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
    assert "unsupported_claims" in guide
    assert "supported_by_source" in guide
    assert "set_C" in guide or "阴性对照" in guide
