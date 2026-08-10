# -*- coding: utf-8 -*-
"""运行时实验契约加载器 (Step 1.2) —— P0-P4 参数的 Single Source of Truth。

从冻结契约 ``docs/question_set_v2_contract.yaml`` 加载 P0-P4 固定路径配置，
返回类型化对象。**任何 runner / Judge / Router / 分析脚本禁止复制 P0-P4
数值** —— 需要参数时 import 本模块。

设计约束：
  - 仅依赖 stdlib + PyYAML，不 import hyperrag 包内其它模块，
    保证可被 importlib 以文件路径方式独立加载。
  - 加载时校验：版本串齐全、stages 恰为 P0..P4、预算数组长度 5 对齐、
    区段配比求和 ~1.0。任何校验失败抛 ContractError，绝不静默容忍。

用法::

    from hyperrag.experiment_contract import load_contract

    contract = load_contract()               # 默认冻结契约路径
    p3 = contract.route("P3")
    p3.entity_vdb_top_k        # 30
    p3.final_context_hard_cap  # 12000
    contract.system_boundary.temperature  # 0.1
"""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT_PATH = REPO_ROOT / "docs" / "question_set_v2_contract.yaml"

EXPECTED_CONTRACT_VERSION = "question-set-v2.1-contract-v1"
EXPECTED_TRACE_SCHEMA_VERSION = "retrieval-trace-v2"

POLICY_STAGES = ["P0", "P1", "P2", "P3", "P4"]
GOLD_ROUTE_ID = "P_gold"   # 非阶梯路径：oracle 证据直填，由 executor 单独处理

REQUIRED_VERSION_KEYS = [
    "contract_version",
    "trace_schema_version",
    "success_rule_version",
    "metric_version",
    "cost_schema_version",
]

# candidate_budget / pre_merge_caps / post_merge_hard_control 中的向量字段
_CANDIDATE_FIELDS = ["chunk_vdb_top_k", "entity_vdb_top_k", "relation_vdb_top_k"]
_PRE_MERGE_FIELDS = ["entity_description_cap", "relation_description_cap",
                     "source_text_cap"]
_POST_MERGE_FIELDS = ["final_context_hard_cap"]


class ContractError(Exception):
    """契约缺失、版本不符或结构不合法。"""


@dataclass(frozen=True)
class RouteConfig:
    """单条固定路径 (P0..P4) 的全部检索参数。"""
    route_id: str
    # candidate_budget
    chunk_vdb_top_k: int
    entity_vdb_top_k: int
    relation_vdb_top_k: int
    # pre_merge_caps（token 上限，Qwen tokenizer 计数）
    entity_description_cap: int
    relation_description_cap: int
    source_text_cap: int
    # post_merge_hard_control
    final_context_hard_cap: int

    @property
    def is_llm_only(self) -> bool:
        """P0：完全不检索。"""
        return (self.chunk_vdb_top_k == 0 and self.entity_vdb_top_k == 0
                and self.relation_vdb_top_k == 0)


@dataclass(frozen=True)
class SystemBoundary:
    """契约 system_boundary 段：模型与系统级冻结配置。"""
    answer_model: str
    embedding_model: str
    judge_model: str
    llm_max_length: int
    temperature: float
    max_response_tokens: int
    router: str                 # "disabled"
    type_aware_weighting: str   # "disabled"
    llm_response_cache: str     # "disabled"


@dataclass(frozen=True)
class ExperimentContract:
    """加载并校验后的完整契约视图。"""
    contract_version: str
    trace_schema_version: str
    success_rule_version: str
    metric_version: str
    cost_schema_version: str
    system_boundary: SystemBoundary
    routes: dict                      # route_id -> RouteConfig
    section_allocation_ratios: dict   # 小写键: {"sources":0.55, "relationships":0.30, "entities":0.15}
    raw: dict = field(repr=False)     # 原始 YAML dict（只读用途）

    def route(self, route_id: str) -> RouteConfig:
        """按 route_id 返回 RouteConfig；P_gold 与未知 id 抛 ContractError。"""
        if route_id == GOLD_ROUTE_ID:
            raise ContractError(
                "P_gold 不是策略阶梯路径，没有检索预算参数；"
                "请在 fixed_route_executor 中走 oracle 证据直填分支。")
        if route_id not in self.routes:
            raise ContractError(
                f"unknown route_id {route_id!r}; valid: {POLICY_STAGES}")
        return self.routes[route_id]

    @property
    def stage_ids(self):
        return list(POLICY_STAGES)


def _require(cond, msg):
    if not cond:
        raise ContractError(msg)


def load_contract(path=None, expected_version=EXPECTED_CONTRACT_VERSION):
    """加载契约 YAML -> ExperimentContract，加载时完成全部结构校验。

    Parameters
    ----------
    path : 契约 YAML 路径，默认冻结契约。
    expected_version : 期望的 contract_version；传 None 跳过版本相等检查
        （仅用于加载历史版本契约做对比分析）。
    """
    import yaml  # 延迟导入，保持模块 import 本身 stdlib-only

    p = Path(path) if path else DEFAULT_CONTRACT_PATH
    _require(p.exists(), f"contract file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ContractError(f"contract YAML failed to parse: {e}") from e
    _require(isinstance(data, dict), "contract YAML root is not a mapping")

    # 1) 版本串
    for key in REQUIRED_VERSION_KEYS:
        _require(isinstance(data.get(key), str) and data[key],
                 f"contract missing/empty version key: {key}")
    if expected_version is not None:
        _require(data["contract_version"] == expected_version,
                 f"contract_version mismatch: expected {expected_version!r}, "
                 f"got {data['contract_version']!r}")

    # 2) 策略阶梯
    ladder = data.get("policy_ladder")
    _require(isinstance(ladder, dict), "policy_ladder missing")
    _require(ladder.get("stages") == POLICY_STAGES,
             f"policy_ladder.stages must be exactly {POLICY_STAGES}")

    n = len(POLICY_STAGES)
    sections = {
        "candidate_budget": _CANDIDATE_FIELDS,
        "pre_merge_caps": _PRE_MERGE_FIELDS,
        "post_merge_hard_control": _POST_MERGE_FIELDS,
    }
    vectors = {}
    for section, fields in sections.items():
        sec = ladder.get(section)
        _require(isinstance(sec, dict), f"policy_ladder.{section} missing")
        for fname in fields:
            vec = sec.get(fname)
            _require(isinstance(vec, list) and len(vec) == n,
                     f"policy_ladder.{section}.{fname} must be a length-{n} list")
            for i, val in enumerate(vec):
                _require(isinstance(val, (int, float)) and not isinstance(val, bool),
                         f"policy_ladder.{section}.{fname}[{i}] must be a number")
                _require(val >= 0,
                         f"policy_ladder.{section}.{fname}[{i}] = {val} is negative")
            vectors[fname] = vec

    # 3) 区段配比
    ratios_raw = ladder.get("post_merge_hard_control", {}) \
                       .get("section_allocation_ratios")
    _require(isinstance(ratios_raw, dict),
             "post_merge_hard_control.section_allocation_ratios missing")
    total = sum(ratios_raw.values())
    _require(abs(total - 1.0) <= 1e-6,
             f"section_allocation_ratios sum={total}, expected 1.0")
    ratios = {str(k).lower(): float(v) for k, v in ratios_raw.items()}

    # 4) system_boundary
    sb = data.get("system_boundary")
    _require(isinstance(sb, dict), "system_boundary missing")
    try:
        boundary = SystemBoundary(
            answer_model=sb["answer_model"],
            embedding_model=sb["embedding_model"],
            judge_model=sb["judge_model"],
            llm_max_length=int(sb["llm_max_length"]),
            temperature=float(sb["temperature"]),
            max_response_tokens=int(sb["max_response_tokens"]),
            router=sb["router"],
            type_aware_weighting=sb["type_aware_weighting"],
            llm_response_cache=sb["llm_response_cache"],
        )
    except KeyError as e:
        raise ContractError(f"system_boundary missing key: {e}") from e

    # 5) 组装 RouteConfig
    routes = {}
    for i, stage in enumerate(POLICY_STAGES):
        routes[stage] = RouteConfig(
            route_id=stage,
            chunk_vdb_top_k=int(vectors["chunk_vdb_top_k"][i]),
            entity_vdb_top_k=int(vectors["entity_vdb_top_k"][i]),
            relation_vdb_top_k=int(vectors["relation_vdb_top_k"][i]),
            entity_description_cap=int(vectors["entity_description_cap"][i]),
            relation_description_cap=int(vectors["relation_description_cap"][i]),
            source_text_cap=int(vectors["source_text_cap"][i]),
            final_context_hard_cap=int(vectors["final_context_hard_cap"][i]),
        )

    return ExperimentContract(
        contract_version=data["contract_version"],
        trace_schema_version=data["trace_schema_version"],
        success_rule_version=data["success_rule_version"],
        metric_version=data["metric_version"],
        cost_schema_version=data["cost_schema_version"],
        system_boundary=boundary,
        routes=routes,
        section_allocation_ratios=ratios,
        raw=data,
    )
