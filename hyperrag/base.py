from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any, Tuple, List, Set, Optional, Dict

from .utils import EmbeddingFunc
from .tokenizer_calibration import qwen_cap_to_tiktoken_budget

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["hyper", "hyper-lite", "graph", "naive", "llm", "adaptive"] = "hyper"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    top_k: int = 60
    # Number of tokens for the original chunks.
    max_token_for_text_unit: int = 1600
    # Number of tokens for the entity descriptions
    max_token_for_entity_context: int = 300
    # Number of tokens for the relationship descriptions
    max_token_for_relation_context: int = 1600
    # Focus types from Query Router (used by adaptive mode for type-aware weighting)
    route_focus_types: Optional[List[str]] = None
    # Master switch for type-aware weighting (only active in adaptive mode)
    enable_type_aware_weighting: bool = False
    # Total context budget (post-merge fuse). None = no truncation (hyper default).
    # Adaptive mode sets this per complexity tier via adaptive_params.py.
    max_total_context_tokens: Optional[int] = None
    # Step 5.2: Router policy for adaptive mode
    # - "llm": call route_query (default, original behavior)
    # - "fixed": skip router, use forced_complexity
    # - "oracle": skip router, forced_complexity set per-question by Step_3
    router_policy: Literal["llm", "fixed", "oracle"] = "llm"
    # Forced complexity for fixed/oracle policies (simple/medium/complex)
    forced_complexity: Optional[str] = None
    # Max tokens for the final response generation (Step 5.3).
    # Only limits the final LLM answer call, not keyword extraction or Router.
    max_response_tokens: int = 3000
    # Step 6: Unified LLM temperature for keyword extraction, Router, and
    # final answer generation. A fixed value (not 0, to keep some diversity
    # while staying reproducible across runs) makes the retrieval pipeline
    # deterministic. Enters the LLM cache key, so old cached responses without
    # temperature are not reused.
    llm_temperature: float = 0.1
    # return type
    return_type: Literal["json", "text"] = "text"
    # Step 5.4: Trace support for reproducibility diagnostics.
    # When save_trace=True, query mode functions populate trace_data dict
    # with retrieval IDs, context hash, and effective params.
    save_trace: bool = False
    trace_data: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Step 1.3 (question-set v2.1): split retrieval parameters.
    # All fields default to None = fall back to the legacy coarse fields
    # (top_k / max_token_for_*), so existing hyper/naive/adaptive behavior
    # is unchanged unless a fixed route (P0-P4) explicitly sets them.
    # Values for P0-P4 MUST come from hyperrag.experiment_contract
    # (single source of truth) -- never hardcode them at call sites.
    # ------------------------------------------------------------------ #
    # candidate_budget: per-VDB top-k. 0 is meaningful (= skip that VDB
    # entirely); None = use legacy top_k.
    chunk_vdb_top_k: Optional[int] = None
    entity_vdb_top_k: Optional[int] = None
    relation_vdb_top_k: Optional[int] = None
    # pre_merge_caps: per-section token caps before merge.
    # None = use legacy max_token_for_entity_context /
    # max_token_for_relation_context / max_token_for_text_unit.
    entity_description_cap: Optional[int] = None
    relation_description_cap: Optional[int] = None
    source_text_cap: Optional[int] = None
    # post_merge_hard_control: final assembled-context hard cap (tokens).
    # None = use legacy max_total_context_tokens.
    final_context_hard_cap: Optional[int] = None
    # Experiment bookkeeping (threaded into trace records; no retrieval effect)
    route_id: Optional[str] = None            # "P0".."P4" / "P_gold"
    repeat_id: Optional[int] = None           # 0-based repeat index
    repeat_seed: Optional[int] = None         # seed for this repeat
    system_snapshot_id: Optional[str] = None  # caches/<data>/snapshots/<id>.json
    contract_version: Optional[str] = None    # e.g. question-set-v2.1-contract-v1
    # Step 1 收尾：区段预算配比（来自契约 section_allocation_ratios，小写键）。
    # None = context_budget 使用模块常量（向后兼容旧 adaptive 路径）。
    section_allocation_ratios: Optional[dict] = None

    # ---- effective-value helpers (None -> legacy fallback) ---- #
    def effective_chunk_top_k(self) -> int:
        return self.top_k if self.chunk_vdb_top_k is None else self.chunk_vdb_top_k

    def effective_entity_top_k(self) -> int:
        return self.top_k if self.entity_vdb_top_k is None else self.entity_vdb_top_k

    def effective_relation_top_k(self) -> int:
        return self.top_k if self.relation_vdb_top_k is None else self.relation_vdb_top_k

    def effective_entity_cap(self) -> int:
        return (self.max_token_for_entity_context
                if self.entity_description_cap is None
                else self.entity_description_cap)

    def effective_relation_cap(self) -> int:
        return (self.max_token_for_relation_context
                if self.relation_description_cap is None
                else self.relation_description_cap)

    def effective_source_cap(self) -> int:
        return (self.max_token_for_text_unit
                if self.source_text_cap is None
                else self.source_text_cap)

    def effective_final_cap(self) -> Optional[int]:
        return (self.max_total_context_tokens
                if self.final_context_hard_cap is None
                else self.final_context_hard_cap)

    # ---- Qwen-unit pre-merge caps -> tiktoken budgets (Step 1 收尾 · 问题 2) ---- #
    # 契约 units.token_unit = qwen_tokenizer_token：**拆分字段**
    # (entity_description_cap / relation_description_cap / source_text_cap)
    # 里的数值是 Qwen token，而热路径 truncate_list_by_token_size 用 tiktoken
    # 计数，必须先经校准保守转换（否则实际保留量偏大，P2-P4 执行参数偏离契约）。
    # legacy 字段 (max_token_for_*) 本来就是 tiktoken 口径的历史基线预算，
    # 原样返回 —— 不转换，保证 hyper / naive / adaptive 旧路径行为不变。
    def _tiktoken_cap(self, qwen_value: Optional[int], legacy_value: int) -> int:
        if qwen_value is None:
            return legacy_value
        return qwen_cap_to_tiktoken_budget(qwen_value)

    def tiktoken_entity_cap(self) -> int:
        return self._tiktoken_cap(
            self.entity_description_cap, self.max_token_for_entity_context)

    def tiktoken_relation_cap(self) -> int:
        return self._tiktoken_cap(
            self.relation_description_cap, self.max_token_for_relation_context)

    def tiktoken_source_cap(self) -> int:
        return self._tiktoken_cap(
            self.source_text_cap, self.max_token_for_text_unit)

    def caps_in_qwen_unit(self) -> bool:
        """是否有任一预合并 cap 来自契约（Qwen 单位）。"""
        return any(v is not None for v in (
            self.entity_description_cap,
            self.relation_description_cap,
            self.source_text_cap,
        ))


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError

"""
    The BaseHypergraphStorage based on hypergraph-DB
"""
@dataclass
class BaseHypergraphStorage(StorageNameSpace):
    async def has_vertex(self, v_id: Any) -> bool:
        raise NotImplementedError

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        raise NotImplementedError

    async def get_vertex(self, v_id: str, default: Any = None) :
        raise NotImplementedError

    async def get_hyperedge(self, e_tuple: Union[List, Set, Tuple], default: Any = None) :
        raise NotImplementedError

    async def get_all_vertices(self):
        raise NotImplementedError

    async def get_all_hyperedges(self):
        raise NotImplementedError

    async def get_num_of_vertices(self):
        raise NotImplementedError

    async def get_num_of_hyperedges(self):
        raise NotImplementedError

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def upsert_hyperedge(self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None) :
        raise NotImplementedError

    async def remove_vertex(self, v_id: Any) :
        raise NotImplementedError

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) :
        raise NotImplementedError

    async def vertex_degree(self, v_id: Any) -> int:
        raise NotImplementedError

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        raise NotImplementedError

    async def get_nbr_e_of_vertex(self, v_id: Any) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> list:
        raise NotImplementedError

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self=True) -> list:
        raise NotImplementedError
