"""Step 1.3 tests: split QueryParam retrieval params + P2 semantics.

Covers:
1. effective_* helpers correctly fall back to legacy fields when split
   fields are None, and return the split value when set.
2. P2 guard: ``relation_vdb_top_k=0`` makes ``_build_relation_query_context``
   return None WITHOUT ever calling the Relation VDB; a positive value does
   call it.
3. P2 semantics: even when ``relation_vdb_top_k=0`` (Relation VDB skipped),
   the entity line still produces Relationships via adjacency expansion
   (constrained by ``relation_description_cap``).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperrag.base import QueryParam
import hyperrag.query_context as qc


def _qp(**overrides) -> QueryParam:
    base = dict(
        mode="hyper",
        top_k=30,
        max_token_for_entity_context=1000,
        max_token_for_relation_context=1000,
        max_token_for_text_unit=1000,
        max_total_context_tokens=None,
    )
    base.update(overrides)
    return QueryParam(**base)


# --------------------------------------------------------------------------
# 1. effective_* helpers
# --------------------------------------------------------------------------
class TestEffectiveHelpers:
    def test_entity_fallback(self):
        assert _qp().effective_entity_top_k() == 30

    def test_entity_override(self):
        assert _qp(entity_vdb_top_k=25).effective_entity_top_k() == 25

    def test_relation_fallback(self):
        assert _qp().effective_relation_top_k() == 30

    def test_relation_override_and_zero(self):
        assert _qp(relation_vdb_top_k=50).effective_relation_top_k() == 50
        # 0 is a valid override (means "skip the line"), not a fallback trigger
        assert _qp(relation_vdb_top_k=0).effective_relation_top_k() == 0

    def test_chunk_fallback(self):
        assert _qp().effective_chunk_top_k() == 30

    def test_chunk_override(self):
        assert _qp(chunk_vdb_top_k=20).effective_chunk_top_k() == 20

    def test_entity_cap_fallback(self):
        assert _qp().effective_entity_cap() == 1000

    def test_entity_cap_override(self):
        assert _qp(entity_description_cap=150).effective_entity_cap() == 150

    def test_relation_cap_fallback(self):
        assert _qp().effective_relation_cap() == 1000

    def test_relation_cap_override(self):
        assert _qp(relation_description_cap=800).effective_relation_cap() == 800

    def test_source_cap_fallback(self):
        assert _qp().effective_source_cap() == 1000

    def test_source_cap_override(self):
        assert _qp(source_text_cap=4000).effective_source_cap() == 4000

    def test_final_cap_fallback_none(self):
        # legacy max_total_context_tokens=None -> effective_final_cap None
        assert _qp().effective_final_cap() is None

    def test_final_cap_override(self):
        assert _qp(final_context_hard_cap=7000).effective_final_cap() == 7000

    def test_final_cap_legacy_passthrough(self):
        # when final_context_hard_cap is None, legacy value is the fallback
        assert _qp(max_total_context_tokens=12000).effective_final_cap() == 12000


# --------------------------------------------------------------------------
# 2. P2 guard on the Relation VDB line
# --------------------------------------------------------------------------
class TestP2RelationGuard:
    def test_rel_vdb_top_k_zero_skips_vdb(self):
        rel_vdb = MagicMock()
        rel_vdb.query = AsyncMock()
        hg = MagicMock()
        hg.get_hyperedge = AsyncMock(return_value={})
        hg.hyperedge_degree = AsyncMock(return_value=1)
        q = _qp(relation_vdb_top_k=0, relation_description_cap=100000)

        result = asyncio.run(
            qc._build_relation_query_context(
                "kw", hg, MagicMock(), rel_vdb, MagicMock(), q
            )
        )
        assert result is None
        rel_vdb.query.assert_not_called()

    def test_rel_vdb_top_k_positive_queries(self):
        rel_vdb = MagicMock()
        rel_vdb.query = AsyncMock(
            return_value=[{"id_set": ("a", "b"), "weight": 1.0, "rank": 1}]
        )
        hg = MagicMock()
        hg.get_hyperedge = AsyncMock(
            return_value={"description": "d", "keywords": "k", "weight": 1.0}
        )
        hg.hyperedge_degree = AsyncMock(return_value=1)
        q = _qp(relation_vdb_top_k=10, relation_description_cap=100000)

        with patch.object(qc, "apply_type_aware_weighting", side_effect=lambda r, **kw: r), \
             patch.object(qc, "_find_most_related_entities_from_relationships", AsyncMock(return_value=([], 0))), \
             patch.object(qc, "_find_related_text_unit_from_relationships", AsyncMock(return_value=([], 0))):
            result = asyncio.run(
                qc._build_relation_query_context(
                    "kw", hg, MagicMock(), rel_vdb, MagicMock(), q
                )
            )
        assert result is not None
        rel_vdb.query.assert_called_once()


# --------------------------------------------------------------------------
# 3. P2 semantics: entity adjacency still yields Relationships
# --------------------------------------------------------------------------
class TestP2EntityAdjacency:
    def test_entity_line_generates_relations_when_rel_vdb_zero(self):
        ent_vdb = MagicMock()
        ent_vdb.query = AsyncMock(
            return_value=[{"entity_name": "E1", "rank": 1, "weight": 1.0}]
        )
        hg = MagicMock()
        hg.get_vertex = AsyncMock(
            return_value={
                "entity_name": "E1",
                "description": "desc",
                "entity_type": "X",
                "additional_properties": "{}",
            }
        )
        hg.vertex_degree = AsyncMock(return_value=1)

        q = _qp(
            relation_vdb_top_k=0,
            entity_vdb_top_k=25,
            entity_description_cap=100000,
            relation_description_cap=100000,
            source_text_cap=100000,
        )

        edges = [
            {
                "src_tgt": ("E1", "E2"),
                "description": "rel",
                "keywords": "k",
                "weight": 1.0,
                "rank": 1,
                "edge_type": "OTHER",
            }
        ]
        chunks = [{"content": "source text"}]
        with patch.object(qc, "apply_type_aware_weighting", side_effect=lambda r, **kw: r), \
             patch.object(qc, "_find_most_related_edges_from_entities", AsyncMock(return_value=(edges, 0))), \
             patch.object(qc, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=(chunks, 0))):
            result = asyncio.run(
                qc._build_entity_query_context("q", hg, ent_vdb, MagicMock(), q)
            )

        assert result is not None
        # P2: Relationships come from entity adjacency, not the skipped Relation VDB
        assert len(result["hyperedges"]) == 1
        assert len(result["entities"]) == 1
        assert "-----Relationships-----" in result["context"]

    def test_entity_line_respects_relation_cap(self):
        """Entity-adjacency relationships are bounded by relation_description_cap."""
        ent_vdb = MagicMock()
        ent_vdb.query = AsyncMock(
            return_value=[{"entity_name": "E1", "rank": 1, "weight": 1.0}]
        )
        hg = MagicMock()
        hg.get_vertex = AsyncMock(
            return_value={
                "entity_name": "E1",
                "description": "desc",
                "entity_type": "X",
                "additional_properties": "{}",
            }
        )
        hg.vertex_degree = AsyncMock(return_value=1)
        # one real neighbor edge so the un-patched adjacency finder runs truncation
        hg.get_nbr_e_of_vertex = AsyncMock(return_value=[["E1", "E2"]])
        hg.get_hyperedge = AsyncMock(
            return_value={"description": "rel", "keywords": "k", "weight": 1.0}
        )
        hg.hyperedge_degree = AsyncMock(return_value=1)

        # relation_description_cap=0 -> all adjacency relationships truncated away
        q = _qp(
            relation_vdb_top_k=0,
            entity_vdb_top_k=25,
            entity_description_cap=100000,
            relation_description_cap=0,
            source_text_cap=100000,
        )

        chunks = [{"content": "source text"}]
        with patch.object(qc, "apply_type_aware_weighting", side_effect=lambda r, **kw: r), \
             patch.object(qc, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=(chunks, 0))):
            result = asyncio.run(
                qc._build_entity_query_context("q", hg, ent_vdb, MagicMock(), q)
            )

        assert result is not None
        # cap=0 removes all adjacency relationships
        assert len(result["hyperedges"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
