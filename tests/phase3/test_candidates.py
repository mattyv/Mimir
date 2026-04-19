"""Phase 3 — candidate detection tests."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository
from mimir.resolution.candidates import find_merge_candidates, find_similar_by_embedding
from mimir.resolution.embedder import EMBEDDING_DIM, update_entity_embedding

# ── helpers ────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_DIM = EMBEDDING_DIM


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _temporal() -> Temporal:
    return Temporal(valid_from=_NOW)


def _visibility() -> Visibility:
    return Visibility(acl=["internal"], sensitivity="internal")


def _entity(entity_id: str, name: str, entity_type: str = "schema:Organization") -> Entity:
    return Entity(
        id=entity_id,
        type=entity_type,
        name=name,
        description="",
        created_at=_NOW,
        confidence=1.0,
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def _uniform_vec() -> list[float]:
    """384-dim unit vector with all equal components."""
    v = 1.0 / math.sqrt(_DIM)
    return [v] * _DIM


def _alt_vec() -> list[float]:
    """384-dim unit vector orthogonal to _uniform_vec (alternating sign)."""
    v = 1.0 / math.sqrt(_DIM)
    return [v if i % 2 == 0 else -v for i in range(_DIM)]


def _insert_with_embedding(
    pg: psycopg.Connection[dict[str, Any]],
    entity_id: str,
    name: str,
    vec: list[float],
    entity_type: str = "schema:Organization",
) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity(entity_id, name, entity_type=entity_type))
    update_entity_embedding(entity_id, vec, pg)


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.phase3
def test_find_similar_empty_when_no_embeddings(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("cand_no_emb_1", "Alpha"))
    repo.upsert(_entity("cand_no_emb_2", "Beta"))

    results = find_similar_by_embedding(
        _uniform_vec(), "schema:Organization", pg, threshold=0.5
    )
    assert results == []


@pytest.mark.phase3
def test_find_similar_finds_identical_vector(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "cand_sim_1", "ACME Corp", vec=vec)
    _insert_with_embedding(pg, "cand_sim_2", "Acme Corp Ltd", vec=vec)

    results = find_similar_by_embedding(vec, "schema:Organization", pg, threshold=0.99)
    ids = {r["id"] for r in results}
    assert "cand_sim_2" in ids


@pytest.mark.phase3
def test_find_similar_excludes_self(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "cand_excl_1", "Self Entity", vec=vec)
    _insert_with_embedding(pg, "cand_excl_2", "Other Entity", vec=vec)

    results = find_similar_by_embedding(
        vec, "schema:Organization", pg, exclude_id="cand_excl_1", threshold=0.99
    )
    ids = {r["id"] for r in results}
    assert "cand_excl_1" not in ids
    assert "cand_excl_2" in ids


@pytest.mark.phase3
def test_find_similar_threshold_filters_orthogonal(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    _insert_with_embedding(pg, "cand_thr_1", "Query Entity", vec=_uniform_vec())
    _insert_with_embedding(pg, "cand_thr_2", "Orthogonal Entity", vec=_alt_vec())

    results = find_similar_by_embedding(
        _uniform_vec(), "schema:Organization", pg, threshold=0.5
    )
    ids = {r["id"] for r in results}
    assert "cand_thr_2" not in ids


@pytest.mark.phase3
def test_find_similar_excludes_different_type(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "cand_type_1", "Org Entity", entity_type="schema:Organization", vec=vec)
    _insert_with_embedding(pg, "cand_type_2", "Person Entity", entity_type="schema:Person", vec=vec)

    results = find_similar_by_embedding(vec, "schema:Organization", pg, threshold=0.0)
    ids = {r["id"] for r in results}
    assert "cand_type_2" not in ids


@pytest.mark.phase3
def test_find_merge_candidates_finds_pair(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "mc_1", "Options Maker", vec=vec)
    _insert_with_embedding(pg, "mc_2", "Options MM", vec=vec)

    candidates = find_merge_candidates(pg, threshold=0.99)
    pairs = {(c.entity_a_id, c.entity_b_id) for c in candidates}
    assert ("mc_1", "mc_2") in pairs or ("mc_2", "mc_1") in pairs


@pytest.mark.phase3
def test_find_merge_candidates_empty_without_embeddings(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    repo = EntityRepository(pg)
    repo.upsert(_entity("mc_no_emb_1", "Alpha"))
    repo.upsert(_entity("mc_no_emb_2", "Beta"))

    candidates = find_merge_candidates(pg, threshold=0.5)
    assert candidates == []


@pytest.mark.phase3
def test_find_merge_candidates_type_filter(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "mc_tf_1", "OrgA", entity_type="schema:Organization", vec=vec)
    _insert_with_embedding(pg, "mc_tf_2", "OrgB", entity_type="schema:Organization", vec=vec)
    _insert_with_embedding(pg, "mc_tf_3", "PersonA", entity_type="schema:Person", vec=vec)
    _insert_with_embedding(pg, "mc_tf_4", "PersonB", entity_type="schema:Person", vec=vec)

    org_cands = find_merge_candidates(pg, entity_type="schema:Organization", threshold=0.99)
    assert all(
        pg.execute("SELECT entity_type FROM entities WHERE id = %s", (c.entity_a_id,))
        .fetchone()["entity_type"] == "schema:Organization"
        for c in org_cands
    )


@pytest.mark.phase3
def test_find_merge_candidates_similarity_value(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    vec = _uniform_vec()
    _insert_with_embedding(pg, "mc_sv_1", "Alpha", vec=vec)
    _insert_with_embedding(pg, "mc_sv_2", "Beta", vec=vec)

    candidates = find_merge_candidates(pg, threshold=0.99)
    matching = [c for c in candidates if set((c.entity_a_id, c.entity_b_id)) == {"mc_sv_1", "mc_sv_2"}]
    assert len(matching) == 1
    assert abs(matching[0].similarity - 1.0) < 1e-4
    assert matching[0].method == "embedding"
