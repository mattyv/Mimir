"""Phase 3 — embedder tests."""

from __future__ import annotations

import math

# ── helpers ────────────────────────────────────────────────────────────────────
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.resolution.embedder import EMBEDDING_DIM, compute_embedding, update_entity_embedding
from tests.conftest import FakeEmbedder


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _temporal(*, offset_days: int = 0) -> Temporal:
    base = datetime(2026, 4, 19, tzinfo=UTC) - timedelta(days=offset_days)
    return Temporal(valid_from=base)


def _visibility() -> Visibility:
    return Visibility(acl=["internal"], sensitivity="internal")


def _entity(entity_id: str, name: str = "Test Entity") -> Entity:
    return Entity(
        id=entity_id,
        type="schema:Organization",
        name=name,
        description="",
        created_at=datetime(2026, 4, 19, tzinfo=UTC),
        confidence=1.0,
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.phase3
def test_compute_embedding_returns_correct_dim(fake_embedder: FakeEmbedder) -> None:
    vec = compute_embedding("risk engine", fake_embedder)
    assert len(vec) == EMBEDDING_DIM


@pytest.mark.phase3
def test_compute_embedding_is_unit_vector(fake_embedder: FakeEmbedder) -> None:
    vec = compute_embedding("order book", fake_embedder)
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.phase3
def test_compute_embedding_deterministic(fake_embedder: FakeEmbedder) -> None:
    text = "options market maker"
    vecs = [compute_embedding(text, fake_embedder) for _ in range(5)]
    assert all(v == vecs[0] for v in vecs[1:])


@pytest.mark.phase3
def test_compute_embedding_different_texts_differ(fake_embedder: FakeEmbedder) -> None:
    v1 = compute_embedding("risk engine", fake_embedder)
    v2 = compute_embedding("trading service", fake_embedder)
    assert v1 != v2


@pytest.mark.phase3
def test_compute_embedding_wrong_dim_raises() -> None:
    class _SmallEmbedder:
        def encode(self, text: str) -> list[float]:
            return [0.5] * 10

    with pytest.raises(ValueError, match="Expected 384"):
        compute_embedding("test", _SmallEmbedder())


@pytest.mark.phase3
def test_update_entity_embedding_stores(pg: psycopg.Connection[dict[str, Any]]) -> None:
    from mimir.persistence.repository import EntityRepository

    repo = EntityRepository(pg)
    repo.upsert(_entity("emb_test_1"))

    embedding = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    result = update_entity_embedding("emb_test_1", embedding, pg)

    assert result is True
    row = pg.execute(
        "SELECT embedding IS NOT NULL AS has_emb FROM entities WHERE id = %s",
        ("emb_test_1",),
    ).fetchone()
    assert row is not None and row["has_emb"]


@pytest.mark.phase3
def test_update_entity_embedding_missing_entity_returns_false(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    embedding = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    result = update_entity_embedding("does_not_exist", embedding, pg)
    assert result is False
