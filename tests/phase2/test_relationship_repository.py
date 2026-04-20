"""Phase 2 — RelationshipRepository tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Relationship
from mimir.persistence.repository import EntityRepository, RelationshipRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_YESTERDAY = _NOW - timedelta(days=1)


def _source() -> Source:
    return Source(type="github", reference="ref://x", retrieved_at=_NOW)  # type: ignore[return-value,arg-type]


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test")


def _temporal(valid_from: datetime = _NOW, valid_until: datetime | None = None) -> Temporal:
    return Temporal(valid_from=valid_from, valid_until=valid_until)


def _visibility() -> Visibility:
    return Visibility(acl=[], sensitivity="internal")


def _seed_entity(conn: psycopg.Connection[Any], entity_id: str, name: str) -> None:
    EntityRepository(conn).upsert(
        Entity(
            id=entity_id,
            type="auros:TradingService",
            name=name,
            description="",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
    )


def _relationship(subject_id: str = "svc_a", object_id: str = "svc_b") -> Relationship:
    return Relationship(
        subject_id=subject_id,
        predicate="auros:dependsOn",
        object_id=object_id,
        confidence=0.8,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def test_relationship_insert_returns_id(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "svc_a", "Alpha")
    _seed_entity(pg, "svc_b", "Beta")
    rel_id = RelationshipRepository(pg).insert(_relationship())
    assert isinstance(rel_id, int)
    assert rel_id > 0


def test_relationship_list_for_subject(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "svc_a", "Alpha")
    _seed_entity(pg, "svc_b", "Beta")
    _seed_entity(pg, "svc_c", "Gamma")
    repo = RelationshipRepository(pg)
    repo.insert(_relationship("svc_a", "svc_b"))
    repo.insert(_relationship("svc_a", "svc_c"))
    rows = repo.list_for_subject("svc_a")
    assert len(rows) == 2
    assert all(r["subject_id"] == "svc_a" for r in rows)


def test_relationship_list_for_object(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "svc_a", "Alpha")
    _seed_entity(pg, "svc_b", "Beta")
    _seed_entity(pg, "svc_c", "Gamma")
    repo = RelationshipRepository(pg)
    repo.insert(_relationship("svc_a", "svc_b"))
    repo.insert(_relationship("svc_c", "svc_b"))
    rows = repo.list_for_object("svc_b")
    assert len(rows) == 2
    assert all(r["object_id"] == "svc_b" for r in rows)


def test_relationship_list_empty_when_none(pg: psycopg.Connection[Any]) -> None:
    rows = RelationshipRepository(pg).list_for_subject("nonexistent")
    assert rows == []


def test_relationship_list_as_of_excludes_expired(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "svc_a", "Alpha")
    _seed_entity(pg, "svc_b", "Beta")
    repo = RelationshipRepository(pg)
    repo.insert(
        Relationship(
            subject_id="svc_a",
            predicate="auros:dependsOn",
            object_id="svc_b",
            confidence=0.8,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(valid_from=_YESTERDAY, valid_until=_YESTERDAY + timedelta(hours=1)),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
    )
    rows = repo.list_for_subject("svc_a", as_of=_NOW)
    assert rows == []


def test_relationship_cascade_delete_with_entity(pg: psycopg.Connection[Any]) -> None:
    """Deleting an entity cascades to its relationships."""
    _seed_entity(pg, "svc_a", "Alpha")
    _seed_entity(pg, "svc_b", "Beta")
    RelationshipRepository(pg).insert(_relationship("svc_a", "svc_b"))
    EntityRepository(pg).delete("svc_a")
    rows = RelationshipRepository(pg).list_for_subject("svc_a")
    assert rows == []
