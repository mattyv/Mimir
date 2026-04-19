"""Phase 2 — PropertyRepository tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Property
from mimir.persistence.repository import EntityRepository, PropertyRepository

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


def _seed_entity(repo: EntityRepository, entity_id: str = "svc_001") -> None:
    repo.upsert(
        Entity(
            id=entity_id,
            type="auros:TradingService",
            name=f"Service {entity_id}",
            description="",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
    )


def _property(entity_id: str = "svc_001", key: str = "schema:name", value: Any = "Acme") -> Property:
    return Property(
        entity_id=entity_id,
        key=key,
        value=value,
        value_type="str",
        confidence=0.9,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def test_property_insert_returns_id(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(EntityRepository(pg))
    prop_id = PropertyRepository(pg).insert(_property())
    assert isinstance(prop_id, int)
    assert prop_id > 0


def test_property_list_for_entity_returns_inserted(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(EntityRepository(pg))
    repo = PropertyRepository(pg)
    repo.insert(_property(key="schema:name", value="Alpha"))
    repo.insert(_property(key="schema:url", value="https://example.com"))
    rows = repo.list_for_entity("svc_001")
    assert len(rows) == 2
    keys = {r["key"] for r in rows}
    assert {"schema:name", "schema:url"} == keys


def test_property_list_empty_for_unknown_entity(pg: psycopg.Connection[Any]) -> None:
    rows = PropertyRepository(pg).list_for_entity("nonexistent")
    assert rows == []


def test_property_list_as_of_active(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(EntityRepository(pg))
    repo = PropertyRepository(pg)
    repo.insert(Property(
        entity_id="svc_001",
        key="schema:name",
        value="Active",
        value_type="str",
        confidence=0.9,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(valid_from=_YESTERDAY, valid_until=_NOW + timedelta(days=1)),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    ))
    rows = repo.list_for_entity("svc_001", as_of=_NOW)
    assert len(rows) == 1


def test_property_list_as_of_excludes_expired(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(EntityRepository(pg))
    repo = PropertyRepository(pg)
    repo.insert(Property(
        entity_id="svc_001",
        key="schema:name",
        value="Old",
        value_type="str",
        confidence=0.9,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(valid_from=_YESTERDAY, valid_until=_YESTERDAY + timedelta(hours=1)),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    ))
    rows = repo.list_for_entity("svc_001", as_of=_NOW)
    assert rows == []


def test_property_list_at_version(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(EntityRepository(pg))
    repo = PropertyRepository(pg)
    p_id = repo.insert(_property(key="schema:name", value="v1"))
    row = pg.execute("SELECT graph_version FROM properties WHERE id = %s", (p_id,)).fetchone()
    assert row is not None
    written_version = row["graph_version"]
    repo.insert(_property(key="schema:url", value="later"))
    # at_version = written_version should see first but not necessarily second
    rows = repo.list_for_entity("svc_001", at_version=written_version)
    keys = {r["key"] for r in rows}
    assert "schema:name" in keys
