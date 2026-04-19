"""Phase 10 — temporal management tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Property, Relationship
from mimir.persistence.repository import (
    EntityRepository,
    PropertyRepository,
    RelationshipRepository,
)
from mimir.temporal.manager import (
    active_entities_at,
    expire_entity,
    expire_property,
    expire_relationship,
    expired_entities,
    supersede_entity,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 4, 19, tzinfo=UTC)
_T2 = datetime(2026, 6, 1, tzinfo=UTC)
_VOCAB = "0.1.0"


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _source() -> Source:
    return Source(type="confluence", reference="https://example.com", retrieved_at=_T1)


def _temporal(from_: datetime = _T0, until: datetime | None = None) -> Temporal:
    return Temporal(valid_from=from_, valid_until=until)


def _visibility() -> Visibility:
    return Visibility(acl=["internal"], sensitivity="internal")


def _make_entity(
    conn: psycopg.Connection[dict[str, Any]],
    name: str,
    valid_from: datetime = _T0,
) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="auros:TradingService",
            name=name,
            description="",
            created_at=_T1,
            confidence=0.9,
            grounding=_grounding(),
            temporal=_temporal(from_=valid_from),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


def _make_rel(
    conn: psycopg.Connection[dict[str, Any]],
    subj: str,
    obj: str,
) -> int:
    return RelationshipRepository(conn).insert(
        Relationship(
            subject_id=subj,
            predicate="auros:dependsOn",
            object_id=obj,
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )


def _make_prop(
    conn: psycopg.Connection[dict[str, Any]],
    entity_id: str,
) -> int:
    return PropertyRepository(conn).insert(
        Property(
            entity_id=entity_id,
            key="auros:hasOwner",
            value="team-a",
            value_type="str",
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version=_VOCAB,
        )
    )


# ── expire_entity ─────────────────────────────────────────────────────────────


@pytest.mark.phase10
def test_expire_entity_sets_valid_until(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_expire")
    assert expire_entity(eid, pg, at=_T2)
    row = pg.execute("SELECT valid_until FROM entities WHERE id = %s", (eid,)).fetchone()
    assert row is not None and row["valid_until"] is not None


@pytest.mark.phase10
def test_expire_entity_already_expired_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_already_expired")
    expire_entity(eid, pg, at=_T2)
    assert not expire_entity(eid, pg, at=_T2)


@pytest.mark.phase10
def test_expire_entity_missing_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    assert not expire_entity("nonexistent-id", pg)


# ── expire_relationship ───────────────────────────────────────────────────────


@pytest.mark.phase10
def test_expire_relationship(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "rel_a")
    b = _make_entity(pg, "rel_b")
    rid = _make_rel(pg, a, b)
    assert expire_relationship(rid, pg, at=_T2)
    row = pg.execute("SELECT valid_until FROM relationships WHERE id = %s", (rid,)).fetchone()
    assert row is not None and row["valid_until"] is not None


@pytest.mark.phase10
def test_expire_relationship_idempotent(pg: psycopg.Connection[dict[str, Any]]) -> None:
    a = _make_entity(pg, "rel_c")
    b = _make_entity(pg, "rel_d")
    rid = _make_rel(pg, a, b)
    expire_relationship(rid, pg)
    assert not expire_relationship(rid, pg)


# ── expire_property ───────────────────────────────────────────────────────────


@pytest.mark.phase10
def test_expire_property(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "prop_svc")
    pid = _make_prop(pg, eid)
    assert expire_property(pid, pg, at=_T2)
    row = pg.execute("SELECT valid_until FROM properties WHERE id = %s", (pid,)).fetchone()
    assert row is not None and row["valid_until"] is not None


# ── supersede_entity ──────────────────────────────────────────────────────────


@pytest.mark.phase10
def test_supersede_entity_sets_payload(pg: psycopg.Connection[dict[str, Any]]) -> None:
    old_id = _make_entity(pg, "old_svc")
    new_id = _make_entity(pg, "new_svc")
    assert supersede_entity(old_id, new_id, pg, at=_T2)
    row = pg.execute("SELECT payload, valid_until FROM entities WHERE id = %s", (old_id,)).fetchone()
    assert row is not None
    assert row["valid_until"] is not None
    assert row["payload"]["superseded_by"] == new_id


@pytest.mark.phase10
def test_supersede_entity_missing_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    assert not supersede_entity("ghost-id", "new-id", pg)


# ── active_entities_at ────────────────────────────────────────────────────────


@pytest.mark.phase10
def test_active_entities_at_includes_active(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "active_svc", valid_from=_T0)
    result = active_entities_at(pg, _T1)
    ids = [r["id"] for r in result]
    assert eid in ids


@pytest.mark.phase10
def test_active_entities_at_excludes_future(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "future_svc", valid_from=_T2)
    result = active_entities_at(pg, _T1)
    ids = [r["id"] for r in result]
    assert eid not in ids


@pytest.mark.phase10
def test_active_entities_at_excludes_expired(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "expired_svc", valid_from=_T0)
    expire_entity(eid, pg, at=_T1 - timedelta(days=1))
    result = active_entities_at(pg, _T1)
    ids = [r["id"] for r in result]
    assert eid not in ids


@pytest.mark.phase10
def test_active_entities_at_type_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "typed_svc")
    result = active_entities_at(pg, _T1, entity_type="auros:TradingService")
    ids = [r["id"] for r in result]
    assert eid in ids

    result2 = active_entities_at(pg, _T1, entity_type="schema:Organization")
    ids2 = [r["id"] for r in result2]
    assert eid not in ids2


# ── expired_entities ──────────────────────────────────────────────────────────


@pytest.mark.phase10
def test_expired_entities_lists_expired(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "going_away")
    expire_entity(eid, pg, at=_T2)
    result = expired_entities(pg)
    ids = [r["id"] for r in result]
    assert eid in ids


@pytest.mark.phase10
def test_expired_entities_excludes_active(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "staying_active")
    result = expired_entities(pg)
    ids = [r["id"] for r in result]
    assert eid not in ids
