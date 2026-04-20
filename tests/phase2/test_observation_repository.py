"""Phase 2 — ObservationRepository tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation
from mimir.persistence.repository import EntityRepository, ObservationRepository

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


def _seed_entity(conn: psycopg.Connection[Any], entity_id: str = "svc_001") -> None:
    EntityRepository(conn).upsert(
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


def _observation(
    obs_type: str = "risk",
    entity_id: str = "svc_001",
    valid_from: datetime = _NOW,
    valid_until: datetime | None = None,
) -> Observation:
    return Observation(
        entity_id=entity_id,
        type=obs_type,  # type: ignore[arg-type]
        description="A test observation",
        confidence=0.75,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(valid_from=valid_from, valid_until=valid_until),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )


def test_observation_insert_returns_id(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg)
    obs_id = ObservationRepository(pg).insert(_observation())
    assert isinstance(obs_id, int)
    assert obs_id > 0


def test_observation_list_for_entity(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg)
    repo = ObservationRepository(pg)
    repo.insert(_observation("risk"))
    repo.insert(_observation("strength"))
    rows = repo.list_for_entity("svc_001")
    assert len(rows) == 2


def test_observation_list_filter_by_type(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg)
    repo = ObservationRepository(pg)
    repo.insert(_observation("risk"))
    repo.insert(_observation("strength"))
    rows = repo.list_for_entity("svc_001", observation_type="risk")
    assert len(rows) == 1
    assert rows[0]["observation_type"] == "risk"


def test_observation_list_empty_for_unknown_entity(pg: psycopg.Connection[Any]) -> None:
    rows = ObservationRepository(pg).list_for_entity("nonexistent")
    assert rows == []


def test_observation_list_as_of_excludes_expired(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg)
    repo = ObservationRepository(pg)
    repo.insert(_observation(valid_from=_YESTERDAY, valid_until=_YESTERDAY + timedelta(hours=1)))
    rows = repo.list_for_entity("svc_001", as_of=_NOW)
    assert rows == []


def test_all_observation_types_storable(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg)
    repo = ObservationRepository(pg)
    for obs_type in (
        "strength",
        "risk",
        "anti_pattern",
        "maturity",
        "smell",
        "opportunity",
        "inconsistency",
        "functional_state",
    ):
        repo.insert(_observation(obs_type))
    rows = repo.list_for_entity("svc_001")
    assert len(rows) == 8
