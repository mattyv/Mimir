"""Phase 2 — ConstraintRepository tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Constraint, Entity
from mimir.persistence.repository import ConstraintRepository, EntityRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_SOURCE = Source(type="code_analysis", reference="code://test", retrieved_at=_NOW)
_GROUNDING = Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test")
_TEMPORAL = Temporal(valid_from=_NOW)
_VISIBILITY = Visibility(acl=["internal"], sensitivity="internal")


def _entity(entity_id: str = "ent_001", name: str = "TestSvc") -> Entity:
    return Entity(
        id=entity_id,
        type="auros:TradingService",
        name=name,
        description="",
        created_at=_NOW,
        confidence=0.9,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )


def _constraint(entity_id: str = "ent_001") -> Constraint:
    return Constraint(
        entity_id=entity_id,
        constraint_type="performance",
        condition="p99 latency",
        threshold={"ms": 5},
        source=_SOURCE,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )


@pytest.fixture
def seeded_entity(pg: psycopg.Connection[dict[str, Any]]) -> str:
    EntityRepository(pg).upsert(_entity())
    return "ent_001"


def test_constraint_insert_returns_id(
    pg: psycopg.Connection[dict[str, Any]], seeded_entity: str
) -> None:
    repo = ConstraintRepository(pg)
    cid = repo.insert(_constraint())
    assert cid > 0


def test_constraint_list_for_entity(
    pg: psycopg.Connection[dict[str, Any]], seeded_entity: str
) -> None:
    repo = ConstraintRepository(pg)
    repo.insert(_constraint())
    rows = repo.list_for_entity("ent_001")
    assert len(rows) == 1
    assert rows[0]["constraint_type"] == "performance"


def test_constraint_list_empty_for_unknown_entity(
    pg: psycopg.Connection[dict[str, Any]], seeded_entity: str
) -> None:
    repo = ConstraintRepository(pg)
    rows = repo.list_for_entity("no_such_entity")
    assert rows == []


def test_constraint_delete(pg: psycopg.Connection[dict[str, Any]], seeded_entity: str) -> None:
    repo = ConstraintRepository(pg)
    cid = repo.insert(_constraint())
    assert repo.delete(cid) is True
    assert repo.list_for_entity("ent_001") == []


def test_constraint_delete_missing_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ConstraintRepository(pg)
    assert repo.delete(99999) is False


def test_constraint_multiple_types(
    pg: psycopg.Connection[dict[str, Any]], seeded_entity: str
) -> None:
    repo = ConstraintRepository(pg)
    for ctype in ("performance", "availability", "legal", "physical", "social"):
        c = Constraint(
            entity_id="ent_001",
            constraint_type=ctype,  # type: ignore[arg-type]
            condition="test",
            threshold=None,
            source=_SOURCE,
            grounding=_GROUNDING,
            temporal=_TEMPORAL,
            visibility=_VISIBILITY,
            vocabulary_version="0.1.0",
        )
        repo.insert(c)
    rows = repo.list_for_entity("ent_001")
    assert len(rows) == 5
