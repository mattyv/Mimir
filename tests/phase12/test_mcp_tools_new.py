"""Phase 12 — new MCP tool tests (health, search fallback, offset, vocabulary, ground_axiom, redact)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.mcp.tools import (
    tool_get_entity,
    tool_get_vocabulary,
    tool_ground_axiom,
    tool_health,
    tool_list_entities,
    tool_search,
)
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_VOCAB = "0.1.0"
_INTERNAL = {"internal"}


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _source() -> Source:
    return Source(type="confluence", reference="https://example.com", retrieved_at=_NOW)


def _make_entity(
    conn: psycopg.Connection[dict[str, Any]],
    name: str,
    *,
    acl: list[str] | None = None,
    sensitivity: str = "internal",
) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService:new"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="auros:TradingService",
            name=name,
            description=f"desc for {name}",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=acl or ["internal"], sensitivity=sensitivity),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


# ── tool_health ────────────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_health_returns_version(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_health({}, pg)
    assert "graph_version" in result
    assert isinstance(result["graph_version"], int)
    assert result["graph_version"] >= 0
    assert "active_entity_count" in result
    assert isinstance(result["active_entity_count"], int)
    # last_update may be None if no bumps happened yet (graph_meta row exists)
    assert "last_update" in result


# ── tool_search ILIKE fallback ────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_search_fallback_ilike(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "IlikeSearchTarget")
    # No embedder supplied → must fall back to ILIKE
    result = tool_search({"query": "ilikesearch"}, pg, _INTERNAL, embedder=None)
    assert result["count"] >= 1
    names = [r["name"] for r in result["results"]]
    assert any("IlikeSearch" in n for n in names)


# ── tool_list_entities offset ─────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_list_entities_offset(pg: psycopg.Connection[dict[str, Any]]) -> None:
    # Create two entities with alphabetically ordered names so ordering is stable
    _make_entity(pg, "OffsetEntityA")
    _make_entity(pg, "OffsetEntityB")

    all_result = tool_list_entities(
        {"entity_type": "auros:TradingService", "limit": 50, "offset": 0},
        pg,
        _INTERNAL,
    )
    all_ids = [e["id"] for e in all_result["entities"]]

    assert len(all_ids) >= 2

    # offset=1 should skip the first result
    offset_result = tool_list_entities(
        {"entity_type": "auros:TradingService", "limit": 50, "offset": 1},
        pg,
        _INTERNAL,
    )
    offset_ids = [e["id"] for e in offset_result["entities"]]

    # The first item from the no-offset list should not appear as the first item
    # in the offset list (it is skipped)
    assert len(offset_ids) == len(all_ids) - 1
    assert offset_result["offset"] == 1


# ── tool_get_vocabulary ───────────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_get_vocabulary_returns_iris(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_get_vocabulary({}, pg)
    assert "entity_types" in result
    assert "predicates" in result
    assert isinstance(result["entity_types"], list)
    assert isinstance(result["predicates"], list)
    assert len(result["entity_types"]) > 0
    assert len(result["predicates"]) > 0
    assert "vocabulary_version" in result
    assert "entity_type_count" in result
    assert result["entity_type_count"] == len(result["entity_types"])


# ── tool_ground_axiom not found ───────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_ground_axiom_not_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_ground_axiom({"entity_id": "nonexistent-entity-id"}, pg, _INTERNAL)
    assert result["error"] == "not_found"
    assert result["entity_id"] == "nonexistent-entity-id"


# ── tool_get_entity redact_restricted ────────────────────────────────────────


@pytest.mark.phase12
def test_tool_get_entity_redact_restricted(pg: psycopg.Connection[dict[str, Any]]) -> None:
    # Create entity with internal sensitivity but restricted to a different ACL group
    eid = _make_entity(
        pg,
        "RedactInternalSvc",
        acl=["team:special"],
        sensitivity="internal",
    )

    # Caller group does not have access; redact_restricted=True should return stub
    result = tool_get_entity(
        {"entity_id": eid, "redact_restricted": True},
        pg,
        {"unrelated-group"},
    )

    assert "entity" in result
    entity = result["entity"]
    assert entity["name"] == "[REDACTED]"
    assert entity["redacted"] is True
    assert entity["id"] == eid
