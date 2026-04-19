"""Phase 12 — MCP tool tests (no live MCP server required)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.mcp.tools import (
    TOOL_REGISTRY,
    tool_classify_entity,
    tool_get_entity,
    tool_graph_metrics,
    tool_list_entities,
    tool_list_observations,
)
from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Observation
from mimir.persistence.repository import EntityRepository, ObservationRepository

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
    acl: list[str] | None = None,
) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService"))
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
            visibility=Visibility(acl=acl or ["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


def _make_obs(conn: psycopg.Connection[dict[str, Any]], entity_id: str, obs_type: str) -> None:
    ObservationRepository(conn).insert(
        Observation(
            entity_id=entity_id,
            type=obs_type,  # type: ignore[arg-type]
            description="test obs",
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )


# ── tool_get_entity ───────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_get_entity_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_get")
    result = tool_get_entity({"entity_id": eid}, pg, _INTERNAL)
    assert "entity" in result
    assert result["entity"]["id"] == eid


@pytest.mark.phase12
def test_get_entity_not_found(pg: psycopg.Connection[dict[str, Any]]) -> None:
    result = tool_get_entity({"entity_id": "ghost"}, pg, _INTERNAL)
    assert result["error"] == "not_found"


@pytest.mark.phase12
def test_get_entity_forbidden(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "svc_forbidden", acl=["team:secret"])
    result = tool_get_entity({"entity_id": eid}, pg, {"other-group"})
    assert result["error"] == "forbidden"


# ── tool_list_entities ────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_list_entities_returns_visible(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "list_svc")
    result = tool_list_entities({}, pg, _INTERNAL)
    ids = [e["id"] for e in result["entities"]]
    assert eid in ids
    assert result["count"] >= 1


@pytest.mark.phase12
def test_list_entities_type_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "typed_svc")
    result = tool_list_entities({"entity_type": "auros:TradingService"}, pg, _INTERNAL)
    assert result["count"] >= 1
    for e in result["entities"]:
        assert e["entity_type"] == "auros:TradingService"


@pytest.mark.phase12
def test_list_entities_acl_filters_out_restricted(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "restricted_svc", acl=["team:secret"])
    result = tool_list_entities({}, pg, {"unrelated-group"})
    ids = [e["id"] for e in result["entities"]]
    restricted_eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "restricted_svc:auros:TradingService"))
    assert restricted_eid not in ids


# ── tool_classify_entity ──────────────────────────────────────────────────────


@pytest.mark.phase12
def test_classify_entity_clear(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "clear_svc")
    result = tool_classify_entity({"entity_id": eid}, pg)
    assert result["entity_id"] == eid
    assert result["domain"] == "clear"
    assert "observation_count" in result


@pytest.mark.phase12
def test_classify_entity_complex(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "complex_svc")
    _make_obs(pg, eid, "risk")
    result = tool_classify_entity({"entity_id": eid}, pg)
    assert result["domain"] == "complex"


# ── tool_list_observations ────────────────────────────────────────────────────


@pytest.mark.phase12
def test_list_observations_returns_obs(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "obs_svc")
    _make_obs(pg, eid, "risk")
    result = tool_list_observations({"entity_id": eid}, pg, _INTERNAL)
    assert result["count"] == 1


@pytest.mark.phase12
def test_list_observations_type_filter(pg: psycopg.Connection[dict[str, Any]]) -> None:
    eid = _make_entity(pg, "obs_svc2")
    _make_obs(pg, eid, "risk")
    _make_obs(pg, eid, "maturity")
    result = tool_list_observations({"entity_id": eid, "observation_type": "risk"}, pg, _INTERNAL)
    assert result["count"] == 1
    assert result["observations"][0]["observation_type"] == "risk"


# ── tool_graph_metrics ────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_graph_metrics_returns_expected_fields(pg: psycopg.Connection[dict[str, Any]]) -> None:
    _make_entity(pg, "gm_svc")
    result = tool_graph_metrics({}, pg)
    assert "node_count" in result
    assert "edge_count" in result
    assert "has_cycles" in result
    assert result["node_count"] >= 1


# ── TOOL_REGISTRY ─────────────────────────────────────────────────────────────


@pytest.mark.phase12
def test_tool_registry_contains_expected_tools() -> None:
    expected = {"get_entity", "list_entities", "classify_entity", "list_observations", "graph_metrics"}
    assert expected <= set(TOOL_REGISTRY.keys())


@pytest.mark.phase12
def test_tool_registry_callables() -> None:
    for name, fn in TOOL_REGISTRY.items():
        assert callable(fn), f"{name} should be callable"
