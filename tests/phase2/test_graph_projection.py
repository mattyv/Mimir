"""Phase 2 — NetworkX graph projection tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Entity, Relationship
from mimir.persistence.graph_projection import build_graph, subgraph_for_entity
from mimir.persistence.repository import EntityRepository, RelationshipRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_YESTERDAY = _NOW - timedelta(days=1)


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


def _seed_rel(
    conn: psycopg.Connection[Any], subj: str, obj: str, pred: str = "auros:dependsOn"
) -> None:
    RelationshipRepository(conn).insert(
        Relationship(
            subject_id=subj,
            predicate=pred,
            object_id=obj,
            confidence=0.8,
            source=Source(type="github", reference="ref://x", retrieved_at=_NOW),  # type: ignore[arg-type]
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
    )


# ── build_graph ────────────────────────────────────────────────────────────────


def test_build_graph_empty_db(pg: psycopg.Connection[Any]) -> None:
    g = build_graph(pg)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_build_graph_nodes_from_entities(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    _seed_entity(pg, "b", "Beta")
    g = build_graph(pg)
    assert g.number_of_nodes() == 2
    assert "a" in g
    assert "b" in g


def test_build_graph_edges_from_relationships(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    _seed_entity(pg, "b", "Beta")
    _seed_rel(pg, "a", "b")
    g = build_graph(pg)
    assert g.number_of_edges() == 1
    assert g.has_edge("a", "b")


def test_build_graph_node_attributes(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    g = build_graph(pg)
    node_data = g.nodes["a"]
    assert node_data["name"] == "Alpha"
    assert node_data["entity_type"] == "auros:TradingService"


def test_build_graph_graph_version_attribute(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    g = build_graph(pg)
    assert "graph_version" in g.graph
    assert g.graph["graph_version"] >= 1


def test_build_graph_as_of_excludes_expired_entity(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "active", "Active")
    EntityRepository(pg).upsert(
        Entity(
            id="expired",
            type="auros:TradingService",
            name="Expired",
            description="",
            created_at=_NOW,
            confidence=0.9,
            grounding=_grounding(),
            temporal=Temporal(valid_from=_YESTERDAY, valid_until=_YESTERDAY + timedelta(hours=1)),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
    )
    g = build_graph(pg, as_of=_NOW)
    assert "active" in g
    assert "expired" not in g


def test_build_graph_at_version(pg: psycopg.Connection[Any]) -> None:
    from mimir.persistence.graph_version import current_graph_version

    _seed_entity(pg, "a", "Alpha")  # written at v_before+1
    v_a = current_graph_version(pg)
    _seed_entity(pg, "b", "Beta")  # written at v0+2
    g_va = build_graph(pg, at_version=v_a)  # snapshot at v_a: only "a"
    g_vab = build_graph(pg, at_version=v_a + 1)  # snapshot includes "b"
    assert "a" in g_va
    assert "b" not in g_va
    assert "b" in g_vab


# ── subgraph_for_entity ────────────────────────────────────────────────────────


def test_subgraph_for_entity_depth_1(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    _seed_entity(pg, "b", "Beta")
    _seed_entity(pg, "c", "Gamma")
    _seed_rel(pg, "a", "b")
    _seed_rel(pg, "b", "c")
    g = build_graph(pg)
    sg = subgraph_for_entity(g, "a", depth=1)
    assert "a" in sg
    assert "b" in sg
    assert "c" not in sg


def test_subgraph_for_entity_depth_2(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    _seed_entity(pg, "b", "Beta")
    _seed_entity(pg, "c", "Gamma")
    _seed_rel(pg, "a", "b")
    _seed_rel(pg, "b", "c")
    g = build_graph(pg)
    sg = subgraph_for_entity(g, "a", depth=2)
    assert "a" in sg
    assert "b" in sg
    assert "c" in sg


def test_subgraph_for_missing_entity_is_empty(pg: psycopg.Connection[Any]) -> None:
    g = build_graph(pg)
    sg = subgraph_for_entity(g, "nonexistent")
    assert sg.number_of_nodes() == 0


def test_subgraph_preserves_edge_attributes(pg: psycopg.Connection[Any]) -> None:
    _seed_entity(pg, "a", "Alpha")
    _seed_entity(pg, "b", "Beta")
    _seed_rel(pg, "a", "b", "auros:dependsOn")
    g = build_graph(pg)
    sg = subgraph_for_entity(g, "a", depth=1)
    edges = list(sg.edges(data=True, keys=True))
    assert len(edges) == 1
    _, _, key, data = edges[0]
    assert key == "auros:dependsOn"
    assert data["predicate"] == "auros:dependsOn"
