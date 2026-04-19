"""MCP tool definitions — pure functions that handle tool calls.

Each tool function takes parsed arguments and a DB connection (or None for
pure-logic tools) and returns a JSON-serialisable dict.

The actual MCP server wiring lives in server.py; these functions are tested
independently without starting the server.
"""

from __future__ import annotations

from typing import Any

import psycopg

from mimir.complexity.metrics import graph_metrics
from mimir.cynefin.classifier import classify_entity
from mimir.permissions.acl import check_access
from mimir.persistence.repository import EntityRepository, ObservationRepository


def tool_get_entity(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Fetch a single entity by id, respecting ACL."""
    entity_id: str = args["entity_id"]
    repo = EntityRepository(conn)
    row = repo.get(entity_id)
    if row is None:
        return {"error": "not_found", "entity_id": entity_id}

    vis = row.get("payload", {}).get("visibility", {})
    decision = check_access(vis.get("acl", []), vis.get("sensitivity", "internal"), caller_groups)
    if not decision.allowed:
        return {"error": "forbidden", "entity_id": entity_id}

    return {"entity": row}


def tool_list_entities(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """List active entities, filtered by type and ACL."""
    entity_type: str | None = args.get("entity_type")
    limit: int = int(args.get("limit", 50))
    repo = EntityRepository(conn)
    rows = repo.list_active(entity_type=entity_type, limit=limit)

    visible = []
    for row in rows:
        vis = row.get("payload", {}).get("visibility", {})
        decision = check_access(vis.get("acl", []), vis.get("sensitivity", "internal"), caller_groups)
        if decision.allowed:
            visible.append(row)

    return {"entities": visible, "count": len(visible)}


def tool_classify_entity(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
) -> dict[str, Any]:
    """Return the Cynefin domain for an entity."""
    entity_id: str = args["entity_id"]
    result = classify_entity(entity_id, conn)
    return {
        "entity_id": result.entity_id,
        "domain": result.domain.value,
        "observation_count": result.observation_count,
        "relationship_count": result.relationship_count,
        "avg_confidence": result.avg_confidence,
    }


def tool_list_observations(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """List observations for an entity."""
    entity_id: str = args["entity_id"]
    obs_type: str | None = args.get("observation_type")
    repo = ObservationRepository(conn)
    rows = repo.list_for_entity(entity_id, observation_type=obs_type)

    visible = []
    for row in rows:
        vis = row.get("payload", {}).get("visibility", {})
        decision = check_access(vis.get("acl", []), vis.get("sensitivity", "internal"), caller_groups)
        if decision.allowed:
            visible.append(row)

    return {"observations": visible, "count": len(visible)}


def tool_graph_metrics(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
) -> dict[str, Any]:
    """Return whole-graph complexity metrics."""
    from mimir.persistence.graph_projection import build_graph

    graph = build_graph(conn)
    m = graph_metrics(graph)
    return {
        "node_count": m.node_count,
        "edge_count": m.edge_count,
        "density": m.density,
        "avg_degree": m.avg_degree,
        "has_cycles": m.has_cycles,
        "strongly_connected_components": m.strongly_connected_components,
        "high_coupling_nodes": m.high_coupling_nodes,
    }


# Registry: maps tool name → callable
TOOL_REGISTRY: dict[str, Any] = {
    "get_entity": tool_get_entity,
    "list_entities": tool_list_entities,
    "classify_entity": tool_classify_entity,
    "list_observations": tool_list_observations,
    "graph_metrics": tool_graph_metrics,
}
