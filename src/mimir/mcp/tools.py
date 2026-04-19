"""MCP tool definitions — pure functions that handle tool calls.

Each tool function takes parsed arguments and a DB connection (or None for
pure-logic tools) and returns a JSON-serialisable dict.

The actual MCP server wiring lives in server.py; these functions are tested
independently without starting the server.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import networkx as nx
import psycopg

from mimir.authority.conflicts import detect_polarity_conflicts
from mimir.complexity.metrics import entity_metrics, graph_metrics
from mimir.cynefin.classifier import classify_entity
from mimir.permissions.acl import check_access
from mimir.persistence.repository import (
    EntityRepository,
    ObservationRepository,
)


def _acl_allowed(row: dict[str, Any], caller_groups: set[str]) -> bool:
    vis = row.get("payload", {}).get("visibility", {})
    return check_access(vis.get("acl", []), vis.get("sensitivity", "internal"), caller_groups).allowed


def _acl_filter(rows: list[dict[str, Any]], caller_groups: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if _acl_allowed(r, caller_groups)]


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
    visible = _acl_filter(rows, caller_groups)
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
    visible = _acl_filter(rows, caller_groups)
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


def tool_find_relationships(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Query relationships with any combination of subject/predicate/object filters."""
    subject_id: str | None = args.get("subject_id")
    predicate: str | None = args.get("predicate")
    object_id: str | None = args.get("object_id")
    limit: int = int(args.get("limit", 50))
    as_of: Any = args.get("as_of")

    if not any([subject_id, predicate, object_id]):
        return {"error": "at_least_one_filter_required"}

    where_clauses: list[str] = ["valid_until IS NULL"]
    params: list[Any] = []

    if as_of:
        where_clauses = ["valid_from <= %s AND (valid_until IS NULL OR valid_until > %s)"]
        params.extend([as_of, as_of])

    if subject_id:
        where_clauses.append("subject_id = %s")
        params.append(subject_id)
    if predicate:
        where_clauses.append("predicate = %s")
        params.append(predicate)
    if object_id:
        where_clauses.append("object_id = %s")
        params.append(object_id)

    rows = conn.execute(
        f"SELECT * FROM relationships WHERE {' AND '.join(where_clauses)} "
        f"ORDER BY confidence DESC LIMIT %s",
        params + [limit],
    ).fetchall()

    raw = [dict(r) for r in rows]
    truncated = len(raw) == limit

    # ACL filter using both endpoints' visibility
    e_repo = EntityRepository(conn)
    visible: list[dict[str, Any]] = []
    for row in raw:
        subj = e_repo.get(row["subject_id"])
        obj = e_repo.get(row["object_id"])
        subj_ok = subj is None or _acl_allowed(subj, caller_groups)
        obj_ok = obj is None or _acl_allowed(obj, caller_groups)
        if subj_ok and obj_ok:
            visible.append(row)

    return {"relationships": visible, "count": len(visible), "truncated": truncated}


def tool_get_neighborhood(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Return the subgraph around entity_id up to depth hops."""
    from mimir.persistence.graph_projection import build_graph, subgraph_for_entity

    entity_id: str = args["entity_id"]
    depth: int = min(int(args.get("depth", 2)), 3)
    predicate_filter: str | None = args.get("predicate")

    graph = build_graph(conn)
    subgraph = subgraph_for_entity(graph, entity_id, depth=depth)

    if predicate_filter:
        edges_to_remove = [
            (u, v, k) for u, v, k in subgraph.edges(keys=True)
            if k != predicate_filter
        ]
        subgraph.remove_edges_from(edges_to_remove)

    MAX_NODES = 500
    all_node_ids = list(subgraph.nodes())
    truncated = len(all_node_ids) > MAX_NODES
    node_ids_capped = all_node_ids[:MAX_NODES]

    nodes_raw = [dict(subgraph.nodes[n]) for n in node_ids_capped]
    visible_nodes = _acl_filter(nodes_raw, caller_groups)
    visible_ids = {n["id"] for n in visible_nodes}

    edges = [
        {"subject": u, "predicate": k, "object": v}
        for u, v, k in subgraph.edges(keys=True)
        if u in visible_ids and v in visible_ids
    ]

    return {
        "center": entity_id,
        "depth": depth,
        "nodes": visible_nodes,
        "edges": edges,
        "truncated": truncated,
    }


def tool_entity_cascade_risk(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Return cascade risk for a single entity, plus downstream entities."""
    from mimir.persistence.graph_projection import build_graph

    entity_id: str = args["entity_id"]
    graph = build_graph(conn)

    if entity_id not in graph:
        return {"error": "not_found", "entity_id": entity_id}

    metrics = entity_metrics(graph, entity_id)
    descendants = list(nx.descendants(graph, entity_id))

    visible_descendants = [
        d for d in descendants
        if d in graph.nodes and _acl_allowed(dict(graph.nodes[d]), caller_groups)
    ]

    return {
        "entity_id": entity_id,
        "cascade_risk": metrics.cascade_risk,
        "fan_in": metrics.fan_in,
        "fan_out": metrics.fan_out,
        "downstream_count": len(visible_descendants),
        "downstream_entities": visible_descendants[:50],
    }


def tool_search(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Full-text search entities by name (embedding search when available)."""
    query: str = args["query"]
    types: list[str] | None = args.get("types")
    limit: int = min(int(args.get("limit", 20)), 50)

    # Name-based fallback search (embedding search requires embedder injection)
    type_clause = "AND entity_type = ANY(%s)" if types else ""
    type_params: list[Any] = [types] if types else []

    rows = conn.execute(
        f"""
        SELECT * FROM entities
         WHERE valid_until IS NULL
           AND (name_normalized ILIKE %s OR description ILIKE %s)
           {type_clause}
         ORDER BY name
         LIMIT %s
        """,
        [f"%{query.lower()}%", f"%{query}%"] + type_params + [limit],
    ).fetchall()

    raw = [dict(r) for r in rows]
    visible = _acl_filter(raw, caller_groups)
    capped = visible[:limit]
    return {"results": capped, "count": len(capped), "query": query}


def tool_get_contradictions(
    args: dict[str, Any],
    conn: psycopg.Connection[dict[str, Any]],
    caller_groups: set[str],
) -> dict[str, Any]:
    """Return polarity conflicts, optionally scoped to an entity."""
    entity_id: str | None = args.get("entity_id")
    conflicts = detect_polarity_conflicts(conn, subject_id=entity_id)
    return {"contradictions": [dataclasses.asdict(c) for c in conflicts]}


# Registry: maps tool name → callable
TOOL_REGISTRY: dict[str, Any] = {
    "get_entity": tool_get_entity,
    "list_entities": tool_list_entities,
    "classify_entity": tool_classify_entity,
    "list_observations": tool_list_observations,
    "graph_metrics": tool_graph_metrics,
    "find_relationships": tool_find_relationships,
    "get_neighborhood": tool_get_neighborhood,
    "entity_cascade_risk": tool_entity_cascade_risk,
    "search": tool_search,
    "get_contradictions": tool_get_contradictions,
}
