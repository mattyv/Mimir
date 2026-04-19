"""NetworkX in-memory graph projection from the Postgres persistence layer.

Builds a directed multigraph where:
  - Nodes are entity rows keyed by entity id.
  - Edges are relationship rows keyed by (subject_id, object_id, predicate).

The projection is read-only and point-in-time: pass as_of/at_version to
reproduce the graph at any historical moment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx
import psycopg

from mimir.persistence.graph_version import current_graph_version


def build_graph(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    as_of: datetime | None = None,
    at_version: int | None = None,
) -> nx.MultiDiGraph[Any]:
    """Build and return a NetworkX MultiDiGraph from active Postgres rows.

    Node attributes: all columns from the entities table.
    Edge attributes: all columns from the relationships table.
    Graph attributes: graph_version at projection time, as_of datetime.
    """
    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    entity_params: list[Any] = []
    entity_clauses: list[str] = []
    if as_of is None:
        entity_clauses.append("valid_until IS NULL")
    else:
        entity_clauses.append("valid_from <= %s AND (valid_until IS NULL OR valid_until > %s)")
        entity_params.extend([as_of, as_of])
    if at_version is not None:
        entity_clauses.append("graph_version <= %s")
        entity_params.append(at_version)

    entity_where = f"WHERE {' AND '.join(entity_clauses)}" if entity_clauses else ""
    entity_rows = conn.execute(
        f"SELECT * FROM entities {entity_where}",
        entity_params,
    ).fetchall()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    rel_params: list[Any] = []
    rel_clauses: list[str] = []
    if as_of is None:
        rel_clauses.append("valid_until IS NULL")
    else:
        rel_clauses.append("valid_from <= %s AND (valid_until IS NULL OR valid_until > %s)")
        rel_params.extend([as_of, as_of])
    if at_version is not None:
        rel_clauses.append("graph_version <= %s")
        rel_params.append(at_version)

    rel_where = f"WHERE {' AND '.join(rel_clauses)}" if rel_clauses else ""
    rel_rows = conn.execute(
        f"SELECT * FROM relationships {rel_where}",
        rel_params,
    ).fetchall()

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    g.graph["graph_version"] = current_graph_version(conn)
    g.graph["as_of"] = as_of

    for row in entity_rows:
        attrs = dict(row)
        g.add_node(attrs["id"], **attrs)

    for row in rel_rows:
        attrs = dict(row)
        g.add_edge(
            attrs["subject_id"],
            attrs["object_id"],
            key=attrs["predicate"],
            **attrs,
        )

    return g


def subgraph_for_entity(
    g: nx.MultiDiGraph[Any],
    entity_id: str,
    *,
    depth: int = 1,
) -> nx.MultiDiGraph[Any]:
    """Return the ego-graph centred on *entity_id* up to *depth* hops."""
    if entity_id not in g:
        return nx.MultiDiGraph()
    nodes = nx.ego_graph(g, entity_id, radius=depth, undirected=True).nodes()
    return g.subgraph(nodes).copy()
