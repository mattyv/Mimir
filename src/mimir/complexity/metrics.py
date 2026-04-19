"""Graph complexity metrics: fan-in, fan-out, coupling, cascade risk.

All functions operate on an in-memory NetworkX graph produced by
mimir.persistence.graph_projection.build_graph().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class EntityMetrics:
    entity_id: str
    fan_in: int = 0
    fan_out: int = 0
    degree: int = 0
    cascade_risk: float = 0.0  # fraction of graph reachable downstream


@dataclass
class GraphMetrics:
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    strongly_connected_components: int = 0
    has_cycles: bool = False
    high_coupling_nodes: list[str] = field(default_factory=list)


def entity_metrics(
    graph: nx.MultiDiGraph[Any],
    entity_id: str,
    *,
    coupling_threshold: int = 5,
) -> EntityMetrics:
    """Compute fan-in, fan-out, degree, and cascade risk for a single entity."""
    if entity_id not in graph:
        return EntityMetrics(entity_id=entity_id)

    fan_out = graph.out_degree(entity_id)
    fan_in = graph.in_degree(entity_id)
    degree = fan_in + fan_out

    # Cascade risk: fraction of nodes reachable from this node (excluding self)
    reachable = len(nx.descendants(graph, entity_id))
    total = graph.number_of_nodes() - 1
    cascade_risk = reachable / total if total > 0 else 0.0

    return EntityMetrics(
        entity_id=entity_id,
        fan_in=int(fan_in),
        fan_out=int(fan_out),
        degree=int(degree),
        cascade_risk=round(cascade_risk, 4),
    )


def graph_metrics(
    graph: nx.MultiDiGraph[Any],
    *,
    coupling_threshold: int = 5,
) -> GraphMetrics:
    """Compute whole-graph complexity metrics."""
    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    if n == 0:
        return GraphMetrics()

    # NetworkX density for directed graphs: e / (n*(n-1))
    density = nx.density(graph)
    avg_degree = (2 * e / n) if n > 0 else 0.0

    sccs = nx.number_strongly_connected_components(graph)
    has_cycles = not nx.is_directed_acyclic_graph(graph)

    high_coupling = [
        nid
        for nid in graph.nodes()
        if (graph.in_degree(nid) + graph.out_degree(nid)) >= coupling_threshold
    ]

    return GraphMetrics(
        node_count=n,
        edge_count=e,
        density=round(density, 6),
        avg_degree=round(avg_degree, 4),
        strongly_connected_components=sccs,
        has_cycles=has_cycles,
        high_coupling_nodes=sorted(high_coupling),
    )


def top_cascade_risk(
    graph: nx.MultiDiGraph[Any],
    *,
    limit: int = 10,
) -> list[EntityMetrics]:
    """Return the top-N entities by cascade risk (highest first)."""
    results = [entity_metrics(graph, nid) for nid in graph.nodes()]
    return sorted(results, key=lambda m: m.cascade_risk, reverse=True)[:limit]
