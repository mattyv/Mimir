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


@dataclass
class ObservabilityDimensions:
    """Seven observable dimensions of graph coverage quality (§10.3)."""

    breadth: int = 0            # entity count
    depth: float = 0.0          # avg shortest-path length (proxy for structural depth)
    connectivity: float = 0.0   # edge_count / node_count ratio
    process_density: float = 0.0    # processes per entity
    decision_density: float = 0.0   # decisions per entity
    constraint_density: float = 0.0 # constraints per entity
    observation_density: float = 0.0  # observations per entity


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


def observability_dimensions(
    graph: nx.MultiDiGraph[Any],
    *,
    process_count: int = 0,
    decision_count: int = 0,
    constraint_count: int = 0,
    observation_count: int = 0,
) -> ObservabilityDimensions:
    """Compute the seven observable dimensions of graph coverage (§10.3).

    The graph-structural dimensions (breadth, depth, connectivity) are derived
    from the NetworkX graph.  The density dimensions require external counts
    passed in by the caller (queried from the DB separately).
    """
    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    if n == 0:
        return ObservabilityDimensions()

    # Depth proxy: mean of all-pairs shortest path would be O(n³); use avg
    # eccentricity of a sample instead.  For large graphs, estimate via BFS
    # from a random subset.
    try:
        undirected = graph.to_undirected()
        if nx.is_connected(undirected) and n <= 500:
            depth = nx.average_shortest_path_length(undirected)
        else:
            # Use average of eccentricities of nodes with highest degree (fast proxy)
            top_nodes = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)[:10]
            lengths: list[float] = []
            for src in top_nodes:
                sp = nx.single_source_shortest_path_length(graph, src)
                if sp:
                    lengths.append(sum(sp.values()) / len(sp))
            depth = sum(lengths) / len(lengths) if lengths else 0.0
    except Exception:
        depth = 0.0

    return ObservabilityDimensions(
        breadth=n,
        depth=round(depth, 4),
        connectivity=round(e / n, 4) if n > 0 else 0.0,
        process_density=round(process_count / n, 4) if n > 0 else 0.0,
        decision_density=round(decision_count / n, 4) if n > 0 else 0.0,
        constraint_density=round(constraint_count / n, 4) if n > 0 else 0.0,
        observation_density=round(observation_count / n, 4) if n > 0 else 0.0,
    )


def target_entity_count(
    cynefin_domain: str,
    *,
    regularity_factor: float = 1.0,
    depth_factor: float = 1.0,
) -> int:
    """Estimate target entity count for a sub-domain (§10.2).

    Base counts per Cynefin domain:
      clear=50, complicated=150, complex=300, chaotic=500, confused=100

    Multiplied by regularity_factor (0.5–2.0) and depth_factor (0.5–2.0).
    """
    _BASES = {
        "clear": 50,
        "complicated": 150,
        "complex": 300,
        "chaotic": 500,
        "confused": 100,
    }
    base = _BASES.get(cynefin_domain, 100)
    return max(1, round(base * regularity_factor * depth_factor))
