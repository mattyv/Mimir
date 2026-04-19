"""Phase 8 — graph complexity analyzer tests."""

from __future__ import annotations

import networkx as nx
import pytest

from mimir.complexity.metrics import (
    EntityMetrics,
    GraphMetrics,
    entity_metrics,
    graph_metrics,
    top_cascade_risk,
)


def _chain(n: int) -> nx.MultiDiGraph[str]:
    """Build a linear chain A→B→C→… of length n."""
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    nodes = [f"n{i}" for i in range(n)]
    g.add_nodes_from(nodes)
    for i in range(n - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def _star(spokes: int) -> nx.MultiDiGraph[str]:
    """Hub with `spokes` outgoing edges."""
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    g.add_node("hub")
    for i in range(spokes):
        g.add_node(f"spoke_{i}")
        g.add_edge("hub", f"spoke_{i}")
    return g


def _cycle(n: int) -> nx.MultiDiGraph[str]:
    """Directed cycle of length n."""
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    nodes = [f"c{i}" for i in range(n)]
    g.add_nodes_from(nodes)
    for i in range(n):
        g.add_edge(nodes[i], nodes[(i + 1) % n])
    return g


# ── entity_metrics ────────────────────────────────────────────────────────────


@pytest.mark.phase8
def test_entity_metrics_unknown_node() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    m = entity_metrics(g, "missing")
    assert m.entity_id == "missing"
    assert m.fan_in == 0 and m.fan_out == 0 and m.cascade_risk == 0.0


@pytest.mark.phase8
def test_entity_metrics_chain_head() -> None:
    g = _chain(5)
    m = entity_metrics(g, "n0")
    assert m.fan_out == 1
    assert m.fan_in == 0
    assert m.degree == 1
    # n0 can reach n1..n4 = 4 out of 4 other nodes
    assert m.cascade_risk == 1.0


@pytest.mark.phase8
def test_entity_metrics_chain_tail() -> None:
    g = _chain(5)
    m = entity_metrics(g, "n4")
    assert m.fan_in == 1
    assert m.fan_out == 0
    assert m.cascade_risk == 0.0


@pytest.mark.phase8
def test_entity_metrics_star_hub() -> None:
    g = _star(6)
    m = entity_metrics(g, "hub")
    assert m.fan_out == 6
    assert m.fan_in == 0
    assert m.cascade_risk == 1.0


@pytest.mark.phase8
def test_entity_metrics_spoke_no_cascade() -> None:
    g = _star(4)
    m = entity_metrics(g, "spoke_0")
    assert m.fan_in == 1
    assert m.fan_out == 0
    assert m.cascade_risk == 0.0


@pytest.mark.phase8
def test_entity_metrics_isolated_node() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    g.add_node("solo")
    g.add_node("other")
    m = entity_metrics(g, "solo")
    assert m.degree == 0
    assert m.cascade_risk == 0.0


# ── graph_metrics ─────────────────────────────────────────────────────────────


@pytest.mark.phase8
def test_graph_metrics_empty() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    m = graph_metrics(g)
    assert isinstance(m, GraphMetrics)
    assert m.node_count == 0 and m.edge_count == 0


@pytest.mark.phase8
def test_graph_metrics_chain() -> None:
    g = _chain(4)
    m = graph_metrics(g)
    assert m.node_count == 4
    assert m.edge_count == 3
    assert not m.has_cycles
    assert m.strongly_connected_components == 4


@pytest.mark.phase8
def test_graph_metrics_cycle_detected() -> None:
    g = _cycle(3)
    m = graph_metrics(g)
    assert m.has_cycles
    assert m.strongly_connected_components == 1


@pytest.mark.phase8
def test_graph_metrics_high_coupling_nodes() -> None:
    g = _star(6)
    m = graph_metrics(g, coupling_threshold=5)
    assert "hub" in m.high_coupling_nodes


@pytest.mark.phase8
def test_graph_metrics_density_range() -> None:
    g = _chain(5)
    m = graph_metrics(g)
    assert 0.0 <= m.density <= 1.0


@pytest.mark.phase8
def test_graph_metrics_avg_degree_positive() -> None:
    g = _chain(4)
    m = graph_metrics(g)
    assert m.avg_degree > 0.0


# ── top_cascade_risk ──────────────────────────────────────────────────────────


@pytest.mark.phase8
def test_top_cascade_risk_ordering() -> None:
    g = _chain(5)
    top = top_cascade_risk(g)
    assert top[0].entity_id == "n0"
    assert top[0].cascade_risk >= top[-1].cascade_risk


@pytest.mark.phase8
def test_top_cascade_risk_limit() -> None:
    g = _star(20)
    top = top_cascade_risk(g, limit=5)
    assert len(top) == 5


@pytest.mark.phase8
def test_top_cascade_risk_empty_graph() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    top = top_cascade_risk(g)
    assert top == []


@pytest.mark.phase8
def test_entity_metrics_dataclass_fields() -> None:
    m = EntityMetrics(entity_id="x", fan_in=2, fan_out=3, degree=5, cascade_risk=0.5)
    assert m.entity_id == "x"
    assert m.degree == 5
