"""Phase 8 — observability dimensions and target entity count tests."""

from __future__ import annotations

import networkx as nx
import pytest

from mimir.complexity.metrics import (
    ObservabilityDimensions,
    observability_dimensions,
    target_entity_count,
)

pytestmark = pytest.mark.phase8


def _chain(n: int) -> nx.MultiDiGraph[str]:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    nodes = [f"n{i}" for i in range(n)]
    g.add_nodes_from(nodes)
    for i in range(n - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def test_observability_dimensions_empty_graph() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    d = observability_dimensions(g)
    assert isinstance(d, ObservabilityDimensions)
    assert d.breadth == 0
    assert d.connectivity == 0.0


def test_observability_dimensions_breadth() -> None:
    g = _chain(5)
    d = observability_dimensions(g)
    assert d.breadth == 5


def test_observability_dimensions_connectivity() -> None:
    g = _chain(4)
    d = observability_dimensions(g)
    # 3 edges / 4 nodes = 0.75
    assert abs(d.connectivity - 0.75) < 0.01


def test_observability_dimensions_density_ratios() -> None:
    g = _chain(4)
    d = observability_dimensions(g, process_count=2, decision_count=1, constraint_count=3, observation_count=8)
    assert d.process_density == pytest.approx(0.5, abs=0.01)
    assert d.decision_density == pytest.approx(0.25, abs=0.01)
    assert d.constraint_density == pytest.approx(0.75, abs=0.01)
    assert d.observation_density == pytest.approx(2.0, abs=0.01)


def test_observability_dimensions_depth_nonnegative() -> None:
    g = _chain(5)
    d = observability_dimensions(g)
    assert d.depth >= 0.0


def test_target_entity_count_clear() -> None:
    assert target_entity_count("clear") == 50


def test_target_entity_count_complex() -> None:
    assert target_entity_count("complex") == 300


def test_target_entity_count_with_factors() -> None:
    # 150 * 2.0 * 1.5 = 450
    assert target_entity_count("complicated", regularity_factor=2.0, depth_factor=1.5) == 450


def test_target_entity_count_unknown_domain() -> None:
    # Unknown domain falls back to base=100
    assert target_entity_count("unknown_domain") == 100


def test_target_entity_count_minimum_one() -> None:
    result = target_entity_count("clear", regularity_factor=0.0001, depth_factor=0.0001)
    assert result >= 1
