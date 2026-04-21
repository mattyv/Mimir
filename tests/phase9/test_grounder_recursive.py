"""Phase 9 — recursive Wikidata anchoring + bootstrap threshold tests."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from mimir.grounder.wikidata import find_ancestor_qids
from mimir.resolution.candidates import get_thresholds
from tests.conftest import FakeSPARQL


def _ancestor_response(*ancestors: tuple[str, str]) -> dict[str, Any]:
    """Build a SPARQL result dict with the given (qid, label) ancestor pairs."""
    bindings = [
        {
            "ancestor": {"value": f"http://www.wikidata.org/entity/{qid}"},
            "ancestorLabel": {"value": label},
        }
        for qid, label in ancestors
    ]
    return {"results": {"bindings": bindings}}


# ── find_ancestor_qids ────────────────────────────────────────────────────────


@pytest.mark.phase9
def test_find_ancestor_qids_empty_result() -> None:
    """Client returns no bindings -> empty list."""
    client = FakeSPARQL()  # default: all queries return empty bindings
    result = find_ancestor_qids("Q42", client)
    assert result == []


@pytest.mark.phase9
def test_find_ancestor_qids_basic() -> None:
    """Client returns 2 ancestors at depth 0 -> list of 2 tuples."""
    client = FakeSPARQL()
    from mimir.grounder.wikidata import _ANCESTOR_QUERY_TEMPLATE

    qid_uri = "http://www.wikidata.org/entity/Q42"
    query = _ANCESTOR_QUERY_TEMPLATE.format(qid_uri=qid_uri)
    client.set_response(query, _ancestor_response(("Q5", "human"), ("Q215627", "person")))

    result = find_ancestor_qids("Q42", client)
    assert len(result) == 2
    qids = [r[0] for r in result]
    labels = [r[1] for r in result]
    assert "Q5" in qids
    assert "Q215627" in qids
    assert "human" in labels
    assert "person" in labels


@pytest.mark.phase9
def test_find_ancestor_qids_depth_cap() -> None:
    """At depth_cap, returns empty list immediately."""
    client = FakeSPARQL()
    result = find_ancestor_qids("Q42", client, depth_cap=4, _depth=4)
    assert result == []


@pytest.mark.phase9
def test_find_ancestor_qids_cycle_detection() -> None:
    """Ancestor already in _seen -> not revisited."""
    client = FakeSPARQL()
    from mimir.grounder.wikidata import _ANCESTOR_QUERY_TEMPLATE

    qid_uri = "http://www.wikidata.org/entity/Q42"
    query = _ANCESTOR_QUERY_TEMPLATE.format(qid_uri=qid_uri)
    # Return Q42 itself as an ancestor (cycle) plus a fresh one
    client.set_response(
        query,
        _ancestor_response(("Q42", "self-loop"), ("Q5", "human")),
    )

    seen: set[str] = {"Q42"}
    result = find_ancestor_qids("Q42", client, _seen=seen)

    qids = [r[0] for r in result]
    # Q42 is already in _seen, so it must NOT appear again
    assert "Q42" not in qids
    # Q5 is fresh, so it should appear
    assert "Q5" in qids


# ── get_thresholds ────────────────────────────────────────────────────────────


@pytest.mark.phase9
def test_get_thresholds_bootstrap_mode(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    """0 entities -> bootstrap thresholds (0.92, 0.80)."""
    auto_merge, review = get_thresholds(pg)
    assert auto_merge == pytest.approx(0.92)
    assert review == pytest.approx(0.80)
