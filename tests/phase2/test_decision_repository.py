"""Phase 2 — DecisionRepository tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import Decision
from mimir.persistence.repository import DecisionRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_SOURCE = Source(type="confluence", reference="https://wiki.example.com/ADR-001", retrieved_at=_NOW)
_GROUNDING = Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test")
_TEMPORAL = Temporal(valid_from=_NOW)
_VISIBILITY = Visibility(acl=["internal"], sensitivity="internal")


def _decision(what: str = "Use Rust for panic_server") -> Decision:
    return Decision(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, what)),
        what=what,
        why="Performance and safety guarantees required for circuit breaker",
        tradeoffs=["Smaller talent pool", "Longer initial dev time"],
        when=_NOW,
        who=["alice", "bob"],
        source=_SOURCE,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )


def test_decision_upsert_returns_true_on_insert(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    assert repo.upsert(_decision()) is True


def test_decision_get_returns_row(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    d = _decision()
    repo.upsert(d)
    row = repo.get(d.id)
    assert row is not None
    assert row["what"] == "Use Rust for panic_server"


def test_decision_get_missing_returns_none(pg: psycopg.Connection[dict[str, Any]]) -> None:
    assert DecisionRepository(pg).get("missing") is None


def test_decision_upsert_updates_existing(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    d = _decision()
    repo.upsert(d)
    updated = Decision(
        id=d.id,
        what=d.what,
        why="Updated reason",
        tradeoffs=d.tradeoffs,
        when=d.when,
        who=d.who,
        source=_SOURCE,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )
    repo.upsert(updated)
    row = repo.get(d.id)
    assert row is not None
    assert row["why"] == "Updated reason"


def test_decision_list_active(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    repo.upsert(_decision("Use Rust for panic_server"))
    repo.upsert(_decision("Adopt gRPC for internal comms"))
    rows = repo.list_active()
    assert len(rows) == 2


def test_decision_list_active_ordered_newest_first(pg: psycopg.Connection[dict[str, Any]]) -> None:
    from datetime import timedelta
    repo = DecisionRepository(pg)
    old = Decision(
        id="old-decision",
        what="Old decision",
        why="reason",
        tradeoffs=[],
        when=_NOW - timedelta(days=30),
        who=["alice"],
        source=_SOURCE,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )
    new = Decision(
        id="new-decision",
        what="New decision",
        why="reason",
        tradeoffs=[],
        when=_NOW,
        who=["bob"],
        source=_SOURCE,
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )
    repo.upsert(old)
    repo.upsert(new)
    rows = repo.list_active()
    assert rows[0]["id"] == "new-decision"


def test_decision_delete(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    d = _decision()
    repo.upsert(d)
    assert repo.delete(d.id) is True
    assert repo.get(d.id) is None


def test_decision_delete_missing_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    assert DecisionRepository(pg).delete("ghost") is False


def test_decision_stores_tradeoffs(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = DecisionRepository(pg)
    d = _decision()
    repo.upsert(d)
    row = repo.get(d.id)
    assert row is not None
    assert "Smaller talent pool" in row["tradeoffs"]
