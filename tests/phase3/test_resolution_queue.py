"""Phase 3 — resolution review queue tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository
from mimir.resolution.queue import enqueue_candidate, get_pending, reject_pair

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_VOCAB = "0.1.0"


def _grounding() -> Grounding:
    return Grounding(tier=GroundingTier.source_cited, depth=1, stop_reason="test")


def _make_entity(conn: psycopg.Connection[dict[str, Any]], suffix: str) -> str:
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"queue_test:{suffix}"))
    EntityRepository(conn).upsert(
        Entity(
            id=eid,
            type="schema:Organization",
            name=f"Queue Test {suffix}",
            description="",
            created_at=_NOW,
            confidence=1.0,
            grounding=_grounding(),
            temporal=Temporal(valid_from=_NOW),
            visibility=Visibility(acl=["internal"], sensitivity="internal"),
            vocabulary_version=_VOCAB,
        )
    )
    return eid


@pytest.mark.phase3
def test_enqueue_candidate_inserts_row(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    a = _make_entity(pg, "enqueue_a")
    b = _make_entity(pg, "enqueue_b")

    row_id = enqueue_candidate(a, b, 0.87, pg)
    assert row_id > 0

    pending = get_pending(pg)
    ids = {e.id for e in pending}
    assert row_id in ids


@pytest.mark.phase3
def test_enqueue_canonical_ordering(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    # Create two entities whose IDs have a known lexicographic order
    a = _make_entity(pg, "canon_a")
    b = _make_entity(pg, "canon_b")

    # Enqueue with b first (reversed order)
    bigger, smaller = (b, a) if b > a else (a, b)
    enqueue_candidate(bigger, smaller, 0.88, pg)

    pending = get_pending(pg)
    assert len(pending) >= 1
    entry = next(e for e in pending if e.similarity == pytest.approx(0.88, abs=1e-4))
    # Canonical: a_id < b_id
    assert entry.entity_a_id < entry.entity_b_id


@pytest.mark.phase3
def test_enqueue_duplicate_is_noop(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    a = _make_entity(pg, "dup_a")
    b = _make_entity(pg, "dup_b")

    first = enqueue_candidate(a, b, 0.90, pg)
    assert first > 0

    second = enqueue_candidate(a, b, 0.90, pg)
    assert second == 0

    # Only one row should exist
    pending = get_pending(pg)
    matching = [e for e in pending if e.id == first]
    assert len(matching) == 1


@pytest.mark.phase3
def test_reject_pair(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    a = _make_entity(pg, "reject_a")
    b = _make_entity(pg, "reject_b")

    row_id = enqueue_candidate(a, b, 0.83, pg)
    assert row_id > 0

    result = reject_pair(row_id, pg)
    assert result is True

    # Should not appear in pending
    pending = get_pending(pg)
    ids = {e.id for e in pending}
    assert row_id not in ids

    # Status should be 'rejected'
    row = pg.execute(
        "SELECT status FROM resolution_queue WHERE id = %s", (row_id,)
    ).fetchone()
    assert row is not None
    assert row["status"] == "rejected"


@pytest.mark.phase3
def test_get_pending_filters_resolved(
    pg: psycopg.Connection[dict[str, Any]],
) -> None:
    a = _make_entity(pg, "filt_a")
    b = _make_entity(pg, "filt_b")
    c = _make_entity(pg, "filt_c")

    id1 = enqueue_candidate(a, b, 0.86, pg)
    id2 = enqueue_candidate(a, c, 0.87, pg)
    id3 = enqueue_candidate(b, c, 0.88, pg)

    # Reject the first one
    reject_pair(id1, pg)

    pending = get_pending(pg)
    pending_ids = {e.id for e in pending}
    assert id1 not in pending_ids
    assert id2 in pending_ids
    assert id3 in pending_ids
