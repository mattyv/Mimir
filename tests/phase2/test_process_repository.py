"""Phase 2 — ProcessRepository tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Process
from mimir.persistence.repository import ProcessRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_GROUNDING = Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test")
_TEMPORAL = Temporal(valid_from=_NOW)
_VISIBILITY = Visibility(acl=["internal"], sensitivity="internal")


def _process(name: str = "Order Lifecycle") -> Process:
    return Process(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
        name=name,
        stages=["receive", "validate", "route", "fill"],
        inputs=["order_request"],
        outputs=["fill_report"],
        slo="<5ms p99",
        grounding=_GROUNDING,
        temporal=_TEMPORAL,
        visibility=_VISIBILITY,
        vocabulary_version="0.1.0",
    )


def test_process_upsert_returns_true_on_insert(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    inserted = repo.upsert(_process())
    assert inserted is True


def test_process_get_returns_row(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    p = _process()
    repo.upsert(p)
    row = repo.get(p.id)
    assert row is not None
    assert row["name"] == "Order Lifecycle"


def test_process_get_missing_returns_none(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    assert repo.get("no_such_id") is None


def test_process_upsert_idempotent(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    p = _process()
    repo.upsert(p)
    repo.upsert(p)
    rows = repo.list_active()
    assert len(rows) == 1


def test_process_list_active(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    repo.upsert(_process("Order Lifecycle"))
    repo.upsert(_process("Risk Check"))
    rows = repo.list_active()
    assert len(rows) == 2


def test_process_delete(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    p = _process()
    repo.upsert(p)
    assert repo.delete(p.id) is True
    assert repo.get(p.id) is None


def test_process_delete_missing_returns_false(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    assert repo.delete("nonexistent") is False


def test_process_stores_stages(pg: psycopg.Connection[dict[str, Any]]) -> None:
    repo = ProcessRepository(pg)
    p = _process()
    repo.upsert(p)
    row = repo.get(p.id)
    assert row is not None
    assert row["stages"] == ["receive", "validate", "route", "fill"]
