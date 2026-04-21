"""Phase 10 — retraction worker tests."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from mimir.temporal.retraction import (
    RetractionResult,
    SourceChecker,
    list_active_source_refs,
    retract_by_source,
    scan_and_retract,
)

pytestmark = pytest.mark.phase10

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_SOURCE_REF = "https://wiki.example.com/spaces/trading-eng/OMMS-overview"


def _insert_entity(
    conn: psycopg.Connection[dict[str, Any]],
    name: str,
    source_ref: str = _SOURCE_REF,
) -> str:
    """Insert a minimal entity row with a source reference in the payload."""
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:auros:TradingService"))
    payload = {
        "source": {
            "type": "confluence",
            "reference": source_ref,
            "retrieved_at": _NOW.isoformat(),
        }
    }
    conn.execute(
        """
        INSERT INTO entities
            (id, entity_type, name, name_normalized, description, confidence,
             valid_from, valid_until, vocabulary_version, payload, graph_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name_normalized, entity_type) DO UPDATE
            SET payload = EXCLUDED.payload
        """,
        (
            eid,
            "auros:TradingService",
            name,
            name.casefold().strip(),
            "",
            1.0,
            _NOW,
            None,
            "0.1.0",
            json.dumps(payload),
            0,
        ),
    )
    return eid


def test_retract_by_source_expires_entity(pg: psycopg.Connection[Any]) -> None:
    eid = _insert_entity(pg, "retract_svc")
    result = retract_by_source(_SOURCE_REF, pg)
    assert isinstance(result, RetractionResult)
    assert result.source_ref == _SOURCE_REF
    assert result.entities_expired >= 1
    row = pg.execute("SELECT valid_until FROM entities WHERE id = %s", (eid,)).fetchone()
    assert row is not None and row["valid_until"] is not None


def test_retract_nonexistent_source_noop(pg: psycopg.Connection[Any]) -> None:
    result = retract_by_source("https://does-not-exist.example.com/page", pg)
    assert result.entities_expired == 0
    assert result.relationships_expired == 0
    assert result.properties_expired == 0


def test_list_active_source_refs(pg: psycopg.Connection[Any]) -> None:
    _insert_entity(pg, "list_svc", source_ref=_SOURCE_REF)
    refs = list_active_source_refs(pg)
    assert _SOURCE_REF in refs


def test_scan_and_retract_calls_checker(pg: psycopg.Connection[Any]) -> None:
    _insert_entity(pg, "scan_svc", source_ref=_SOURCE_REF)

    class _AlwaysGone:
        def exists(self, reference: str) -> bool:
            return False

    checker: SourceChecker = _AlwaysGone()
    results = scan_and_retract([_SOURCE_REF], checker, pg)
    assert len(results) == 1
    assert results[0].source_ref == _SOURCE_REF
    assert results[0].entities_expired >= 1
