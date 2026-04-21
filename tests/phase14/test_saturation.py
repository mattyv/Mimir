"""Phase 14 — source saturation tracking tests (§10.1)."""

from __future__ import annotations

import os
from typing import Any

import psycopg
import pytest

from mimir.observability.saturation import (
    SaturationRecord,
    get_saturation,
    is_saturated,
    record_saturation,
)

pytestmark = pytest.mark.phase14

_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")


def _conn() -> psycopg.Connection[Any]:
    return psycopg.connect(_DSN, row_factory=psycopg.rows.dict_row, autocommit=True)


def test_record_saturation_inserts_row(_pg_schema: None) -> None:
    with _conn() as conn:
        record_saturation("confluence", "https://wiki/page1", 10, 5, conn)
        row = conn.execute(
            "SELECT saturation_pct FROM source_saturation WHERE source_ref = 'https://wiki/page1'"
        ).fetchone()
        assert row is not None
        # 5/15 ≈ 0.3333
        assert abs(float(row["saturation_pct"]) - 0.3333) < 0.01
        # Cleanup
        conn.execute("DELETE FROM source_saturation WHERE source_ref = 'https://wiki/page1'")


def test_get_saturation_returns_records(_pg_schema: None) -> None:
    with _conn() as conn:
        record_saturation("slack", "https://slack/channel1", 2, 18, conn)
        records = get_saturation(conn, source_type="slack")
        assert any(r.source_ref == "https://slack/channel1" for r in records)
        r = next(r for r in records if r.source_ref == "https://slack/channel1")
        assert r.saturation_pct == pytest.approx(0.9, abs=0.01)
        assert isinstance(r, SaturationRecord)
        # Cleanup
        conn.execute("DELETE FROM source_saturation WHERE source_ref = 'https://slack/channel1'")


def test_is_saturated_below_threshold(_pg_schema: None) -> None:
    with _conn() as conn:
        record_saturation("github", "repo/abc", 50, 50, conn)
        assert is_saturated("github", "repo/abc", conn, threshold=0.99) is False
        conn.execute("DELETE FROM source_saturation WHERE source_ref = 'repo/abc'")


def test_is_saturated_above_threshold(_pg_schema: None) -> None:
    with _conn() as conn:
        record_saturation("confluence", "wiki/xyz", 0, 100, conn)
        assert is_saturated("confluence", "wiki/xyz", conn, threshold=0.95) is True
        conn.execute("DELETE FROM source_saturation WHERE source_ref = 'wiki/xyz'")


def test_is_saturated_no_record(_pg_schema: None) -> None:
    with _conn() as conn:
        assert is_saturated("confluence", "nonexistent/ref", conn) is False
