"""Phase 2 — tests for resolution_queue, source_saturation, and audit_log tables."""
from __future__ import annotations

from typing import Any

import psycopg
import pytest

pytestmark = pytest.mark.phase2


def test_resolution_queue_table_exists(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT to_regclass('public.resolution_queue') AS t"
    ).fetchone()
    assert row is not None and row["t"] is not None


def test_source_saturation_table_exists(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT to_regclass('public.source_saturation') AS t"
    ).fetchone()
    assert row is not None and row["t"] is not None


def test_audit_log_table_exists(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT to_regclass('public.audit_log') AS t"
    ).fetchone()
    assert row is not None and row["t"] is not None
