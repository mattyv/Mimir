"""Phase 2 — Concurrency and isolation level tests.

Tests that concurrent upserts on the same (name_normalized, entity_type) safely
converge to a single row rather than raising or inserting duplicates.
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.rows import dict_row

from mimir.models.base import Grounding, GroundingTier, Temporal, Visibility
from mimir.models.nodes import Entity
from mimir.persistence.repository import EntityRepository

pytestmark = pytest.mark.phase2

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")


def _make_entity(entity_id: str, name: str = "Shared Service") -> Entity:
    return Entity(
        id=entity_id,
        type="auros:TradingService",
        name=name,
        description="Concurrent entity",
        created_at=_NOW,
        confidence=0.9,
        grounding=Grounding(tier=GroundingTier.source_cited, depth=0, stop_reason="test"),
        temporal=Temporal(valid_from=_NOW),
        visibility=Visibility(acl=[], sensitivity="internal"),
        vocabulary_version="0.1.0",
    )


def test_concurrent_upsert_same_name_type(_pg_schema: None) -> None:
    """Concurrent upserts with the same (name_normalized, entity_type) produce exactly one row.

    Each thread opens its own connection and upserts the same logical entity.
    Because ON CONFLICT DO UPDATE is atomic, we expect exactly 1 row after all
    threads complete — not 0 (lost write) or N (duplicate rows).
    """
    n_threads = 4
    errors: list[Exception] = []

    # Record initial version so we can restore it after (upsert bumps graph_version)
    with psycopg.connect(_DSN, row_factory=dict_row) as _vc:
        init_ver: int = _vc.execute("SELECT version FROM graph_meta WHERE id=1").fetchone()[
            "version"
        ]  # type: ignore[index]

    def worker(i: int) -> None:
        try:
            with psycopg.connect(_DSN, row_factory=dict_row, autocommit=False) as conn:
                conn.execute("BEGIN")
                EntityRepository(conn).upsert(_make_entity(f"conc_shared_{i}", "Concurrent Shared"))
                conn.commit()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised errors: {errors}"

    # Verify exactly one row for this (name_normalized, entity_type) combination
    with psycopg.connect(_DSN, row_factory=dict_row) as check:
        row = check.execute(
            "SELECT COUNT(*) AS n FROM entities "
            "WHERE name_normalized = 'concurrent shared' AND entity_type = 'auros:TradingService'"
        ).fetchone()
        assert row is not None
        assert row["n"] == 1, f"Expected 1 entity row but got {row['n']}"

    # Cleanup: remove test entities and reset version to pre-test value
    with psycopg.connect(_DSN, row_factory=dict_row) as cleanup:
        cleanup.execute("DELETE FROM entities WHERE name_normalized = 'concurrent shared'")
        cleanup.execute("UPDATE graph_meta SET version = %s WHERE id = 1", (init_ver,))


def test_concurrent_graph_version_monotonic(_pg_schema: None) -> None:
    """Graph versions assigned by concurrent transactions are monotonically increasing.

    We collect all assigned versions and verify they form a set of unique
    integers — no two transactions received the same version.
    """
    n_threads = 5
    versions: list[int] = []
    lock = threading.Lock()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            with psycopg.connect(_DSN, row_factory=dict_row, autocommit=False) as conn:
                conn.execute("BEGIN")
                from mimir.persistence.graph_version import bump_graph_version

                v = bump_graph_version(conn)
                conn.commit()
                with lock:
                    versions.append(v)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised errors: {errors}"
    assert len(versions) == n_threads
    # All versions must be unique — the row-level lock on graph_meta serializes
    assert len(set(versions)) == n_threads, f"Duplicate versions detected: {sorted(versions)}"

    # Cleanup: reset graph_meta version to pre-test value
    min_ver = min(versions) - 1
    with psycopg.connect(_DSN, row_factory=dict_row) as cleanup:
        cleanup.execute("UPDATE graph_meta SET version = %s WHERE id = 1", (min_ver,))


def test_isolation_repeatable_read_sees_snapshot(_pg_schema: None) -> None:
    """A REPEATABLE READ transaction does not see changes committed after it started."""
    with psycopg.connect(_DSN, row_factory=dict_row, autocommit=False) as reader:
        reader.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        reader.execute("BEGIN")
        # Read initial state
        n_before = reader.execute(
            "SELECT COUNT(*) AS n FROM entities WHERE id LIKE 'iso_%'"
        ).fetchone()["n"]  # type: ignore[index]

        # Another connection inserts an entity after the reader opened its transaction
        with psycopg.connect(_DSN, row_factory=dict_row, autocommit=True) as writer:
            writer.execute(
                """
                INSERT INTO entities
                    (id, entity_type, name, name_normalized, description,
                     confidence, valid_from, vocabulary_version, payload, graph_version)
                VALUES ('iso_test_snap', 'auros:TradingService', 'ISO Test Snap', 'iso test snap',
                        '', 1.0, NOW(), '0.1.0', '{}', 0)
                ON CONFLICT DO NOTHING
                """
            )

        # Reader should still see the old count (snapshot isolation)
        n_after = reader.execute(
            "SELECT COUNT(*) AS n FROM entities WHERE id LIKE 'iso_%'"
        ).fetchone()["n"]  # type: ignore[index]
        reader.rollback()

    assert n_before == n_after, (
        f"REPEATABLE READ snapshot broken: before={n_before}, after={n_after}"
    )

    # Cleanup inserted row
    with psycopg.connect(_DSN, row_factory=dict_row) as cleanup:
        cleanup.execute("DELETE FROM entities WHERE id = 'iso_test_snap'")
