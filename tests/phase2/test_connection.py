"""Phase 2 — Connection pool and transaction context manager tests."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.phase2

_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")


def test_init_pool_creates_pool() -> None:
    from mimir.persistence.connection import init_pool

    pool = init_pool(_DSN, min_size=1, max_size=2)
    assert pool is not None
    pool.close()


def test_get_pool_raises_before_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import mimir.persistence.connection as _conn_mod

    original = _conn_mod._pool
    _conn_mod._pool = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            _conn_mod.get_pool()
    finally:
        _conn_mod._pool = original


def test_init_pool_replaces_existing() -> None:
    from mimir.persistence.connection import get_pool, init_pool

    pool1 = init_pool(_DSN, min_size=1, max_size=2)
    pool2 = init_pool(_DSN, min_size=1, max_size=3)
    assert get_pool() is pool2
    assert pool1 is not pool2
    pool2.close()


def test_transaction_commits_on_clean_exit(_pg_schema: None) -> None:
    """A clean transaction exit commits changes visible to other connections."""
    from mimir.persistence.connection import init_pool, transaction

    pool = init_pool(_DSN, min_size=1, max_size=2)
    try:
        # Read initial graph_meta version
        with pool.connection() as conn:
            row = conn.execute("SELECT version FROM graph_meta WHERE id=1").fetchone()
            assert row is not None
            initial_version = row["version"]

        # Bump version inside a transaction
        with transaction(pool, isolation="read committed") as conn:
            conn.execute("UPDATE graph_meta SET version = version + 100 WHERE id = 1")

        # Verify the commit is visible to a fresh connection
        with pool.connection() as conn:
            row = conn.execute("SELECT version FROM graph_meta WHERE id=1").fetchone()
            assert row is not None
            assert row["version"] == initial_version + 100

        # Clean up the change we made (so test isolation is preserved)
        with transaction(pool, isolation="read committed") as conn:
            conn.execute("UPDATE graph_meta SET version = %s WHERE id = 1", (initial_version,))
    finally:
        pool.close()


def test_transaction_rollbacks_on_exception(_pg_schema: None) -> None:
    """An exception inside the transaction context manager triggers a rollback."""
    from mimir.persistence.connection import init_pool, transaction

    pool = init_pool(_DSN, min_size=1, max_size=2)
    try:
        with pool.connection() as conn:
            row = conn.execute("SELECT version FROM graph_meta WHERE id=1").fetchone()
            assert row is not None
            initial_version = row["version"]

        try:
            with transaction(pool, isolation="read committed") as conn:
                conn.execute("UPDATE graph_meta SET version = 9999 WHERE id = 1")
                raise ValueError("deliberate rollback")
        except ValueError:
            pass

        with pool.connection() as conn:
            row = conn.execute("SELECT version FROM graph_meta WHERE id=1").fetchone()
            assert row is not None
            assert row["version"] == initial_version, "Rollback should have reverted the change"
    finally:
        pool.close()


def test_transaction_isolation_level_set(_pg_schema: None) -> None:
    """transaction() sets the requested isolation level."""
    from mimir.persistence.connection import init_pool, transaction

    pool = init_pool(_DSN, min_size=1, max_size=2)
    try:
        with transaction(pool, isolation="repeatable read") as conn:
            row = conn.execute("SELECT current_setting('transaction_isolation') AS lvl").fetchone()
            assert row is not None
            assert "repeatable" in row["lvl"]
    finally:
        pool.close()
