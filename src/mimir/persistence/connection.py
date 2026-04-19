"""Postgres connection pool management for Mimir."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool as ConnectionPool

_pool: ConnectionPool | None = None


def init_pool(conninfo: str, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """Initialise the module-level connection pool.

    Must be called once at application startup before any repository is used.
    Safe to call repeatedly — subsequent calls replace the pool (old pool is
    closed first).
    """
    global _pool
    if _pool is not None:
        _pool.close()
    _pool = ConnectionPool(
        conninfo,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=True,
    )
    return _pool


def get_pool() -> ConnectionPool:
    """Return the module-level pool, raising if not initialised."""
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialised. Call init_pool() before using repositories."
        )
    return _pool


@contextmanager
def transaction(
    pool: ConnectionPool | None = None,
    isolation: str = "read committed",
) -> Generator[psycopg.Connection[dict[str, Any]], None, None]:
    """Context manager that yields an open connection inside a transaction.

    Commits on clean exit, rolls back on exception.

    Args:
        pool: Pool to borrow from; defaults to the module-level pool.
        isolation: Postgres isolation level string.  Valid values:
            "read committed" (default, MCP reads),
            "repeatable read" (crawler writes),
            "serializable" (entity resolver).
    """
    p = pool or get_pool()
    with p.connection() as raw_conn:
        conn: psycopg.Connection[dict[str, Any]] = raw_conn  # type: ignore[assignment]
        conn.autocommit = False
        conn.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation.upper()}")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
