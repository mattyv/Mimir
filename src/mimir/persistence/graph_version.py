"""Graph version counter — atomic increment on every write transaction."""

from __future__ import annotations

from typing import Any

import psycopg


def bump_graph_version(conn: psycopg.Connection[dict[str, Any]]) -> int:
    """Atomically increment the graph version and return the new value.

    Must be called inside an open transaction.  The UPDATE acquires a
    row-level lock on the single graph_meta row, so concurrent writers
    serialize here rather than silently skipping a version.
    """
    row = conn.execute(
        """
        UPDATE graph_meta
           SET version    = version + 1,
               updated_at = NOW()
         WHERE id = 1
        RETURNING version
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("graph_meta sentinel row is missing; run apply_schema() first")
    return int(row["version"])


def current_graph_version(conn: psycopg.Connection[dict[str, Any]]) -> int:
    """Return the current graph version without incrementing."""
    row = conn.execute("SELECT version FROM graph_meta WHERE id = 1").fetchone()
    if row is None:
        raise RuntimeError("graph_meta sentinel row is missing; run apply_schema() first")
    return int(row["version"])
