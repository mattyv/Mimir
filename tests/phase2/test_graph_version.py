"""Phase 2 — graph_version counter tests."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from mimir.persistence.graph_version import bump_graph_version, current_graph_version

pytestmark = pytest.mark.phase2


def test_version_is_nonnegative(pg: psycopg.Connection[Any]) -> None:
    """Graph version is always a non-negative integer."""
    assert current_graph_version(pg) >= 0


def test_bump_increments_by_one(pg: psycopg.Connection[Any]) -> None:
    before = current_graph_version(pg)
    v = bump_graph_version(pg)
    assert v == before + 1


def test_bump_twice_increments_twice(pg: psycopg.Connection[Any]) -> None:
    before = current_graph_version(pg)
    bump_graph_version(pg)
    v2 = bump_graph_version(pg)
    assert v2 == before + 2


def test_current_after_bump(pg: psycopg.Connection[Any]) -> None:
    before = current_graph_version(pg)
    bump_graph_version(pg)
    assert current_graph_version(pg) == before + 1


def test_current_does_not_increment(pg: psycopg.Connection[Any]) -> None:
    v1 = current_graph_version(pg)
    v2 = current_graph_version(pg)
    assert v1 == v2


def test_bump_missing_sentinel_raises() -> None:
    """bump_graph_version raises RuntimeError when sentinel row is absent."""
    import psycopg
    from psycopg.rows import dict_row

    from mimir.persistence.schema import apply_schema

    with psycopg.connect(
        "dbname=mimir_test user=root", row_factory=dict_row, autocommit=False
    ) as conn:
        conn.execute("BEGIN")
        apply_schema(conn)
        conn.execute("DELETE FROM graph_meta")
        with pytest.raises(RuntimeError, match="graph_meta sentinel"):
            bump_graph_version(conn)
        conn.rollback()


def test_current_missing_sentinel_raises() -> None:
    """current_graph_version raises RuntimeError when sentinel row is absent."""
    import psycopg
    from psycopg.rows import dict_row

    from mimir.persistence.schema import apply_schema

    with psycopg.connect(
        "dbname=mimir_test user=root", row_factory=dict_row, autocommit=False
    ) as conn:
        conn.execute("BEGIN")
        apply_schema(conn)
        conn.execute("DELETE FROM graph_meta")
        with pytest.raises(RuntimeError, match="graph_meta sentinel"):
            current_graph_version(conn)
        conn.rollback()
