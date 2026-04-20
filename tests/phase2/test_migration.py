"""Phase 2 — Alembic migration tests.

Verifies that the 0001 migration can be applied and reversed without error.
Each test starts with a completely clean database (no mimir tables, no
alembic_version) so alembic can run its DDL from scratch without colliding
with the session-scoped _pg_schema fixture.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

pytestmark = pytest.mark.phase2

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")

# Convert DATABASE_URL to the SQLAlchemy+psycopg dialect that alembic env.py expects.
_raw_db_url = os.environ.get("DATABASE_URL", "")
if _raw_db_url.startswith("postgresql://"):
    _alembic_url = _raw_db_url.replace("postgresql://", "postgresql+psycopg://", 1)
elif _raw_db_url.startswith("postgresql+psycopg://"):
    _alembic_url = _raw_db_url
else:
    _alembic_url = "postgresql+psycopg://root@/mimir_test"

_ENV = {**os.environ, "MIMIR_DATABASE_URL": _alembic_url}


def _run_alembic(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_PROJECT_ROOT,
        env=_ENV,
        capture_output=True,
        text=True,
    )


@pytest.fixture(autouse=True)
def _clean_for_migration() -> Iterator[None]:
    """Drop all mimir tables + alembic_version before each test; restore after.

    Migration tests need a clean slate so alembic can create tables from
    scratch without hitting DuplicateTable errors from the session-scoped
    _pg_schema fixture.  After each test the schema is restored via
    apply_schema() so subsequent non-migration tests can use the pg fixture.
    """
    from mimir.persistence.schema import apply_schema, drop_schema

    with psycopg.connect(_DSN, row_factory=dict_row, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS alembic_version")
        drop_schema(conn)

    yield

    with psycopg.connect(_DSN, row_factory=dict_row, autocommit=True) as conn:
        drop_schema(conn)
        apply_schema(conn)


def test_migration_upgrade_creates_tables() -> None:
    """Running alembic upgrade head creates all expected tables."""
    result = _run_alembic("upgrade", "head")
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"


def test_migration_downgrade_removes_tables() -> None:
    """Running alembic downgrade base drops all Mimir tables."""
    _run_alembic("upgrade", "head")

    result = _run_alembic("downgrade", "base")
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}"


def test_migration_idempotent_upgrade() -> None:
    """Running upgrade head twice in a row is safe (alembic_version guards it)."""
    for _ in range(2):
        result = _run_alembic("upgrade", "head")
        assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"
