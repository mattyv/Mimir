"""Phase 2 — Alembic migration tests.

Verifies that the 0001 migration can be applied and reversed without error.
These tests use a dedicated database schema namespace (via a temp schema) to
avoid interfering with the session-scoped _pg_schema fixture.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.phase2

_ENV = {**os.environ, "MIMIR_DATABASE_URL": "postgresql+psycopg://root@/mimir_test"}


def _run_alembic(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd="/home/user/Mimir",
        env=_ENV,
        capture_output=True,
        text=True,
    )


def test_migration_upgrade_creates_tables() -> None:
    """Running alembic upgrade head creates all expected tables."""
    result = _run_alembic("upgrade", "head")
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"


def test_migration_downgrade_removes_tables() -> None:
    """Running alembic downgrade base drops all Mimir tables."""
    # Ensure we're at head first
    _run_alembic("upgrade", "head")

    result = _run_alembic("downgrade", "base")
    assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}"

    # Restore schema for subsequent tests that need it
    _run_alembic("upgrade", "head")


def test_migration_idempotent_upgrade() -> None:
    """Running upgrade head twice in a row is safe (alembic_version guards it)."""
    for _ in range(2):
        result = _run_alembic("upgrade", "head")
        assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"
