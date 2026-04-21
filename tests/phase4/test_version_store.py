"""Phase 4 — version_store tests."""

from __future__ import annotations

import os
from typing import Any

import psycopg
import psycopg.rows
import pytest

from mimir.adapters.version_store import get_version, has_changed, set_version

pytestmark = pytest.mark.phase4

_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")


def _conn() -> psycopg.Connection[Any]:
    return psycopg.connect(_DSN, row_factory=psycopg.rows.dict_row, autocommit=True)


def test_get_version_no_record(_pg_schema: None) -> None:
    with _conn() as conn:
        assert get_version("confluence", "http://wiki/page1", conn) is None


def test_set_and_get_version(_pg_schema: None) -> None:
    with _conn() as conn:
        set_version("confluence", "http://wiki/page1", "42", conn)
        assert get_version("confluence", "http://wiki/page1", conn) == "42"
        conn.execute(
            "DELETE FROM source_versions WHERE source_ref = 'http://wiki/page1'"
        )


def test_set_version_upserts(_pg_schema: None) -> None:
    with _conn() as conn:
        set_version("confluence", "http://wiki/page2", "1", conn)
        set_version("confluence", "http://wiki/page2", "7", conn)
        assert get_version("confluence", "http://wiki/page2", conn) == "7"
        conn.execute(
            "DELETE FROM source_versions WHERE source_ref = 'http://wiki/page2'"
        )


def test_set_version_different_source_types_independent(_pg_schema: None) -> None:
    with _conn() as conn:
        set_version("confluence", "http://shared/ref", "10", conn)
        set_version("github", "http://shared/ref", "abc123", conn)
        assert get_version("confluence", "http://shared/ref", conn) == "10"
        assert get_version("github", "http://shared/ref", conn) == "abc123"
        conn.execute("DELETE FROM source_versions WHERE source_ref = 'http://shared/ref'")


def test_has_changed_no_stored_returns_true(_pg_schema: None) -> None:
    with _conn() as conn:
        assert has_changed("confluence", "http://wiki/new", "99", conn) is True


def test_has_changed_same_version_returns_false(_pg_schema: None) -> None:
    with _conn() as conn:
        set_version("github", "https://github.com/a/b/blob/main/f.py", "deadbeef", conn)
        assert (
            has_changed("github", "https://github.com/a/b/blob/main/f.py", "deadbeef", conn)
            is False
        )
        conn.execute(
            "DELETE FROM source_versions WHERE source_ref = 'https://github.com/a/b/blob/main/f.py'"
        )


def test_has_changed_different_version_returns_true(_pg_schema: None) -> None:
    with _conn() as conn:
        set_version("confluence", "http://wiki/old", "3", conn)
        assert has_changed("confluence", "http://wiki/old", "7", conn) is True
        conn.execute("DELETE FROM source_versions WHERE source_ref = 'http://wiki/old'")
