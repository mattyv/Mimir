"""Phase 4 — Confluence update detection tests."""

from __future__ import annotations

import os
from typing import Any

import httpx
import psycopg
import pytest
import respx

from mimir.adapters.confluence import ConfluenceAdapter
from mimir.adapters.version_store import set_version

pytestmark = pytest.mark.phase4

_BASE = "https://wiki.example.com"
_TOKEN = "test-token"
_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")

_VERSION_PAYLOAD = {
    "id": "12345",
    "title": "OMMS Overview",
    "version": {"number": 7},
    "_links": {"webui": "/spaces/trading-eng/pages/12345/OMMS+Overview"},
}

_PAGE_PAYLOAD = {
    "id": "12345",
    "title": "OMMS Overview",
    "space": {"key": "trading-eng"},
    "body": {"storage": {"value": "<p>Updated content.</p>"}},
    "version": {"number": 7},
    "_links": {"webui": "/spaces/trading-eng/pages/12345/OMMS+Overview"},
}


def _adapter() -> ConfluenceAdapter:
    return ConfluenceAdapter(_BASE, _TOKEN, client=httpx.Client())


def _conn() -> psycopg.Connection[Any]:
    import psycopg.rows
    return psycopg.connect(_DSN, row_factory=psycopg.rows.dict_row, autocommit=True)


@respx.mock
def test_fetch_page_version_returns_number_and_reference() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_VERSION_PAYLOAD)
    )
    result = _adapter().fetch_page_version("12345")
    assert result is not None
    version_number, reference = result
    assert version_number == 7
    assert reference.startswith(_BASE)


@respx.mock
def test_fetch_page_version_not_found_returns_none() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/99999").mock(return_value=httpx.Response(404))
    assert _adapter().fetch_page_version("99999") is None


@respx.mock
def test_fetch_page_version_fallback_reference_when_no_webui() -> None:
    payload = {"id": "12345", "version": {"number": 3}, "_links": {}}
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=payload)
    )
    result = _adapter().fetch_page_version("12345")
    assert result is not None
    _, reference = result
    assert "12345" in reference


@respx.mock
def test_to_chunk_includes_version_number() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_PAGE_PAYLOAD)
    )
    chunk = _adapter().fetch_page("12345")
    assert chunk is not None
    assert chunk.metadata["version_number"] == 7


@respx.mock
def test_fetch_changed_returns_new_page(_pg_schema: None) -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_VERSION_PAYLOAD)
    )
    with _conn() as conn:
        changed = _adapter().fetch_changed(["12345"], conn)
    assert "12345" in changed


@respx.mock
def test_fetch_changed_skips_unchanged_page(_pg_schema: None) -> None:
    reference = f"{_BASE}/spaces/trading-eng/pages/12345/OMMS+Overview"
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_VERSION_PAYLOAD)
    )
    with _conn() as conn:
        set_version("confluence", reference, "7", conn)
        changed = _adapter().fetch_changed(["12345"], conn)
        assert "12345" not in changed
        conn.execute("DELETE FROM source_versions WHERE source_ref = %s", (reference,))


@respx.mock
def test_fetch_changed_returns_page_with_newer_version(_pg_schema: None) -> None:
    reference = f"{_BASE}/spaces/trading-eng/pages/12345/OMMS+Overview"
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_VERSION_PAYLOAD)  # version 7
    )
    with _conn() as conn:
        set_version("confluence", reference, "3", conn)  # stored = 3, current = 7
        changed = _adapter().fetch_changed(["12345"], conn)
        assert "12345" in changed
        conn.execute("DELETE FROM source_versions WHERE source_ref = %s", (reference,))


@respx.mock
def test_fetch_changed_excludes_404_pages(_pg_schema: None) -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/deleted").mock(return_value=httpx.Response(404))
    with _conn() as conn:
        changed = _adapter().fetch_changed(["deleted"], conn)
    assert changed == []
