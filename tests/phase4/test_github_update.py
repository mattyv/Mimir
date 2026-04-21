"""Phase 4 — GitHub update detection tests."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx
import psycopg
import pytest
import respx

from mimir.adapters.github import _API_BASE, GitHubAdapter
from mimir.adapters.version_store import set_version

pytestmark = pytest.mark.phase4

_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")

_FILE_PAYLOAD = {
    "name": "service.py",
    "path": "src/service.py",
    "sha": "deadbeef1234",
    "html_url": "https://github.com/example/panic_server/blob/main/src/service.py",
    "content": base64.b64encode(b"print('hello')").decode() + "\n",
    "encoding": "base64",
}


def _adapter() -> GitHubAdapter:
    return GitHubAdapter(token="ghp_test", client=httpx.Client())


def _conn() -> psycopg.Connection[Any]:
    import psycopg.rows
    return psycopg.connect(_DSN, row_factory=psycopg.rows.dict_row, autocommit=True)


@respx.mock
def test_fetch_file_sha_returns_sha_and_url() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/src/service.py").mock(
        return_value=httpx.Response(200, json=_FILE_PAYLOAD)
    )
    result = _adapter().fetch_file_sha("example", "panic_server", "src/service.py")
    assert result is not None
    sha, html_url = result
    assert sha == "deadbeef1234"
    assert "example/panic_server" in html_url


@respx.mock
def test_fetch_file_sha_not_found_returns_none() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/missing.py").mock(
        return_value=httpx.Response(404)
    )
    assert _adapter().fetch_file_sha("example", "panic_server", "missing.py") is None


@respx.mock
def test_fetch_file_sha_fallback_url_when_no_html_url() -> None:
    payload = {"sha": "abc123", "path": "README.md"}
    respx.get(f"{_API_BASE}/repos/example/repo/contents/README.md").mock(
        return_value=httpx.Response(200, json=payload)
    )
    result = _adapter().fetch_file_sha("example", "repo", "README.md")
    assert result is not None
    sha, html_url = result
    assert sha == "abc123"
    assert "example/repo" in html_url


@respx.mock
def test_fetch_changed_files_returns_new_file(_pg_schema: None) -> None:
    respx.get(
        f"{_API_BASE}/repos/example/panic_server/contents/src/service.py"
    ).mock(return_value=httpx.Response(200, json=_FILE_PAYLOAD))
    with _conn() as conn:
        changed = _adapter().fetch_changed_files(
            "example", "panic_server", ["src/service.py"], conn
        )
    assert "src/service.py" in changed


@respx.mock
def test_fetch_changed_files_skips_unchanged(_pg_schema: None) -> None:
    html_url = "https://github.com/example/panic_server/blob/main/src/service.py"
    respx.get(
        f"{_API_BASE}/repos/example/panic_server/contents/src/service.py"
    ).mock(return_value=httpx.Response(200, json=_FILE_PAYLOAD))
    with _conn() as conn:
        set_version("github", html_url, "deadbeef1234", conn)
        changed = _adapter().fetch_changed_files(
            "example", "panic_server", ["src/service.py"], conn
        )
        assert "src/service.py" not in changed
        conn.execute("DELETE FROM source_versions WHERE source_ref = %s", (html_url,))


@respx.mock
def test_fetch_changed_files_returns_file_with_new_sha(_pg_schema: None) -> None:
    html_url = "https://github.com/example/panic_server/blob/main/src/service.py"
    respx.get(
        f"{_API_BASE}/repos/example/panic_server/contents/src/service.py"
    ).mock(return_value=httpx.Response(200, json=_FILE_PAYLOAD))
    with _conn() as conn:
        set_version("github", html_url, "oldsha999", conn)
        changed = _adapter().fetch_changed_files(
            "example", "panic_server", ["src/service.py"], conn
        )
        assert "src/service.py" in changed
        conn.execute("DELETE FROM source_versions WHERE source_ref = %s", (html_url,))


@respx.mock
def test_fetch_changed_files_excludes_deleted(_pg_schema: None) -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/gone.py").mock(
        return_value=httpx.Response(404)
    )
    with _conn() as conn:
        changed = _adapter().fetch_changed_files(
            "example", "panic_server", ["gone.py"], conn
        )
    assert changed == []


def test_to_chunk_sha_in_metadata() -> None:
    adapter = _adapter()
    # Test _to_chunk directly via fetch_file response
    with respx.mock:
        respx.get(
            f"{_API_BASE}/repos/example/panic_server/contents/src/service.py"
        ).mock(return_value=httpx.Response(200, json=_FILE_PAYLOAD))
        chunk = adapter.fetch_file("example", "panic_server", "src/service.py")
    assert chunk is not None
    assert chunk.metadata["sha"] == "deadbeef1234"
