"""Phase 4 — GitHubAdapter tests."""

from __future__ import annotations

import base64

import httpx
import pytest
import respx

from mimir.adapters.github import _API_BASE, GitHubAdapter

_README_CONTENT = "# panic_server\n\nSafety circuit breaker."
_README_PAYLOAD = {
    "name": "README.md",
    "path": "README.md",
    "content": base64.b64encode(_README_CONTENT.encode()).decode() + "\n",
    "encoding": "base64",
    "html_url": "https://github.com/example/panic_server/blob/main/README.md",
    "sha": "abc123",
}


def _adapter() -> GitHubAdapter:
    return GitHubAdapter(token="ghp_test", client=httpx.Client())


@pytest.mark.phase4
@respx.mock
def test_fetch_file_returns_chunk() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/README.md").mock(
        return_value=httpx.Response(200, json=_README_PAYLOAD)
    )
    chunk = _adapter().fetch_file("example", "panic_server", "README.md")
    assert chunk is not None
    assert chunk.source_type == "github"
    assert _README_CONTENT in chunk.content


@pytest.mark.phase4
@respx.mock
def test_fetch_file_decodes_base64() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/README.md").mock(
        return_value=httpx.Response(200, json=_README_PAYLOAD)
    )
    chunk = _adapter().fetch_file("example", "panic_server", "README.md")
    assert chunk is not None
    assert "panic_server" in chunk.content


@pytest.mark.phase4
@respx.mock
def test_fetch_file_sets_acl() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/README.md").mock(
        return_value=httpx.Response(200, json=_README_PAYLOAD)
    )
    chunk = _adapter().fetch_file("example", "panic_server", "README.md")
    assert chunk is not None
    assert "repo:example/panic_server" in chunk.acl


@pytest.mark.phase4
@respx.mock
def test_fetch_file_not_found_returns_none() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/missing.md").mock(
        return_value=httpx.Response(404)
    )
    chunk = _adapter().fetch_file("example", "panic_server", "missing.md")
    assert chunk is None


@pytest.mark.phase4
@respx.mock
def test_fetch_readme_returns_chunk() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/readme").mock(
        return_value=httpx.Response(200, json=_README_PAYLOAD)
    )
    chunk = _adapter().fetch_readme("example", "panic_server")
    assert chunk is not None
    assert chunk.source_type == "github"


@pytest.mark.phase4
@respx.mock
def test_fetch_readme_not_found_returns_none() -> None:
    respx.get(f"{_API_BASE}/repos/example/no_readme/readme").mock(return_value=httpx.Response(404))
    chunk = _adapter().fetch_readme("example", "no_readme")
    assert chunk is None


@pytest.mark.phase4
@respx.mock
def test_fetch_file_metadata_preserved() -> None:
    respx.get(f"{_API_BASE}/repos/example/panic_server/contents/README.md").mock(
        return_value=httpx.Response(200, json=_README_PAYLOAD)
    )
    chunk = _adapter().fetch_file("example", "panic_server", "README.md")
    assert chunk is not None
    assert chunk.metadata["owner"] == "example"
    assert chunk.metadata["repo"] == "panic_server"
    assert chunk.metadata["sha"] == "abc123"
