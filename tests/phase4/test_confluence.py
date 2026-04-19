"""Phase 4 — ConfluenceAdapter tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from mimir.adapters.confluence import ConfluenceAdapter

_BASE = "https://wiki.example.com"
_TOKEN = "test-token"
_PAGE_PAYLOAD = {
    "id": "12345",
    "title": "OMMS Overview",
    "space": {"key": "trading-eng"},
    "body": {"storage": {"value": "<p>The OMMS service handles <b>options</b> market making.</p>"}},
    "_links": {"webui": "/spaces/trading-eng/pages/12345/OMMS+Overview"},
}


def _adapter() -> ConfluenceAdapter:
    client = httpx.Client()
    return ConfluenceAdapter(_BASE, _TOKEN, client=client)


@pytest.mark.phase4
@respx.mock
def test_fetch_page_returns_chunk() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_PAGE_PAYLOAD)
    )
    chunk = _adapter().fetch_page("12345")
    assert chunk is not None
    assert chunk.source_type == "confluence"
    assert chunk.id == "confluence_12345"


@pytest.mark.phase4
@respx.mock
def test_fetch_page_strips_html() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_PAGE_PAYLOAD)
    )
    chunk = _adapter().fetch_page("12345")
    assert chunk is not None
    assert "<p>" not in chunk.content
    assert "<b>" not in chunk.content
    assert "options" in chunk.content


@pytest.mark.phase4
@respx.mock
def test_fetch_page_sets_acl_from_space() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_PAGE_PAYLOAD)
    )
    chunk = _adapter().fetch_page("12345")
    assert chunk is not None
    assert "space:trading-eng" in chunk.acl


@pytest.mark.phase4
@respx.mock
def test_fetch_page_not_found_returns_none() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/99999").mock(
        return_value=httpx.Response(404)
    )
    chunk = _adapter().fetch_page("99999")
    assert chunk is None


@pytest.mark.phase4
@respx.mock
def test_fetch_page_http_error_raises() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(500)
    )
    with pytest.raises(httpx.HTTPStatusError):
        _adapter().fetch_page("12345")


@pytest.mark.phase4
@respx.mock
def test_search_returns_chunks() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(200, json={"results": [_PAGE_PAYLOAD, _PAGE_PAYLOAD]})
    )
    chunks = _adapter().search("trading-eng", "OMMS")
    assert len(chunks) == 2
    assert all(c.source_type == "confluence" for c in chunks)


@pytest.mark.phase4
@respx.mock
def test_search_empty_results() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    chunks = _adapter().search("trading-eng", "nonexistent")
    assert chunks == []


@pytest.mark.phase4
@respx.mock
def test_fetch_page_reference_uses_webui_link() -> None:
    respx.get(f"{_BASE}/wiki/rest/api/content/12345").mock(
        return_value=httpx.Response(200, json=_PAGE_PAYLOAD)
    )
    chunk = _adapter().fetch_page("12345")
    assert chunk is not None
    assert chunk.reference.startswith(_BASE)
