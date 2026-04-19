"""Phase 4 — SlackAdapter tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from mimir.adapters.slack import _API_BASE, SlackAdapter

_MESSAGES = [
    {"type": "message", "user": "U002", "text": "Restarting now.", "ts": "1700000002.000"},
    {"type": "message", "user": "U001", "text": "The risk engine is down.", "ts": "1700000001.000"},
]


def _adapter() -> SlackAdapter:
    return SlackAdapter(token="xoxb-test", client=httpx.Client())


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_returns_chunk() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": True, "messages": _MESSAGES})
    )
    chunks = _adapter().fetch_channel("C123ABC")
    assert len(chunks) == 1
    assert chunks[0].source_type == "slack"


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_content_includes_messages() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": True, "messages": _MESSAGES})
    )
    chunks = _adapter().fetch_channel("C123ABC")
    assert "risk engine" in chunks[0].content


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_acl_contains_channel() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": True, "messages": _MESSAGES})
    )
    chunks = _adapter().fetch_channel("C123ABC")
    assert "channel:C123ABC" in chunks[0].acl


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_empty_returns_empty_list() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": True, "messages": []})
    )
    chunks = _adapter().fetch_channel("C123ABC")
    assert chunks == []


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_slack_error_raises() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
    )
    with pytest.raises(ValueError, match="channel_not_found"):
        _adapter().fetch_channel("CINVALID")


@pytest.mark.phase4
@respx.mock
def test_fetch_channel_messages_in_chronological_order() -> None:
    respx.get(f"{_API_BASE}/conversations.history").mock(
        return_value=httpx.Response(200, json={"ok": True, "messages": _MESSAGES})
    )
    chunks = _adapter().fetch_channel("C123ABC")
    lines = chunks[0].content.splitlines()
    assert lines[0].startswith("U001")
    assert lines[1].startswith("U002")
