"""Slack source adapter — fetches channel history via the Web API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from mimir.adapters.base import Chunk

_API_BASE = "https://slack.com/api"


class SlackAdapter:
    """Fetch Slack channel message history as Chunk objects.

    Args:
        token:  Slack Bot OAuth token (``xoxb-…``).
        client: Optional pre-configured httpx.Client; primarily for testing.
    """

    def __init__(self, token: str, client: httpx.Client | None = None) -> None:
        self._token = token
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {token}"}
        )

    def fetch_channel(self, channel_id: str, *, limit: int = 100) -> list[Chunk]:
        """Fetch recent messages from a Slack channel.

        Returns one Chunk per batch of messages.  An empty channel returns
        an empty list.  Raises httpx.HTTPStatusError on HTTP failures; raises
        ValueError if the Slack API returns ``ok: false``.
        """
        resp = self._client.get(
            f"{_API_BASE}/conversations.history",
            params={"channel": channel_id, "limit": limit},
        )
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            raise ValueError(f"Slack API error: {body.get('error', 'unknown')}")
        messages: list[dict[str, Any]] = body.get("messages", [])
        if not messages:
            return []
        content = self._format_messages(messages)
        ts = messages[0].get("ts", "")
        return [
            Chunk(
                id=f"slack_{channel_id}_{ts}",
                source_type="slack",
                content=content,
                acl=[f"channel:{channel_id}"],
                retrieved_at=datetime.now(UTC),
                reference=f"https://slack.com/archives/{channel_id}",
                metadata={"channel_id": channel_id, "message_count": len(messages)},
            )
        ]

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for msg in reversed(messages):  # chronological order
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            if text:
                lines.append(f"{user}: {text}")
        return "\n".join(lines)
