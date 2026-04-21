"""Confluence source adapter — fetches pages via the REST API v2."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg

from mimir.adapters.base import Chunk

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", text).strip()


class ConfluenceAdapter:
    """Fetch Confluence pages as Chunk objects.

    Args:
        base_url: Root URL of the Confluence instance, e.g.
                  ``https://wiki.example.com``.
        token:    Confluence personal access token (Bearer auth).
        client:   Optional pre-configured httpx.Client; primarily for testing.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self._client = client or httpx.Client(headers=self._headers)

    def fetch_page(self, page_id: str) -> Chunk | None:
        """Fetch a single Confluence page by its numeric ID.

        Returns None if the page is not found (404).
        Raises httpx.HTTPStatusError for other error codes.
        """
        resp = self._client.get(
            f"{self._base}/wiki/rest/api/content/{page_id}",
            params={"expand": "body.storage,space,version"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._to_chunk(resp.json())

    def fetch_page_version(self, page_id: str) -> tuple[int, str] | None:
        """Return (version_number, reference_url) without downloading the page body.

        Uses a metadata-only fetch (expand=version only) — much cheaper than
        fetch_page() when you only need to check whether the page changed.
        Returns None if the page is not found (404).
        """
        resp = self._client.get(
            f"{self._base}/wiki/rest/api/content/{page_id}",
            params={"expand": "version"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        version_number: int = data.get("version", {}).get("number", 0)
        web_ui: str = data.get("_links", {}).get("webui", "")
        reference = f"{self._base}{web_ui}" if web_ui else f"{self._base}/pages/{page_id}"
        return version_number, reference

    def fetch_changed(
        self,
        page_ids: list[str],
        conn: psycopg.Connection[Any],
    ) -> list[str]:
        """Return the subset of page_ids whose Confluence version differs from stored.

        Makes one cheap metadata-only request per page_id.  Pages that return
        404 (deleted) are excluded from the result — callers should handle
        deletions separately via retract_by_source().
        """
        from mimir.adapters.version_store import get_version

        changed: list[str] = []
        for pid in page_ids:
            result = self.fetch_page_version(pid)
            if result is None:
                continue
            version_number, reference = result
            stored = get_version("confluence", reference, conn)
            if stored is None or stored != str(version_number):
                changed.append(pid)
        return changed

    def search(self, space_key: str, query: str, limit: int = 25) -> list[Chunk]:
        """Full-text search within a Confluence space."""
        resp = self._client.get(
            f"{self._base}/wiki/rest/api/content/search",
            params={
                "cql": f'space="{space_key}" AND text~"{query}"',
                "expand": "body.storage,space,version",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        return [self._to_chunk(r) for r in resp.json().get("results", [])]

    def _to_chunk(self, page: dict[str, Any]) -> Chunk:
        space_key = page.get("space", {}).get("key", "")
        html = page.get("body", {}).get("storage", {}).get("value", "")
        title = page.get("title", "")
        page_id = page.get("id", "")
        web_ui = page.get("_links", {}).get("webui", "")
        reference = f"{self._base}{web_ui}" if web_ui else f"{self._base}/pages/{page_id}"
        version_number: int = page.get("version", {}).get("number", 0)
        return Chunk(
            id=f"confluence_{page_id}",
            source_type="confluence",
            content=f"# {title}\n\n{_strip_html(html)}",
            acl=[f"space:{space_key}"] if space_key else [],
            retrieved_at=datetime.now(UTC),
            reference=reference,
            metadata={
                "page_id": page_id,
                "space_key": space_key,
                "title": title,
                "version_number": version_number,
            },
        )
