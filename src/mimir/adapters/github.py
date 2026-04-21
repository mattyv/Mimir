"""GitHub source adapter — fetches file contents via the REST API."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg

from mimir.adapters.base import Chunk

_API_BASE = "https://api.github.com"


class GitHubAdapter:
    """Fetch GitHub file contents and READMEs as Chunk objects.

    Args:
        token:  GitHub personal access token (optional, but rate-limited without one).
        client: Optional pre-configured httpx.Client; primarily for testing.
    """

    def __init__(
        self,
        token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(headers=headers)

    def fetch_file(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "HEAD",
    ) -> Chunk | None:
        """Fetch a file from a GitHub repository.

        Returns None if the file is not found (404).
        """
        resp = self._client.get(
            f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._to_chunk(owner, repo, resp.json())

    def fetch_readme(self, owner: str, repo: str) -> Chunk | None:
        """Fetch the default README for a repository."""
        resp = self._client.get(f"{_API_BASE}/repos/{owner}/{repo}/readme")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._to_chunk(owner, repo, resp.json())

    def fetch_file_sha(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "HEAD",
    ) -> tuple[str, str] | None:
        """Return (blob_sha, html_url) for a file without decoding content.

        Makes the same API call as fetch_file() but skips base64 decoding,
        making it suitable for cheap change-detection before committing to
        a full re-ingest.  Returns None if the file is not found (404).
        """
        resp = self._client.get(
            f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        sha: str = data.get("sha", "")
        html_url: str = data.get(
            "html_url", f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"
        )
        return sha, html_url

    def fetch_changed_files(
        self,
        owner: str,
        repo: str,
        paths: list[str],
        conn: psycopg.Connection[Any],
        ref: str = "HEAD",
    ) -> list[str]:
        """Return the subset of paths whose blob SHA differs from stored.

        Makes one cheap API request per path.  Paths that return 404
        (deleted files) are excluded — handle deletions via retract_by_source().
        """
        from mimir.adapters.version_store import get_version

        changed: list[str] = []
        for path in paths:
            result = self.fetch_file_sha(owner, repo, path, ref)
            if result is None:
                continue
            sha, html_url = result
            stored = get_version("github", html_url, conn)
            if stored is None or stored != sha:
                changed.append(path)
        return changed

    def _to_chunk(self, owner: str, repo: str, data: dict[str, Any]) -> Chunk:
        raw = data.get("content", "")
        encoding = data.get("encoding", "base64")
        if encoding == "base64":
            content = base64.b64decode(raw.replace("\n", "")).decode("utf-8", errors="replace")
        else:
            content = raw
        path = data.get("path", "")
        html_url = data.get("html_url", f"https://github.com/{owner}/{repo}/blob/HEAD/{path}")
        sha = data.get("sha", "")
        return Chunk(
            id=f"github_{owner}_{repo}_{path}".replace("/", "_"),
            source_type="github",
            content=content,
            acl=[f"repo:{owner}/{repo}"],
            retrieved_at=datetime.now(UTC),
            reference=html_url,
            metadata={"owner": owner, "repo": repo, "path": path, "sha": sha},
        )
