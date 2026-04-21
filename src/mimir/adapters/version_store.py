"""Source-document version store — tracks last-seen version per (source_type, source_ref).

Used by update-detection logic so the crawler can skip unchanged documents
without querying entity payloads.  version_key is an opaque string:
- Confluence: str(version.number)  e.g. "42"
- GitHub:     blob SHA              e.g. "a3f8c1d..."
"""

from __future__ import annotations

from typing import Any

import psycopg


def get_version(
    source_type: str,
    source_ref: str,
    conn: psycopg.Connection[Any],
) -> str | None:
    """Return the stored version key, or None if never ingested."""
    row = conn.execute(
        "SELECT version_key FROM source_versions WHERE source_type = %s AND source_ref = %s",
        (source_type, source_ref),
    ).fetchone()
    if row is None:
        return None
    return str(row["version_key"])


def set_version(
    source_type: str,
    source_ref: str,
    version_key: str,
    conn: psycopg.Connection[Any],
) -> None:
    """Upsert the version key for a source document."""
    conn.execute(
        """
        INSERT INTO source_versions (source_type, source_ref, version_key)
        VALUES (%s, %s, %s)
        ON CONFLICT (source_type, source_ref) DO UPDATE SET
            version_key = EXCLUDED.version_key,
            updated_at  = NOW()
        """,
        (source_type, source_ref, version_key),
    )


def has_changed(
    source_type: str,
    source_ref: str,
    new_version: str,
    conn: psycopg.Connection[Any],
) -> bool:
    """Return True if new_version differs from the stored version (or was never stored)."""
    return get_version(source_type, source_ref, conn) != new_version
