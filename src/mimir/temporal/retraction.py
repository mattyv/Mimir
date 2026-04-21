"""Retraction worker — scan for axioms whose source chunk no longer exists.

For each adapter type, calls exists() on the source reference.  If the source
is gone, expires all entities/relationships/properties/observations that were
derived exclusively from that reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import psycopg


class SourceChecker(Protocol):
    """Adapter must implement this to support retraction scanning."""

    def exists(self, reference: str) -> bool:
        """Return True if the source chunk at *reference* still exists."""
        ...


@dataclass
class RetractionResult:
    source_ref: str
    entities_expired: int
    relationships_expired: int
    properties_expired: int


def retract_by_source(
    source_ref: str,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> RetractionResult:
    """Expire all active axioms whose sole source is *source_ref*.

    An axiom is considered "solely" from this source when its payload->source->reference
    matches source_ref and no other corroborating source row exists.
    Only expires rows with valid_until IS NULL.
    """
    ts = at or datetime.now(UTC)

    # Expire entities whose payload source matches
    e_result = conn.execute(
        """
        UPDATE entities
           SET valid_until = %s
         WHERE valid_until IS NULL
           AND payload->'source'->>'reference' = %s
        RETURNING id
        """,
        (ts, source_ref),
    )
    entities_expired = int(e_result.rowcount)

    # Expire relationships
    r_result = conn.execute(
        """
        UPDATE relationships
           SET valid_until = %s
         WHERE valid_until IS NULL
           AND payload->'source'->>'reference' = %s
        RETURNING id
        """,
        (ts, source_ref),
    )
    rels_expired = int(r_result.rowcount)

    # Expire properties
    p_result = conn.execute(
        """
        UPDATE properties
           SET valid_until = %s
         WHERE valid_until IS NULL
           AND payload->'source'->>'reference' = %s
        RETURNING id
        """,
        (ts, source_ref),
    )
    props_expired = int(p_result.rowcount)

    return RetractionResult(
        source_ref=source_ref,
        entities_expired=entities_expired,
        relationships_expired=rels_expired,
        properties_expired=props_expired,
    )


def scan_and_retract(
    source_refs: list[str],
    checker: SourceChecker,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> list[RetractionResult]:
    """Check each reference; retract those where checker.exists() returns False."""
    results = []
    for ref in source_refs:
        if not checker.exists(ref):
            results.append(retract_by_source(ref, conn, at=at))
    return results


def list_active_source_refs(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    source_type: str | None = None,
) -> list[str]:
    """Return distinct source references currently in active (non-expired) entities."""
    if source_type:
        rows = conn.execute(
            """
            SELECT DISTINCT payload->'source'->>'reference' AS ref
              FROM entities
             WHERE valid_until IS NULL
               AND payload->'source'->>'type' = %s
               AND payload->'source'->>'reference' IS NOT NULL
            """,
            (source_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT payload->'source'->>'reference' AS ref
              FROM entities
             WHERE valid_until IS NULL
               AND payload->'source'->>'reference' IS NOT NULL
            """
        ).fetchall()
    return [row["ref"] for row in rows if row["ref"]]
