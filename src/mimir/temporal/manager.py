"""Temporal lifecycle operations for entities, properties, and relationships."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg


@dataclass
class ExpiryResult:
    table: str
    row_id: str
    expired_at: datetime


def expire_entity(
    entity_id: str,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> bool:
    """Set valid_until on an entity row. Returns True if the row was updated."""
    ts = at or datetime.now(UTC)
    result = conn.execute(
        "UPDATE entities SET valid_until = %s WHERE id = %s AND valid_until IS NULL RETURNING id",
        (ts, entity_id),
    )
    return int(result.rowcount) > 0


def expire_relationship(
    relationship_id: int,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> bool:
    """Set valid_until on a relationship row."""
    ts = at or datetime.now(UTC)
    result = conn.execute(
        "UPDATE relationships SET valid_until = %s WHERE id = %s AND valid_until IS NULL RETURNING id",
        (ts, relationship_id),
    )
    return int(result.rowcount) > 0


def expire_property(
    property_id: int,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> bool:
    """Set valid_until on a property row."""
    ts = at or datetime.now(UTC)
    result = conn.execute(
        "UPDATE properties SET valid_until = %s WHERE id = %s AND valid_until IS NULL RETURNING id",
        (ts, property_id),
    )
    return int(result.rowcount) > 0


def supersede_entity(
    old_id: str,
    new_id: str,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    at: datetime | None = None,
) -> bool:
    """Expire *old_id* and record *new_id* as its successor in payload."""
    ts = at or datetime.now(UTC)
    result = conn.execute(
        """
        UPDATE entities
        SET valid_until = %s,
            payload = payload || jsonb_build_object('superseded_by', %s::text)
        WHERE id = %s AND valid_until IS NULL
        RETURNING id
        """,
        (ts, new_id, old_id),
    )
    return int(result.rowcount) > 0


def active_entities_at(
    conn: psycopg.Connection[dict[str, Any]],
    as_of: datetime,
    *,
    entity_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return entity rows that were active at *as_of*."""
    if entity_type:
        rows = conn.execute(
            """
            SELECT * FROM entities
            WHERE valid_from <= %s
              AND (valid_until IS NULL OR valid_until > %s)
              AND entity_type = %s
            ORDER BY name LIMIT %s
            """,
            (as_of, as_of, entity_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM entities
            WHERE valid_from <= %s
              AND (valid_until IS NULL OR valid_until > %s)
            ORDER BY name LIMIT %s
            """,
            (as_of, as_of, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def expired_entities(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return all entity rows that have been expired (valid_until IS NOT NULL)."""
    rows = conn.execute(
        "SELECT * FROM entities WHERE valid_until IS NOT NULL ORDER BY valid_until DESC LIMIT %s",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
