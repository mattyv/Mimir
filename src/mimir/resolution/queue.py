"""Resolution review queue — holds merge candidates for human approval.

Pairs with similarity in the 'borderline' range (configured thresholds)
are enqueued rather than auto-merged.  The queue table must exist
(schema.py / migration 0003 creates it).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg


@dataclass
class QueueEntry:
    id: int
    entity_a_id: str
    entity_b_id: str
    similarity: float
    method: str
    status: str  # 'pending', 'approved', 'rejected'
    created_at: datetime
    resolved_at: datetime | None


def enqueue_candidate(
    entity_a_id: str,
    entity_b_id: str,
    similarity: float,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    method: str = "embedding",
) -> int:
    """Insert a merge candidate into the review queue.

    Returns the new row id.  If the (a_id, b_id) pair already exists,
    does nothing (ON CONFLICT DO NOTHING) and returns 0.
    Ensures a_id < b_id lexicographically for canonical ordering.
    """
    a_id, b_id = (
        (entity_a_id, entity_b_id)
        if entity_a_id < entity_b_id
        else (entity_b_id, entity_a_id)
    )
    row = conn.execute(
        """
        INSERT INTO resolution_queue (entity_a_id, entity_b_id, similarity, method)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (entity_a_id, entity_b_id) DO NOTHING
        RETURNING id
        """,
        (a_id, b_id, similarity, method),
    ).fetchone()
    return int(row["id"]) if row else 0


def get_pending(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    limit: int = 50,
) -> list[QueueEntry]:
    """Return pending queue entries, oldest first."""
    rows = conn.execute(
        """
        SELECT id, entity_a_id, entity_b_id, similarity, method, status, created_at, resolved_at
          FROM resolution_queue
         WHERE status = 'pending'
         ORDER BY created_at
         LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def approve_merge(
    queue_id: int,
    conn: psycopg.Connection[dict[str, Any]],
) -> bool:
    """Mark a queue entry as approved and execute the merge.

    Returns True if the entry was found and processed.
    """
    from mimir.resolution.merger import merge_entities

    row = conn.execute(
        "SELECT entity_a_id, entity_b_id FROM resolution_queue WHERE id = %s AND status = 'pending'",
        (queue_id,),
    ).fetchone()
    if row is None:
        return False

    merge_entities(row["entity_a_id"], row["entity_b_id"], conn)
    conn.execute(
        "UPDATE resolution_queue SET status = 'approved', resolved_at = %s WHERE id = %s",
        (datetime.now(UTC), queue_id),
    )
    return True


def reject_pair(
    queue_id: int,
    conn: psycopg.Connection[dict[str, Any]],
) -> bool:
    """Mark a queue entry as rejected (no merge performed)."""
    result = conn.execute(
        "UPDATE resolution_queue SET status = 'rejected', resolved_at = %s WHERE id = %s AND status = 'pending' RETURNING id",
        (datetime.now(UTC), queue_id),
    )
    return int(result.rowcount) > 0


def _row_to_entry(row: dict[str, Any]) -> QueueEntry:
    return QueueEntry(
        id=int(row["id"]),
        entity_a_id=row["entity_a_id"],
        entity_b_id=row["entity_b_id"],
        similarity=float(row["similarity"]),
        method=row["method"],
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row.get("resolved_at"),
    )
