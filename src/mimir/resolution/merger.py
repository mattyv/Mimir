"""Merge two Entity nodes into one, preserving all provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

from mimir.persistence.graph_version import bump_graph_version


@dataclass
class MergeResult:
    kept_id: str
    dropped_id: str
    graph_version: int
    properties_rerouted: int
    observations_rerouted: int
    relationships_rerouted: int


def merge_entities(
    kept_id: str,
    dropped_id: str,
    conn: psycopg.Connection[dict[str, Any]],
) -> MergeResult:
    """Merge *dropped_id* into *kept_id* within the caller's transaction.

    Steps performed atomically:
    1. Validate both entities exist and are currently active.
    2. Reroute all properties, observations, and relationships from dropped → kept.
    3. Remove any self-referential relationships introduced by the rerouting.
    4. Expire the dropped entity (valid_until = NOW(), superseded_by = kept_id).
    5. Bump the graph version once.

    Raises ValueError if either entity is missing or already expired.
    """
    rows = conn.execute(
        "SELECT id FROM entities WHERE id = ANY(%s) AND valid_until IS NULL",
        ([kept_id, dropped_id],),
    ).fetchall()
    found = {row["id"] for row in rows}
    if kept_id not in found:
        raise ValueError(f"Entity {kept_id!r} not found or already superseded")
    if dropped_id not in found:
        raise ValueError(f"Entity {dropped_id!r} not found or already superseded")

    props = conn.execute(
        "UPDATE properties SET entity_id = %s WHERE entity_id = %s",
        (kept_id, dropped_id),
    ).rowcount

    obs = conn.execute(
        "UPDATE observations SET entity_id = %s WHERE entity_id = %s",
        (kept_id, dropped_id),
    ).rowcount

    rel_sub = conn.execute(
        "UPDATE relationships SET subject_id = %s WHERE subject_id = %s",
        (kept_id, dropped_id),
    ).rowcount
    rel_obj = conn.execute(
        "UPDATE relationships SET object_id = %s WHERE object_id = %s",
        (kept_id, dropped_id),
    ).rowcount

    # Self-loops arise when a relationship connected the two merged entities.
    conn.execute(
        "DELETE FROM relationships WHERE subject_id = %s AND object_id = %s",
        (kept_id, kept_id),
    )

    conn.execute(
        """
        UPDATE entities
           SET valid_until = %s,
               payload     = payload || jsonb_build_object('superseded_by', %s::text)
         WHERE id = %s
        """,
        (datetime.now(UTC), kept_id, dropped_id),
    )

    version = bump_graph_version(conn)

    return MergeResult(
        kept_id=kept_id,
        dropped_id=dropped_id,
        graph_version=version,
        properties_rerouted=props,
        observations_rerouted=obs,
        relationships_rerouted=rel_sub + rel_obj,
    )
