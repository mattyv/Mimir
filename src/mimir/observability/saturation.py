"""Per-source ingestion saturation tracking (§10.1).

Records how many new vs. seen entities each ingestion run produces
per source reference.  When new/seen → 0, the source is saturated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg


@dataclass
class SaturationRecord:
    source_type: str
    source_ref: str
    entities_new: int
    entities_seen: int
    saturation_pct: float  # 0.0 = all new, 1.0 = fully saturated


def record_saturation(
    source_type: str,
    source_ref: str,
    entities_new: int,
    entities_seen: int,
    conn: psycopg.Connection[dict[str, Any]],
) -> None:
    """Persist one saturation measurement for a source reference."""
    total = entities_new + entities_seen
    pct = (entities_seen / total) if total > 0 else 0.0
    conn.execute(
        """
        INSERT INTO source_saturation
            (source_type, source_ref, entities_new, entities_seen, saturation_pct)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_type, source_ref, run_at) DO NOTHING
        """,
        (source_type, source_ref, entities_new, entities_seen, round(pct, 4)),
    )


def get_saturation(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    source_type: str | None = None,
    limit: int = 100,
) -> list[SaturationRecord]:
    """Return most recent saturation record per (source_type, source_ref).

    Ordered by saturation_pct descending — most saturated sources first.
    """
    type_clause = "AND source_type = %s" if source_type else ""
    type_params: list[Any] = [source_type] if source_type else []

    rows = conn.execute(
        f"""
        SELECT DISTINCT ON (source_type, source_ref)
               source_type, source_ref, entities_new, entities_seen, saturation_pct
          FROM source_saturation
         WHERE true {type_clause}
         ORDER BY source_type, source_ref, run_at DESC
        """,
        type_params,
    ).fetchall()

    records = [
        SaturationRecord(
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            entities_new=int(row["entities_new"]),
            entities_seen=int(row["entities_seen"]),
            saturation_pct=float(row["saturation_pct"]),
        )
        for row in rows
    ]
    return sorted(records, key=lambda r: r.saturation_pct, reverse=True)[:limit]


def is_saturated(
    source_type: str,
    source_ref: str,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    threshold: float = 0.95,
) -> bool:
    """Return True if the most recent run for this source hit the saturation threshold."""
    row = conn.execute(
        """
        SELECT saturation_pct FROM source_saturation
         WHERE source_type = %s AND source_ref = %s
         ORDER BY run_at DESC LIMIT 1
        """,
        (source_type, source_ref),
    ).fetchone()
    if row is None:
        return False
    return float(row["saturation_pct"]) >= threshold
