"""Conflict detection — property and polarity conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg


@dataclass
class PropertyConflict:
    entity_id: str
    key: str
    values: list[dict[str, Any]]  # each: {value, source_type, source_ref, property_id}


@dataclass
class PolarityConflict:
    subject_id: str
    object_id: str
    predicates: list[str]  # two opposite predicates asserted simultaneously


def detect_property_conflicts(
    entity_id: str,
    conn: psycopg.Connection[dict[str, Any]],
) -> list[PropertyConflict]:
    """Return property keys where >1 distinct active value exists for *entity_id*."""
    rows = conn.execute(
        """
        SELECT
            key,
            json_agg(
                json_build_object(
                    'value', value,
                    'source_type', payload->'source'->>'type',
                    'source_ref', payload->'source'->>'reference',
                    'property_id', id::text
                )
            ) AS values
        FROM properties
        WHERE entity_id = %s
          AND valid_until IS NULL
        GROUP BY key
        HAVING COUNT(DISTINCT value::text) > 1
        """,
        (entity_id,),
    ).fetchall()

    return [
        PropertyConflict(
            entity_id=entity_id,
            key=row["key"],
            values=row["values"],
        )
        for row in rows
    ]


def detect_polarity_conflicts(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    subject_id: str | None = None,
) -> list[PolarityConflict]:
    """Return pairs of opposite predicates asserted between the same subject/object."""
    from pathlib import Path

    from mimir.vocabulary.loader import load_vocabulary

    _vocab_path = Path(__file__).parent.parent / "vocabulary" / "vocabulary.yaml"
    vocab = load_vocabulary(_vocab_path)

    polarity_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pred in vocab.predicates:
        opp = vocab.get_polarity_opposite(pred.iri)
        if opp:
            pair: tuple[str, str] = tuple(sorted([pred.iri, opp]))  # type: ignore[assignment]
            if pair not in seen:
                seen.add(pair)
                polarity_pairs.append(pair)

    conflicts: list[PolarityConflict] = []
    for pred_a, pred_b in polarity_pairs:
        if subject_id is not None:
            rows = conn.execute(
                """
                SELECT DISTINCT r1.subject_id, r1.object_id
                FROM relationships r1
                JOIN relationships r2
                  ON r1.subject_id = r2.subject_id
                 AND r1.object_id = r2.object_id
                WHERE r1.subject_id = %s
                  AND r1.predicate = %s
                  AND r2.predicate = %s
                  AND r1.valid_until IS NULL
                  AND r2.valid_until IS NULL
                """,
                (subject_id, pred_a, pred_b),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT r1.subject_id, r1.object_id
                FROM relationships r1
                JOIN relationships r2
                  ON r1.subject_id = r2.subject_id
                 AND r1.object_id = r2.object_id
                WHERE r1.predicate = %s
                  AND r2.predicate = %s
                  AND r1.valid_until IS NULL
                  AND r2.valid_until IS NULL
                """,
                (pred_a, pred_b),
            ).fetchall()

        for row in rows:
            conflicts.append(
                PolarityConflict(
                    subject_id=row["subject_id"],
                    object_id=row["object_id"],
                    predicates=[pred_a, pred_b],
                )
            )

    return conflicts
