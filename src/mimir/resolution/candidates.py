"""Similarity-based merge-candidate detection for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg

from mimir.resolution.embedder import _vec_sql


@dataclass
class MergeCandidate:
    entity_a_id: str
    entity_b_id: str
    similarity: float  # cosine similarity, 0.0–1.0
    method: str  # currently always "embedding"


def find_similar_by_embedding(
    embedding: list[float],
    entity_type: str,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    exclude_id: str | None = None,
    threshold: float = 0.85,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return active entities of *entity_type* whose stored embedding is
    cosine-similar to *embedding* at or above *threshold*.

    Results are sorted by descending similarity.  Pass *exclude_id* to omit
    the source entity itself from the results.
    """
    vec_lit = _vec_sql(embedding)
    exclude_clause = "AND id != %s" if exclude_id is not None else ""
    extra_params: list[Any] = [exclude_id] if exclude_id is not None else []

    rows = conn.execute(
        f"""
        SELECT id, name, entity_type,
               (1 - (embedding <=> %s::vector)) AS similarity
          FROM entities
         WHERE entity_type = %s
           AND valid_until IS NULL
           AND embedding IS NOT NULL
           AND (1 - (embedding <=> %s::vector)) >= %s
           {exclude_clause}
         ORDER BY similarity DESC
         LIMIT %s
        """,
        [vec_lit, entity_type, vec_lit, threshold] + extra_params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def find_merge_candidates(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    entity_type: str | None = None,
    threshold: float = 0.85,
    limit: int = 100,
) -> list[MergeCandidate]:
    """Return scored pairs of active entities that are likely duplicates.

    Performs a self-join restricted to entities sharing the same entity_type
    and both having a non-null embedding.  Only pairs whose cosine similarity
    meets *threshold* are returned, sorted by descending similarity.
    """
    type_clause = "AND a.entity_type = %s" if entity_type else ""
    type_params: list[Any] = [entity_type] if entity_type else []

    rows = conn.execute(
        f"""
        SELECT a.id AS a_id, b.id AS b_id,
               (1 - (a.embedding <=> b.embedding)) AS similarity
          FROM entities a
          JOIN entities b
            ON a.entity_type = b.entity_type
           AND a.id < b.id
         WHERE a.valid_until IS NULL
           AND b.valid_until IS NULL
           AND a.embedding IS NOT NULL
           AND b.embedding IS NOT NULL
           AND (1 - (a.embedding <=> b.embedding)) >= %s
           {type_clause}
         ORDER BY similarity DESC
         LIMIT %s
        """,
        [threshold] + type_params + [limit],
    ).fetchall()

    return [
        MergeCandidate(
            entity_a_id=row["a_id"],
            entity_b_id=row["b_id"],
            similarity=float(row["similarity"]),
            method="embedding",
        )
        for row in rows
    ]
