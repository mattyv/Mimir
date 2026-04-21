"""Known-entity context injection for extraction pipeline.

Fetches existing graph entities to inject into the LLM prompt so that
cross-chunk coreference works reliably.  Four strategies are combined,
deduped by entity id, and capped at 50 entries.
"""

from __future__ import annotations

from typing import Any

import psycopg

from mimir.adapters.base import Chunk


def _strategy_source_adjacent(
    chunk: Chunk,
    conn: psycopg.Connection[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Entities extracted from the same source reference."""
    rows = conn.execute(
        """
        SELECT id, name, entity_type, description
          FROM entities
         WHERE valid_until IS NULL
           AND payload->'source'->>'reference' = %s
         ORDER BY name
         LIMIT %s
        """,
        (chunk.reference, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _strategy_token_prefix(
    chunk: Chunk,
    conn: psycopg.Connection[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Entities whose normalized name appears as a token in the chunk content."""
    words = set(chunk.content.casefold().split())
    if not words:
        return []
    # Fetch a broader set and filter in Python to avoid huge SQL IN clauses
    rows = conn.execute(
        """
        SELECT id, name, entity_type, description, name_normalized
          FROM entities
         WHERE valid_until IS NULL
         ORDER BY name
         LIMIT 500
        """
    ).fetchall()
    matched = [dict(r) for r in rows if r["name_normalized"] in words]
    return matched[:limit]


def _strategy_embedding_nn(
    chunk: Chunk,
    conn: psycopg.Connection[dict[str, Any]],
    embedder: Any,
    limit: int = 20,
    threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Entities whose embedding is close to the chunk text embedding."""
    from mimir.resolution.embedder import _vec_sql, compute_embedding

    vec = compute_embedding(chunk.content[:512], embedder)
    vec_lit = _vec_sql(vec)
    rows = conn.execute(
        """
        SELECT id, name, entity_type, description,
               (1 - (embedding <=> %s::vector)) AS sim
          FROM entities
         WHERE valid_until IS NULL
           AND embedding IS NOT NULL
           AND (1 - (embedding <=> %s::vector)) >= %s
         ORDER BY sim DESC
         LIMIT %s
        """,
        (vec_lit, vec_lit, threshold, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _strategy_graph_walk(
    seed_ids: list[str],
    conn: psycopg.Connection[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Entities one hop away from already-known seed entities."""
    if not seed_ids:
        return []
    rows = conn.execute(
        """
        SELECT DISTINCT e.id, e.name, e.entity_type, e.description
          FROM entities e
          JOIN relationships r
            ON (r.subject_id = e.id OR r.object_id = e.id)
         WHERE (r.subject_id = ANY(%s) OR r.object_id = ANY(%s))
           AND e.id != ALL(%s)
           AND e.valid_until IS NULL
           AND r.valid_until IS NULL
         LIMIT %s
        """,
        (seed_ids, seed_ids, seed_ids, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_context_entities(
    chunk: Chunk,
    conn: psycopg.Connection[dict[str, Any]],
    *,
    embedder: Any | None = None,
    seed_ids: list[str] | None = None,
    cap: int = 50,
) -> list[dict[str, Any]]:
    """Combine all four strategies, dedup by id, and cap at *cap* entities.

    Returns a list of entity dicts with keys: id, name, entity_type, description.
    Embedding strategy is skipped if embedder is None.
    """
    seen: dict[str, dict[str, Any]] = {}

    for entity in _strategy_source_adjacent(chunk, conn):
        seen.setdefault(entity["id"], entity)

    for entity in _strategy_token_prefix(chunk, conn):
        seen.setdefault(entity["id"], entity)

    if embedder is not None:
        try:
            for entity in _strategy_embedding_nn(chunk, conn, embedder):
                seen.setdefault(entity["id"], entity)
        except Exception:
            pass  # embedding unavailable — degrade gracefully

    for entity in _strategy_graph_walk(seed_ids or [], conn):
        seen.setdefault(entity["id"], entity)

    return list(seen.values())[:cap]


def format_context_for_prompt(entities: list[dict[str, Any]]) -> str:
    """Serialise context entity list as a compact string for prompt injection."""
    if not entities:
        return ""
    lines = ["Known entities in the knowledge graph (for reference):"]
    for e in entities:
        lines.append(
            f"  - {e['name']} ({e.get('entity_type', 'unknown')}): {e.get('description', '')[:80]}"
        )
    return "\n".join(lines)
