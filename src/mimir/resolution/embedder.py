"""Sentence-level embedding utilities for the entity resolution pipeline."""

from __future__ import annotations

import math
from typing import Any, Protocol

import psycopg

EMBEDDING_DIM = 384


class Embedder(Protocol):
    """Minimal interface satisfied by sentence-transformers SentenceTransformer."""

    def encode(self, text: str) -> Any:
        """Return a sequence of floats of length EMBEDDING_DIM."""
        ...


def _vec_sql(v: list[float]) -> str:
    """Format a float list as a pgvector literal string, e.g. '[0.1,0.2,…]'."""
    return "[" + ",".join(str(x) for x in v) + "]"


def compute_embedding(text: str, embedder: Embedder) -> list[float]:
    """Return a unit-normalized EMBEDDING_DIM-vector for *text*."""
    raw = embedder.encode(text)
    vec = [float(x) for x in raw]
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim vector, got {len(vec)}")
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def update_entity_embedding(
    entity_id: str,
    embedding: list[float],
    conn: psycopg.Connection[dict[str, Any]],
) -> bool:
    """Persist *embedding* on the entity row. Returns True if the row was found."""
    result = conn.execute(
        "UPDATE entities SET embedding = %s::vector WHERE id = %s RETURNING id",
        (_vec_sql(embedding), entity_id),
    )
    return result.rowcount > 0
