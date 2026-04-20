"""Phase 5 — context injection tests (fetch_context_entities, format_context_for_prompt)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from mimir.adapters.base import Chunk
from mimir.crawler.context import (
    _strategy_source_adjacent,
    _strategy_token_prefix,
    fetch_context_entities,
    format_context_for_prompt,
)

_TEST_DSN = os.environ.get("DATABASE_URL", "dbname=mimir_test user=root")
_NOW = datetime(2026, 4, 19, tzinfo=UTC)


def _chunk(
    content: str = "some content",
    reference: str = "https://wiki.example.com/page",
    chunk_id: str = "ctx_chunk",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="confluence",
        content=content,
        acl=["internal"],
        retrieved_at=_NOW,
        reference=reference,
    )


def _insert_entity(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    name: str,
    entity_type: str = "schema:Thing",
    description: str = "",
    reference: str = "https://wiki.example.com/page",
    name_normalized: str | None = None,
) -> str:
    """Insert a minimal entity row directly and return its id."""
    e_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}:{entity_type}"))
    norm = name_normalized if name_normalized is not None else name.casefold().strip()
    payload = {
        "source": {
            "type": "confluence",
            "reference": reference,
            "retrieved_at": _NOW.isoformat(),
        }
    }
    conn.execute(
        """
        INSERT INTO entities
            (id, entity_type, name, name_normalized, description,
             confidence, valid_from, valid_until, vocabulary_version, payload, graph_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, 1)
        ON CONFLICT (name_normalized, entity_type) DO NOTHING
        """,
        (
            e_id,
            entity_type,
            name,
            norm,
            description,
            0.8,
            _NOW,
            "0.1.0",
            json.dumps(payload),
        ),
    )
    return e_id


# ---------------------------------------------------------------------------
# Basic / empty-DB tests
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_fetch_context_empty_db(_pg_schema: None) -> None:
    """An empty database returns an empty context list."""
    with psycopg.connect(_TEST_DSN, row_factory=dict_row, autocommit=False) as conn:
        conn.execute("BEGIN")
        try:
            chunk = _chunk()
            result = fetch_context_entities(chunk, conn)
            assert result == []
        finally:
            conn.rollback()


# ---------------------------------------------------------------------------
# Strategy: source_adjacent
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_strategy_source_adjacent(_pg_schema: None) -> None:
    """Entity with matching source reference appears via source_adjacent strategy."""
    ref = "https://wiki.example.com/target-page"
    with psycopg.connect(_TEST_DSN, row_factory=dict_row, autocommit=False) as conn:
        conn.execute("BEGIN")
        try:
            _insert_entity(conn, name="Alpha", reference=ref)
            # Entity with a different reference — should NOT appear
            _insert_entity(
                conn,
                name="Beta",
                reference="https://other.example.com/page",
            )
            chunk = _chunk(reference=ref)
            rows = _strategy_source_adjacent(chunk, conn)
            names = [r["name"] for r in rows]
            assert "Alpha" in names
            assert "Beta" not in names
        finally:
            conn.rollback()


# ---------------------------------------------------------------------------
# Strategy: token_prefix
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_strategy_token_prefix(_pg_schema: None) -> None:
    """Entity whose name_normalized matches a token in the chunk appears via token_prefix."""
    with psycopg.connect(_TEST_DSN, row_factory=dict_row, autocommit=False) as conn:
        conn.execute("BEGIN")
        try:
            # name_normalized = "omms", chunk content contains the word "omms"
            _insert_entity(
                conn,
                name="OMMS",
                name_normalized="omms",
                reference="https://other.example.com",
            )
            _insert_entity(
                conn,
                name="UnrelatedService",
                name_normalized="unrelatedservice",
                reference="https://other.example.com",
            )
            chunk = _chunk(content="The omms service handles trading.")
            rows = _strategy_token_prefix(chunk, conn)
            names = [r["name"] for r in rows]
            assert "OMMS" in names
            assert "UnrelatedService" not in names
        finally:
            conn.rollback()


# ---------------------------------------------------------------------------
# format_context_for_prompt
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_format_context_for_prompt_empty() -> None:
    """Empty entity list returns an empty string."""
    result = format_context_for_prompt([])
    assert result == ""


@pytest.mark.phase5
def test_format_context_for_prompt_entities() -> None:
    """A list of two entities produces a multi-line string with a header."""
    entities = [
        {"id": "id-1", "name": "ServiceA", "entity_type": "schema:Thing", "description": "desc a"},
        {"id": "id-2", "name": "ServiceB", "entity_type": "auros:TradingService", "description": ""},
    ]
    result = format_context_for_prompt(entities)
    lines = result.splitlines()
    # First line is the header
    assert lines[0].startswith("Known entities")
    # Both entity names appear somewhere in the output
    assert "ServiceA" in result
    assert "ServiceB" in result
    # Should be at least 3 lines: header + 2 entity lines
    assert len(lines) >= 3


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_fetch_context_deduplication(_pg_schema: None) -> None:
    """Same entity matched by multiple strategies appears only once in result."""
    ref = "https://wiki.example.com/shared-page"
    with psycopg.connect(_TEST_DSN, row_factory=dict_row, autocommit=False) as conn:
        conn.execute("BEGIN")
        try:
            # Entity whose name_normalized matches a chunk token AND whose
            # source reference matches the chunk reference — both strategies fire.
            _insert_entity(
                conn,
                name="omms",
                name_normalized="omms",
                reference=ref,
            )
            chunk = _chunk(content="The omms service.", reference=ref)
            result = fetch_context_entities(chunk, conn)
            ids = [r["id"] for r in result]
            assert len(ids) == len(set(ids)), "duplicate entity ids in result"
            assert len([r for r in result if r["name"] == "omms"]) == 1
        finally:
            conn.rollback()


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


@pytest.mark.phase5
def test_fetch_context_cap(_pg_schema: None) -> None:
    """Result is capped at 50 entities even when more than 50 match."""
    with psycopg.connect(_TEST_DSN, row_factory=dict_row, autocommit=False) as conn:
        conn.execute("BEGIN")
        try:
            # Insert 60 entities whose name_normalized matches a token in the chunk.
            # Use a shared token so all 60 match via token_prefix.
            token = "sharedtoken"
            for i in range(60):
                unique_name = f"Entity{i:03d}{token}"
                _insert_entity(
                    conn,
                    name=unique_name,
                    name_normalized=f"entity{i:03d}{token}",
                    reference="https://other.example.com",
                )
            # The chunk content contains the token, which is a suffix of each
            # name_normalized — we need exact word match.  Insert 60 single-word
            # normalized names that equal the token.
            for i in range(60):
                word = f"tok{i:03d}"
                _insert_entity(
                    conn,
                    name=f"Tok{i:03d}",
                    name_normalized=word,
                    reference="https://other.example.com",
                    entity_type=f"schema:Thing{i}",
                )
            # Build chunk content containing all 60 words
            words = " ".join(f"tok{i:03d}" for i in range(60))
            chunk = _chunk(content=words, reference="https://nowhere.example.com")
            result = fetch_context_entities(chunk, conn, cap=50)
            assert len(result) <= 50
        finally:
            conn.rollback()
