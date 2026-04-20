"""Phase 2 — Schema DDL and graph_meta tests."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

pytestmark = pytest.mark.phase2


def test_schema_creates_entities_table(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'entities' ORDER BY column_name"
    ).fetchall()
    columns = {r["column_name"] for r in row}
    assert {
        "id",
        "entity_type",
        "name",
        "name_normalized",
        "confidence",
        "valid_from",
        "valid_until",
        "payload",
        "embedding",
        "graph_version",
    } <= columns


def test_schema_creates_properties_table(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'properties' ORDER BY column_name"
    ).fetchall()
    columns = {r["column_name"] for r in row}
    assert {"id", "entity_id", "key", "value", "value_type", "confidence"} <= columns


def test_schema_creates_relationships_table(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'relationships' ORDER BY column_name"
    ).fetchall()
    columns = {r["column_name"] for r in row}
    assert {"id", "subject_id", "predicate", "object_id", "confidence"} <= columns


def test_schema_creates_observations_table(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'observations' ORDER BY column_name"
    ).fetchall()
    columns = {r["column_name"] for r in row}
    assert {"id", "entity_id", "observation_type", "description", "confidence"} <= columns


def test_schema_graph_meta_sentinel_exists(pg: psycopg.Connection[Any]) -> None:
    row = pg.execute("SELECT version FROM graph_meta WHERE id = 1").fetchone()
    assert row is not None
    assert row["version"] == 0


def test_schema_idempotent_apply(pg: psycopg.Connection[Any]) -> None:
    """apply_schema can be called multiple times without error (IF NOT EXISTS)."""
    from mimir.persistence.schema import apply_schema

    apply_schema(pg)  # second call inside same transaction
    row = pg.execute("SELECT COUNT(*) AS n FROM entities").fetchone()
    assert row is not None
    assert row["n"] == 0


def test_drop_schema_removes_tables(pg: psycopg.Connection[Any]) -> None:
    from mimir.persistence.schema import drop_schema

    drop_schema(pg)
    row = pg.execute(
        "SELECT COUNT(*) AS n FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name IN "
        "('entities','properties','relationships','observations','graph_meta')"
    ).fetchone()
    assert row is not None
    assert row["n"] == 0
