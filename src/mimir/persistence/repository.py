"""Repository pattern for Mimir's persistence layer.

Each repository is a thin stateless object that accepts a psycopg
connection and converts between Pydantic models and Postgres rows.

Bitemporal query semantics
--------------------------
as_of : datetime | None
    Filter by the *real-world* interval.  A row is visible when:
        valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)
    When None, only rows with valid_until IS NULL (currently active) are
    returned.
at_version : int | None
    Filter by the graph_version stamp applied at write time.  When None,
    no version filter is applied.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from typing import Any

import psycopg

from mimir.models.nodes import Entity, Observation, Property, Relationship
from mimir.persistence.graph_version import bump_graph_version

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Casefold + NFC-normalize entity name for dedup index."""
    return unicodedata.normalize("NFC", name).casefold().strip()


def _as_of_clause(alias: str, as_of: datetime | None) -> tuple[str, list[Any]]:
    """Return (WHERE fragment, params) for a bitemporal as_of filter."""
    if as_of is None:
        return f"{alias}.valid_until IS NULL", []
    return (
        f"{alias}.valid_from <= %s AND ({alias}.valid_until IS NULL OR {alias}.valid_until > %s)",
        [as_of, as_of],
    )


def _version_clause(alias: str, at_version: int | None) -> tuple[str, list[Any]]:
    """Return (WHERE fragment, params) for an at_version filter."""
    if at_version is None:
        return "", []
    return f"{alias}.graph_version <= %s", [at_version]


def _build_where(
    clauses: list[tuple[str, list[Any]]],
) -> tuple[str, list[Any]]:
    """Combine non-empty (fragment, params) pairs into a single WHERE clause."""
    non_empty = [(c, p) for c, p in clauses if c]
    if not non_empty:
        return "", []
    fragments = " AND ".join(c for c, _ in non_empty)
    params: list[Any] = []
    for _, p in non_empty:
        params.extend(p)
    return f"WHERE {fragments}", params


# ---------------------------------------------------------------------------
# EntityRepository
# ---------------------------------------------------------------------------


class EntityRepository:
    """CRUD + bitemporal queries for Entity nodes."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def upsert(self, entity: Entity) -> bool:
        """Insert or update an Entity row.

        Returns True if a new row was inserted, False if an existing row was
        updated.  Uses the (name_normalized, entity_type) unique index for
        conflict detection, so callers do not need to pre-check for existence.
        The graph version is bumped exactly once per call.
        """
        version = bump_graph_version(self._conn)
        payload = entity.model_dump(
            mode="json",
            exclude={"id", "type", "name", "description", "confidence", "temporal", "vocabulary_version"},
        )
        result = self._conn.execute(
            """
            INSERT INTO entities
                (id, entity_type, name, name_normalized, description, confidence,
                 valid_from, valid_until, vocabulary_version, payload, graph_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name_normalized, entity_type)
            DO UPDATE SET
                description       = EXCLUDED.description,
                confidence        = EXCLUDED.confidence,
                valid_from        = EXCLUDED.valid_from,
                valid_until       = EXCLUDED.valid_until,
                vocabulary_version = EXCLUDED.vocabulary_version,
                payload           = EXCLUDED.payload,
                graph_version     = EXCLUDED.graph_version
            RETURNING (xmax = 0) AS inserted
            """,
            (
                entity.id,
                entity.type,
                entity.name,
                _normalize_name(entity.name),
                entity.description,
                entity.confidence,
                entity.temporal.valid_from,
                entity.temporal.valid_until,
                entity.vocabulary_version,
                json.dumps(payload),
                version,
            ),
        )
        row = result.fetchone()
        return bool(row["inserted"]) if row else True

    def get(
        self,
        entity_id: str,
        *,
        as_of: datetime | None = None,
        at_version: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a single entity row by id with optional bitemporal filters."""
        a_clause, a_params = _as_of_clause("e", as_of)
        v_clause, v_params = _version_clause("e", at_version)
        where, params = _build_where(
            [("e.id = %s", [entity_id]), (a_clause, a_params), (v_clause, v_params)]
        )
        row = self._conn.execute(
            f"SELECT * FROM entities e {where} LIMIT 1",
            params,
        ).fetchone()
        return dict(row) if row else None

    def list_active(
        self,
        *,
        entity_type: str | None = None,
        as_of: datetime | None = None,
        at_version: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return active entity rows with optional type + bitemporal filters."""
        a_clause, a_params = _as_of_clause("e", as_of)
        v_clause, v_params = _version_clause("e", at_version)
        t_clause: tuple[str, list[Any]] = (
            ("e.entity_type = %s", [entity_type]) if entity_type else ("", [])
        )
        where, params = _build_where([t_clause, (a_clause, a_params), (v_clause, v_params)])
        rows = self._conn.execute(
            f"SELECT * FROM entities e {where} ORDER BY e.name LIMIT %s OFFSET %s",
            params + [limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, entity_id: str) -> bool:
        """Hard-delete an entity and all dependent rows (via CASCADE).

        Returns True if a row was deleted, False if not found.
        """
        result = self._conn.execute(
            "DELETE FROM entities WHERE id = %s RETURNING id",
            (entity_id,),
        )
        return result.rowcount > 0

    def count(self) -> int:
        """Return total number of entity rows."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()
        return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# PropertyRepository
# ---------------------------------------------------------------------------


class PropertyRepository:
    """CRUD + bitemporal queries for Property nodes."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def insert(self, prop: Property) -> int:
        """Insert a Property row and return its auto-assigned id."""
        version = bump_graph_version(self._conn)
        payload = prop.model_dump(
            mode="json",
            exclude={"entity_id", "key", "value", "value_type", "confidence", "temporal", "vocabulary_version"},
        )
        row = self._conn.execute(
            """
            INSERT INTO properties
                (entity_id, key, value, value_type, confidence,
                 valid_from, valid_until, vocabulary_version, payload, graph_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                prop.entity_id,
                prop.key,
                json.dumps(prop.value),
                prop.value_type,
                prop.confidence,
                prop.temporal.valid_from,
                prop.temporal.valid_until,
                prop.vocabulary_version,
                json.dumps(payload),
                version,
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def list_for_entity(
        self,
        entity_id: str,
        *,
        as_of: datetime | None = None,
        at_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all property rows for a given entity."""
        a_clause, a_params = _as_of_clause("p", as_of)
        v_clause, v_params = _version_clause("p", at_version)
        where, params = _build_where(
            [("p.entity_id = %s", [entity_id]), (a_clause, a_params), (v_clause, v_params)]
        )
        rows = self._conn.execute(
            f"SELECT * FROM properties p {where} ORDER BY p.key, p.id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# RelationshipRepository
# ---------------------------------------------------------------------------


class RelationshipRepository:
    """CRUD + bitemporal queries for Relationship nodes."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def insert(self, rel: Relationship) -> int:
        """Insert a Relationship row and return its auto-assigned id."""
        version = bump_graph_version(self._conn)
        payload = rel.model_dump(
            mode="json",
            exclude={"subject_id", "predicate", "object_id", "confidence", "temporal", "vocabulary_version"},
        )
        row = self._conn.execute(
            """
            INSERT INTO relationships
                (subject_id, predicate, object_id, confidence,
                 valid_from, valid_until, vocabulary_version, payload, graph_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                rel.subject_id,
                rel.predicate,
                rel.object_id,
                rel.confidence,
                rel.temporal.valid_from,
                rel.temporal.valid_until,
                rel.vocabulary_version,
                json.dumps(payload),
                version,
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def list_for_subject(
        self,
        subject_id: str,
        *,
        as_of: datetime | None = None,
        at_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all relationships where subject_id matches."""
        a_clause, a_params = _as_of_clause("r", as_of)
        v_clause, v_params = _version_clause("r", at_version)
        where, params = _build_where(
            [("r.subject_id = %s", [subject_id]), (a_clause, a_params), (v_clause, v_params)]
        )
        rows = self._conn.execute(
            f"SELECT * FROM relationships r {where} ORDER BY r.predicate, r.id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def list_for_object(
        self,
        object_id: str,
        *,
        as_of: datetime | None = None,
        at_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all relationships where object_id matches."""
        a_clause, a_params = _as_of_clause("r", as_of)
        v_clause, v_params = _version_clause("r", at_version)
        where, params = _build_where(
            [("r.object_id = %s", [object_id]), (a_clause, a_params), (v_clause, v_params)]
        )
        rows = self._conn.execute(
            f"SELECT * FROM relationships r {where} ORDER BY r.predicate, r.id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# ObservationRepository
# ---------------------------------------------------------------------------


class ObservationRepository:
    """CRUD + bitemporal queries for Observation nodes."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def insert(self, obs: Observation) -> int:
        """Insert an Observation row and return its auto-assigned id."""
        version = bump_graph_version(self._conn)
        payload = obs.model_dump(
            mode="json",
            exclude={"entity_id", "type", "description", "confidence", "temporal", "vocabulary_version"},
        )
        row = self._conn.execute(
            """
            INSERT INTO observations
                (entity_id, observation_type, description, confidence,
                 valid_from, valid_until, vocabulary_version, payload, graph_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                obs.entity_id,
                obs.type,
                obs.description,
                obs.confidence,
                obs.temporal.valid_from,
                obs.temporal.valid_until,
                obs.vocabulary_version,
                json.dumps(payload),
                version,
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def list_for_entity(
        self,
        entity_id: str,
        *,
        observation_type: str | None = None,
        as_of: datetime | None = None,
        at_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return observation rows for a given entity."""
        a_clause, a_params = _as_of_clause("o", as_of)
        v_clause, v_params = _version_clause("o", at_version)
        t_clause: tuple[str, list[Any]] = (
            ("o.observation_type = %s", [observation_type]) if observation_type else ("", [])
        )
        where, params = _build_where(
            [("o.entity_id = %s", [entity_id]), t_clause, (a_clause, a_params), (v_clause, v_params)]
        )
        rows = self._conn.execute(
            f"SELECT * FROM observations o {where} ORDER BY o.id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
