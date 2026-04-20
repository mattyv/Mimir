"""DDL constants and schema-creation helpers for Mimir's Postgres schema."""

from __future__ import annotations

from typing import Any

import psycopg

# ---------------------------------------------------------------------------
# DDL — executed by Alembic migrations and test fixtures
# ---------------------------------------------------------------------------

# Enable pgvector extension (idempotent)
ENABLE_PGVECTOR = "CREATE EXTENSION IF NOT EXISTS vector"

# graph_meta tracks a monotonically increasing version counter that is
# bumped inside every write transaction via bump_graph_version().
CREATE_GRAPH_META = """
CREATE TABLE IF NOT EXISTS graph_meta (
    id             INTEGER PRIMARY KEY DEFAULT 1,
    version        BIGINT  NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
)
"""

INSERT_GRAPH_META_SENTINEL = """
INSERT INTO graph_meta (id, version) VALUES (1, 0)
ON CONFLICT (id) DO NOTHING
"""

CREATE_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT         PRIMARY KEY,
    entity_type     TEXT         NOT NULL,
    name            TEXT         NOT NULL,
    name_normalized TEXT         NOT NULL,
    description     TEXT         NOT NULL DEFAULT '',
    confidence      REAL         NOT NULL DEFAULT 1.0,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    embedding       VECTOR(384),
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    CONSTRAINT entities_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
)
"""

# Prevents duplicate (name, type) combinations from different ingestion runs.
# The resolver uses ON CONFLICT on this index for upsert-or-increment logic.
CREATE_ENTITIES_UNIQUE_NAME_TYPE = """
CREATE UNIQUE INDEX IF NOT EXISTS entities_name_type_uidx
    ON entities (name_normalized, entity_type)
"""

CREATE_PROPERTIES = """
CREATE TABLE IF NOT EXISTS properties (
    id              BIGSERIAL    PRIMARY KEY,
    entity_id       TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    key             TEXT         NOT NULL,
    value           JSONB        NOT NULL,
    value_type      TEXT         NOT NULL,
    confidence      REAL         NOT NULL DEFAULT 1.0,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    CONSTRAINT properties_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
)
"""

CREATE_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS relationships (
    id              BIGSERIAL    PRIMARY KEY,
    subject_id      TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate       TEXT         NOT NULL,
    object_id       TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    confidence      REAL         NOT NULL DEFAULT 1.0,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    CONSTRAINT relationships_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
)
"""

CREATE_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS observations (
    id              BIGSERIAL    PRIMARY KEY,
    entity_id       TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    observation_type TEXT        NOT NULL,
    description     TEXT         NOT NULL,
    confidence      REAL         NOT NULL DEFAULT 1.0,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    CONSTRAINT observations_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
)
"""

CREATE_CONSTRAINTS = """
CREATE TABLE IF NOT EXISTS constraints (
    id              BIGSERIAL    PRIMARY KEY,
    entity_id       TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    constraint_type TEXT         NOT NULL,
    condition       TEXT         NOT NULL,
    threshold       JSONB,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    CONSTRAINT constraints_type CHECK (
        constraint_type IN ('performance', 'availability', 'legal', 'physical', 'social')
    )
)
"""

CREATE_PROCESSES = """
CREATE TABLE IF NOT EXISTS processes (
    id              TEXT         PRIMARY KEY,
    name            TEXT         NOT NULL,
    name_normalized TEXT         NOT NULL,
    stages          JSONB        NOT NULL,
    inputs          JSONB        NOT NULL,
    outputs         JSONB        NOT NULL,
    slo             TEXT,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0
)
"""

CREATE_PROCESSES_UNIQUE_NAME = """
CREATE UNIQUE INDEX IF NOT EXISTS processes_name_uidx ON processes (name_normalized)
"""

CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT         PRIMARY KEY,
    what            TEXT         NOT NULL,
    why             TEXT         NOT NULL,
    tradeoffs       JSONB        NOT NULL,
    decided_when    TIMESTAMPTZ  NOT NULL,
    who             JSONB        NOT NULL,
    valid_from      TIMESTAMPTZ  NOT NULL,
    valid_until     TIMESTAMPTZ,
    vocabulary_version TEXT      NOT NULL,
    payload         JSONB        NOT NULL DEFAULT '{}',
    graph_version   BIGINT       NOT NULL DEFAULT 0
)
"""

CREATE_RESOLUTION_QUEUE = """
CREATE TABLE IF NOT EXISTS resolution_queue (
    id            BIGSERIAL    PRIMARY KEY,
    entity_a_id   TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id   TEXT         NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    similarity    REAL         NOT NULL,
    method        TEXT         NOT NULL DEFAULT 'embedding',
    status        TEXT         NOT NULL DEFAULT 'pending'
                               CHECK (status IN ('pending','approved','rejected')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at   TIMESTAMPTZ,
    UNIQUE (entity_a_id, entity_b_id)
)
"""

CREATE_SOURCE_SATURATION = """
CREATE TABLE IF NOT EXISTS source_saturation (
    id            BIGSERIAL    PRIMARY KEY,
    source_type   TEXT         NOT NULL,
    source_ref    TEXT         NOT NULL,
    run_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    entities_new  INTEGER      NOT NULL DEFAULT 0,
    entities_seen INTEGER      NOT NULL DEFAULT 0,
    saturation_pct REAL        NOT NULL DEFAULT 0.0,
    UNIQUE (source_type, source_ref, run_at)
)
"""

CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL    PRIMARY KEY,
    table_name      TEXT         NOT NULL,
    row_id          TEXT         NOT NULL,
    operation       TEXT         NOT NULL CHECK (operation IN ('insert','update','delete')),
    graph_version   BIGINT       NOT NULL DEFAULT 0,
    worker_id       TEXT,
    run_id          TEXT,
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
)
"""

# Ordered list of all DDL statements to apply when creating schema from scratch
ALL_DDL: tuple[str, ...] = (
    ENABLE_PGVECTOR,
    CREATE_GRAPH_META,
    INSERT_GRAPH_META_SENTINEL,
    CREATE_ENTITIES,
    CREATE_ENTITIES_UNIQUE_NAME_TYPE,
    CREATE_PROPERTIES,
    CREATE_RELATIONSHIPS,
    CREATE_OBSERVATIONS,
    CREATE_CONSTRAINTS,
    CREATE_PROCESSES,
    CREATE_PROCESSES_UNIQUE_NAME,
    CREATE_DECISIONS,
    CREATE_RESOLUTION_QUEUE,
    CREATE_SOURCE_SATURATION,
    CREATE_AUDIT_LOG,
)

# Teardown order respects FK constraints
DROP_ALL_DDL: tuple[str, ...] = (
    "DROP TABLE IF EXISTS audit_log CASCADE",
    "DROP TABLE IF EXISTS source_saturation CASCADE",
    "DROP TABLE IF EXISTS resolution_queue CASCADE",
    "DROP TABLE IF EXISTS decisions CASCADE",
    "DROP TABLE IF EXISTS processes CASCADE",
    "DROP TABLE IF EXISTS constraints CASCADE",
    "DROP TABLE IF EXISTS observations CASCADE",
    "DROP TABLE IF EXISTS relationships CASCADE",
    "DROP TABLE IF EXISTS properties CASCADE",
    "DROP TABLE IF EXISTS entities CASCADE",
    "DROP TABLE IF EXISTS graph_meta CASCADE",
    "DROP EXTENSION IF EXISTS vector CASCADE",
)


def apply_schema(conn: psycopg.Connection[Any]) -> None:
    """Apply ALL_DDL against *conn* inside the caller's transaction."""
    for ddl in ALL_DDL:
        conn.execute(ddl)


def drop_schema(conn: psycopg.Connection[Any]) -> None:
    """Drop all Mimir tables from *conn* inside the caller's transaction."""
    for ddl in DROP_ALL_DDL:
        conn.execute(ddl)
