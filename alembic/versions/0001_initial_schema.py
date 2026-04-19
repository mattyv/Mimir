"""Initial schema — entities, properties, relationships, observations, graph_meta.

Revision ID: 0001
Revises:
Create Date: 2026-04-19
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE graph_meta (
            id         INTEGER PRIMARY KEY DEFAULT 1,
            version    BIGINT  NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT single_row CHECK (id = 1)
        )
        """
    )
    op.execute("INSERT INTO graph_meta (id, version) VALUES (1, 0) ON CONFLICT (id) DO NOTHING")

    op.execute(
        """
        CREATE TABLE entities (
            id                 TEXT        PRIMARY KEY,
            entity_type        TEXT        NOT NULL,
            name               TEXT        NOT NULL,
            name_normalized    TEXT        NOT NULL,
            description        TEXT        NOT NULL DEFAULT '',
            confidence         REAL        NOT NULL DEFAULT 1.0,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            embedding          VECTOR(384),
            graph_version      BIGINT      NOT NULL DEFAULT 0,
            CONSTRAINT entities_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX entities_name_type_uidx
            ON entities (name_normalized, entity_type)
        """
    )

    op.execute(
        """
        CREATE TABLE properties (
            id                 BIGSERIAL   PRIMARY KEY,
            entity_id          TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            key                TEXT        NOT NULL,
            value              JSONB       NOT NULL,
            value_type         TEXT        NOT NULL,
            confidence         REAL        NOT NULL DEFAULT 1.0,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0,
            CONSTRAINT properties_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE relationships (
            id                 BIGSERIAL   PRIMARY KEY,
            subject_id         TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            predicate          TEXT        NOT NULL,
            object_id          TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            confidence         REAL        NOT NULL DEFAULT 1.0,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0,
            CONSTRAINT relationships_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE observations (
            id                 BIGSERIAL   PRIMARY KEY,
            entity_id          TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            observation_type   TEXT        NOT NULL,
            description        TEXT        NOT NULL,
            confidence         REAL        NOT NULL DEFAULT 1.0,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0,
            CONSTRAINT observations_confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS observations CASCADE")
    op.execute("DROP TABLE IF EXISTS relationships CASCADE")
    op.execute("DROP TABLE IF EXISTS properties CASCADE")
    op.execute("DROP TABLE IF EXISTS entities CASCADE")
    op.execute("DROP TABLE IF EXISTS graph_meta CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector CASCADE")
