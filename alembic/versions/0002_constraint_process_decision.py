"""Add constraints, processes, decisions tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-19
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE constraints (
            id                 BIGSERIAL   PRIMARY KEY,
            entity_id          TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            constraint_type    TEXT        NOT NULL,
            condition          TEXT        NOT NULL,
            threshold          JSONB,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0,
            CONSTRAINT constraints_type CHECK (
                constraint_type IN ('performance', 'availability', 'legal', 'physical', 'social')
            )
        )
        """
    )

    op.execute(
        """
        CREATE TABLE processes (
            id                 TEXT        PRIMARY KEY,
            name               TEXT        NOT NULL,
            name_normalized    TEXT        NOT NULL,
            stages             JSONB       NOT NULL,
            inputs             JSONB       NOT NULL,
            outputs            JSONB       NOT NULL,
            slo                TEXT,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX processes_name_uidx ON processes (name_normalized)"
    )

    op.execute(
        """
        CREATE TABLE decisions (
            id                 TEXT        PRIMARY KEY,
            what               TEXT        NOT NULL,
            why                TEXT        NOT NULL,
            tradeoffs          JSONB       NOT NULL,
            decided_when       TIMESTAMPTZ NOT NULL,
            who                JSONB       NOT NULL,
            valid_from         TIMESTAMPTZ NOT NULL,
            valid_until        TIMESTAMPTZ,
            vocabulary_version TEXT        NOT NULL,
            payload            JSONB       NOT NULL DEFAULT '{}',
            graph_version      BIGINT      NOT NULL DEFAULT 0
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS decisions CASCADE")
    op.execute("DROP TABLE IF EXISTS processes CASCADE")
    op.execute("DROP TABLE IF EXISTS constraints CASCADE")
