"""add resolution_queue, source_saturation and audit_log tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-20
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
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
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS source_saturation (
            id             BIGSERIAL    PRIMARY KEY,
            source_type    TEXT         NOT NULL,
            source_ref     TEXT         NOT NULL,
            run_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            entities_new   INTEGER      NOT NULL DEFAULT 0,
            entities_seen  INTEGER      NOT NULL DEFAULT 0,
            saturation_pct REAL         NOT NULL DEFAULT 0.0,
            UNIQUE (source_type, source_ref, run_at)
        )
    """)
    op.execute("""
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
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP TABLE IF EXISTS source_saturation")
    op.execute("DROP TABLE IF EXISTS resolution_queue")
