"""add source_versions table for crawl-level version tracking

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS source_versions (
            source_type  TEXT        NOT NULL,
            source_ref   TEXT        NOT NULL,
            version_key  TEXT        NOT NULL,
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (source_type, source_ref)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS source_versions")
