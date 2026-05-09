"""Add regime_classification column to macro_calibrations.

Issue #530: cache rule-based MacroRegime classifier output alongside the
BAML LLM phase.  Backward-compatible — column is nullable; no data migration.

Revision ID: b1c2d3e4f5g6
Revises: 2a3be07fc983
Create Date: 2026-05-09
"""

import sqlalchemy as sa

from alembic import op

revision: str = "b1c2d3e4f5g6"
down_revision: str | None = "2a3be07fc983"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "macro_calibrations",
        sa.Column("regime_classification", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("macro_calibrations", "regime_classification")
