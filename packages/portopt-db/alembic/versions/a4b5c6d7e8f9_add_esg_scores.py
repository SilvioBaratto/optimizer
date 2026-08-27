"""Add esg_scores table.

Task A4 (Gap A): persist the latest ESG / sustainability snapshot from
yf.Ticker.sustainability.

Revision ID: a4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a4b5c6d7e8f9"
down_revision: str | Sequence[str] | None = "f3a4b5c6d7e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "esg_scores",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("total_esg", sa.Numeric(20, 6), nullable=True),
        sa.Column("environment_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("social_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("governance_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("highest_controversy", sa.Numeric(20, 6), nullable=True),
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("instrument_id", name="uq_esg_score_instrument"),
    )
    op.create_index("ix_esg_scores_instrument_id", "esg_scores", ["instrument_id"])


def downgrade() -> None:
    op.drop_index("ix_esg_scores_instrument_id", table_name="esg_scores")
    op.drop_table("esg_scores")
