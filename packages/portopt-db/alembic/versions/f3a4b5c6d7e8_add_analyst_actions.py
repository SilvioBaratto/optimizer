"""Add analyst_actions table (upgrades/downgrades).

Task A3 (Gap A): persist individual analyst upgrade/downgrade actions from
yf.Ticker.upgrades_downgrades.

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f3a4b5c6d7e8"
down_revision: str | Sequence[str] | None = "e2f3a4b5c6d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analyst_actions",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action_date", sa.Date(), nullable=False),
        sa.Column("firm", sa.String(200), nullable=False),
        sa.Column("from_grade", sa.String(100), nullable=True),
        sa.Column("to_grade", sa.String(100), nullable=False),
        sa.Column("action", sa.String(50), nullable=True),
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
        sa.UniqueConstraint(
            "instrument_id", "action_date", "firm", "to_grade", name="uq_analyst_action"
        ),
    )
    op.create_index(
        "ix_analyst_actions_instrument_id", "analyst_actions", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_analyst_actions_instrument_id", table_name="analyst_actions")
    op.drop_table("analyst_actions")
