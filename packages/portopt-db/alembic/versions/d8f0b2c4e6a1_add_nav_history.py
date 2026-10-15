"""Add the persisted marked-to-market NAV series table (``nav_history``).

One row per ``(portfolio_id, date)`` holding the base-currency portfolio value
plus an optional base-normalized benchmark level. Written by the fund daily
NAV-mark step; read by the performance tracker. No FK — the ``portfolios`` table
was dropped in the ingestion strip — so ``portfolio_id`` is a bare portable UUID.

Revision ID: d8f0b2c4e6a1
Revises: c7e9a1b3d5f0
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d8f0b2c4e6a1"
down_revision: str | Sequence[str] | None = "c7e9a1b3d5f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``nav_history`` table with its portfolio-date uniqueness guard."""
    op.create_table(
        "nav_history",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("portfolio_id", sa.Uuid(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("nav", sa.Numeric(20, 6), nullable=False),
        sa.Column("benchmark", sa.Numeric(20, 6), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "portfolio_id", "date", name="uq_nav_history_portfolio_date"
        ),
    )
    op.create_index("ix_nav_history_portfolio_id", "nav_history", ["portfolio_id"])
    op.create_index("ix_nav_history_date", "nav_history", ["date"])


def downgrade() -> None:
    """Drop the ``nav_history`` table and its indexes."""
    op.drop_index("ix_nav_history_date", table_name="nav_history")
    op.drop_index("ix_nav_history_portfolio_id", table_name="nav_history")
    op.drop_table("nav_history")
