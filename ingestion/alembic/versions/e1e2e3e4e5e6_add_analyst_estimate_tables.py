"""Add analyst-estimate tables: earnings_estimate, revenue_estimate, growth_estimates.

Dedicated typed tables (SPEC OQ5) for the forward-period analyst estimates the
client already fetches (earnings/revenue/growth).

Revision ID: e1e2e3e4e5e6
Revises: f7b8c9d0e1a2
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e1e2e3e4e5e6"
down_revision: str | Sequence[str] | None = "f7b8c9d0e1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _id_and_timestamps() -> list[sa.Column]:
    return [
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
    ]


def _instrument_fk() -> sa.Column:
    return sa.Column(
        "instrument_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "earnings_estimate",
        _instrument_fk(),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("num_analysts", sa.Integer(), nullable=True),
        sa.Column("avg", sa.Numeric(20, 6), nullable=True),
        sa.Column("low", sa.Numeric(20, 6), nullable=True),
        sa.Column("high", sa.Numeric(20, 6), nullable=True),
        sa.Column("year_ago_eps", sa.Numeric(20, 6), nullable=True),
        sa.Column("growth", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period", name="uq_earnings_estimate_instrument_period"
        ),
    )
    op.create_index(
        "ix_earnings_estimate_instrument_id", "earnings_estimate", ["instrument_id"]
    )

    op.create_table(
        "revenue_estimate",
        _instrument_fk(),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("num_analysts", sa.Integer(), nullable=True),
        sa.Column("avg", sa.Numeric(38, 2), nullable=True),
        sa.Column("low", sa.Numeric(38, 2), nullable=True),
        sa.Column("high", sa.Numeric(38, 2), nullable=True),
        sa.Column("year_ago_revenue", sa.Numeric(38, 2), nullable=True),
        sa.Column("growth", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period", name="uq_revenue_estimate_instrument_period"
        ),
    )
    op.create_index(
        "ix_revenue_estimate_instrument_id", "revenue_estimate", ["instrument_id"]
    )

    op.create_table(
        "growth_estimates",
        _instrument_fk(),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("stock_trend", sa.Numeric(20, 6), nullable=True),
        sa.Column("industry_trend", sa.Numeric(20, 6), nullable=True),
        sa.Column("sector_trend", sa.Numeric(20, 6), nullable=True),
        sa.Column("index_trend", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period", name="uq_growth_estimate_instrument_period"
        ),
    )
    op.create_index(
        "ix_growth_estimates_instrument_id", "growth_estimates", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_growth_estimates_instrument_id", table_name="growth_estimates")
    op.drop_table("growth_estimates")
    op.drop_index("ix_revenue_estimate_instrument_id", table_name="revenue_estimate")
    op.drop_table("revenue_estimate")
    op.drop_index("ix_earnings_estimate_instrument_id", table_name="earnings_estimate")
    op.drop_table("earnings_estimate")
