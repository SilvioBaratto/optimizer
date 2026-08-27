"""Add earnings_history + earnings_dates tables.

Task A2 (Gap A): persist historical EPS surprise (yf.Ticker.earnings_history)
and past/upcoming earnings dates (yf.Ticker.get_earnings_dates).

Revision ID: e2f3a4b5c6d7
Revises: e1e2e3e4e5e6
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: str | Sequence[str] | None = "e1e2e3e4e5e6"
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
        "earnings_history",
        _instrument_fk(),
        sa.Column("period_date", sa.Date(), nullable=False),
        sa.Column("eps_estimate", sa.Numeric(20, 6), nullable=True),
        sa.Column("eps_actual", sa.Numeric(20, 6), nullable=True),
        sa.Column("eps_difference", sa.Numeric(20, 6), nullable=True),
        sa.Column("surprise_percent", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period_date", name="uq_earnings_history_instrument_period"
        ),
    )
    op.create_index(
        "ix_earnings_history_instrument_id", "earnings_history", ["instrument_id"]
    )

    op.create_table(
        "earnings_dates",
        _instrument_fk(),
        sa.Column("earnings_date", sa.Date(), nullable=False),
        sa.Column("eps_estimate", sa.Numeric(20, 6), nullable=True),
        sa.Column("eps_actual", sa.Numeric(20, 6), nullable=True),
        sa.Column("surprise_percent", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "earnings_date", name="uq_earnings_date_instrument_date"
        ),
    )
    op.create_index(
        "ix_earnings_dates_instrument_id", "earnings_dates", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_earnings_dates_instrument_id", table_name="earnings_dates")
    op.drop_table("earnings_dates")
    op.drop_index("ix_earnings_history_instrument_id", table_name="earnings_history")
    op.drop_table("earnings_history")
