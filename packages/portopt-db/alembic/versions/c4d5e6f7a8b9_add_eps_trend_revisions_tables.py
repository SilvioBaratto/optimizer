"""Add eps_trend and eps_revisions typed tables.

The eps_trend / eps_revisions panels from yfinance are indexed by forward-period
label ("0q", "+1q", "0y", "+1y") with metric columns (snapshot ages / revision
counts) — neither axis is a date, so they never belonged in the date-keyed
``financial_statements`` EAV table (the old code coerced their metric-name columns
to datetimes, which turned every column to NaT, dropped them all, and stored
nothing while logging a pandas warning on each ticker). Give them dedicated typed
tables, mirroring the sibling earnings_estimate / revenue_estimate /
growth_estimates tables.

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7a8"
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
        "eps_trend",
        _instrument_fk(),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("current_estimate", sa.Numeric(20, 6), nullable=True),
        sa.Column("days_ago_7", sa.Numeric(20, 6), nullable=True),
        sa.Column("days_ago_30", sa.Numeric(20, 6), nullable=True),
        sa.Column("days_ago_60", sa.Numeric(20, 6), nullable=True),
        sa.Column("days_ago_90", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period", name="uq_eps_trend_instrument_period"
        ),
    )
    op.create_index("ix_eps_trend_instrument_id", "eps_trend", ["instrument_id"])

    op.create_table(
        "eps_revisions",
        _instrument_fk(),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("up_last_7days", sa.Integer(), nullable=True),
        sa.Column("up_last_30days", sa.Integer(), nullable=True),
        sa.Column("down_last_7days", sa.Integer(), nullable=True),
        sa.Column("down_last_30days", sa.Integer(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "period", name="uq_eps_revisions_instrument_period"
        ),
    )
    op.create_index(
        "ix_eps_revisions_instrument_id", "eps_revisions", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_eps_revisions_instrument_id", table_name="eps_revisions")
    op.drop_table("eps_revisions")
    op.drop_index("ix_eps_trend_instrument_id", table_name="eps_trend")
    op.drop_table("eps_trend")
