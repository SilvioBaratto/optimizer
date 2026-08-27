"""Add market-wide calendar tables.

Task B2 (Gap B): earnings / IPO / splits / economic-event calendars from
yf.Calendars.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c2d3e4f5a6b7"
down_revision: str | Sequence[str] | None = "b1c2d3e4f5a6"
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


def upgrade() -> None:
    op.create_table(
        "earnings_calendar",
        sa.Column("ticker", sa.String(30), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("eps_estimate", sa.Numeric(20, 6), nullable=True),
        sa.Column("eps_actual", sa.Numeric(20, 6), nullable=True),
        sa.Column("eps_surprise_pct", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("ticker", "event_date", name="uq_earnings_calendar"),
    )
    op.create_index(
        "ix_earnings_calendar_event_date", "earnings_calendar", ["event_date"]
    )

    op.create_table(
        "ipo_calendar",
        sa.Column("ticker", sa.String(30), nullable=False),
        sa.Column("ipo_date", sa.Date(), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("exchange", sa.String(50), nullable=True),
        sa.Column("price_range", sa.String(100), nullable=True),
        sa.Column("currency", sa.String(10), nullable=True),
        sa.Column("shares", sa.BigInteger(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("ticker", "ipo_date", name="uq_ipo_calendar"),
    )
    op.create_index("ix_ipo_calendar_ipo_date", "ipo_calendar", ["ipo_date"])

    op.create_table(
        "split_calendar",
        sa.Column("ticker", sa.String(30), nullable=False),
        sa.Column("split_date", sa.Date(), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("ratio", sa.String(50), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("ticker", "split_date", name="uq_split_calendar"),
    )
    op.create_index("ix_split_calendar_split_date", "split_calendar", ["split_date"])

    op.create_table(
        "economic_event_calendar",
        sa.Column("event", sa.String(255), nullable=False),
        sa.Column("country", sa.String(50), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("actual", sa.String(50), nullable=True),
        sa.Column("forecast", sa.String(50), nullable=True),
        sa.Column("prior", sa.String(50), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "event", "country", "event_date", name="uq_economic_event_calendar"
        ),
    )
    op.create_index(
        "ix_economic_event_calendar_event_date",
        "economic_event_calendar",
        ["event_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_economic_event_calendar_event_date",
        table_name="economic_event_calendar",
    )
    op.drop_table("economic_event_calendar")
    op.drop_index("ix_split_calendar_split_date", table_name="split_calendar")
    op.drop_table("split_calendar")
    op.drop_index("ix_ipo_calendar_ipo_date", table_name="ipo_calendar")
    op.drop_table("ipo_calendar")
    op.drop_index("ix_earnings_calendar_event_date", table_name="earnings_calendar")
    op.drop_table("earnings_calendar")
