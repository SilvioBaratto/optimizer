"""Add the daily foreign-exchange rate table (``fx_rates``).

One row per ``(base_currency, quote_currency, date)`` holding the base-per-quote
rate. Written by the ingestion FX feed; read by the fund performance tracker,
which pivots the rows into the ``optimizer.fx`` rates frame. Market-wide — no
instrument FK. Currency codes are ISO-4217 *major* units (``GBP``, never the
``GBp`` sub-unit).

Revision ID: c7e9a1b3d5f0
Revises: e7d6c5b4a3f2
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c7e9a1b3d5f0"
down_revision: str | Sequence[str] | None = "e7d6c5b4a3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``fx_rates`` table with its pair-date uniqueness guard."""
    op.create_table(
        "fx_rates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("base_currency", sa.String(3), nullable=False),
        sa.Column("quote_currency", sa.String(3), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("rate", sa.Numeric(20, 10), nullable=False),
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
            "base_currency",
            "quote_currency",
            "date",
            name="uq_fx_rates_pair_date",
        ),
    )
    op.create_index("ix_fx_rates_date", "fx_rates", ["date"])
    op.create_index("ix_fx_rates_pair", "fx_rates", ["base_currency", "quote_currency"])


def downgrade() -> None:
    """Drop the ``fx_rates`` table and its indexes."""
    op.drop_index("ix_fx_rates_pair", table_name="fx_rates")
    op.drop_index("ix_fx_rates_date", table_name="fx_rates")
    op.drop_table("fx_rates")
