"""Add shares_outstanding + capital_gains tables and price_history.capital_gains.

Task A6 (Gap A): persist point-in-time shares outstanding
(``get_shares_full``), fund capital-gain distributions (``capital_gains``),
and the per-row "Capital Gains" history column.

Revision ID: c6d7e8f9a0b1
Revises: b5c6d7e8f9a0
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c6d7e8f9a0b1"
down_revision: str | Sequence[str] | None = "b5c6d7e8f9a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "price_history",
        sa.Column("capital_gains", sa.Numeric(20, 6), nullable=True),
    )

    op.create_table(
        "shares_outstanding",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("shares", sa.BigInteger(), nullable=True),
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
            "instrument_id", "date", name="uq_shares_outstanding_instrument_date"
        ),
    )
    op.create_index(
        "ix_shares_outstanding_instrument_id",
        "shares_outstanding",
        ["instrument_id"],
    )

    op.create_table(
        "capital_gains",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
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
            "instrument_id", "date", name="uq_capital_gain_instrument_date"
        ),
    )
    op.create_index(
        "ix_capital_gains_instrument_id", "capital_gains", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_capital_gains_instrument_id", table_name="capital_gains")
    op.drop_table("capital_gains")
    op.drop_index(
        "ix_shares_outstanding_instrument_id", table_name="shares_outstanding"
    )
    op.drop_table("shares_outstanding")
    op.drop_column("price_history", "capital_gains")
