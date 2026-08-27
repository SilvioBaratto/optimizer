"""Add options_chain table.

Task A10 (Gap A / OQ1): full option-chain snapshots from
yf.Ticker.option_chain. High-volume — written by its own low-frequency
scheduler step, one row per (instrument, snapshot date, contract).

Revision ID: a0b1c2d3e4f5
Revises: f9a0b1c2d3e4
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a0b1c2d3e4f5"
down_revision: str | Sequence[str] | None = "f9a0b1c2d3e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "options_chain",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("expiry", sa.Date(), nullable=False),
        sa.Column("option_type", sa.String(4), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("contract_symbol", sa.String(50), nullable=False),
        sa.Column("last_price", sa.Numeric(20, 6), nullable=True),
        sa.Column("bid", sa.Numeric(20, 6), nullable=True),
        sa.Column("ask", sa.Numeric(20, 6), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("open_interest", sa.BigInteger(), nullable=True),
        sa.Column("implied_volatility", sa.Numeric(20, 10), nullable=True),
        sa.Column("in_the_money", sa.Boolean(), nullable=True),
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
            "instrument_id", "as_of", "contract_symbol", name="uq_option_contract"
        ),
    )
    op.create_index(
        "ix_options_chain_instrument_id", "options_chain", ["instrument_id"]
    )
    op.create_index("ix_options_chain_expiry", "options_chain", ["expiry"])


def downgrade() -> None:
    op.drop_index("ix_options_chain_expiry", table_name="options_chain")
    op.drop_index("ix_options_chain_instrument_id", table_name="options_chain")
    op.drop_table("options_chain")
