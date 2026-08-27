"""Add major_holders + insider_purchases + insider_roster tables.

Task A7 (Gap A): persist ownership breakdown (``major_holders``), the
6-month insider buy/sell summary (``insider_purchases``), and the insider
roster (``insider_roster_holders``).

Revision ID: d7e8f9a0b1c2
Revises: c6d7e8f9a0b1
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d7e8f9a0b1c2"
down_revision: str | Sequence[str] | None = "c6d7e8f9a0b1"
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
        "major_holders",
        _instrument_fk(),
        sa.Column("insiders_percent_held", sa.Float(), nullable=True),
        sa.Column("institutions_percent_held", sa.Float(), nullable=True),
        sa.Column("institutions_float_percent_held", sa.Float(), nullable=True),
        sa.Column("institutions_count", sa.BigInteger(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("instrument_id", name="uq_major_holders_instrument"),
    )
    op.create_index(
        "ix_major_holders_instrument_id", "major_holders", ["instrument_id"]
    )

    op.create_table(
        "insider_purchases",
        _instrument_fk(),
        sa.Column("purchase_shares", sa.BigInteger(), nullable=True),
        sa.Column("sale_shares", sa.BigInteger(), nullable=True),
        sa.Column("net_shares", sa.BigInteger(), nullable=True),
        sa.Column("total_insider_shares", sa.BigInteger(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("instrument_id", name="uq_insider_purchases_instrument"),
    )
    op.create_index(
        "ix_insider_purchases_instrument_id", "insider_purchases", ["instrument_id"]
    )

    op.create_table(
        "insider_roster",
        _instrument_fk(),
        sa.Column("insider_name", sa.String(500), nullable=False),
        sa.Column("position", sa.String(500), nullable=True),
        sa.Column("most_recent_transaction", sa.String(200), nullable=True),
        sa.Column("latest_transaction_date", sa.Date(), nullable=True),
        sa.Column("shares_owned_directly", sa.BigInteger(), nullable=True),
        sa.Column("shares_owned_indirectly", sa.BigInteger(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "insider_name", name="uq_insider_roster_instrument_name"
        ),
    )
    op.create_index(
        "ix_insider_roster_instrument_id", "insider_roster", ["instrument_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_insider_roster_instrument_id", table_name="insider_roster")
    op.drop_table("insider_roster")
    op.drop_index("ix_insider_purchases_instrument_id", table_name="insider_purchases")
    op.drop_table("insider_purchases")
    op.drop_index("ix_major_holders_instrument_id", table_name="major_holders")
    op.drop_table("major_holders")
