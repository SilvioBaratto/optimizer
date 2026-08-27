"""Add ticker_profile_extras table.

Task A9 (Gap A / OQ3): 1:1 extra yf.Ticker.info fields (short interest,
52-week change vs S&P, sector/industry keys, governance-risk scores) mapped
from the same info dict as ticker_profiles.

Revision ID: f9a0b1c2d3e4
Revises: e8f9a0b1c2d3
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f9a0b1c2d3e4"
down_revision: str | Sequence[str] | None = "e8f9a0b1c2d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ticker_profile_extras",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("shares_short", sa.BigInteger(), nullable=True),
        sa.Column("shares_short_prior_month", sa.BigInteger(), nullable=True),
        sa.Column("short_ratio", sa.Float(), nullable=True),
        sa.Column("short_percent_of_float", sa.Float(), nullable=True),
        sa.Column("shares_percent_shares_out", sa.Float(), nullable=True),
        sa.Column("held_percent_insiders", sa.Float(), nullable=True),
        sa.Column("held_percent_institutions", sa.Float(), nullable=True),
        sa.Column("fifty_two_week_change", sa.Float(), nullable=True),
        sa.Column("sandp_52_week_change", sa.Float(), nullable=True),
        sa.Column("sector_key", sa.String(100), nullable=True),
        sa.Column("industry_key", sa.String(150), nullable=True),
        sa.Column("audit_risk", sa.Integer(), nullable=True),
        sa.Column("board_risk", sa.Integer(), nullable=True),
        sa.Column("compensation_risk", sa.Integer(), nullable=True),
        sa.Column("shareholder_rights_risk", sa.Integer(), nullable=True),
        sa.Column("overall_risk", sa.Integer(), nullable=True),
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
            "instrument_id", name="uq_ticker_profile_extras_instrument"
        ),
    )
    op.create_index(
        "ix_ticker_profile_extras_instrument_id",
        "ticker_profile_extras",
        ["instrument_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ticker_profile_extras_instrument_id",
        table_name="ticker_profile_extras",
    )
    op.drop_table("ticker_profile_extras")
