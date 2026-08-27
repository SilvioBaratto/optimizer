"""Add ETF funds_data depth tables + overview columns.

Task A8 (Gap A): equity_holdings, bond_holdings, bond_ratings, fund_operations
depth tables plus category/description on etf_metadata (fund_overview).

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e8f9a0b1c2d3"
down_revision: str | Sequence[str] | None = "d7e8f9a0b1c2"
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
    op.add_column("etf_metadata", sa.Column("category", sa.String(255), nullable=True))
    op.add_column("etf_metadata", sa.Column("description", sa.Text(), nullable=True))

    op.create_table(
        "etf_equity_holdings",
        _instrument_fk(),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("price_to_earnings", sa.Numeric(20, 6), nullable=True),
        sa.Column("price_to_book", sa.Numeric(20, 6), nullable=True),
        sa.Column("price_to_sales", sa.Numeric(20, 6), nullable=True),
        sa.Column("price_to_cashflow", sa.Numeric(20, 6), nullable=True),
        sa.Column("median_market_cap", sa.Numeric(24, 2), nullable=True),
        sa.Column("three_year_earnings_growth", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_equity_holdings_instrument_asof"
        ),
    )
    op.create_index(
        "ix_etf_equity_holdings_instrument_id",
        "etf_equity_holdings",
        ["instrument_id"],
    )

    op.create_table(
        "etf_bond_holdings",
        _instrument_fk(),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("duration", sa.Numeric(20, 6), nullable=True),
        sa.Column("maturity", sa.Numeric(20, 6), nullable=True),
        sa.Column("credit_quality", sa.Numeric(20, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_bond_holdings_instrument_asof"
        ),
    )
    op.create_index(
        "ix_etf_bond_holdings_instrument_id", "etf_bond_holdings", ["instrument_id"]
    )

    op.create_table(
        "etf_bond_ratings",
        _instrument_fk(),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("rating", sa.String(50), nullable=False),
        sa.Column("weight", sa.Numeric(10, 6), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id",
            "as_of",
            "rating",
            name="uq_etf_bond_ratings_instrument_asof_rating",
        ),
    )
    op.create_index(
        "ix_etf_bond_ratings_instrument_id", "etf_bond_ratings", ["instrument_id"]
    )

    op.create_table(
        "etf_fund_operations",
        _instrument_fk(),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("annual_report_expense_ratio", sa.Numeric(10, 6), nullable=True),
        sa.Column("annual_holdings_turnover", sa.Numeric(10, 6), nullable=True),
        sa.Column("total_net_assets", sa.Numeric(24, 2), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_fund_operations_instrument_asof"
        ),
    )
    op.create_index(
        "ix_etf_fund_operations_instrument_id",
        "etf_fund_operations",
        ["instrument_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_etf_fund_operations_instrument_id", table_name="etf_fund_operations"
    )
    op.drop_table("etf_fund_operations")
    op.drop_index("ix_etf_bond_ratings_instrument_id", table_name="etf_bond_ratings")
    op.drop_table("etf_bond_ratings")
    op.drop_index("ix_etf_bond_holdings_instrument_id", table_name="etf_bond_holdings")
    op.drop_table("etf_bond_holdings")
    op.drop_index(
        "ix_etf_equity_holdings_instrument_id", table_name="etf_equity_holdings"
    )
    op.drop_table("etf_equity_holdings")
    op.drop_column("etf_metadata", "description")
    op.drop_column("etf_metadata", "category")
