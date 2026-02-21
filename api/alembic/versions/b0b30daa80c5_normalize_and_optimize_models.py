"""normalize_and_optimize_models

Revision ID: b0b30daa80c5
Revises: ecf0a9a2bfdc
Create Date: 2026-01-30 18:21:06.510976

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b0b30daa80c5"
down_revision: str | Sequence[str] | None = "ecf0a9a2bfdc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    # ----------------------------------------------------------------
    # 1. Rename mutualfund_holders → mutual_fund_holders (preserve data)
    # ----------------------------------------------------------------
    op.rename_table("mutualfund_holders", "mutual_fund_holders")

    # Rename the index
    op.execute(
        "ALTER INDEX ix_mutualfund_holders_instrument_id "
        "RENAME TO ix_mutual_fund_holders_instrument_id"
    )

    # Rename the unique constraint
    op.execute(
        "ALTER TABLE mutual_fund_holders "
        "RENAME CONSTRAINT uq_mf_holder_instrument_name "
        "TO uq_mutual_fund_holder_instrument_name"
    )

    # ----------------------------------------------------------------
    # 2. Add ondelete CASCADE to instruments.exchange_id FK
    # ----------------------------------------------------------------
    op.drop_constraint(
        "instruments_exchange_id_fkey", "instruments", type_="foreignkey"
    )
    op.create_foreign_key(
        "instruments_exchange_id_fkey",
        "instruments",
        "exchanges",
        ["exchange_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # ----------------------------------------------------------------
    # 3. Add missing indexes on ticker_profiles
    # ----------------------------------------------------------------
    op.create_index("ix_ticker_profiles_sector", "ticker_profiles", ["sector"])
    op.create_index("ix_ticker_profiles_industry", "ticker_profiles", ["industry"])
    op.create_index("ix_ticker_profiles_country", "ticker_profiles", ["country"])

    # ----------------------------------------------------------------
    # 4. Add missing index on analyst_price_targets.instrument_id
    # ----------------------------------------------------------------
    op.create_index(
        "ix_analyst_price_targets_instrument_id",
        "analyst_price_targets",
        ["instrument_id"],
    )

    # ----------------------------------------------------------------
    # 5. Add missing index on insider_transactions.start_date
    # ----------------------------------------------------------------
    op.create_index(
        "ix_insider_transactions_start_date", "insider_transactions", ["start_date"]
    )

    # ----------------------------------------------------------------
    # 6. Add missing index on financial_statements.period_date
    # ----------------------------------------------------------------
    op.create_index(
        "ix_financial_statements_period_date", "financial_statements", ["period_date"]
    )

    # ----------------------------------------------------------------
    # 7. Change Float → Numeric(20,6) on AnalystPriceTarget price columns
    # ----------------------------------------------------------------
    for col in ("current", "low", "high", "mean", "median"):
        op.alter_column(
            "analyst_price_targets",
            col,
            existing_type=sa.DOUBLE_PRECISION(precision=53),
            type_=sa.Numeric(precision=20, scale=6),
            existing_nullable=True,
        )

    # ----------------------------------------------------------------
    # 8. Change Numeric(30,6) → Numeric(20,6) on financial_statements.value
    # ----------------------------------------------------------------
    op.alter_column(
        "financial_statements",
        "value",
        existing_type=sa.NUMERIC(precision=30, scale=6),
        type_=sa.Numeric(precision=20, scale=6),
        existing_nullable=True,
    )

    # ----------------------------------------------------------------
    # 9. Add country indexes on macro tables
    # ----------------------------------------------------------------
    op.create_index(
        "ix_economic_indicators_country", "economic_indicators", ["country"]
    )
    op.create_index(
        "ix_trading_economics_indicators_country",
        "trading_economics_indicators",
        ["country"],
    )
    op.create_index("ix_bond_yields_country", "bond_yields", ["country"])

    # ----------------------------------------------------------------
    # 10. Remove scraped_at from all 3 macro tables
    # ----------------------------------------------------------------
    op.drop_column("economic_indicators", "scraped_at")
    op.drop_column("trading_economics_indicators", "scraped_at")
    op.drop_column("bond_yields", "scraped_at")

    # ----------------------------------------------------------------
    # 11. Change reference_date from VARCHAR(100) to Date
    #     Use USING clause to convert existing string values.
    #     Existing values are like "Dec 2024" — we parse via SQL:
    #       to_date('Dec 2024', 'Mon YYYY')
    #     NULLs and empty strings are kept as NULL.
    # ----------------------------------------------------------------
    # economic_indicators.reference_date
    op.execute(
        "UPDATE economic_indicators "
        "SET reference_date = NULL "
        "WHERE reference_date IS NOT NULL AND TRIM(reference_date) = ''"
    )
    op.execute(
        "ALTER TABLE economic_indicators "
        "ALTER COLUMN reference_date TYPE date "
        "USING CASE "
        "  WHEN reference_date IS NULL THEN NULL "
        "  WHEN TRIM(reference_date) = '' THEN NULL "
        "  ELSE to_date(reference_date, 'Mon YYYY') "
        "END"
    )

    # bond_yields.reference_date
    op.execute(
        "UPDATE bond_yields "
        "SET reference_date = NULL "
        "WHERE reference_date IS NOT NULL AND TRIM(reference_date) = ''"
    )
    op.execute(
        "ALTER TABLE bond_yields "
        "ALTER COLUMN reference_date TYPE date "
        "USING CASE "
        "  WHEN reference_date IS NULL THEN NULL "
        "  WHEN TRIM(reference_date) = '' THEN NULL "
        "  ELSE to_date(reference_date, 'Mon YYYY') "
        "END"
    )

    # ----------------------------------------------------------------
    # 12. Make insider_transactions.start_date NOT NULL
    #     First backfill NULLs with sentinel date 1970-01-01.
    # ----------------------------------------------------------------
    op.execute(
        "UPDATE insider_transactions SET start_date = '1970-01-01' WHERE start_date IS NULL"
    )
    op.alter_column(
        "insider_transactions",
        "start_date",
        existing_type=sa.DATE(),
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    # 12. Revert start_date to nullable
    op.alter_column(
        "insider_transactions",
        "start_date",
        existing_type=sa.DATE(),
        nullable=True,
    )

    # 11. Revert reference_date back to VARCHAR(100)
    op.execute(
        "ALTER TABLE bond_yields "
        "ALTER COLUMN reference_date TYPE varchar(100) "
        "USING CASE "
        "  WHEN reference_date IS NULL THEN NULL "
        "  ELSE to_char(reference_date, 'Mon YYYY') "
        "END"
    )
    op.execute(
        "ALTER TABLE economic_indicators "
        "ALTER COLUMN reference_date TYPE varchar(100) "
        "USING CASE "
        "  WHEN reference_date IS NULL THEN NULL "
        "  ELSE to_char(reference_date, 'Mon YYYY') "
        "END"
    )

    # 10. Restore scraped_at on all 3 macro tables
    op.add_column(
        "bond_yields",
        sa.Column(
            "scraped_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "trading_economics_indicators",
        sa.Column(
            "scraped_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "economic_indicators",
        sa.Column(
            "scraped_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # 9. Drop country indexes on macro tables
    op.drop_index("ix_bond_yields_country", table_name="bond_yields")
    op.drop_index(
        "ix_trading_economics_indicators_country",
        table_name="trading_economics_indicators",
    )
    op.drop_index("ix_economic_indicators_country", table_name="economic_indicators")

    # 8. Revert financial_statements.value to Numeric(30,6)
    op.alter_column(
        "financial_statements",
        "value",
        existing_type=sa.Numeric(precision=20, scale=6),
        type_=sa.NUMERIC(precision=30, scale=6),
        existing_nullable=True,
    )

    # 7. Revert AnalystPriceTarget columns to Float
    for col in ("current", "low", "high", "mean", "median"):
        op.alter_column(
            "analyst_price_targets",
            col,
            existing_type=sa.Numeric(precision=20, scale=6),
            type_=sa.DOUBLE_PRECISION(precision=53),
            existing_nullable=True,
        )

    # 6. Drop financial_statements.period_date index
    op.drop_index(
        "ix_financial_statements_period_date", table_name="financial_statements"
    )

    # 5. Drop insider_transactions.start_date index
    op.drop_index(
        "ix_insider_transactions_start_date", table_name="insider_transactions"
    )

    # 4. Drop analyst_price_targets.instrument_id index
    op.drop_index(
        "ix_analyst_price_targets_instrument_id", table_name="analyst_price_targets"
    )

    # 3. Drop ticker_profiles indexes
    op.drop_index("ix_ticker_profiles_country", table_name="ticker_profiles")
    op.drop_index("ix_ticker_profiles_industry", table_name="ticker_profiles")
    op.drop_index("ix_ticker_profiles_sector", table_name="ticker_profiles")

    # 2. Revert instruments FK (remove ondelete CASCADE)
    op.drop_constraint(
        "instruments_exchange_id_fkey", "instruments", type_="foreignkey"
    )
    op.create_foreign_key(
        "instruments_exchange_id_fkey",
        "instruments",
        "exchanges",
        ["exchange_id"],
        ["id"],
    )

    # 1. Rename mutual_fund_holders back to mutualfund_holders
    op.execute(
        "ALTER TABLE mutual_fund_holders "
        "RENAME CONSTRAINT uq_mutual_fund_holder_instrument_name "
        "TO uq_mf_holder_instrument_name"
    )
    op.execute(
        "ALTER INDEX ix_mutual_fund_holders_instrument_id "
        "RENAME TO ix_mutualfund_holders_instrument_id"
    )
    op.rename_table("mutual_fund_holders", "mutualfund_holders")
