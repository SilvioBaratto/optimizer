"""drop_ilsole_real_columns

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | Sequence[str] | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Delete all ilsole_real rows (redundant with TradingEconomics data)
    op.execute("DELETE FROM economic_indicators WHERE source = 'ilsole_real'")

    # 2. Drop the old unique constraint
    op.drop_constraint(
        "uq_economic_indicator_country_source",
        "economic_indicators",
        type_="unique",
    )

    # 3. Drop real indicator columns and source column
    op.drop_column("economic_indicators", "source")
    op.drop_column("economic_indicators", "gdp_growth_qq")
    op.drop_column("economic_indicators", "industrial_production")
    op.drop_column("economic_indicators", "unemployment")
    op.drop_column("economic_indicators", "consumer_prices")
    op.drop_column("economic_indicators", "deficit")
    op.drop_column("economic_indicators", "debt")
    op.drop_column("economic_indicators", "st_rate")
    op.drop_column("economic_indicators", "lt_rate")

    # 5. Drop st_rate_forecast (always NULL — source never publishes it)
    op.drop_column("economic_indicators", "st_rate_forecast")

    # 4. Create new unique constraint on country alone
    op.create_unique_constraint(
        "uq_economic_indicator_country",
        "economic_indicators",
        ["country"],
    )


def downgrade() -> None:
    import sqlalchemy as sa

    # Drop new constraint
    op.drop_constraint(
        "uq_economic_indicator_country",
        "economic_indicators",
        type_="unique",
    )

    # Re-add dropped columns
    op.add_column(
        "economic_indicators",
        sa.Column("lt_rate", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("st_rate", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("debt", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("deficit", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("consumer_prices", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("unemployment", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("industrial_production", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("gdp_growth_qq", sa.Float(), nullable=True),
    )
    op.add_column(
        "economic_indicators",
        sa.Column("source", sa.String(50), nullable=False, server_default="ilsole_forecast"),
    )

    # Restore old unique constraint
    op.create_unique_constraint(
        "uq_economic_indicator_country_source",
        "economic_indicators",
        ["country", "source"],
    )
