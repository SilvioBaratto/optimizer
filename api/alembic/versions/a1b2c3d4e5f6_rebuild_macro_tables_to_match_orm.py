"""Rebuild macro tables to match ORM models.

The previous migration 9242f7444435 was deleted, leaving the
economic_indicators and trading_economics_indicators tables with
stale schemas.  All affected tables are empty, so we drop and
recreate them to match the current ORM definitions.

Revision ID: a1b2c3d4e5f6
Revises: 6b106e82ce20
Create Date: 2026-02-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "6b106e82ce20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = set(inspector.get_table_names())

    # Drop FK from country_regime_assessments → economic_indicators first
    if "country_regime_assessments" in existing:
        op.drop_constraint(
            "country_regime_assessments_economic_indicators_id_fkey",
            "country_regime_assessments",
            type_="foreignkey",
        )

    # Drop old tables if they exist
    for tbl in (
        "economic_indicators",
        "trading_economics_indicators",
        "trading_economics_bond_yields",
        "trading_economics_snapshots",
    ):
        if tbl in existing:
            op.drop_table(tbl)

    # Recreate economic_indicators to match ORM
    op.create_table(
        "economic_indicators",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("gdp_growth_qq", sa.Float, nullable=True),
        sa.Column("industrial_production", sa.Float, nullable=True),
        sa.Column("unemployment", sa.Float, nullable=True),
        sa.Column("consumer_prices", sa.Float, nullable=True),
        sa.Column("deficit", sa.Float, nullable=True),
        sa.Column("debt", sa.Float, nullable=True),
        sa.Column("st_rate", sa.Float, nullable=True),
        sa.Column("lt_rate", sa.Float, nullable=True),
        sa.Column("last_inflation", sa.Float, nullable=True),
        sa.Column("inflation_6m", sa.Float, nullable=True),
        sa.Column("inflation_10y_avg", sa.Float, nullable=True),
        sa.Column("gdp_growth_6m", sa.Float, nullable=True),
        sa.Column("earnings_12m", sa.Float, nullable=True),
        sa.Column("eps_expected_12m", sa.Float, nullable=True),
        sa.Column("peg_ratio", sa.Float, nullable=True),
        sa.Column("st_rate_forecast", sa.Float, nullable=True),
        sa.Column("lt_rate_forecast", sa.Float, nullable=True),
        sa.Column("reference_date", sa.Date, nullable=True),
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
            "country", "source", name="uq_economic_indicator_country_source"
        ),
    )
    op.create_index(
        "ix_economic_indicators_country", "economic_indicators", ["country"]
    )

    # Recreate trading_economics_indicators to match ORM
    op.create_table(
        "trading_economics_indicators",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("indicator_key", sa.String(100), nullable=False),
        sa.Column("value", sa.Float, nullable=True),
        sa.Column("previous", sa.Float, nullable=True),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("reference", sa.String(100), nullable=True),
        sa.Column("raw_name", sa.String(200), nullable=True),
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
            "country", "indicator_key", name="uq_te_indicator_country_key"
        ),
    )
    op.create_index(
        "ix_trading_economics_indicators_country",
        "trading_economics_indicators",
        ["country"],
    )

    # Re-add FK from country_regime_assessments → economic_indicators
    # (only if the table exists — it was removed in later migrations)
    existing_after = set(sa.inspect(conn).get_table_names())
    if "country_regime_assessments" in existing_after:
        op.create_foreign_key(
            "country_regime_assessments_economic_indicators_id_fkey",
            "country_regime_assessments",
            "economic_indicators",
            ["economic_indicators_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    # This is a destructive migration; downgrade is not supported.
    raise NotImplementedError(
        "Downgrade not supported — old table schemas are not preserved."
    )
