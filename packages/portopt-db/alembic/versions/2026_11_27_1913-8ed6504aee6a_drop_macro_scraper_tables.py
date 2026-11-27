"""Drop the six tables the removed macro web scrapers filled.

The Il Sole 24 Ore forecast scraper and the Trading Economics scraper (macro
indicators and government bond yields) are gone, so nothing reads or writes
``economic_indicators``, ``economic_indicator_observations``,
``trading_economics_indicators``, ``trading_economics_observations``, ``bond_yields``
or ``bond_yield_observations``. The upgrade deletes the stored macro and bond-yield
history in them. The downgrade recreates the six tables empty, with the columns,
constraints and indexes they had at the previous head: the rows came from the scrapers
and are not restored. Take a database dump before upgrading if the history matters.

The FRED observations and the macro news tables are not touched.

Revision ID: 8ed6504aee6a
Revises: 2266e24c509f
Create Date: 2026-11-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "8ed6504aee6a"
down_revision: str | Sequence[str] | None = "2266e24c509f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COUNTRY_LENGTH = 100
_MATURITY_LENGTH = 10
_INDICATOR_KEY_LENGTH = 100
_UNIT_LENGTH = 50
_REFERENCE_LENGTH = 100
_RAW_NAME_LENGTH = 200


def _float_columns(*names: str) -> list[sa.Column]:
    return [sa.Column(name, sa.Float(), nullable=True) for name in names]


def _country_column() -> sa.Column:
    return sa.Column("country", sa.String(_COUNTRY_LENGTH), nullable=False)


def _id_column() -> sa.Column:
    return sa.Column("id", sa.Uuid(), primary_key=True)


def _timestamp_columns() -> list[sa.Column]:
    return [
        sa.Column(
            name,
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        )
        for name in ("created_at", "updated_at")
    ]


def _forecast_columns() -> list[sa.Column]:
    return [
        *_float_columns(
            "last_inflation",
            "inflation_6m",
            "inflation_10y_avg",
            "gdp_growth_6m",
            "earnings_12m",
            "eps_expected_12m",
            "peg_ratio",
            "lt_rate_forecast",
        ),
        sa.Column("reference_date", sa.Date(), nullable=True),
    ]


def _create_economic_indicators() -> None:
    op.create_table(
        "economic_indicators",
        _id_column(),
        _country_column(),
        *_forecast_columns(),
        *_timestamp_columns(),
        sa.UniqueConstraint("country", name="uq_economic_indicator_country"),
    )


def _create_economic_indicator_observations() -> None:
    op.create_table(
        "economic_indicator_observations",
        _id_column(),
        _country_column(),
        sa.Column("date", sa.Date(), nullable=False),
        *_forecast_columns(),
        *_timestamp_columns(),
        sa.UniqueConstraint("country", "date", name="uq_econ_obs_country_date"),
    )
    op.create_index(
        "ix_econ_observations_country", "economic_indicator_observations", ["country"]
    )
    op.create_index(
        "ix_econ_observations_date", "economic_indicator_observations", ["date"]
    )


def _create_trading_economics_indicators() -> None:
    op.create_table(
        "trading_economics_indicators",
        _id_column(),
        _country_column(),
        sa.Column("indicator_key", sa.String(_INDICATOR_KEY_LENGTH), nullable=False),
        *_float_columns("value", "previous"),
        sa.Column("unit", sa.String(_UNIT_LENGTH), nullable=True),
        sa.Column("reference", sa.String(_REFERENCE_LENGTH), nullable=True),
        sa.Column("raw_name", sa.String(_RAW_NAME_LENGTH), nullable=True),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "country", "indicator_key", name="uq_te_indicator_country_key"
        ),
    )
    op.create_index(
        "ix_trading_economics_indicators_country",
        "trading_economics_indicators",
        ["country"],
    )


def _create_trading_economics_observations() -> None:
    op.create_table(
        "trading_economics_observations",
        _id_column(),
        _country_column(),
        sa.Column("indicator_key", sa.String(_INDICATOR_KEY_LENGTH), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        *_float_columns("value"),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "country", "indicator_key", "date", name="uq_te_obs_country_key_date"
        ),
    )
    op.create_index(
        "ix_te_observations_country", "trading_economics_observations", ["country"]
    )
    op.create_index(
        "ix_te_observations_date", "trading_economics_observations", ["date"]
    )


def _create_bond_yields() -> None:
    op.create_table(
        "bond_yields",
        _id_column(),
        _country_column(),
        sa.Column("maturity", sa.String(_MATURITY_LENGTH), nullable=False),
        *_float_columns("yield_value", "day_change", "month_change", "year_change"),
        sa.Column("reference_date", sa.Date(), nullable=True),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "country", "maturity", name="uq_bond_yield_country_maturity"
        ),
    )
    op.create_index("ix_bond_yields_country", "bond_yields", ["country"])


def _create_bond_yield_observations() -> None:
    op.create_table(
        "bond_yield_observations",
        _id_column(),
        _country_column(),
        sa.Column("maturity", sa.String(_MATURITY_LENGTH), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        *_float_columns("yield_value"),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "country", "maturity", "date", name="uq_bond_obs_country_mat_date"
        ),
    )
    op.create_index(
        "ix_bond_observations_country", "bond_yield_observations", ["country"]
    )
    op.create_index("ix_bond_observations_date", "bond_yield_observations", ["date"])


def upgrade() -> None:
    """Drop the six tables; the macro and bond-yield history in them is deleted."""
    op.drop_table("bond_yield_observations")
    op.drop_table("bond_yields")
    op.drop_table("trading_economics_observations")
    op.drop_table("trading_economics_indicators")
    op.drop_table("economic_indicator_observations")
    op.drop_table("economic_indicators")


def downgrade() -> None:
    """Recreate the six tables empty; the deleted history is not restored."""
    _create_economic_indicators()
    _create_economic_indicator_observations()
    _create_trading_economics_indicators()
    _create_trading_economics_observations()
    _create_bond_yields()
    _create_bond_yield_observations()
