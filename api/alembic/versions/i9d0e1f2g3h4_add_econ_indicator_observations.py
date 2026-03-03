"""add_econ_indicator_observations

Revision ID: i9d0e1f2g3h4
Revises: h8c9d0e1f2g3
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "i9d0e1f2g3h4"
down_revision: str | Sequence[str] | None = "h8c9d0e1f2g3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "economic_indicator_observations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("last_inflation", sa.Float(), nullable=True),
        sa.Column("inflation_6m", sa.Float(), nullable=True),
        sa.Column("inflation_10y_avg", sa.Float(), nullable=True),
        sa.Column("gdp_growth_6m", sa.Float(), nullable=True),
        sa.Column("earnings_12m", sa.Float(), nullable=True),
        sa.Column("eps_expected_12m", sa.Float(), nullable=True),
        sa.Column("peg_ratio", sa.Float(), nullable=True),
        sa.Column("lt_rate_forecast", sa.Float(), nullable=True),
        sa.Column("reference_date", sa.Date(), nullable=True),
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
            "country", "date",
            name="uq_econ_obs_country_date",
        ),
    )
    op.create_index("ix_econ_observations_country", "economic_indicator_observations", ["country"])
    op.create_index("ix_econ_observations_date", "economic_indicator_observations", ["date"])

    # Seed from existing snapshot table
    op.execute(
        """
        INSERT INTO economic_indicator_observations
            (id, country, date, last_inflation, inflation_6m, inflation_10y_avg,
             gdp_growth_6m, earnings_12m, eps_expected_12m, peg_ratio,
             lt_rate_forecast, reference_date, created_at, updated_at)
        SELECT gen_random_uuid(), country, CURRENT_DATE,
               last_inflation, inflation_6m, inflation_10y_avg,
               gdp_growth_6m, earnings_12m, eps_expected_12m, peg_ratio,
               lt_rate_forecast, reference_date, NOW(), NOW()
        FROM economic_indicators
        ON CONFLICT DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_table("economic_indicator_observations")
