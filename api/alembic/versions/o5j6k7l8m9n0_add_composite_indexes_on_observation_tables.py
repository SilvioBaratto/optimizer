"""add_composite_indexes_on_observation_tables

Revision ID: o5j6k7l8m9n0
Revises: n4i5j6k7l8m9
Create Date: 2026-03-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "o5j6k7l8m9n0"
down_revision: str | Sequence[str] | None = "n4i5j6k7l8m9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_te_obs_country_key_date",
        "trading_economics_observations",
        ["country", "indicator_key", "date"],
    )
    op.create_index(
        "ix_bond_obs_country_maturity_date",
        "bond_yield_observations",
        ["country", "maturity", "date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_bond_obs_country_maturity_date",
        table_name="bond_yield_observations",
    )
    op.drop_index(
        "ix_te_obs_country_key_date",
        table_name="trading_economics_observations",
    )
