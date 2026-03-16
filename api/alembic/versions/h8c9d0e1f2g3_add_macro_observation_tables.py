"""add_macro_observation_tables

Revision ID: h8c9d0e1f2g3
Revises: g7b8c9d0e1f2
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "h8c9d0e1f2g3"
down_revision: str | Sequence[str] | None = "g7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -- trading_economics_observations --
    op.create_table(
        "trading_economics_observations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("indicator_key", sa.String(100), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
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
            "country", "indicator_key", "date",
            name="uq_te_obs_country_key_date",
        ),
    )
    op.create_index("ix_te_observations_country", "trading_economics_observations", ["country"])
    op.create_index("ix_te_observations_date", "trading_economics_observations", ["date"])

    # -- bond_yield_observations --
    op.create_table(
        "bond_yield_observations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("maturity", sa.String(10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("yield_value", sa.Float(), nullable=True),
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
            "country", "maturity", "date",
            name="uq_bond_obs_country_mat_date",
        ),
    )
    op.create_index("ix_bond_observations_country", "bond_yield_observations", ["country"])
    op.create_index("ix_bond_observations_date", "bond_yield_observations", ["date"])

    # -- Seed from existing snapshot tables (if they exist) --
    conn = op.get_bind()
    existing = set(sa.inspect(conn).get_table_names())

    if "trading_economics_indicators" in existing:
        op.execute(
            """
            INSERT INTO trading_economics_observations (id, country, indicator_key, date, value, created_at, updated_at)
            SELECT gen_random_uuid(), country, indicator_key, CURRENT_DATE, value, NOW(), NOW()
            FROM trading_economics_indicators
            WHERE value IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        )
    if "bond_yields" in existing:
        op.execute(
            """
            INSERT INTO bond_yield_observations (id, country, maturity, date, yield_value, created_at, updated_at)
            SELECT gen_random_uuid(), country, maturity, CURRENT_DATE, yield_value, NOW(), NOW()
            FROM bond_yields
            WHERE yield_value IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        )


def downgrade() -> None:
    op.drop_table("bond_yield_observations")
    op.drop_table("trading_economics_observations")
