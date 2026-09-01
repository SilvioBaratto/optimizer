"""Drop the dead industry_trend / sector_trend columns from growth_estimates.

yfinance 1.6.0's ``Ticker.growth_estimates`` emits only ``stockTrend`` and
``indexTrend`` — the industry and sector trend series were removed upstream, so
``growth_estimates.industry_trend`` / ``sector_trend`` can never be populated
(verified: 0 non-null across all rows). ``stock_trend`` (analyst-covered names)
and ``index_trend`` remain live, so the table stays. Dropping the two columns is
non-destructive (all NULL). ``downgrade`` re-adds them (nullable) for reversibility.

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f4a5b6c7d8e9"
down_revision: str | Sequence[str] | None = "e3f4a5b6c7d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("growth_estimates", "industry_trend")
    op.drop_column("growth_estimates", "sector_trend")


def downgrade() -> None:
    op.add_column(
        "growth_estimates",
        sa.Column("industry_trend", sa.Numeric(20, 6), nullable=True),
    )
    op.add_column(
        "growth_estimates",
        sa.Column("sector_trend", sa.Numeric(20, 6), nullable=True),
    )
