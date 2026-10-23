"""Add preferred and convertible allocation columns to ``etf_asset_classes``.

Yahoo reports six asset-class positions; only stock/bond/cash/other were stored,
so a preferred-stock or convertible-bond ETF persisted weights summing to ~6%.
Existing rows keep NULL in the new columns (the original payload is gone) and
are filled on the next fund-data refetch.

Revision ID: e9a1c3d5f7b2
Revises: d8f0b2c4e6a1
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e9a1c3d5f7b2"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable ``preferred_pct`` and ``convertible_pct`` columns."""
    op.add_column(
        "etf_asset_classes",
        sa.Column("preferred_pct", sa.Numeric(10, 6), nullable=True),
    )
    op.add_column(
        "etf_asset_classes",
        sa.Column("convertible_pct", sa.Numeric(10, 6), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``preferred_pct`` and ``convertible_pct`` columns."""
    op.drop_column("etf_asset_classes", "convertible_pct")
    op.drop_column("etf_asset_classes", "preferred_pct")
