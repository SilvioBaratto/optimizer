"""Add price_unit to price_history (raw listing/sub-unit currency; SPEC OQ2).

yfinance 1.6.0 keeps GBp/ZAc/ILA sub-unit prices as-quoted under repair=True. We
store the listing currency as ``price_unit`` and leave the OHLCV values raw; the
optimizer fx layer converts at analysis time. Nullable — populated on next fetch.

Revision ID: f7b8c9d0e1a2
Revises: e6f7a8b9c0d1
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f7b8c9d0e1a2"
down_revision: str | Sequence[str] | None = "e6f7a8b9c0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "price_history",
        sa.Column("price_unit", sa.String(length=10), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("price_history", "price_unit")
