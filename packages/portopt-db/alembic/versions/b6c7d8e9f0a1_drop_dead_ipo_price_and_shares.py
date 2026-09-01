"""Drop the dead ipo_calendar.price_range and shares columns.

yfinance's ``Calendars.get_ipo_info_calendar`` carries no price or share-count
data — ``Price From`` / ``Price`` / ``Shares`` come back NaN for every IPO, past
or upcoming (verified: 0 non-null across the window). So ``ipo_calendar.price_range``
and ``shares`` can never be populated. ``ticker`` / ``ipo_date`` / ``company_name``
/ ``exchange`` / ``currency`` remain live. Dropping is non-destructive (all NULL).
``downgrade`` re-adds the columns (nullable) for reversibility.

Revision ID: b6c7d8e9f0a1
Revises: a5b6c7d8e9f0
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b6c7d8e9f0a1"
down_revision: str | Sequence[str] | None = "a5b6c7d8e9f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("ipo_calendar", "price_range")
    op.drop_column("ipo_calendar", "shares")


def downgrade() -> None:
    op.add_column(
        "ipo_calendar",
        sa.Column("price_range", sa.String(100), nullable=True),
    )
    op.add_column(
        "ipo_calendar",
        sa.Column("shares", sa.BigInteger(), nullable=True),
    )
