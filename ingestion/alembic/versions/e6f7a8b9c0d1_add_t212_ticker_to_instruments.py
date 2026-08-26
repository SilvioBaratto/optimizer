"""Add t212_ticker to instruments; backfill ticker from yfinance_ticker.

yfinance is now the universe source (SPEC D9/D13/D14): ``ticker`` holds the Yahoo
ticker and a nullable ``t212_ticker`` carries the optional Trading 212 mapping.
The backfill sets ``ticker = yfinance_ticker`` for existing rows so the column's
meaning is consistent going forward.

The backfill is one-way — ``downgrade`` only drops the new column and does not
restore the previous ``ticker`` values (acceptable: pre-pivot universes are
rebuilt from the Screener). Restore from a pre-upgrade dump if the old tickers
are needed.

Revision ID: e6f7a8b9c0d1
Revises: b9c8d7e6f5a4
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: str | Sequence[str] | None = "b9c8d7e6f5a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "instruments",
        sa.Column("t212_ticker", sa.String(length=100), nullable=True),
    )
    # ticker becomes the Yahoo ticker (D13). Backfill from yfinance_ticker where
    # one exists so existing rows carry the new meaning.
    op.execute(
        "UPDATE instruments SET ticker = yfinance_ticker "
        "WHERE yfinance_ticker IS NOT NULL AND yfinance_ticker <> ''"
    )


def downgrade() -> None:
    op.drop_column("instruments", "t212_ticker")
