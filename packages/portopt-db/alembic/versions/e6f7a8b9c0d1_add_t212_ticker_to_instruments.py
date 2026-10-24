"""Add t212_ticker to instruments; backfill ticker from yfinance_ticker.

yfinance is now the universe source (SPEC D9/D13/D14): ``ticker`` holds the Yahoo
ticker and a nullable ``t212_ticker`` carries the optional Trading 212 mapping.
The backfill sets ``ticker = yfinance_ticker`` for existing rows so the column's
meaning is consistent going forward. ``uq_instrument_ticker_exchange`` is on
``(ticker, exchange_id)`` while ``yfinance_ticker`` is not unique, so several
Trading 212 lines on one exchange can share a Yahoo symbol. Only the oldest line
of each such group is backfilled, and a row is also skipped when another row on
its exchange already holds the symbol as its ``ticker``. Skipped rows keep their
previous ticker, so the upgrade completes instead of aborting on the constraint.

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
    # one exists so existing rows carry the new meaning. The constraint is
    # checked row by row, so both guards must be decided from the pre-update
    # state: the oldest row of a (exchange, Yahoo symbol) group wins, and a
    # symbol already used as another row's ticker is never taken over.
    op.execute(
        "UPDATE instruments SET ticker = yfinance_ticker "
        "WHERE yfinance_ticker IS NOT NULL AND yfinance_ticker <> '' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM instruments AS holder "
        "WHERE holder.exchange_id = instruments.exchange_id "
        "AND holder.ticker = instruments.yfinance_ticker "
        "AND holder.id <> instruments.id) "
        "AND NOT EXISTS ("
        "SELECT 1 FROM instruments AS earlier "
        "WHERE earlier.exchange_id = instruments.exchange_id "
        "AND earlier.yfinance_ticker = instruments.yfinance_ticker "
        "AND (earlier.created_at < instruments.created_at "
        "OR (earlier.created_at = instruments.created_at "
        "AND earlier.id < instruments.id)))"
    )


def downgrade() -> None:
    op.drop_column("instruments", "t212_ticker")
