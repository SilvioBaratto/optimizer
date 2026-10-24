"""Repair ETF fund snapshots written before the yfinance funds-client fixes.

The funds client now stores Yahoo's fund valuation rows (Price/Earnings, /Book,
/Sales, /Cashflow) as multiples instead of the reciprocals Yahoo serves, drops
its ``0.0`` no-data sentinel, and no longer writes an all-NULL asset-class
snapshot. Those fixes only cover new writes, and snapshots are keyed on
``(instrument_id, as_of)``, so without this revision an instrument's history
would mix ``0.04``-style yields with ``24``-style multiples, and an all-NULL
``etf_asset_classes`` row would stay the newest ``as_of`` for any fund that now
yields no allocation.

Every row in the DB at upgrade time predates the fix: ``alembic upgrade head``
runs from the container entrypoint before the daemon starts, so no multiple has
been written yet. The inversion is not idempotent, so a database that ran the
fixed ingestion code before this revision must not apply it.

Revision ID: 3aa457e38da0
Revises: d8f0b2c4e6a1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "3aa457e38da0"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VALUATION_COLUMNS = (
    "price_to_earnings",
    "price_to_book",
    "price_to_sales",
    "price_to_cashflow",
)

_equity_holdings = sa.table(
    "etf_equity_holdings", *(sa.column(c, sa.Numeric) for c in _VALUATION_COLUMNS)
)
_asset_classes = sa.table(
    "etf_asset_classes",
    sa.column("stock_pct", sa.Numeric),
    sa.column("bond_pct", sa.Numeric),
    sa.column("cash_pct", sa.Numeric),
    sa.column("other_pct", sa.Numeric),
)


def _invert_positive_valuations() -> None:
    # A float literal keeps SQLite (tests) from truncating to integer division
    # when a value happens to be stored as an integer.
    one = sa.literal_column("1.0")
    for name in _VALUATION_COLUMNS:
        col = _equity_holdings.c[name]
        op.execute(_equity_holdings.update().where(col > 0).values({name: one / col}))


def upgrade() -> None:
    """Turn stored valuation reciprocals into multiples and drop empty snapshots."""
    for name in _VALUATION_COLUMNS:
        col = _equity_holdings.c[name]
        # 0.0 was Yahoo's no-data sentinel and a negative reciprocal (negative
        # earnings) has no meaningful multiple; neither survives inversion.
        op.execute(_equity_holdings.update().where(col <= 0).values({name: None}))
    _invert_positive_valuations()

    op.execute(
        _asset_classes.delete().where(
            sa.and_(
                _asset_classes.c.stock_pct.is_(None),
                _asset_classes.c.bond_pct.is_(None),
                _asset_classes.c.cash_pct.is_(None),
                _asset_classes.c.other_pct.is_(None),
            )
        )
    )


def downgrade() -> None:
    """Re-invert multiples to the reciprocal form the pre-fix writer produced.

    Sentinels nulled and empty snapshots deleted by ``upgrade()`` carried no
    information and are not restored.
    """
    _invert_positive_valuations()
