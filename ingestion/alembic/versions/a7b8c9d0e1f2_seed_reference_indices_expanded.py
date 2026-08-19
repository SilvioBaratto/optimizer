"""Expand reference-index seed to 10 canonical benchmarks (issue #430).

The preceding migration ``x4y5z6a7b8c9`` seeded only SPY. This migration adds
the remaining nine benchmarks that power the Dashboard benchmark selector
and the Backtesting Lab benchmark dropdown: QQQ, IWM, EFA, EEM, AGG, VGK,
VWO, TLT, GLD.

Design choice: we **add a new migration** rather than mutating
``x4y5z6a7b8c9`` retroactively, since that revision is already applied in
deployed environments. Combined with the predecessor, the production DB
ends up with all 10 canonical benchmarks.

Each insert mirrors the SPY pattern exactly:
  * ``gen_random_uuid()`` primary key
  * ``instrument_type='ETF'``
  * exchange looked up by name via a subquery
  * a ``WHERE EXISTS`` guard so the INSERT is a no-op when the target
    exchange (NYSE or NASDAQ) is not yet seeded on a fresh DB
  * ``ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange DO NOTHING``
    so re-running the migration is safe.

Revision ID: a7b8c9d0e1f2
Revises: z6a7b8c9d0e1
Create Date: 2026-04-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "z6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Benchmark catalogue
# ---------------------------------------------------------------------------
#
# Tuple layout: (ticker, short_name, long_name, isin, exchange_name, yfinance_ticker)
#
# Exchange mapping:
#   * QQQ trades on NASDAQ (the Invesco QQQ Trust primary listing).
#   * The rest are NYSE Arca ETFs; seeded under the pre-existing "NYSE"
#     exchange name to match the SPY row in migration x4y5z6a7b8c9.
#
# All ISINs are public and widely documented on Bloomberg / ETF.com /
# issuer fact sheets (iShares, Vanguard, Invesco, SPDR).
INDICES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("QQQ", "QQQ", "Invesco QQQ Trust", "US46090E1038", "NASDAQ", "QQQ"),
    ("IWM", "IWM", "iShares Russell 2000 ETF", "US4642876555", "NYSE", "IWM"),
    ("EFA", "EFA", "iShares MSCI EAFE ETF", "US4642874659", "NYSE", "EFA"),
    ("EEM", "EEM", "iShares MSCI Emerging Markets ETF", "US4642872349", "NYSE", "EEM"),
    (
        "AGG",
        "AGG",
        "iShares Core U.S. Aggregate Bond ETF",
        "US4642872265",
        "NYSE",
        "AGG",
    ),
    ("VGK", "VGK", "Vanguard FTSE Europe ETF", "US9220427671", "NYSE", "VGK"),
    ("VWO", "VWO", "Vanguard FTSE Emerging Markets ETF", "US9220428588", "NYSE", "VWO"),
    ("TLT", "TLT", "iShares 20+ Year Treasury Bond ETF", "US4642874329", "NYSE", "TLT"),
    ("GLD", "GLD", "SPDR Gold Shares", "US78463V1070", "NYSE", "GLD"),
)


def upgrade() -> None:
    for ticker, short_name, long_name, isin, exchange_name, yfinance_ticker in INDICES:
        op.execute(
            _build_insert_sql(
                ticker=ticker,
                short_name=short_name,
                long_name=long_name,
                isin=isin,
                exchange_name=exchange_name,
                yfinance_ticker=yfinance_ticker,
            )
        )


def downgrade() -> None:
    """Remove the nine rows seeded by this migration.

    SPY is left untouched — it is owned by the predecessor migration
    ``x4y5z6a7b8c9`` and that migration's own ``downgrade()`` handles it.
    """
    tickers = ", ".join(f"'{t[0]}'" for t in INDICES)
    op.execute(f"""
        DELETE FROM instruments
        WHERE ticker IN ({tickers})
          AND instrument_type = 'ETF'
    """)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_insert_sql(
    *,
    ticker: str,
    short_name: str,
    long_name: str,
    isin: str,
    exchange_name: str,
    yfinance_ticker: str,
) -> str:
    """Return the idempotent, exchange-guarded INSERT statement for one ETF."""
    return f"""
        INSERT INTO instruments (
            id,
            ticker,
            short_name,
            name,
            isin,
            instrument_type,
            currency_code,
            yfinance_ticker,
            exchange_id,
            created_at,
            updated_at
        )
        SELECT
            gen_random_uuid(),
            '{ticker}',
            '{short_name}',
            '{long_name}',
            '{isin}',
            'ETF',
            'USD',
            '{yfinance_ticker}',
            (SELECT id FROM exchanges WHERE name = '{exchange_name}'),
            now(),
            now()
        WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = '{exchange_name}')
        ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange DO NOTHING
    """
