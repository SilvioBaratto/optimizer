"""Seed SPY reference index instrument.

Benchmarks are ingested outside the T212 universe builder pipeline because
they are reference indices, not tradable universe members.  The T212 builder
filters for ``type == 'STOCK'`` and would reject ETFs.

Revision ID: x4y5z6a7b8c9
Revises: w3x4y5z6a7b8
Create Date: 2026-04-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "x4y5z6a7b8c9"
down_revision: str | Sequence[str] | None = "w3x4y5z6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Guard: no-op on fresh databases where NYSE exchange is not yet populated.
    # Exchanges are seeded at runtime by the Trading212 universe builder, not by
    # any migration. Without the WHERE EXISTS clause the subquery returns NULL,
    # causing a NOT NULL constraint violation on exchange_id.
    op.execute("""
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
            'SPY',
            'SPY',
            'SPDR S&P 500 ETF Trust',
            'US78462F1030',
            'ETF',
            'USD',
            'SPY',
            (SELECT id FROM exchanges WHERE name = 'NYSE'),
            now(),
            now()
        WHERE EXISTS (SELECT 1 FROM exchanges WHERE name = 'NYSE')
        ON CONFLICT ON CONSTRAINT uq_instrument_ticker_exchange DO NOTHING
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM instruments
        WHERE ticker = 'SPY'
          AND yfinance_ticker = 'SPY'
          AND instrument_type = 'ETF'
    """)
