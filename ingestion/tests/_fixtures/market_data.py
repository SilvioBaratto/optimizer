"""DB seed builder for the market_data domain.

Seeds ``Exchange`` + ``Instrument`` + ``TickerProfile`` only — no bulk
``PriceHistory`` (kept cheap for SQLite test runs).  Reuses
``seed_universe`` (get-or-create) so it composes with it without a
unique-constraint collision.
"""

from __future__ import annotations

from typing import NamedTuple

from portopt_db.models.market_data.yfinance_data import TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy.orm import Session

from tests._fixtures._helpers import add_and_flush
from tests._fixtures.universe import seed_universe


class MarketDataSeed(NamedTuple):
    exchange: Exchange
    instrument: Instrument
    profile: TickerProfile


def _make_profile(instrument: Instrument, sector: str) -> TickerProfile:
    return TickerProfile(
        instrument_id=instrument.id,
        symbol=instrument.ticker,
        sector=sector,
        industry="Consumer Electronics",
        country="United States",
        currency="USD",
    )


def seed_market_data(
    session: Session,
    *,
    exchange_name: str = "Test Exchange",
    ticker: str = "AAPL",
    sector: str = "Technology",
) -> MarketDataSeed:
    universe = seed_universe(session, exchange_name=exchange_name, ticker=ticker)
    profile = add_and_flush(session, _make_profile(universe.instrument, sector))
    return MarketDataSeed(
        exchange=universe.exchange,
        instrument=universe.instrument,
        profile=profile,
    )
