"""Branch-coverage tests for ViewGenerationRepository.

Covers the missed lines reported at 76%:
  26      get_instrument_by_ticker — None branch
  36-50   get_close_prices — rows in window / empty / None-close filtered
  54      get_ticker_profile — None branch
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from app.models.market_data.yfinance_data import PriceHistory, TickerProfile
from app.repositories.views.view_generation_repository import ViewGenerationRepository
from tests._fixtures._helpers import add_and_flush
from tests._fixtures.universe import seed_universe

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_instrument_with_ticker(session: Session, ticker: str):
    seed = seed_universe(session, ticker=ticker, exchange_name=f"EX-{ticker}")
    return seed.instrument


def _seed_price(
    session: Session,
    instrument,
    price_date: date,
    close: float | None,
) -> PriceHistory:
    return add_and_flush(
        session,
        PriceHistory(instrument_id=instrument.id, date=price_date, close=close),
    )


def _seed_profile(session: Session, instrument, sector: str = "Technology") -> TickerProfile:
    return add_and_flush(
        session,
        TickerProfile(
            instrument_id=instrument.id,
            symbol=instrument.ticker,
            sector=sector,
            country="United States",
            currency="USD",
        ),
    )


# ---------------------------------------------------------------------------
# get_instrument_by_ticker
# ---------------------------------------------------------------------------


class TestGetInstrumentByTicker:
    def test_when_ticker_absent_returns_none(self, db_session: Session) -> None:
        repo = ViewGenerationRepository(db_session)
        result = repo.get_instrument_by_ticker("NOTHERE")
        assert result is None

    def test_when_ticker_present_returns_instrument(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument_with_ticker(db_session, "AAPL")
        repo = ViewGenerationRepository(db_session)
        result = repo.get_instrument_by_ticker("AAPL")
        assert result is not None
        assert result.id == inst.id
        assert result.ticker == "AAPL"


# ---------------------------------------------------------------------------
# get_close_prices
# ---------------------------------------------------------------------------


class TestGetClosePrices:
    def test_when_no_prices_returns_empty_list(self, db_session: Session) -> None:
        inst = _seed_instrument_with_ticker(db_session, "MSFT")
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=30)
        assert result == []

    def test_when_price_within_window_returned_as_float(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument_with_ticker(db_session, "GOOG")
        recent = date.today() - timedelta(days=5)
        _seed_price(db_session, inst, recent, 175.50)
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=30)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert result[0] == pytest.approx(175.50)

    def test_when_price_outside_window_excluded(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument_with_ticker(db_session, "NVDA")
        old_date = date.today() - timedelta(days=400)
        _seed_price(db_session, inst, old_date, 500.0)
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=260)
        assert result == []

    def test_when_null_close_is_filtered_out(self, db_session: Session) -> None:
        inst = _seed_instrument_with_ticker(db_session, "META")
        recent = date.today() - timedelta(days=3)
        _seed_price(db_session, inst, recent, None)
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=30)
        assert result == []

    def test_when_mix_of_null_and_valid_only_valid_returned(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument_with_ticker(db_session, "TSLA")
        today = date.today()
        _seed_price(db_session, inst, today - timedelta(days=2), None)
        _seed_price(db_session, inst, today - timedelta(days=1), 245.0)
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=10)
        assert result == pytest.approx([245.0])

    def test_prices_returned_in_ascending_date_order(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument_with_ticker(db_session, "AMZN")
        today = date.today()
        _seed_price(db_session, inst, today - timedelta(days=3), 180.0)
        _seed_price(db_session, inst, today - timedelta(days=2), 182.0)
        _seed_price(db_session, inst, today - timedelta(days=1), 184.0)
        repo = ViewGenerationRepository(db_session)
        result = repo.get_close_prices(inst.id, lookback_days=30)
        assert result == pytest.approx([180.0, 182.0, 184.0])


# ---------------------------------------------------------------------------
# get_ticker_profile
# ---------------------------------------------------------------------------


class TestGetTickerProfile:
    def test_when_no_profile_returns_none(self, db_session: Session) -> None:
        inst = _seed_instrument_with_ticker(db_session, "IBM")
        repo = ViewGenerationRepository(db_session)
        result = repo.get_ticker_profile(inst.id)
        assert result is None

    def test_when_profile_present_returns_it(self, db_session: Session) -> None:
        inst = _seed_instrument_with_ticker(db_session, "ORCL")
        profile = _seed_profile(db_session, inst, sector="Technology")
        repo = ViewGenerationRepository(db_session)
        result = repo.get_ticker_profile(inst.id)
        assert result is not None
        assert result.id == profile.id
        assert result.sector == "Technology"
