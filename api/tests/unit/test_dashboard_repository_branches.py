"""Branch-coverage tests for DashboardRepository.

Covers the missed lines reported at 32%:
  35-59   get_latest_fred_observations (empty-series_ids, rows, <2 values omitted)
  68-79   get_latest_fred_observation_dates (rows / empty)
  87-96   get_benchmark_prices (rows / empty)
  100-107 get_benchmark_latest_date (found / None)
  120     get_benchmark_coverage (empty tickers early-return)
  137     get_benchmark_coverage (rows present)
  145-153 get_ten_year_yield_usa (found / None)
  157-162 get_ten_year_yield_reference_date (found / None)
  170-173 instrument_exists (True / False)
  187-211 get_multi_ticker_prices (populated / empty / empty-tickers early-return)
  219-224 get_etf_instruments (found / empty)
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from app.models.macro.macro_regime import BondYield, FredObservation
from app.models.market_data.yfinance_data import PriceHistory
from app.models.universe.universe import Instrument
from app.repositories.dashboard.dashboard_repository import DashboardRepository
from tests._fixtures._helpers import add_and_flush
from tests._fixtures.universe import seed_universe

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_instrument(
    session: Session,
    *,
    ticker: str,
    yfinance_ticker: str | None = None,
    instrument_type: str | None = None,
) -> Instrument:
    seed = seed_universe(session, ticker=ticker, exchange_name=f"EX-{ticker}")
    inst = seed.instrument
    if yfinance_ticker is not None:
        inst.yfinance_ticker = yfinance_ticker
        session.flush()
    if instrument_type is not None:
        inst.instrument_type = instrument_type
        session.flush()
    return inst


def _seed_price(
    session: Session,
    instrument: Instrument,
    price_date: date,
    close: float,
) -> PriceHistory:
    return add_and_flush(
        session,
        PriceHistory(
            instrument_id=instrument.id,
            date=price_date,
            close=close,
        ),
    )


def _seed_fred(
    session: Session,
    series_id: str,
    obs_date: date,
    value: float | None,
) -> FredObservation:
    return add_and_flush(
        session,
        FredObservation(series_id=series_id, date=obs_date, value=value),
    )


def _seed_bond_yield(
    session: Session,
    country: str,
    maturity: str,
    yield_value: float | None,
    day_change: float | None = 0.05,
) -> BondYield:
    return add_and_flush(
        session,
        BondYield(
            country=country,
            maturity=maturity,
            yield_value=yield_value,
            day_change=day_change,
        ),
    )


# ---------------------------------------------------------------------------
# get_latest_fred_observations
# ---------------------------------------------------------------------------


class TestGetLatestFredObservations:
    def test_when_empty_series_ids_returns_empty_dict(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observations([])
        assert result == {}

    def test_when_two_observations_returns_latest_and_previous(
        self, db_session: Session
    ) -> None:
        _seed_fred(db_session, "DGS10", date(2024, 1, 2), 4.10)
        _seed_fred(db_session, "DGS10", date(2024, 1, 1), 4.00)
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observations(["DGS10"])
        assert "DGS10" in result
        latest, previous = result["DGS10"]
        assert latest == pytest.approx(4.10)
        assert previous == pytest.approx(4.00)

    def test_when_only_one_observation_series_omitted(
        self, db_session: Session
    ) -> None:
        _seed_fred(db_session, "UNRATE", date(2024, 1, 1), 3.9)
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observations(["UNRATE"])
        assert "UNRATE" not in result

    def test_when_unknown_series_id_returns_empty_dict(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observations(["NOSUCHSERIES"])
        assert result == {}

    def test_when_null_value_row_is_skipped(self, db_session: Session) -> None:
        _seed_fred(db_session, "T10Y2Y", date(2024, 1, 3), 0.55)
        _seed_fred(db_session, "T10Y2Y", date(2024, 1, 2), None)
        _seed_fred(db_session, "T10Y2Y", date(2024, 1, 1), 0.40)
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observations(["T10Y2Y"])
        # Null row is filtered; only (0.55, 0.40) survive
        assert "T10Y2Y" in result
        assert result["T10Y2Y"] == pytest.approx((0.55, 0.40))


# ---------------------------------------------------------------------------
# get_latest_fred_observation_dates
# ---------------------------------------------------------------------------


class TestGetLatestFredObservationDates:
    def test_when_empty_series_ids_returns_none(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_latest_fred_observation_dates([]) is None

    def test_when_rows_present_returns_most_recent_date(
        self, db_session: Session
    ) -> None:
        _seed_fred(db_session, "DGS10", date(2024, 3, 1), 4.20)
        _seed_fred(db_session, "DGS10", date(2024, 2, 1), 4.10)
        repo = DashboardRepository(db_session)
        result = repo.get_latest_fred_observation_dates(["DGS10"])
        assert result == date(2024, 3, 1)

    def test_when_no_matching_rows_returns_none(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_latest_fred_observation_dates(["GHOST"]) is None


# ---------------------------------------------------------------------------
# get_benchmark_prices
# ---------------------------------------------------------------------------


class TestGetBenchmarkPrices:
    def test_when_ticker_missing_returns_empty_list(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_benchmark_prices("MISSING") == []

    def test_when_prices_present_returns_oldest_first(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(
            db_session, ticker="SPY", yfinance_ticker="SPY"
        )
        _seed_price(db_session, inst, date(2024, 1, 1), 450.0)
        _seed_price(db_session, inst, date(2024, 1, 2), 455.0)
        repo = DashboardRepository(db_session)
        prices = repo.get_benchmark_prices("SPY", n=2)
        assert prices == pytest.approx([450.0, 455.0])

    def test_when_n_limits_result(self, db_session: Session) -> None:
        inst = _seed_instrument(
            db_session, ticker="QQQ", yfinance_ticker="QQQ"
        )
        for i, close in enumerate([400.0, 405.0, 410.0]):
            _seed_price(db_session, inst, date(2024, 1, i + 1), close)
        repo = DashboardRepository(db_session)
        prices = repo.get_benchmark_prices("QQQ", n=2)
        assert len(prices) == 2
        # n=2 fetches latest 2 then reverses → oldest of the two first
        assert prices[0] < prices[1]


# ---------------------------------------------------------------------------
# get_benchmark_latest_date
# ---------------------------------------------------------------------------


class TestGetBenchmarkLatestDate:
    def test_when_no_prices_returns_none(self, db_session: Session) -> None:
        _seed_instrument(db_session, ticker="IWM", yfinance_ticker="IWM")
        repo = DashboardRepository(db_session)
        assert repo.get_benchmark_latest_date("IWM") is None

    def test_when_prices_present_returns_most_recent_date(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(
            db_session, ticker="DIA", yfinance_ticker="DIA"
        )
        _seed_price(db_session, inst, date(2024, 1, 1), 380.0)
        _seed_price(db_session, inst, date(2024, 1, 5), 385.0)
        repo = DashboardRepository(db_session)
        assert repo.get_benchmark_latest_date("DIA") == date(2024, 1, 5)


# ---------------------------------------------------------------------------
# get_benchmark_coverage
# ---------------------------------------------------------------------------


class TestGetBenchmarkCoverage:
    def test_when_empty_tickers_returns_empty_dict(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_benchmark_coverage([]) == {}

    def test_when_ticker_has_no_prices_returns_zero_count_none_date(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, ticker="VTI", yfinance_ticker="VTI")
        repo = DashboardRepository(db_session)
        result = repo.get_benchmark_coverage(["VTI"])
        assert result["VTI"] == (0, None)

    def test_when_prices_present_returns_count_and_latest_date(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(
            db_session, ticker="GLD", yfinance_ticker="GLD"
        )
        _seed_price(db_session, inst, date(2024, 1, 1), 185.0)
        _seed_price(db_session, inst, date(2024, 1, 2), 186.0)
        repo = DashboardRepository(db_session)
        result = repo.get_benchmark_coverage(["GLD"])
        count, latest = result["GLD"]
        assert count == 2
        assert latest == date(2024, 1, 2)

    def test_when_ticker_not_in_db_returns_default_zero_none(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        result = repo.get_benchmark_coverage(["GHOST"])
        assert result["GHOST"] == (0, None)


# ---------------------------------------------------------------------------
# get_ten_year_yield_usa
# ---------------------------------------------------------------------------


class TestGetTenYearYieldUsa:
    def test_when_no_row_returns_none(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_ten_year_yield_usa() is None

    def test_when_row_exists_returns_yield_and_day_change(
        self, db_session: Session
    ) -> None:
        _seed_bond_yield(
            db_session, country="United States", maturity="10Y",
            yield_value=4.25, day_change=0.03
        )
        repo = DashboardRepository(db_session)
        result = repo.get_ten_year_yield_usa()
        assert result is not None
        yval, dchange = result
        assert yval == pytest.approx(4.25)
        assert dchange == pytest.approx(0.03)

    def test_when_row_has_null_yield_returns_none(
        self, db_session: Session
    ) -> None:
        _seed_bond_yield(
            db_session, country="United States", maturity="10Y",
            yield_value=None, day_change=0.01
        )
        repo = DashboardRepository(db_session)
        assert repo.get_ten_year_yield_usa() is None

    def test_when_day_change_is_null_defaults_to_zero(
        self, db_session: Session
    ) -> None:
        _seed_bond_yield(
            db_session, country="USA", maturity="10Y",
            yield_value=4.50, day_change=None
        )
        repo = DashboardRepository(db_session)
        result = repo.get_ten_year_yield_usa()
        assert result is not None
        _, dchange = result
        assert dchange == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_ten_year_yield_reference_date
# ---------------------------------------------------------------------------


class TestGetTenYearYieldReferenceDate:
    def test_when_no_row_returns_none(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.get_ten_year_yield_reference_date() is None

    def test_when_row_exists_returns_reference_date(
        self, db_session: Session
    ) -> None:
        row = _seed_bond_yield(
            db_session, country="United States", maturity="10Y",
            yield_value=4.0
        )
        row.reference_date = date(2024, 6, 1)
        db_session.flush()
        repo = DashboardRepository(db_session)
        assert repo.get_ten_year_yield_reference_date() == date(2024, 6, 1)


# ---------------------------------------------------------------------------
# instrument_exists
# ---------------------------------------------------------------------------


class TestInstrumentExists:
    def test_when_ticker_absent_returns_false(self, db_session: Session) -> None:
        repo = DashboardRepository(db_session)
        assert repo.instrument_exists("NOTHERE") is False

    def test_when_ticker_present_returns_true(self, db_session: Session) -> None:
        _seed_instrument(db_session, ticker="NVDA", yfinance_ticker="NVDA")
        repo = DashboardRepository(db_session)
        assert repo.instrument_exists("NVDA") is True


# ---------------------------------------------------------------------------
# get_multi_ticker_prices
# ---------------------------------------------------------------------------


class TestGetMultiTickerPrices:
    def test_when_empty_tickers_returns_empty_dataframe(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        df = repo.get_multi_ticker_prices([], date(2024, 1, 1), date(2024, 1, 31))
        assert df.empty

    def test_when_no_matching_rows_returns_empty_dataframe(
        self, db_session: Session
    ) -> None:
        repo = DashboardRepository(db_session)
        df = repo.get_multi_ticker_prices(
            ["GHOST"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert df.empty

    def test_when_prices_present_returns_pivoted_dataframe(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(
            db_session, ticker="MSFT", yfinance_ticker="MSFT"
        )
        _seed_price(db_session, inst, date(2024, 1, 2), 370.0)
        _seed_price(db_session, inst, date(2024, 1, 3), 375.0)
        repo = DashboardRepository(db_session)
        df = repo.get_multi_ticker_prices(
            ["MSFT"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert not df.empty
        assert "MSFT" in df.columns
        assert df["MSFT"].iloc[0] == pytest.approx(370.0)
        assert df["MSFT"].iloc[1] == pytest.approx(375.0)

    def test_when_outside_date_range_returns_empty_dataframe(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(
            db_session, ticker="AMZN", yfinance_ticker="AMZN"
        )
        _seed_price(db_session, inst, date(2023, 12, 31), 150.0)
        repo = DashboardRepository(db_session)
        df = repo.get_multi_ticker_prices(
            ["AMZN"], date(2024, 1, 1), date(2024, 1, 31)
        )
        assert df.empty


# ---------------------------------------------------------------------------
# get_etf_instruments
# ---------------------------------------------------------------------------


class TestGetEtfInstruments:
    def test_when_no_etf_rows_returns_empty_list(
        self, db_session: Session
    ) -> None:
        # Seed a non-ETF instrument to confirm it is excluded
        _seed_instrument(
            db_session, ticker="AAPL", instrument_type="Stock"
        )
        repo = DashboardRepository(db_session)
        result = repo.get_etf_instruments()
        assert result == []

    def test_when_etf_rows_present_returns_them_ordered_by_yfinance_ticker(
        self, db_session: Session
    ) -> None:
        # The repo orders by yfinance_ticker — set it explicitly so ORDER BY is
        # deterministic on SQLite (NULLs have undefined sort position).
        _seed_instrument(
            db_session, ticker="XLK", yfinance_ticker="XLK", instrument_type="ETF"
        )
        _seed_instrument(
            db_session, ticker="GLD", yfinance_ticker="GLD", instrument_type="ETF"
        )
        repo = DashboardRepository(db_session)
        result = repo.get_etf_instruments()
        yf_tickers = [i.yfinance_ticker for i in result]
        assert yf_tickers == sorted(yf_tickers)
        assert {i.ticker for i in result} == {"XLK", "GLD"}
