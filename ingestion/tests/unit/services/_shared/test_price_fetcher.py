"""Unit tests for ``_shared/_price_fetcher.fetch_close_prices`` (issue #826).

NOTE on AC: the criterion 'zero-row DB path raises the expected error type'
does NOT match this module. ``fetch_close_prices`` SILENTLY returns an empty
DataFrame by contract (docstring: "Tickers absent from the database are
silently dropped"). The raise-on-empty guard lives in the *caller*
(``backtest_service.validate_prices`` / ``run_and_persist`` raise ``ValueError``
on empty prices). These tests therefore assert the documented silent-empty
contract; the repository is mocked at the import site (no DB).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services._shared._price_fetcher import fetch_close_prices

_REPO = "app.services._shared._price_fetcher.YFinanceRepository"


def _row(d: date, close: float | None) -> SimpleNamespace:
    return SimpleNamespace(date=d, close=close)


class TestFetchClosePrices:
    def test_when_no_instruments_found_then_returns_empty_frame(self) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_instrument_by_yfinance_ticker.return_value = None
            result = fetch_close_prices(["AAPL", "MSFT"], MagicMock())

        assert result.empty  # silent drop, NOT a raise

    def test_when_instrument_has_prices_then_column_built(self) -> None:
        rows = [_row(date(2024, 1, 2), 100.0), _row(date(2024, 1, 3), 101.0)]
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_instrument_by_yfinance_ticker.return_value = SimpleNamespace(
                id="inst-1"
            )
            repo.get_price_history.return_value = rows
            result = fetch_close_prices(["AAPL"], MagicMock())

        assert list(result.columns) == ["AAPL"]
        assert result["AAPL"].tolist() == [100.0, 101.0]

    def test_when_close_is_none_then_row_excluded(self) -> None:
        rows = [_row(date(2024, 1, 2), None), _row(date(2024, 1, 3), 101.0)]
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_instrument_by_yfinance_ticker.return_value = SimpleNamespace(
                id="inst-1"
            )
            repo.get_price_history.return_value = rows
            result = fetch_close_prices(["AAPL"], MagicMock())

        assert result["AAPL"].tolist() == [101.0]  # None close dropped

    def test_when_one_of_two_tickers_missing_then_only_present_kept(self) -> None:
        def _instrument(ticker: str):
            return SimpleNamespace(id="inst-1") if ticker == "AAPL" else None

        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.get_instrument_by_yfinance_ticker.side_effect = _instrument
            repo.get_price_history.return_value = [_row(date(2024, 1, 2), 100.0)]
            result = fetch_close_prices(["AAPL", "GHOST"], MagicMock())

        assert list(result.columns) == ["AAPL"]
