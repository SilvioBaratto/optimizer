"""Unit tests for _price_fetcher.fetch_close_prices (issue #382).

TDD: Red phase — tests written before implementation.
All tests are isolated via mocks; no real DB calls.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(ticker_to_rows: dict[str, list]) -> MagicMock:
    """Return a mock YFinanceRepository backed by ticker_to_rows mapping."""
    repo = MagicMock()

    def get_instrument(ticker: str) -> MagicMock | None:
        if ticker not in ticker_to_rows:
            return None
        inst = MagicMock()
        inst.id = ticker
        return inst

    def get_price_history(instrument_id, start_date=None, end_date=None, **_):
        rows = ticker_to_rows.get(instrument_id, [])
        return [
            r
            for r in rows
            if (start_date is None or r.date >= start_date)
            and (end_date is None or r.date <= end_date)
        ]

    repo.get_instrument_by_yfinance_ticker.side_effect = get_instrument
    repo.get_price_history.side_effect = get_price_history
    return repo


def _make_row(d: date, close: float | None) -> MagicMock:
    row = MagicMock()
    row.date = d
    row.close = close
    return row


# ---------------------------------------------------------------------------
# fetch_close_prices — happy path
# ---------------------------------------------------------------------------


class TestFetchClosePricesReturnsDataFrame:
    """fetch_close_prices returns a DataFrame indexed by date."""

    def test_returns_dataframe(self) -> None:
        from app.services._shared import fetch_close_prices

        rows = [_make_row(date(2024, 1, 2), 100.0), _make_row(date(2024, 1, 3), 101.0)]
        repo = _make_repo({"AAPL": rows})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["AAPL"], session)

        assert isinstance(result, pd.DataFrame)
        assert "AAPL" in result.columns

    def test_prices_sorted_ascending_by_date(self) -> None:
        from app.services._shared import fetch_close_prices

        rows = [
            _make_row(date(2024, 1, 3), 101.0),
            _make_row(date(2024, 1, 2), 100.0),
        ]
        repo = _make_repo({"AAPL": rows})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["AAPL"], session)

        assert list(result.index) == sorted(result.index)

    def test_multiple_tickers_become_columns(self) -> None:
        from app.services._shared import fetch_close_prices

        repo = _make_repo(
            {
                "AAPL": [_make_row(date(2024, 1, 2), 100.0)],
                "MSFT": [_make_row(date(2024, 1, 2), 200.0)],
            }
        )
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["AAPL", "MSFT"], session)

        assert set(result.columns) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# fetch_close_prices — date range filtering
# ---------------------------------------------------------------------------


class TestFetchClosePricesDateFiltering:
    """start_date / end_date are passed to the repository."""

    def test_date_range_forwarded_to_repository(self) -> None:
        from app.services._shared import fetch_close_prices

        rows = [
            _make_row(date(2024, 1, 1), 99.0),
            _make_row(date(2024, 1, 2), 100.0),
            _make_row(date(2024, 1, 3), 101.0),
        ]
        repo = _make_repo({"AAPL": rows})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(
                ["AAPL"],
                session,
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 3),
            )

        # only rows within range
        assert len(result) == 2

    def test_no_dates_returns_all_rows(self) -> None:
        from app.services._shared import fetch_close_prices

        rows = [_make_row(date(2024, 1, i), float(i)) for i in range(1, 6)]
        repo = _make_repo({"AAPL": rows})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["AAPL"], session)

        assert len(result) == 5


# ---------------------------------------------------------------------------
# fetch_close_prices — edge cases
# ---------------------------------------------------------------------------


class TestFetchClosePricesEdgeCases:
    """Handles missing tickers and None close values gracefully."""

    def test_unknown_ticker_silently_dropped(self) -> None:
        from app.services._shared import fetch_close_prices

        repo = _make_repo({})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["UNKNOWN"], session)

        assert result.empty

    def test_none_close_value_excluded(self) -> None:
        from app.services._shared import fetch_close_prices

        rows = [
            _make_row(date(2024, 1, 2), None),
            _make_row(date(2024, 1, 3), 101.0),
        ]
        repo = _make_repo({"AAPL": rows})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices(["AAPL"], session)

        assert result.loc[date(2024, 1, 3), "AAPL"] == pytest.approx(101.0)
        assert date(2024, 1, 2) not in result.index

    def test_empty_ticker_list_returns_empty_dataframe(self) -> None:
        from app.services._shared import fetch_close_prices

        repo = _make_repo({})
        session = MagicMock()

        with _patch_repo(repo):
            result = fetch_close_prices([], session)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_repo(repo: MagicMock):
    from unittest.mock import patch

    return patch(
        "app.services._shared._price_fetcher.YFinanceRepository",
        return_value=repo,
    )
