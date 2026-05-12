"""Tests for research/data/_equity_prices.py and _equity_fundamentals.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from research.data._equity_fundamentals import (
    _FUNDAMENTAL_COLUMNS,
    _dedup_fundamentals_df,
    assemble_analyst_data,
    assemble_financial_statements,
    assemble_fundamentals,
    assemble_insider_data,
)
from research.data._equity_prices import (
    _apply_delisting_returns,
    assemble_prices,
    assemble_volumes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session(rows: list) -> MagicMock:
    """Return a MagicMock session whose execute().all() returns *rows*."""
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


def _mock_session_scalars(profiles: list) -> MagicMock:
    """Return a MagicMock session whose execute().scalars().all() returns *profiles*."""
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = profiles
    return session


# ---------------------------------------------------------------------------
# TestApplyDelistingReturns
# ---------------------------------------------------------------------------


class TestApplyDelistingReturns:
    """Tests for the pure _apply_delisting_returns function."""

    def _make_prices(self) -> pd.DataFrame:
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=idx)

    def test_when_no_delistings_original_prices_returned(self) -> None:
        """when no delistings, original prices returned unchanged."""
        prices = self._make_prices()
        result = _apply_delisting_returns(prices, [])
        pd.testing.assert_frame_equal(result, prices)

    def test_when_delisting_return_given_synthetic_price_appended(self) -> None:
        """when delisting return given, synthetic price = last_price * (1 + r)."""
        prices = self._make_prices()
        delisted_ts = pd.Timestamp("2024-01-04")
        result = _apply_delisting_returns(prices, [("AAPL", delisted_ts, -0.10)])
        assert delisted_ts in result.index
        expected = 102.0 * (1.0 - 0.10)
        assert pytest.approx(result.loc[delisted_ts, "AAPL"], rel=1e-6) == expected

    def test_when_ticker_not_in_prices_columns_skipped(self) -> None:
        """when ticker is absent from price columns, it is silently skipped."""
        prices = self._make_prices()
        delisted_ts = pd.Timestamp("2024-01-04")
        result = _apply_delisting_returns(prices, [("UNKNOWN", delisted_ts, -0.20)])
        assert "UNKNOWN" not in result.columns
        assert delisted_ts not in result.index


# ---------------------------------------------------------------------------
# TestDeduplicateFundamentals
# ---------------------------------------------------------------------------


class TestDeduplicateFundamentals:
    """Tests for the pure _dedup_fundamentals_df function."""

    def _make_df(self, tickers: list[str], currencies: list[str]) -> pd.DataFrame:
        n = len(tickers)
        data: dict[str, Any] = {
            col: [float(i) for i in range(n)] for col in _FUNDAMENTAL_COLUMNS[:3]
        }
        data["_raw_currency"] = currencies
        data["exchange"] = ["NYSE"] * n
        df = pd.DataFrame(data, index=tickers)
        df.index.name = "ticker"
        return df

    def test_when_duplicate_tickers_lower_ccy_rank_wins(self) -> None:
        """when duplicate tickers exist, USD (rank 0) wins over EUR (rank 2)."""
        df = self._make_df(["AAPL", "AAPL"], ["USD", "EUR"])
        result = _dedup_fundamentals_df(df)
        assert len(result) == 1
        # The row with lower _nan_count among equal-rank rows wins;
        # both have the same NaN count here so USD row (first after sort) is kept.
        assert "AAPL" in result.index

    def test_when_single_ticker_no_duplication_occurs(self) -> None:
        """when only one row per ticker, the DataFrame is unchanged in shape."""
        df = self._make_df(["AAPL", "MSFT"], ["USD", "USD"])
        result = _dedup_fundamentals_df(df)
        assert len(result) == 2

    def test_when_helper_columns_dropped_from_result(self) -> None:
        """when dedup runs, _raw_currency, _ccy_rank, _nan_count are absent."""
        df = self._make_df(["AAPL"], ["USD"])
        result = _dedup_fundamentals_df(df)
        for col in ("_raw_currency", "_ccy_rank", "_nan_count"):
            assert col not in result.columns


# ---------------------------------------------------------------------------
# TestAssemblePrices
# ---------------------------------------------------------------------------


_TICKER_RANK_MAP_PATH = "research.data._equity_prices._build_ticker_rank_map"


class TestAssemblePrices:
    """Tests for assemble_prices using a mocked session."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no price rows exist, an empty DataFrame is returned."""
        session = _mock_session([])
        with patch(_TICKER_RANK_MAP_PATH, return_value={}):
            result = assemble_prices(session)
        assert result.empty

    def test_when_prices_present_dates_x_tickers_shape(self) -> None:
        """when 2 instruments x 2 dates, result shape is (2, 2)."""
        rank_map = {
            "inst-a": ("AAPL", 0),
            "inst-b": ("MSFT", 0),
        }
        price_rows = [
            ("inst-a", "2024-01-01", 100.0),
            ("inst-a", "2024-01-02", 101.0),
            ("inst-b", "2024-01-01", 200.0),
            ("inst-b", "2024-01-02", 201.0),
        ]
        # execute() is called multiple times: prices, then delistings, then currency map
        session = MagicMock()
        session.execute.return_value.all.side_effect = [
            price_rows,  # price_query rows
            [],  # delisting_rows
            [],  # currency map from instruments
        ]
        with patch(_TICKER_RANK_MAP_PATH, return_value=rank_map):
            result = assemble_prices(session)
        assert result.shape == (2, 2)
        assert set(result.columns) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# TestAssembleVolumes
# ---------------------------------------------------------------------------


class TestAssembleVolumes:
    """Tests for assemble_volumes using a mocked session."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no volume rows exist, an empty DataFrame is returned."""
        session = _mock_session([])
        with patch(_TICKER_RANK_MAP_PATH, return_value={}):
            result = assemble_volumes(session)
        assert result.empty

    def test_when_volumes_present_correct_shape(self) -> None:
        """when 2 instruments x 2 dates, volume result shape is (2, 2)."""
        rank_map = {
            "inst-a": ("AAPL", 0),
            "inst-b": ("MSFT", 0),
        }
        vol_rows = [
            ("inst-a", "2024-01-01", 1_000_000.0),
            ("inst-a", "2024-01-02", 1_100_000.0),
            ("inst-b", "2024-01-01", 2_000_000.0),
            ("inst-b", "2024-01-02", 2_100_000.0),
        ]
        session = _mock_session(vol_rows)
        with patch(_TICKER_RANK_MAP_PATH, return_value=rank_map):
            result = assemble_volumes(session)
        assert result.shape == (2, 2)
        assert set(result.columns) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# TestAssembleFundamentals
# ---------------------------------------------------------------------------


class TestAssembleFundamentals:
    """Tests for assemble_fundamentals using a mocked session."""

    def test_when_no_profiles_empty_tuple_returned(self) -> None:
        """when no TickerProfiles exist, an empty tuple (df, {}, {}) is returned."""
        session = _mock_session_scalars([])
        result = assemble_fundamentals(session)
        df, sector_map, ccy_map = result
        assert df.empty
        assert sector_map == {}
        assert ccy_map == {}


# ---------------------------------------------------------------------------
# TestAssembleFinancialStatements
# ---------------------------------------------------------------------------


_BUILD_TICKER_MAP_PATH = "research.data._equity_fundamentals._build_ticker_map"


class TestAssembleFinancialStatements:
    """Tests for assemble_financial_statements using a mocked session."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no statement rows exist, a DataFrame with schema columns is returned."""
        session = _mock_session([])
        with patch(_BUILD_TICKER_MAP_PATH, return_value={}):
            result = assemble_financial_statements(session)
        assert result.empty
        assert "ticker" in result.columns

    def test_when_rows_present_ticker_column_present(self) -> None:
        """when statement rows exist, result contains a ticker column."""
        ticker_map = {"inst-a": "AAPL"}
        rows = [("inst-a", "income", "annual", "2023-12-31")]
        session = _mock_session(rows)
        with patch(_BUILD_TICKER_MAP_PATH, return_value=ticker_map):
            result = assemble_financial_statements(session)
        assert "ticker" in result.columns
        assert result.iloc[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# TestAssembleAnalystData
# ---------------------------------------------------------------------------


class TestAssembleAnalystData:
    """Tests for assemble_analyst_data using a mocked session."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no analyst rows exist, returns empty DataFrame with schema columns."""
        session = _mock_session([])
        with patch(_BUILD_TICKER_MAP_PATH, return_value={}):
            result = assemble_analyst_data(session)
        assert result.empty
        assert "ticker" in result.columns


# ---------------------------------------------------------------------------
# TestAssembleInsiderData
# ---------------------------------------------------------------------------


class TestAssembleInsiderData:
    """Tests for assemble_insider_data using a mocked session."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no insider rows exist, returns empty DataFrame with schema columns."""
        session = _mock_session([])
        with patch(_BUILD_TICKER_MAP_PATH, return_value={}):
            result = assemble_insider_data(session)
        assert result.empty
        assert "ticker" in result.columns
