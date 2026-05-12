"""Tests for research/data/_helpers.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from research.data._helpers import (
    _DEDUP_DROP_THRESHOLD_PCT,
    _TRADING_DAYS,
    REGION_MAP,
    _build_ticker_map,
    _pivot_with_dedup,
    _to_float,
)

# ---------------------------------------------------------------------------
# REGION_MAP
# ---------------------------------------------------------------------------


class TestRegionMap:
    """Tests for the REGION_MAP constant."""

    def test_region_map_import_succeeds(self) -> None:
        """REGION_MAP can be imported from research.data._helpers."""
        assert REGION_MAP is not None

    def test_region_map_covers_exactly_40_countries(self) -> None:
        """REGION_MAP has exactly 40 country entries."""
        assert len(REGION_MAP) == 40

    def test_region_map_values_are_exactly_four_regions(self) -> None:
        """REGION_MAP maps to exactly 4 canonical region strings."""
        expected_regions = {
            "Americas",
            "Europe",
            "Asia-Pacific",
            "Middle East & Africa",
        }
        actual_regions = set(REGION_MAP.values())
        assert actual_regions == expected_regions

    def test_americas_countries_map_correctly(self) -> None:
        """Known Americas countries resolve to 'Americas'."""
        assert REGION_MAP["United States"] == "Americas"
        assert REGION_MAP["Canada"] == "Americas"
        assert REGION_MAP["Brazil"] == "Americas"

    def test_europe_countries_map_correctly(self) -> None:
        """Known Europe countries resolve to 'Europe'."""
        assert REGION_MAP["Germany"] == "Europe"
        assert REGION_MAP["United Kingdom"] == "Europe"
        assert REGION_MAP["France"] == "Europe"

    def test_asia_pacific_countries_map_correctly(self) -> None:
        """Known Asia-Pacific countries resolve to 'Asia-Pacific'."""
        assert REGION_MAP["Japan"] == "Asia-Pacific"
        assert REGION_MAP["Australia"] == "Asia-Pacific"
        assert REGION_MAP["China"] == "Asia-Pacific"

    def test_middle_east_africa_countries_map_correctly(self) -> None:
        """Known Middle East & Africa countries resolve to 'Middle East & Africa'."""
        assert REGION_MAP["Israel"] == "Middle East & Africa"
        assert REGION_MAP["South Africa"] == "Middle East & Africa"
        assert REGION_MAP["Saudi Arabia"] == "Middle East & Africa"

    def test_other_is_not_a_key(self) -> None:
        """'Other' is not a key; callers default unmapped countries themselves."""
        assert "Other" not in REGION_MAP


# ---------------------------------------------------------------------------
# _to_float
# ---------------------------------------------------------------------------


class TestToFloat:
    """Tests for _to_float coercion helper."""

    def test_none_returns_none(self) -> None:
        """_to_float(None) returns None."""
        assert _to_float(None) is None

    def test_decimal_returns_float(self) -> None:
        """_to_float(Decimal('1.5')) returns 1.5."""
        result = _to_float(Decimal("1.5"))
        assert result == 1.5
        assert isinstance(result, float)

    def test_integer_returns_float(self) -> None:
        """_to_float(3) returns 3.0."""
        result = _to_float(3)
        assert result == 3.0
        assert isinstance(result, float)

    def test_float_returns_float(self) -> None:
        """_to_float(2.5) returns 2.5."""
        result = _to_float(2.5)
        assert result == 2.5
        assert isinstance(result, float)

    def test_zero_returns_zero_float(self) -> None:
        """_to_float(0) returns 0.0."""
        assert _to_float(0) == 0.0

    def test_negative_decimal_returns_float(self) -> None:
        """_to_float(Decimal('-3.14')) returns -3.14."""
        result = _to_float(Decimal("-3.14"))
        assert pytest.approx(result) == -3.14


# ---------------------------------------------------------------------------
# _pivot_with_dedup
# ---------------------------------------------------------------------------


class TestPivotWithDedup:
    """Tests for _pivot_with_dedup."""

    def _make_df(
        self,
        dates: list[str],
        tickers: list[str],
        values: list[float],
        ccy_ranks: list[int],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": tickers,
                "close": values,
                "_ccy_rank": ccy_ranks,
            }
        )

    def test_no_duplicates_pivots_correctly(self) -> None:
        """With no duplicates, the pivot produces correct shape and values."""
        df = self._make_df(
            dates=["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            tickers=["AAPL", "MSFT", "AAPL", "MSFT"],
            values=[180.0, 370.0, 182.0, 372.0],
            ccy_ranks=[0, 0, 0, 0],
        )
        result = _pivot_with_dedup(df, "date", "ticker", "close")
        assert result.shape == (2, 2)
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert result.loc["2024-01-01", "AAPL"] == 180.0

    def test_duplicates_keeps_lower_ccy_rank_row(self) -> None:
        """When two rows share (date, ticker), the row with lower _ccy_rank wins.

        Uses 21 rows total (1 duplicate = ~4.8% drop rate, below the 5% threshold)
        so the dedup logic is exercised without triggering the ValueError guard.
        """
        # Build 20 unique (date, ticker) rows + 1 duplicate for VOD on 2024-01-01
        # Total rows = 21; dropping 1 = 4.76% < 5% threshold.
        dates = [f"2024-01-{i:02d}" for i in range(2, 22)]  # 20 unique dates
        tickers = ["AAPL"] * 20
        values = [float(100 + i) for i in range(20)]
        ranks = [0] * 20

        # Append the duplicate pair for VOD
        dates += ["2024-01-01", "2024-01-01"]
        tickers += ["VOD", "VOD"]
        values += [100.0, 9900.0]  # 100 = USD price (rank 0 wins), 9900 = GBX price
        ranks += [0, 3]

        df = self._make_df(dates=dates, tickers=tickers, values=values, ccy_ranks=ranks)
        result = _pivot_with_dedup(df, "date", "ticker", "close", "test_dedup")
        assert result.loc["2024-01-01", "VOD"] == 100.0

    def test_raises_value_error_when_drop_fraction_exceeds_threshold(self) -> None:
        """ValueError is raised when dropped fraction > _DEDUP_DROP_THRESHOLD_PCT."""
        # Create 100 rows for same (date, ticker) pair — 99% will be dropped
        n_rows = 100
        df = self._make_df(
            dates=["2024-01-01"] * n_rows,
            tickers=["AAPL"] * n_rows,
            values=[float(i) for i in range(n_rows)],
            ccy_ranks=list(range(n_rows)),
        )
        with pytest.raises(ValueError, match="dedup dropped"):
            _pivot_with_dedup(df, "date", "ticker", "close", "test_excess_drop")

    def test_threshold_constant_is_five_percent(self) -> None:
        """_DEDUP_DROP_THRESHOLD_PCT equals 0.05."""
        assert _DEDUP_DROP_THRESHOLD_PCT == 0.05


# ---------------------------------------------------------------------------
# _build_ticker_map
# ---------------------------------------------------------------------------


class TestBuildTickerMap:
    """Tests for _build_ticker_map using a mocked SQLAlchemy Session."""

    def _make_row(self, id_val: str, ticker: str) -> MagicMock:
        row = MagicMock()
        row.__getitem__ = lambda _, idx: id_val if idx == 0 else ticker
        return row

    def test_returns_correct_mapping(self) -> None:
        """_build_ticker_map returns {instrument_id_hex: yfinance_ticker}."""
        session = MagicMock()
        # Simulate two DB rows: (uuid_bytes_like, ticker_str)
        row_a = ("uuid-aaaa", "AAPL")
        row_b = ("uuid-bbbb", "MSFT")
        session.execute.return_value.all.return_value = [row_a, row_b]

        result = _build_ticker_map(session)

        assert result == {"uuid-aaaa": "AAPL", "uuid-bbbb": "MSFT"}

    def test_empty_db_returns_empty_dict(self) -> None:
        """When no instruments exist, an empty dict is returned."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []

        result = _build_ticker_map(session)

        assert result == {}

    def test_include_delisted_false_adds_filter(self) -> None:
        """When include_delisted=False, the query is executed (filter applied)."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []

        _build_ticker_map(session, include_delisted=False)

        # Verify session.execute was called exactly once with a statement
        session.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants in _helpers."""

    def test_trading_days_is_252(self) -> None:
        """_TRADING_DAYS equals 252 (equity convention)."""
        assert _TRADING_DAYS == 252
