"""Tests for recovering as-traded closes from dividend-adjusted closes."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.preprocessing import unadjust_close


def _yahoo_adjust(raw: pd.Series, dividends: pd.Series) -> pd.Series:
    """Dividend-adjusted close built the way Yahoo builds it.

    Each ex-date ``i`` scales every earlier close by ``1 - D_i / C_{i-1}``,
    where ``C_{i-1}`` is the previous available as-traded close.
    """
    prev = raw.ffill()
    factor = pd.Series(1.0, index=raw.index)
    running = 1.0
    for i in range(len(raw) - 1, 0, -1):
        running *= 1.0 - dividends.iloc[i] / prev.iloc[i - 1]
        factor.iloc[i - 1] = running
    return raw * factor


@pytest.fixture()
def raw_close() -> pd.DataFrame:
    """Two tickers of as-traded closes over 40 business days."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-02", periods=40)
    return pd.DataFrame(
        {
            "PAYER": 50 * np.exp(rng.normal(0, 0.01, len(dates)).cumsum()),
            "NOPAY": 20 * np.exp(rng.normal(0, 0.01, len(dates)).cumsum()),
        },
        index=dates,
    )


@pytest.fixture()
def dividends(raw_close: pd.DataFrame) -> pd.DataFrame:
    """Cash dividends: PAYER goes ex-dividend on days 10 and 25."""
    div = pd.DataFrame(0.0, index=raw_close.index, columns=raw_close.columns)
    div.loc[raw_close.index[10], "PAYER"] = 0.5
    div.loc[raw_close.index[25], "PAYER"] = 0.5
    return div


def _adjusted(raw: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: _yahoo_adjust(raw[c], div[c]) for c in raw.columns})


class TestUnadjustClose:
    """``unadjust_close`` inverts the dividend adjustment baked into closes."""

    def test_recovers_as_traded_close_from_dividend_adjusted_close(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """Adjusted closes map back to the prices that actually traded."""
        adjusted = _adjusted(raw_close, dividends)
        assert (adjusted["PAYER"].iloc[0] / raw_close["PAYER"].iloc[0]) < 0.98

        result = unadjust_close(adjusted, dividends)

        pd.testing.assert_frame_equal(result, raw_close, rtol=1e-12)

    def test_without_dividends_the_close_is_unchanged(
        self, raw_close: pd.DataFrame
    ) -> None:
        """With no dividends there is nothing to undo."""
        zero = pd.DataFrame(0.0, index=raw_close.index, columns=raw_close.columns)
        pd.testing.assert_frame_equal(unadjust_close(raw_close, zero), raw_close)

    def test_missing_close_stays_missing_and_does_not_break_neighbours(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """A gap stays NaN and the adjustment still lines up around it."""
        raw_gap = raw_close.copy()
        # Day 9 is the previous close of the day-10 ex-date: the adjustment
        # must fall back to the last available close, as Yahoo does.
        raw_gap.iloc[9, 0] = np.nan
        raw_gap.iloc[17, 0] = np.nan
        adjusted = _adjusted(raw_gap, dividends)

        result = unadjust_close(adjusted, dividends)

        pd.testing.assert_frame_equal(result, raw_gap, rtol=1e-12)

    def test_dividend_columns_and_dates_that_are_absent_count_as_zero(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """A ticker or date absent from the dividend frame had no dividend."""
        adjusted = _adjusted(raw_close, dividends)
        partial = dividends[["PAYER"]].iloc[5:]

        result = unadjust_close(adjusted, partial)

        pd.testing.assert_frame_equal(result, raw_close, rtol=1e-12)

    def test_null_dividends_count_as_zero(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """NULL dividend cells (non-ex-dates in the DB) are not dividends."""
        adjusted = _adjusted(raw_close, dividends)
        nullable = dividends.mask(dividends == 0.0)

        result = unadjust_close(adjusted, nullable)

        pd.testing.assert_frame_equal(result, raw_close, rtol=1e-12)

    def test_decimal_frames_from_the_database_are_accepted(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """Numeric(20, 6) columns arrive as Decimal objects and must work."""
        adjusted = _adjusted(raw_close, dividends)
        as_decimal = adjusted.map(lambda v: Decimal(str(v)))

        result = unadjust_close(as_decimal, dividends)

        pd.testing.assert_frame_equal(result, raw_close, rtol=1e-9)

    def test_non_numeric_prices_raise_a_data_error(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """Garbage prices fail loudly instead of producing NaN factors."""
        bad = raw_close.astype(object)
        bad.iloc[0, 0] = "n/a"
        with pytest.raises(DataError):
            unadjust_close(bad, dividends)

    def test_input_frames_are_not_modified(
        self, raw_close: pd.DataFrame, dividends: pd.DataFrame
    ) -> None:
        """Callers keep their stored frames untouched."""
        adjusted = _adjusted(raw_close, dividends)
        before_close, before_div = adjusted.copy(), dividends.copy()

        unadjust_close(adjusted, dividends)

        pd.testing.assert_frame_equal(adjusted, before_close)
        pd.testing.assert_frame_equal(dividends, before_div)
