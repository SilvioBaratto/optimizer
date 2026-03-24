"""Tests for cli/_currency.py — minor-unit currency normalization."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from cli._currency import (
    MINOR_CURRENCY_DIVISORS,
    build_currency_map,
    normalize_fundamentals,
    normalize_prices,
    normalize_to_major_currency,
)

# ---------------------------------------------------------------------------
# MINOR_CURRENCY_DIVISORS
# ---------------------------------------------------------------------------


class TestMinorCurrencyDivisors:
    def test_gbx_present(self) -> None:
        assert "GBX" in MINOR_CURRENCY_DIVISORS
        assert MINOR_CURRENCY_DIVISORS["GBX"] == 100.0

    def test_gbp_lowercase_variant(self) -> None:
        """yfinance sometimes uses 'GBp' for pence."""
        assert "GBp" in MINOR_CURRENCY_DIVISORS
        assert MINOR_CURRENCY_DIVISORS["GBp"] == 100.0

    def test_ila_present(self) -> None:
        assert "ILA" in MINOR_CURRENCY_DIVISORS
        assert MINOR_CURRENCY_DIVISORS["ILA"] == 100.0

    def test_zac_present(self) -> None:
        assert "ZAC" in MINOR_CURRENCY_DIVISORS
        assert MINOR_CURRENCY_DIVISORS["ZAC"] == 100.0

    def test_major_currencies_absent(self) -> None:
        for ccy in ("USD", "GBP", "EUR", "JPY", "CHF"):
            assert ccy not in MINOR_CURRENCY_DIVISORS


# ---------------------------------------------------------------------------
# normalize_to_major_currency
# ---------------------------------------------------------------------------


class TestNormalizeToMajorCurrency:
    @pytest.mark.parametrize(
        ("minor", "expected"),
        [
            ("GBX", "GBP"),
            ("GBp", "GBP"),
            ("ILA", "ILS"),
            ("ZAC", "ZAR"),
        ],
    )
    def test_minor_to_major(self, minor: str, expected: str) -> None:
        assert normalize_to_major_currency(minor) == expected

    @pytest.mark.parametrize("major", ["USD", "EUR", "GBP", "JPY"])
    def test_major_unchanged(self, major: str) -> None:
        assert normalize_to_major_currency(major) == major


# ---------------------------------------------------------------------------
# build_currency_map
# ---------------------------------------------------------------------------


def _mock_profile(
    yf_ticker: str,
    currency_code: str | None,
    profile_currency: str | None = None,
) -> MagicMock:
    """Create a mock TickerProfile with an associated Instrument."""
    instrument = MagicMock()
    instrument.yfinance_ticker = yf_ticker
    instrument.currency_code = currency_code

    profile = MagicMock()
    profile.instrument = instrument
    profile.currency = profile_currency
    return profile


class TestBuildCurrencyMap:
    def test_basic(self) -> None:
        profiles = [
            _mock_profile("BARC.L", "GBX"),
            _mock_profile("AAPL", "USD"),
        ]
        result = build_currency_map(profiles)
        assert result == {"BARC.L": "GBX", "AAPL": "USD"}

    def test_fallback_to_profile_currency(self) -> None:
        """When Instrument.currency_code is None, use TickerProfile.currency."""
        profiles = [
            _mock_profile("BARC.L", None, "GBp"),
        ]
        result = build_currency_map(profiles)
        assert result == {"BARC.L": "GBp"}

    def test_skip_no_instrument(self) -> None:
        profile = MagicMock()
        profile.instrument = None
        profile.currency = "USD"
        result = build_currency_map([profile])
        assert result == {}

    def test_skip_no_ticker(self) -> None:
        profile = _mock_profile("", "USD")
        profile.instrument.yfinance_ticker = ""
        result = build_currency_map([profile])
        assert result == {}

    def test_skip_no_currency(self) -> None:
        profiles = [_mock_profile("AAPL", None, None)]
        result = build_currency_map(profiles)
        assert result == {}


# ---------------------------------------------------------------------------
# normalize_fundamentals
# ---------------------------------------------------------------------------


class TestNormalizeFundamentals:
    def test_gbx_market_cap_divided(self) -> None:
        """A £20M micro-cap stored as 2B GBX should become 20M GBP."""
        df = pd.DataFrame(
            {"market_cap": [2_000_000_000.0], "enterprise_value": [3_000_000_000.0]},
            index=["BARC.L"],
        )
        currency_map = {"BARC.L": "GBX"}
        result_df, result_map = normalize_fundamentals(df, currency_map)

        assert result_df.loc["BARC.L", "market_cap"] == pytest.approx(20_000_000.0)
        assert result_df.loc["BARC.L", "enterprise_value"] == pytest.approx(
            30_000_000.0
        )
        assert result_map["BARC.L"] == "GBP"

    def test_usd_unchanged(self) -> None:
        df = pd.DataFrame(
            {"market_cap": [150_000_000_000.0]},
            index=["AAPL"],
        )
        currency_map = {"AAPL": "USD"}
        result_df, result_map = normalize_fundamentals(df, currency_map)

        assert result_df.loc["AAPL", "market_cap"] == pytest.approx(150_000_000_000.0)
        assert result_map["AAPL"] == "USD"

    def test_mixed_currencies(self) -> None:
        df = pd.DataFrame(
            {"market_cap": [2_000_000_000.0, 150_000_000_000.0]},
            index=["BARC.L", "AAPL"],
        )
        currency_map = {"BARC.L": "GBX", "AAPL": "USD"}
        result_df, _ = normalize_fundamentals(df, currency_map)

        assert result_df.loc["BARC.L", "market_cap"] == pytest.approx(20_000_000.0)
        assert result_df.loc["AAPL", "market_cap"] == pytest.approx(150_000_000_000.0)

    def test_ila_normalization(self) -> None:
        """Israeli Agorot should be divided by 100."""
        df = pd.DataFrame(
            {"current_price": [5000.0]},
            index=["TEVA.TA"],
        )
        currency_map = {"TEVA.TA": "ILA"}
        result_df, result_map = normalize_fundamentals(df, currency_map)

        assert result_df.loc["TEVA.TA", "current_price"] == pytest.approx(50.0)
        assert result_map["TEVA.TA"] == "ILS"

    def test_gbp_variant(self) -> None:
        """yfinance 'GBp' variant should also be divided by 100."""
        df = pd.DataFrame({"market_cap": [1_000_000_000.0]}, index=["VOD.L"])
        currency_map = {"VOD.L": "GBp"}
        result_df, result_map = normalize_fundamentals(df, currency_map)

        assert result_df.loc["VOD.L", "market_cap"] == pytest.approx(10_000_000.0)
        assert result_map["VOD.L"] == "GBP"

    def test_does_not_modify_input(self) -> None:
        """Ensure the original DataFrame is not mutated."""
        df = pd.DataFrame({"market_cap": [2_000_000_000.0]}, index=["BARC.L"])
        original_val = df.loc["BARC.L", "market_cap"]
        currency_map = {"BARC.L": "GBX"}
        normalize_fundamentals(df, currency_map)
        assert df.loc["BARC.L", "market_cap"] == original_val

    def test_empty_currency_map(self) -> None:
        df = pd.DataFrame({"market_cap": [100.0]}, index=["AAPL"])
        result_df, result_map = normalize_fundamentals(df, {})
        assert result_df.loc["AAPL", "market_cap"] == pytest.approx(100.0)
        assert result_map == {}

    def test_ticker_not_in_df_ignored(self) -> None:
        """Currency map entry for a ticker not in the DataFrame is harmless."""
        df = pd.DataFrame({"market_cap": [100.0]}, index=["AAPL"])
        currency_map = {"BARC.L": "GBX", "AAPL": "USD"}
        result_df, _ = normalize_fundamentals(df, currency_map)
        assert result_df.loc["AAPL", "market_cap"] == pytest.approx(100.0)

    def test_column_not_in_df_ignored(self) -> None:
        """Columns listed but absent from DataFrame don't cause errors."""
        df = pd.DataFrame({"market_cap": [2_000_000_000.0]}, index=["BARC.L"])
        currency_map = {"BARC.L": "GBX"}
        # enterprise_value is in default columns but not in df — should not error
        result_df, _ = normalize_fundamentals(df, currency_map)
        assert result_df.loc["BARC.L", "market_cap"] == pytest.approx(20_000_000.0)

    def test_nan_values_preserved(self) -> None:
        """NaN values should remain NaN after normalization."""
        df = pd.DataFrame(
            {"market_cap": [np.nan], "current_price": [500.0]},
            index=["BARC.L"],
        )
        currency_map = {"BARC.L": "GBX"}
        result_df, _ = normalize_fundamentals(df, currency_map)
        assert pd.isna(result_df.loc["BARC.L", "market_cap"])
        assert result_df.loc["BARC.L", "current_price"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# normalize_prices
# ---------------------------------------------------------------------------


class TestNormalizePrices:
    def test_gbx_prices_divided(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame(
            {"BARC.L": [150.0, 152.0, 148.0], "AAPL": [180.0, 182.0, 179.0]},
            index=dates,
        )
        currency_map = {"BARC.L": "GBX", "AAPL": "USD"}
        result = normalize_prices(df, currency_map)

        np.testing.assert_array_almost_equal(
            result["BARC.L"].values, [1.50, 1.52, 1.48]
        )
        np.testing.assert_array_almost_equal(
            result["AAPL"].values, [180.0, 182.0, 179.0]
        )

    def test_no_minor_currencies(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"AAPL": [180.0, 182.0, 179.0]}, index=dates)
        currency_map = {"AAPL": "USD"}
        result = normalize_prices(df, currency_map)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_modify_input(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"BARC.L": [150.0, 152.0, 148.0]}, index=dates)
        original = df.copy()
        currency_map = {"BARC.L": "GBX"}
        normalize_prices(df, currency_map)
        pd.testing.assert_frame_equal(df, original)

    def test_empty_currency_map(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"AAPL": [180.0, 182.0, 179.0]}, index=dates)
        result = normalize_prices(df, {})
        pd.testing.assert_frame_equal(result, df)

    def test_ticker_not_in_prices_ignored(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"AAPL": [180.0, 182.0, 179.0]}, index=dates)
        currency_map = {"BARC.L": "GBX"}  # not in df columns
        result = normalize_prices(df, currency_map)
        pd.testing.assert_frame_equal(result, df)
