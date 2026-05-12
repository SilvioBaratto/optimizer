"""Parity tests for research.data._currency — bit-exact fixture verification.

These tests verify that the module relocated to ``research.data._currency``
produces identical output to the original ``research._currency`` module.
All numeric assertions use ``numpy.testing.assert_array_equal`` for
bit-exact comparisons where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.data._currency import (
    CURRENCY_DEDUP_PRIORITY,
    MINOR_CURRENCY_DIVISORS,
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
    normalize_to_major_currency,
)

# ---------------------------------------------------------------------------
# MINOR_CURRENCY_DIVISORS — structure tests
# ---------------------------------------------------------------------------


class TestMinorCurrencyDivisors:
    def test_when_checking_keys_exactly_four_keys_are_present(self) -> None:
        assert set(MINOR_CURRENCY_DIVISORS) == {"GBX", "GBp", "ILA", "ZAC"}

    def test_when_checking_divisors_each_equals_100(self) -> None:
        for key, value in MINOR_CURRENCY_DIVISORS.items():
            np.testing.assert_array_equal(
                np.float64(value),
                np.float64(100.0),
                err_msg=f"Expected 100.0 for {key}, got {value}",
            )


# ---------------------------------------------------------------------------
# CURRENCY_DEDUP_PRIORITY — rank fixtures
# ---------------------------------------------------------------------------


class TestCurrencyDedupPriority:
    def test_when_usd_rank_is_0(self) -> None:
        assert CURRENCY_DEDUP_PRIORITY["USD"] == 0

    def test_when_gbp_rank_is_1(self) -> None:
        assert CURRENCY_DEDUP_PRIORITY["GBP"] == 1

    def test_when_eur_rank_is_2(self) -> None:
        assert CURRENCY_DEDUP_PRIORITY["EUR"] == 2


# ---------------------------------------------------------------------------
# normalize_to_major_currency
# ---------------------------------------------------------------------------


class TestNormalizeToMajorCurrency:
    def test_when_gbx_gbp_is_returned(self) -> None:
        assert normalize_to_major_currency("GBX") == "GBP"

    def test_when_usd_usd_is_returned_unchanged(self) -> None:
        assert normalize_to_major_currency("USD") == "USD"

    def test_when_gbp_lowercase_variant_gbp_is_returned(self) -> None:
        assert normalize_to_major_currency("GBp") == "GBP"

    def test_when_ila_ils_is_returned(self) -> None:
        assert normalize_to_major_currency("ILA") == "ILS"

    def test_when_zac_zar_is_returned(self) -> None:
        assert normalize_to_major_currency("ZAC") == "ZAR"


# ---------------------------------------------------------------------------
# currency_dedup_rank
# ---------------------------------------------------------------------------


class TestCurrencyDedupRank:
    def test_when_usd_0_is_returned(self) -> None:
        assert currency_dedup_rank("USD") == 0

    def test_when_none_99_is_returned(self) -> None:
        assert currency_dedup_rank(None) == 99

    def test_when_unknown_currency_99_is_returned(self) -> None:
        assert currency_dedup_rank("SEK") == 99


# ---------------------------------------------------------------------------
# normalize_prices — bit-exact fixtures + immutability
# ---------------------------------------------------------------------------


class TestNormalizePrices:
    def test_when_gbx_ticker_present_prices_divided_by_100_bit_exact(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {"BARC.L": [15000.0, 15200.0, 14800.0], "AAPL": [180.0, 182.0, 179.0]},
            index=dates,
        )
        result = normalize_prices(prices, {"BARC.L": "GBX", "AAPL": "USD"})

        np.testing.assert_array_equal(
            result["BARC.L"].values,
            np.array([15000.0, 15200.0, 14800.0]) / 100.0,
        )
        np.testing.assert_array_equal(result["AAPL"].values, prices["AAPL"].values)

    def test_when_no_minor_currencies_original_object_is_returned(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2)
        prices = pd.DataFrame({"AAPL": [180.0, 182.0]}, index=dates)
        result = normalize_prices(prices, {"AAPL": "USD"})
        pd.testing.assert_frame_equal(result, prices)

    def test_when_gbx_present_original_dataframe_is_unchanged(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        prices = pd.DataFrame({"BARC.L": [15000.0, 15200.0, 14800.0]}, index=dates)
        original_values = prices["BARC.L"].values.copy()
        normalize_prices(prices, {"BARC.L": "GBX"})
        np.testing.assert_array_equal(prices["BARC.L"].values, original_values)


# ---------------------------------------------------------------------------
# normalize_fundamentals — bit-exact fixtures + immutability
# ---------------------------------------------------------------------------


class TestNormalizeFundamentals:
    def test_when_gbx_ticker_market_cap_divided_by_100_bit_exact(self) -> None:
        df = pd.DataFrame(
            {"market_cap": [2_000_000_000.0]},
            index=["BARC.L"],
        )
        result_df, result_map = normalize_fundamentals(df, {"BARC.L": "GBX"})

        np.testing.assert_array_equal(
            result_df.loc["BARC.L", "market_cap"],
            2_000_000_000.0 / 100.0,
        )
        assert result_map["BARC.L"] == "GBP"

    def test_when_usd_ticker_values_unchanged(self) -> None:
        df = pd.DataFrame({"market_cap": [3_000_000_000.0]}, index=["AAPL"])
        result_df, result_map = normalize_fundamentals(df, {"AAPL": "USD"})

        np.testing.assert_array_equal(
            result_df.loc["AAPL", "market_cap"], 3_000_000_000.0
        )
        assert result_map["AAPL"] == "USD"

    def test_when_gbx_ticker_present_original_dataframe_is_unchanged(self) -> None:
        df = pd.DataFrame({"market_cap": [2_000_000_000.0]}, index=["BARC.L"])
        original_value = df.loc["BARC.L", "market_cap"]
        normalize_fundamentals(df, {"BARC.L": "GBX"})
        np.testing.assert_array_equal(df.loc["BARC.L", "market_cap"], original_value)


# ---------------------------------------------------------------------------
# build_currency_map — smoke test (no DB dependency)
# ---------------------------------------------------------------------------


class TestBuildCurrencyMap:
    def test_when_empty_profiles_empty_map_is_returned(self) -> None:
        assert build_currency_map([]) == {}
