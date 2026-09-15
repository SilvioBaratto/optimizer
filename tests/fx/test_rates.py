"""Tests for FX rate utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.fx import (
    align_fx_rates,
    build_fx_pair_ticker,
    compute_cross_rate,
    invert_fx_rates,
    required_fx_currencies,
)


class TestBuildFxPairTicker:
    """Tests for build_fx_pair_ticker()."""

    def test_same_currency_returns_none(self) -> None:
        assert build_fx_pair_ticker("EUR", "EUR") is None
        assert build_fx_pair_ticker("usd", "USD") is None

    def test_from_usd(self) -> None:
        result = build_fx_pair_ticker("USD", "EUR")
        assert result == "EURUSD=X"

    def test_to_usd(self) -> None:
        result = build_fx_pair_ticker("GBP", "USD")
        assert result == "GBPUSD=X"

    def test_cross_via_usd(self) -> None:
        result = build_fx_pair_ticker("GBP", "EUR", cross_via_usd=True)
        assert result == ("GBPUSD=X", "EURUSD=X")

    def test_direct_cross(self) -> None:
        result = build_fx_pair_ticker("GBP", "EUR", cross_via_usd=False)
        assert result == "GBPEUR=X"

    def test_case_insensitive(self) -> None:
        result = build_fx_pair_ticker("gbp", "usd")
        assert result == "GBPUSD=X"


class TestComputeCrossRate:
    """Tests for compute_cross_rate()."""

    def test_basic_cross(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        gbp_usd = pd.Series([1.27, 1.28, 1.26, 1.29, 1.27], index=dates)
        eur_usd = pd.Series([1.09, 1.10, 1.08, 1.11, 1.09], index=dates)

        cross = compute_cross_rate(gbp_usd, eur_usd)

        # GBP/EUR = GBP/USD / EUR/USD
        expected = gbp_usd / eur_usd
        pd.testing.assert_series_equal(cross, expected)

    def test_misaligned_dates(self) -> None:
        dates_a = pd.date_range("2024-01-01", periods=5, freq="B")
        dates_b = pd.date_range("2024-01-02", periods=5, freq="B")
        a = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=dates_a)
        b = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4], index=dates_b)

        cross = compute_cross_rate(a, b)
        # Inner join: only overlapping dates
        assert len(cross) == 4


class TestAlignFxRates:
    """Tests for align_fx_rates()."""

    def test_basic_alignment(self) -> None:
        fx_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        fx = pd.DataFrame({"GBP": np.linspace(1.15, 1.20, 10)}, index=fx_dates)

        result = align_fx_rates(fx, price_dates)

        assert len(result) == len(price_dates)
        assert list(result.columns) == ["GBP"]

    def test_forward_fill_gaps(self) -> None:
        fx_dates = pd.to_datetime(["2024-01-02", "2024-01-05"])
        price_dates = pd.bdate_range("2024-01-02", periods=4)
        fx = pd.DataFrame({"GBP": [1.15, 1.18]}, index=fx_dates)

        result = align_fx_rates(fx, price_dates, fill_limit=3)

        # Jan 3, Jan 4 should be forward-filled from Jan 2
        assert not result["GBP"].isna().any()

    def test_fill_limit_respected(self) -> None:
        fx_dates = pd.to_datetime(["2024-01-02"])
        price_dates = pd.bdate_range("2024-01-02", periods=10)
        fx = pd.DataFrame({"GBP": [1.15]}, index=fx_dates)

        result = align_fx_rates(fx, price_dates, fill_limit=3)

        # First 4 should be filled (original + 3 forward), rest NaN
        assert result["GBP"].notna().sum() == 4

    def test_empty_raises(self) -> None:
        from optimizer.exceptions import DataError

        price_dates = pd.bdate_range("2024-01-01", periods=5)
        with pytest.raises(DataError, match="empty"):
            align_fx_rates(pd.DataFrame(), price_dates)


class TestInvertFxRates:
    """Tests for invert_fx_rates()."""

    def test_series_reciprocal(self) -> None:
        s = pd.Series([1.25, 2.0, 0.5])
        result = invert_fx_rates(s)
        pd.testing.assert_series_equal(result, pd.Series([0.8, 0.5, 2.0]))

    def test_dataframe_reciprocal(self) -> None:
        df = pd.DataFrame({"GBP": [1.25, 2.0], "USD": [4.0, 0.5]})
        result = invert_fx_rates(df)
        expected = pd.DataFrame({"GBP": [0.8, 0.5], "USD": [0.25, 2.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_zero_maps_to_nan_not_inf(self) -> None:
        s = pd.Series([0.0, 2.0])
        result = invert_fx_rates(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 0.5

    def test_double_inversion_roundtrip(self) -> None:
        s = pd.Series([1.15, 1.17, 0.92])
        pd.testing.assert_series_equal(invert_fx_rates(invert_fx_rates(s)), s)


class TestRequiredFxCurrencies:
    """Tests for required_fx_currencies()."""

    def test_mixed_currencies(self) -> None:
        cmap = {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == {"GBP", "USD"}

    def test_all_base(self) -> None:
        cmap = {"ORA.PA": "EUR", "BNP.PA": "EUR"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == set()

    def test_case_insensitive(self) -> None:
        cmap = {"LLOY.L": "gbp"}
        result = required_fx_currencies(cmap, "eur")
        assert result == {"GBP"}

    def test_minor_units_resolve_to_major(self) -> None:
        # GBp -> GBP, ZAc -> ZAR; the caller must fetch the major-unit pairs.
        cmap = {"LLOY.L": "GBp", "NPN.JO": "ZAc", "ORA.PA": "EUR"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == {"GBP", "ZAR"}

    def test_pence_ticker_with_gbp_base_needs_no_fx(self) -> None:
        # GBp resolves to GBP == base, so no FX pair is required.
        cmap = {"LLOY.L": "GBp"}
        result = required_fx_currencies(cmap, "GBP")
        assert result == set()


class TestBuildFxPairTickerMinorUnits:
    """build_fx_pair_ticker resolves sub-unit codes to their major pair."""

    def test_pence_resolves_to_gbp_pair(self) -> None:
        assert build_fx_pair_ticker("GBp", "USD") == "GBPUSD=X"

    def test_cents_resolves_to_zar_pair(self) -> None:
        assert build_fx_pair_ticker("ZAc", "USD") == "ZARUSD=X"

    def test_pence_to_gbp_base_returns_none(self) -> None:
        # GBp -> GBP; same major currency as target -> no pair.
        assert build_fx_pair_ticker("GBp", "GBP") is None
