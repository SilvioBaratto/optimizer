"""Tests for FX rate utilities."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from optimizer.fx import (
    align_fx_rates,
    build_fx_pair_ticker,
    compute_cross_rate,
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


class TestAssembleFxRates:
    """Tests for assemble_fx_rates() in cli/data_assembly.py."""

    def test_eur_base_fetches_gbp_and_usd(self) -> None:
        """Mock yfinance and verify cross-rate computation for EUR base."""
        from cli.data_assembly import assemble_fx_rates

        price_index = pd.bdate_range("2024-01-02", periods=10)
        cmap = {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}

        # Mock yfinance download: returns GBPUSD=X and EURUSD=X
        mock_data = pd.DataFrame(
            {
                ("Close", "GBPUSD=X"): np.linspace(1.27, 1.29, 10),
                ("Close", "EURUSD=X"): np.linspace(1.09, 1.11, 10),
            },
            index=price_index,
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = mock_data
            result = assemble_fx_rates(cmap, "EUR", price_index)

        # Should have GBP and USD columns
        assert "GBP" in result.columns
        assert "USD" in result.columns
        # EUR should NOT be in the result (it's the base currency)
        assert "EUR" not in result.columns

        # GBP rate = GBPUSD / EURUSD (cross rate)
        expected_gbp = np.linspace(1.27, 1.29, 10) / np.linspace(1.09, 1.11, 10)
        np.testing.assert_allclose(
            result["GBP"].values, expected_gbp, rtol=1e-10
        )

        # USD rate = 1 / EURUSD (reciprocal)
        expected_usd = 1.0 / np.linspace(1.09, 1.11, 10)
        np.testing.assert_allclose(
            result["USD"].values, expected_usd, rtol=1e-10
        )

    def test_all_base_currency_returns_empty(self) -> None:
        """When all tickers are in the base currency, return empty DataFrame."""
        from cli.data_assembly import assemble_fx_rates

        price_index = pd.bdate_range("2024-01-02", periods=5)
        cmap = {"ORA.PA": "EUR", "BNP.PA": "EUR"}

        result = assemble_fx_rates(cmap, "EUR", price_index)

        assert result.empty or len(result.columns) == 0

    def test_usd_base_direct_rate(self) -> None:
        """USD base with GBP foreign — direct rate, no reciprocal needed."""
        from cli.data_assembly import assemble_fx_rates

        price_index = pd.bdate_range("2024-01-02", periods=5)
        cmap = {"LLOY.L": "GBP", "SPY": "USD"}

        mock_data = pd.DataFrame(
            {"Close": np.linspace(1.27, 1.29, 5)},
            index=price_index,
        )

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = mock_data
            result = assemble_fx_rates(cmap, "USD", price_index)

        assert "GBP" in result.columns
        np.testing.assert_allclose(
            result["GBP"].values, np.linspace(1.27, 1.29, 5), rtol=1e-10
        )

    def test_download_failure_returns_empty(self) -> None:
        """When yfinance download fails, return empty DataFrame gracefully."""
        from cli.data_assembly import assemble_fx_rates

        price_index = pd.bdate_range("2024-01-02", periods=5)
        cmap = {"LLOY.L": "GBP"}

        with patch("yfinance.download") as mock_download:
            mock_download.side_effect = Exception("Network error")
            result = assemble_fx_rates(cmap, "EUR", price_index)

        assert result.empty or len(result.columns) == 0
