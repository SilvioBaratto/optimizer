"""Unit tests for compute_asset_class_returns — pure computation, no DB."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services.dashboard_service import (
    _sector_return_for_period,
    compute_asset_class_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    tickers: list[str],
    n_days: int = 60,
    start: str = "2026-01-02",
) -> pd.DataFrame:
    """Deterministic price DataFrame starting from a given date."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, periods=n_days)
    data = {}
    for t in tickers:
        data[t] = (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# _sector_return_for_period
# ---------------------------------------------------------------------------


class TestSectorReturnForPeriod:
    def test_normal_case(self):
        prices = _make_prices(["AAPL", "MSFT"], n_days=10)
        result = _sector_return_for_period(
            ["AAPL", "MSFT"], [0.6, 0.4], prices, -2
        )
        assert isinstance(result, float)

    def test_missing_ticker_skipped(self):
        prices = _make_prices(["AAPL"], n_days=10)
        result = _sector_return_for_period(
            ["AAPL", "MISSING"], [0.5, 0.5], prices, -2
        )
        # Only AAPL contributes; should not raise
        assert isinstance(result, float)

    def test_all_missing_returns_zero(self):
        prices = _make_prices(["AAPL"], n_days=10)
        result = _sector_return_for_period(
            ["MISSING1", "MISSING2"], [0.5, 0.5], prices, -2
        )
        assert result == 0.0

    def test_zero_weights_returns_zero(self):
        prices = _make_prices(["AAPL", "MSFT"], n_days=10)
        result = _sector_return_for_period(
            ["AAPL", "MSFT"], [0.0, 0.0], prices, -2
        )
        assert result == 0.0


# ---------------------------------------------------------------------------
# compute_asset_class_returns
# ---------------------------------------------------------------------------


class TestComputeAssetClassReturns:
    def test_basic_two_sectors(self):
        prices = _make_prices(["AAPL", "MSFT", "XOM", "CVX"])
        weights = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.3, "CVX": 0.2}
        sector_mapping = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "XOM": "Energy",
            "CVX": "Energy",
        }
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        assert len(result["returns"]) == 2
        assert "as_of" in result

    def test_sectors_ordered_by_weight_descending(self):
        prices = _make_prices(["AAPL", "XOM"])
        weights = {"AAPL": 0.1, "XOM": 0.9}
        sector_mapping = {"AAPL": "Technology", "XOM": "Energy"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        # Energy (0.9) should come first
        assert result["returns"][0]["name"] == "Energy"

    def test_each_row_has_all_period_keys(self):
        prices = _make_prices(["AAPL", "MSFT"])
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        sector_mapping = {"AAPL": "Tech", "MSFT": "Tech"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        row = result["returns"][0]
        assert "1D" in row
        assert "1W" in row
        assert "1M" in row
        assert "YTD" in row
        for key in ("1D", "1W", "1M", "YTD"):
            assert isinstance(row[key], float)

    def test_as_of_equals_last_price_date(self):
        prices = _make_prices(["AAPL"])
        weights = {"AAPL": 1.0}
        sector_mapping = {"AAPL": "Tech"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        expected_date = prices.index[-1].date()
        assert result["as_of"] == expected_date

    def test_unknown_sector_for_unmapped_ticker(self):
        prices = _make_prices(["AAPL", "NEWCO"])
        weights = {"AAPL": 0.6, "NEWCO": 0.4}
        sector_mapping = {"AAPL": "Tech"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        names = {r["name"] for r in result["returns"]}
        assert "Unknown" in names

    def test_insufficient_rows_raises(self):
        dates = pd.bdate_range("2026-03-17", periods=1)
        prices = pd.DataFrame({"AAPL": [100.0]}, index=dates)
        with pytest.raises(ValueError, match="Insufficient price data"):
            compute_asset_class_returns(
                {"AAPL": 1.0},
                {"AAPL": "Tech"},
                prices,
                date(2026, 3, 18),
            )

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError, match="Insufficient price data"):
            compute_asset_class_returns(
                {"AAPL": 1.0},
                {"AAPL": "Tech"},
                pd.DataFrame(),
                date(2026, 3, 18),
            )

    def test_returns_list_length_equals_sector_count(self):
        prices = _make_prices(["AAPL", "MSFT", "XOM"])
        weights = {"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3}
        sector_mapping = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        assert len(result["returns"]) == 2

    def test_values_are_rounded_floats(self):
        prices = _make_prices(["AAPL"])
        weights = {"AAPL": 1.0}
        sector_mapping = {"AAPL": "Tech"}
        result = compute_asset_class_returns(
            weights, sector_mapping, prices, date(2026, 3, 18)
        )
        row = result["returns"][0]
        for key in ("1D", "1W", "1M", "YTD"):
            # Should have at most 6 decimal places
            val = row[key]
            assert val == round(val, 6)
