"""Unit tests for drift analysis — pure computation, no DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.dashboard_service import (
    _actual_weights_from_positions,
    _actual_weights_from_prices,
    compute_drift,
)


# ---------------------------------------------------------------------------
# _actual_weights_from_positions
# ---------------------------------------------------------------------------


class TestActualWeightsFromPositions:
    def test_valid_positions(self):
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 100.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 20, "current_price": 50.0},
        ]
        result = _actual_weights_from_positions(positions)
        assert result is not None
        assert result["AAPL"] == pytest.approx(0.5)
        assert result["MSFT"] == pytest.approx(0.5)

    def test_excludes_none_current_price(self):
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 100.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 20, "current_price": None},
        ]
        result = _actual_weights_from_positions(positions)
        assert result is not None
        assert "MSFT" not in result
        assert result["AAPL"] == pytest.approx(1.0)

    def test_excludes_none_yfinance_ticker(self):
        positions = [
            {"yfinance_ticker": None, "name": "Unknown", "quantity": 10, "current_price": 100.0},
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 100.0},
        ]
        result = _actual_weights_from_positions(positions)
        assert result is not None
        assert result["AAPL"] == pytest.approx(1.0)

    def test_all_excluded_returns_none(self):
        positions = [
            {"yfinance_ticker": None, "name": "X", "quantity": 10, "current_price": 100.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 20, "current_price": None},
        ]
        assert _actual_weights_from_positions(positions) is None

    def test_empty_positions_returns_none(self):
        assert _actual_weights_from_positions([]) is None


# ---------------------------------------------------------------------------
# _actual_weights_from_prices
# ---------------------------------------------------------------------------


class TestActualWeightsFromPrices:
    def test_normal_case(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10), "MSFT": np.linspace(200, 200, 10)},
            index=dates,
        )
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        result = _actual_weights_from_prices(weights, prices)
        # AAPL went up 10%, MSFT stayed flat → AAPL's weight should increase
        assert result["AAPL"] > 0.6
        assert result["MSFT"] < 0.4
        assert sum(result.values()) == pytest.approx(1.0)

    def test_single_row_no_drift(self):
        dates = pd.bdate_range("2024-01-02", periods=1)
        prices = pd.DataFrame({"AAPL": [100.0], "MSFT": [200.0]}, index=dates)
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        result = _actual_weights_from_prices(weights, prices)
        assert result["AAPL"] == pytest.approx(0.6)
        assert result["MSFT"] == pytest.approx(0.4)

    def test_missing_tickers_ignored(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame({"AAPL": np.linspace(100, 110, 5)}, index=dates)
        weights = {"AAPL": 0.6, "MISSING": 0.4}
        result = _actual_weights_from_prices(weights, prices)
        assert "AAPL" in result
        assert "MISSING" not in result

    def test_no_common_tickers_raises(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame({"XYZ": [100.0] * 5}, index=dates)
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        with pytest.raises(ValueError, match="No price data"):
            _actual_weights_from_prices(weights, prices)


# ---------------------------------------------------------------------------
# compute_drift
# ---------------------------------------------------------------------------


class TestComputeDrift:
    def test_primary_path_with_positions(self):
        target = {"AAPL": 0.5, "MSFT": 0.5}
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 120.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 10, "current_price": 80.0},
        ]
        result = compute_drift(target, positions, threshold=0.05)
        assert len(result["entries"]) == 2
        assert result["threshold"] == 0.05

        # AAPL: actual=0.6, target=0.5 → drift=0.1 → breached
        aapl = next(e for e in result["entries"] if e["ticker"] == "AAPL")
        assert aapl["actual"] == pytest.approx(0.6)
        assert aapl["drift"] == pytest.approx(0.1)
        assert aapl["breached"] is True
        assert aapl["name"] == "Apple"

    def test_fallback_to_prices(self):
        target = {"AAPL": 0.6, "MSFT": 0.4}
        positions: list[dict] = []
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices_df = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10), "MSFT": np.linspace(200, 200, 10)},
            index=dates,
        )
        result = compute_drift(target, positions, threshold=0.05, prices_df=prices_df)
        assert len(result["entries"]) == 2
        assert result["total_drift"] > 0

    def test_sorted_by_abs_drift_descending(self):
        target = {"A": 0.5, "B": 0.3, "C": 0.2}
        positions = [
            {"yfinance_ticker": "A", "name": "A Inc", "quantity": 10, "current_price": 50.0},
            {"yfinance_ticker": "B", "name": "B Inc", "quantity": 10, "current_price": 40.0},
            {"yfinance_ticker": "C", "name": "C Inc", "quantity": 10, "current_price": 10.0},
        ]
        result = compute_drift(target, positions, threshold=0.01)
        drifts = [abs(e["drift"]) for e in result["entries"]]
        assert drifts == sorted(drifts, reverse=True)

    def test_breached_count(self):
        target = {"AAPL": 0.5, "MSFT": 0.5}
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 120.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 10, "current_price": 80.0},
        ]
        result = compute_drift(target, positions, threshold=0.05)
        assert result["breached_count"] == 2  # both drift ±0.1 > 0.05

    def test_total_drift(self):
        target = {"AAPL": 0.5, "MSFT": 0.5}
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 120.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 10, "current_price": 80.0},
        ]
        result = compute_drift(target, positions, threshold=0.05)
        assert result["total_drift"] == pytest.approx(0.2)

    def test_zero_threshold_all_breached(self):
        target = {"AAPL": 0.5, "MSFT": 0.5}
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 11, "current_price": 100.0},
            {"yfinance_ticker": "MSFT", "name": "Microsoft", "quantity": 10, "current_price": 100.0},
        ]
        result = compute_drift(target, positions, threshold=0.0)
        assert result["breached_count"] == 2

    def test_ticker_in_target_not_in_actual(self):
        target = {"AAPL": 0.6, "MISSING": 0.4}
        positions = [
            {"yfinance_ticker": "AAPL", "name": "Apple", "quantity": 10, "current_price": 100.0},
        ]
        result = compute_drift(target, positions, threshold=0.05)
        missing = next(e for e in result["entries"] if e["ticker"] == "MISSING")
        assert missing["actual"] == pytest.approx(0.0)
        assert missing["drift"] == pytest.approx(-0.4)

    def test_no_positions_no_prices_raises(self):
        with pytest.raises(ValueError, match="No price data"):
            compute_drift({"AAPL": 1.0}, [], threshold=0.05)

    def test_no_positions_empty_prices_raises(self):
        with pytest.raises(ValueError, match="No price data"):
            compute_drift({"AAPL": 1.0}, [], threshold=0.05, prices_df=pd.DataFrame())
