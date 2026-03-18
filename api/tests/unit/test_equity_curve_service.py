"""Unit tests for compute_equity_curve — pure computation, no DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.dashboard_service import compute_equity_curve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices_df() -> pd.DataFrame:
    """Price DataFrame with 3 portfolio tickers + benchmark, 756 trading days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-03-01", periods=756)
    data = {}
    for ticker in ["AAPL", "MSFT", "GOOG", "SPY"]:
        cum = (1 + rng.normal(0.0003, 0.012, 756)).cumprod() * 100
        data[ticker] = cum
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def weights() -> dict[str, float]:
    return {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeEquityCurve:
    def test_success(self, prices_df: pd.DataFrame, weights: dict[str, float]):
        result = compute_equity_curve(weights, prices_df, "SPY")

        assert "points" in result
        assert "portfolio_total_return" in result
        assert "benchmark_total_return" in result
        assert len(result["points"]) > 0

    def test_points_rebased_to_100(
        self, prices_df: pd.DataFrame, weights: dict[str, float]
    ):
        result = compute_equity_curve(weights, prices_df, "SPY")
        first = result["points"][0]
        assert first["portfolio"] == pytest.approx(100.0)
        assert first["benchmark"] == pytest.approx(100.0)

    def test_total_returns_are_floats(
        self, prices_df: pd.DataFrame, weights: dict[str, float]
    ):
        result = compute_equity_curve(weights, prices_df, "SPY")
        assert isinstance(result["portfolio_total_return"], float)
        assert isinstance(result["benchmark_total_return"], float)

    def test_points_have_date_objects(
        self, prices_df: pd.DataFrame, weights: dict[str, float]
    ):
        import datetime

        result = compute_equity_curve(weights, prices_df, "SPY")
        for pt in result["points"]:
            assert isinstance(pt["date"], datetime.date)

    def test_missing_tickers_renormalize(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.5, "MISSING": 0.3, "GOOG": 0.2}
        result = compute_equity_curve(weights, prices_df, "SPY")
        assert len(result["points"]) > 0

    def test_all_tickers_missing_raises(self, prices_df: pd.DataFrame):
        weights = {"FAKE1": 0.5, "FAKE2": 0.5}
        with pytest.raises(ValueError, match="No price data"):
            compute_equity_curve(weights, prices_df, "SPY")

    def test_insufficient_overlap_raises(self):
        """Fewer than 2 common dates after alignment."""
        dates = pd.bdate_range("2024-01-02", periods=2)
        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "SPY": [200.0, 202.0]},
            index=dates,
        )
        weights = {"AAPL": 1.0}
        with pytest.raises(ValueError, match="Insufficient overlapping"):
            compute_equity_curve(weights, prices, "SPY")

    def test_last_point_matches_total_return(
        self, prices_df: pd.DataFrame, weights: dict[str, float]
    ):
        result = compute_equity_curve(weights, prices_df, "SPY")
        last = result["points"][-1]
        expected_port_ret = last["portfolio"] / 100 - 1
        assert result["portfolio_total_return"] == pytest.approx(
            expected_port_ret, abs=1e-3
        )
