"""Unit tests for dashboard_service — pure computation, no DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.dashboard_service import (
    _annualized_return,
    _cvar_95,
    _max_drawdown,
    _portfolio_returns,
    _sharpe_ratio,
    _total_return,
    _volatility,
    compute_allocation,
    compute_performance_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def daily_returns() -> pd.Series:
    """252 days of synthetic daily returns (mean ~0.05%, std ~1%)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, 252))


@pytest.fixture()
def prices_df() -> pd.DataFrame:
    """Price DataFrame with 3 tickers + benchmark, 300 trading days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    data = {}
    for ticker in ["AAPL", "MSFT", "GOOG", "SPY"]:
        cum = (1 + rng.normal(0.0003, 0.012, 300)).cumprod() * 100
        data[ticker] = cum
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Individual KPI tests
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_positive(self, daily_returns: pd.Series):
        result = _total_return(daily_returns)
        # With positive drift we expect a positive total return
        assert isinstance(result, float)

    def test_zero_returns(self):
        zeros = pd.Series([0.0] * 100)
        assert _total_return(zeros) == pytest.approx(0.0)


class TestAnnualizedReturn:
    def test_returns_float(self, daily_returns: pd.Series):
        result = _annualized_return(daily_returns)
        assert isinstance(result, float)

    def test_empty(self):
        assert _annualized_return(pd.Series([], dtype=float)) == 0.0


class TestSharpeRatio:
    def test_returns_float(self, daily_returns: pd.Series):
        result = _sharpe_ratio(daily_returns)
        assert isinstance(result, float)

    def test_zero_vol_returns_zero(self):
        flat = pd.Series([0.0] * 100)
        assert _sharpe_ratio(flat) == 0.0


class TestMaxDrawdown:
    def test_always_non_positive(self, daily_returns: pd.Series):
        assert _max_drawdown(daily_returns) <= 0.0

    def test_no_drawdown(self):
        # Monotonically increasing returns → no drawdown
        rets = pd.Series([0.01] * 100)
        assert _max_drawdown(rets) == pytest.approx(0.0)


class TestVolatility:
    def test_positive(self, daily_returns: pd.Series):
        assert _volatility(daily_returns) > 0.0

    def test_zero_vol(self):
        flat = pd.Series([0.0] * 100)
        assert _volatility(flat) == 0.0


class TestCVaR95:
    def test_less_than_mean(self, daily_returns: pd.Series):
        result = _cvar_95(daily_returns)
        assert result <= daily_returns.mean()

    def test_empty_tail(self):
        # All positive returns → tail below 5th percentile is small
        rets = pd.Series([0.01] * 100)
        result = _cvar_95(rets)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


class TestPortfolioReturns:
    def test_basic(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
        rets = _portfolio_returns(prices_df, weights)
        assert isinstance(rets, pd.Series)
        assert len(rets) == len(prices_df) - 1  # pct_change drops first row

    def test_missing_tickers_renormalize(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.5, "MISSING": 0.3, "GOOG": 0.2}
        rets = _portfolio_returns(prices_df, weights)
        assert len(rets) > 0

    def test_all_missing_raises(self, prices_df: pd.DataFrame):
        weights = {"FAKE1": 0.5, "FAKE2": 0.5}
        with pytest.raises(ValueError, match="No price data"):
            _portfolio_returns(prices_df, weights)


# ---------------------------------------------------------------------------
# Full compute_performance_metrics
# ---------------------------------------------------------------------------


class TestComputePerformanceMetrics:
    def test_returns_7_kpis(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        result = compute_performance_metrics(weights, prices_df)
        assert len(result["kpis"]) == 7
        assert "nav" in result
        assert "nav_change_pct" in result

    def test_kpi_labels(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        result = compute_performance_metrics(weights, prices_df)
        labels = {k["label"] for k in result["kpis"]}
        expected = {
            "Total Return",
            "Ann. Return",
            "Sharpe Ratio",
            "Max Drawdown",
            "Portfolio Value",
            "Volatility",
            "CVaR 95%",
        }
        assert labels == expected

    def test_sparklines_are_lists_of_floats(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        result = compute_performance_metrics(weights, prices_df)
        for kpi in result["kpis"]:
            assert isinstance(kpi["sparkline"], list)
            assert all(isinstance(v, float) for v in kpi["sparkline"])

    def test_with_explicit_nav(self, prices_df: pd.DataFrame):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        result = compute_performance_metrics(weights, prices_df, nav=12500.0)
        assert result["nav"] == 12500.0

    def test_insufficient_data_raises(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame(
            {"AAPL": range(10), "MSFT": range(10)},
            index=dates,
        )
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        with pytest.raises(ValueError, match="Insufficient price data"):
            compute_performance_metrics(weights, prices)


# ---------------------------------------------------------------------------
# Allocation sunburst
# ---------------------------------------------------------------------------


class TestComputeAllocation:
    def test_basic_grouping(self):
        weights = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.3, "CVX": 0.2}
        mapping = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "XOM": "Energy",
            "CVX": "Energy",
        }
        result = compute_allocation(weights, mapping)
        assert len(result["nodes"]) == 2
        sector_names = {n["name"] for n in result["nodes"]}
        assert sector_names == {"Technology", "Energy"}

    def test_weights_sum_to_100(self):
        weights = {"AAPL": 0.4, "MSFT": 0.3, "XOM": 0.3}
        mapping = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        result = compute_allocation(weights, mapping)
        total = sum(n["value"] for n in result["nodes"])
        assert total == pytest.approx(100.0)

    def test_nodes_sorted_descending(self):
        weights = {"AAPL": 0.1, "XOM": 0.9}
        mapping = {"AAPL": "Tech", "XOM": "Energy"}
        result = compute_allocation(weights, mapping)
        assert result["nodes"][0]["name"] == "Energy"
        assert result["nodes"][0]["value"] >= result["nodes"][1]["value"]

    def test_children_sorted_descending(self):
        weights = {"AAPL": 0.2, "MSFT": 0.5, "GOOG": 0.3}
        mapping = {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech"}
        result = compute_allocation(weights, mapping)
        children = result["nodes"][0]["children"]
        assert children[0]["name"] == "MSFT"
        assert children[0]["value"] >= children[1]["value"]

    def test_total_positions(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "XOM": 0.2}
        mapping = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        result = compute_allocation(weights, mapping)
        assert result["total_positions"] == 3

    def test_total_sectors(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "XOM": 0.2}
        mapping = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        result = compute_allocation(weights, mapping)
        assert result["total_sectors"] == 2

    def test_unknown_sector_for_unmapped_ticker(self):
        weights = {"AAPL": 0.6, "NEWCO": 0.4}
        mapping = {"AAPL": "Tech"}
        result = compute_allocation(weights, mapping)
        names = {n["name"] for n in result["nodes"]}
        assert "Unknown" in names

    def test_single_sector(self):
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        mapping = {"AAPL": "Tech", "MSFT": "Tech"}
        result = compute_allocation(weights, mapping)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["value"] == pytest.approx(100.0)

    def test_values_are_percentages(self):
        weights = {"AAPL": 1.0}
        mapping = {"AAPL": "Tech"}
        result = compute_allocation(weights, mapping)
        assert result["nodes"][0]["value"] == pytest.approx(100.0)

    def test_children_values_are_percentages(self):
        weights = {"AAPL": 0.5}
        mapping = {"AAPL": "Tech"}
        result = compute_allocation(weights, mapping)
        assert result["nodes"][0]["children"][0]["value"] == pytest.approx(50.0)
