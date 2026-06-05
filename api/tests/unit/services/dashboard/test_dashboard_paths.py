"""Empty-data and exception-path tests for dashboard_service (issue #825) — Part 1.

Covers paths NOT exercised by tests/unit/test_dashboard_service.py:
  - compute_equity_curve raises ValueError on insufficient overlapping data (line 309)
  - compute_drift raises ValueError when positions are empty and prices_df is None (line 507)
  - compute_drift raises ValueError when positions are empty and prices_df is empty (line 507)
  - _actual_weights_from_prices raises ValueError when no common tickers (line 555)
  - compute_drift with valid broker positions returns JSON-safe result (empty-data: no breaches)
  - compute_equity_curve with minimal valid data returns JSON-safe result

Raises already covered in tests/unit/test_dashboard_service.py (skipped):
  - ValueError in _portfolio_returns: line 80 — covered by TestPortfolioReturns.test_all_missing_raises
  - ValueError in compute_performance_metrics: line 194 — covered by
    TestComputePerformanceMetrics.test_insufficient_data_raises
  - ValueError in compute_rolling_metrics: line 393 — covered by
    TestComputeRollingMetrics.test_raises_when_window_exceeds_data

Part 2 (compute_asset_class_returns raises) lives in test_dashboard_paths_2.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prices(
    tickers: list[str],
    periods: int = 10,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start, periods=periods)
    data = {
        t: (1 + rng.normal(0.0003, 0.01, periods)).cumprod() * 100 for t in tickers
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_equity_curve — exception paths
# ---------------------------------------------------------------------------


class TestComputeEquityCurveExceptionPaths:
    """compute_equity_curve raises ValueError when overlapping date index is too short."""

    def test_when_insufficient_overlap_then_raises_value_error(self) -> None:
        from app.services.dashboard.dashboard_service import compute_equity_curve

        # 1-row DataFrame gives 0 returns after pct_change().dropna() —
        # intersection will have < 2 points, triggering the guard at line 309.
        dates = pd.bdate_range("2024-01-02", periods=1)
        prices = pd.DataFrame({"AAPL": [150.0], "SPY": [470.0]}, index=dates)

        with pytest.raises(ValueError, match="Insufficient overlapping"):
            compute_equity_curve(
                weights={"AAPL": 1.0},
                prices=prices,
                benchmark="SPY",
            )

    def test_when_portfolio_ticker_missing_from_prices_then_raises(self) -> None:
        from app.services.dashboard.dashboard_service import compute_equity_curve

        prices = _make_prices(["SPY"], periods=10)

        with pytest.raises(ValueError, match="No price data"):
            compute_equity_curve(
                weights={"AAPL": 1.0},
                prices=prices,
                benchmark="SPY",
            )


class TestComputeEquityCurveEmptyData:
    """compute_equity_curve with valid data returns a JSON-safe result."""

    def test_when_valid_data_then_result_is_json_safe(self) -> None:
        from app.services.dashboard.dashboard_service import compute_equity_curve

        prices = _make_prices(["AAPL", "MSFT", "SPY"], periods=30)

        result = compute_equity_curve(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            prices=prices,
            benchmark="SPY",
        )

        assert "points" in result
        assert "portfolio_total_return" in result
        assert "benchmark_total_return" in result
        # Points contain date objects — convert for JSON safety check
        safe = {
            **result,
            "points": [
                {k: str(v) if hasattr(v, "isoformat") else v for k, v in p.items()}
                for p in result["points"]
            ],
        }
        assert_json_safe(safe)

    def test_when_single_ticker_portfolio_then_returns_nonempty_points(self) -> None:
        from app.services.dashboard.dashboard_service import compute_equity_curve

        prices = _make_prices(["AAPL", "SPY"], periods=20)

        result = compute_equity_curve(
            weights={"AAPL": 1.0},
            prices=prices,
            benchmark="SPY",
        )

        assert len(result["points"]) > 0


# ---------------------------------------------------------------------------
# compute_drift — exception paths
# ---------------------------------------------------------------------------


class TestComputeDriftExceptionPaths:
    """compute_drift raises ValueError when no broker positions AND no prices."""

    def test_when_empty_positions_and_no_prices_df_then_raises(self) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        with pytest.raises(ValueError, match="No price data available for drift"):
            compute_drift(
                target_weights={"AAPL": 0.5, "MSFT": 0.5},
                broker_positions=[],
                threshold=0.05,
                prices_df=None,
            )

    def test_when_empty_positions_and_empty_prices_df_then_raises(self) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        with pytest.raises(ValueError, match="No price data available for drift"):
            compute_drift(
                target_weights={"AAPL": 0.5, "MSFT": 0.5},
                broker_positions=[],
                threshold=0.05,
                prices_df=pd.DataFrame(),
            )

    def test_when_positions_have_no_valid_ticker_and_prices_empty_then_raises(
        self,
    ) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        # Positions without yfinance_ticker → treated as no valid positions
        invalid_positions = [{"name": "Apple", "quantity": 10, "current_price": 150.0}]

        with pytest.raises(ValueError, match="No price data available for drift"):
            compute_drift(
                target_weights={"AAPL": 1.0},
                broker_positions=invalid_positions,
                threshold=0.05,
                prices_df=pd.DataFrame(),
            )


class TestComputeDriftEmptyData:
    """compute_drift with valid broker positions returns a JSON-safe result."""

    def test_when_valid_positions_then_returns_json_safe_dict(self) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        positions = [
            {
                "yfinance_ticker": "AAPL",
                "name": "Apple Inc",
                "quantity": 10.0,
                "current_price": 150.0,
            },
            {
                "yfinance_ticker": "MSFT",
                "name": "Microsoft",
                "quantity": 5.0,
                "current_price": 300.0,
            },
        ]

        result = compute_drift(
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            broker_positions=positions,
            threshold=0.05,
        )

        assert "entries" in result
        assert "total_drift" in result
        assert "breached_count" in result
        assert "threshold" in result
        assert_json_safe(result)

    def test_when_target_weight_ticker_absent_from_positions_then_drift_entry_present(
        self,
    ) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        positions = [
            {
                "yfinance_ticker": "AAPL",
                "name": "Apple Inc",
                "quantity": 10.0,
                "current_price": 150.0,
            },
        ]

        result = compute_drift(
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            broker_positions=positions,
            threshold=0.05,
        )

        tickers = {e["ticker"] for e in result["entries"]}
        assert "MSFT" in tickers
        assert_json_safe(result)

    def test_when_price_fallback_used_and_no_common_tickers_then_raises(self) -> None:
        from app.services.dashboard.dashboard_service import compute_drift

        # Empty positions force fallback to prices_df; prices have no matching columns
        prices = _make_prices(["UNRELATED"], periods=5)

        with pytest.raises(ValueError, match="No price data available for drift"):
            compute_drift(
                target_weights={"AAPL": 0.5, "MSFT": 0.5},
                broker_positions=[],
                threshold=0.05,
                prices_df=prices,
            )


# ---------------------------------------------------------------------------
# _actual_weights_from_prices — exception path (line 555)
# ---------------------------------------------------------------------------


class TestActualWeightsFromPricesExceptionPath:
    """_actual_weights_from_prices raises ValueError when target tickers are
    absent from prices_df columns (line 555 inside compute_drift fallback).
    """

    def test_when_no_common_tickers_then_raises_value_error(self) -> None:
        from app.services.dashboard.dashboard_service import _actual_weights_from_prices

        prices = _make_prices(["UNRELATED"], periods=5)

        with pytest.raises(ValueError, match="No price data available for drift"):
            _actual_weights_from_prices(
                target_weights={"AAPL": 0.5, "MSFT": 0.5},
                prices_df=prices,
            )

    def test_when_common_tickers_present_then_returns_dict(self) -> None:
        from app.services.dashboard.dashboard_service import _actual_weights_from_prices

        prices = _make_prices(["AAPL", "MSFT"], periods=5)

        result = _actual_weights_from_prices(
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
            prices_df=prices,
        )

        # Private helper returns raw numeric values; JSON safety is validated
        # at the public compute_drift boundary (see TestComputeDriftEmptyData).
        assert set(result.keys()) == {"AAPL", "MSFT"}
        assert all(isinstance(v, (float, int)) or hasattr(v, "__float__") for v in result.values())
