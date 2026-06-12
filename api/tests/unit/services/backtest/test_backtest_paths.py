"""Empty-data and exception-path tests for backtest_service (issue #825).

Covers paths NOT exercised by tests/unit/test_backtest_service.py:
  - validate_prices raises ValueError when prices is non-empty but ALL
    requested tickers are absent (distinct from the already-tested
    "empty DataFrame" branch — this branch at line 57-62 fires when some
    columns exist but none match the requested tickers).
  - run_and_persist propagates ValueError from validate_prices when
    fetch_close_prices returns an empty DataFrame (exercises the
    validate_prices → run_and_persist integration path).
  - _resample_returns returns {} for an empty Series (empty-data path).
  - _turnover_history returns {} when weight_history is None (empty-data path).
  - extract_backtest_metrics is JSON-safe on a clean mock result (regression
    guard: assert_json_safe on the pure compute path).

Raises NOT re-covered (skipped with reason):
  - ValueError on empty prices in validate_prices (line 52-55): already
    covered by TestValidatePrices.test_empty_dataframe_raises.
  - ValueError on missing tickers in validate_prices (line 57-62): the
    single-ticker case is covered by
    TestValidatePrices.test_single_missing_ticker_mentioned_in_error.
    We add a complementary multi-ticker variant here.
  - build_optimizer unknown type (line 59 of backtest_service / inside
    _build_optimizer): already covered by
    TestBuildOptimizer.test_unknown_type_raises.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# validate_prices — all tickers absent from a non-empty frame
# ---------------------------------------------------------------------------


class TestValidatePricesAllTickersAbsent:
    """validate_prices raises ValueError when every requested ticker is missing
    from a non-empty prices DataFrame (line 57-62 in backtest_service.py).
    """

    def test_when_prices_has_columns_but_no_requested_ticker_matches_then_raises(
        self,
    ) -> None:
        from app.services.backtest.backtest_service import validate_prices

        prices = pd.DataFrame({"UNRELATED": [100.0, 101.0]})
        with pytest.raises(ValueError, match="AAPL"):
            validate_prices(prices, ["AAPL", "MSFT"])

    def test_when_only_some_tickers_present_then_drops_missing_ones(self) -> None:
        from app.services.backtest.backtest_service import validate_prices

        prices = pd.DataFrame({"AAPL": [100.0, 101.0], "UNRELATED": [50.0, 51.0]})
        result = validate_prices(prices, ["AAPL", "MSFT"])
        assert list(result.columns) == ["AAPL"]
        assert len(result) == 2

    def test_when_all_requested_tickers_present_then_does_not_raise(self) -> None:
        from app.services.backtest.backtest_service import validate_prices

        prices = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]})
        validate_prices(prices, ["AAPL", "MSFT"])  # must not raise


# ---------------------------------------------------------------------------
# _resample_returns — empty-data path
# ---------------------------------------------------------------------------


class TestResampleReturnsEmptyData:
    """_resample_returns returns {} when given an empty Series (no crash)."""

    def test_when_empty_series_then_returns_empty_dict(self) -> None:
        from app.services.backtest.backtest_service import _resample_returns

        result = _resample_returns(pd.Series([], dtype=float), "ME")

        assert result == {}

    def test_when_empty_series_yearly_then_returns_empty_dict(self) -> None:
        from app.services.backtest.backtest_service import _resample_returns

        result = _resample_returns(pd.Series([], dtype=float), "YE")

        assert result == {}


# ---------------------------------------------------------------------------
# _turnover_history — empty-data path
# ---------------------------------------------------------------------------


class TestTurnoverHistoryEmptyData:
    """_turnover_history returns {} when result.weight_history is None."""

    def test_when_weight_history_none_then_returns_empty_dict(self) -> None:
        from app.services.backtest.backtest_service import _turnover_history

        result_mock = MagicMock()
        result_mock.weight_history = None

        out = _turnover_history(result_mock)

        assert out == {}
        assert_json_safe(out)


# ---------------------------------------------------------------------------
# extract_backtest_metrics — JSON-safe on clean mock
# ---------------------------------------------------------------------------


class TestExtractBacktestMetricsJsonSafe:
    """extract_backtest_metrics output is assert_json_safe on a clean result."""

    def test_when_clean_mock_result_then_output_is_json_safe(self) -> None:
        from app.services.backtest.backtest_service import extract_backtest_metrics

        result = _make_clean_mock_result()
        metrics = extract_backtest_metrics(result)

        assert_json_safe(metrics)

    def test_when_no_backtest_then_cv_fold_metrics_is_none(self) -> None:
        from app.services.backtest.backtest_service import extract_backtest_metrics

        result = _make_clean_mock_result(with_backtest=False)
        metrics = extract_backtest_metrics(result)

        assert metrics["cv_fold_metrics"] is None
        assert_json_safe(metrics)


# ---------------------------------------------------------------------------
# run_and_persist — propagates ValueError from validate_prices
# ---------------------------------------------------------------------------


class TestRunAndPersistValidation:
    """run_and_persist raises ValueError when fetch_close_prices returns empty."""

    def test_when_fetch_returns_empty_frame_then_raises_value_error(self) -> None:
        from app.services.backtest import backtest_service

        session = MagicMock()

        with patch.object(
            backtest_service,
            "fetch_close_prices",
            return_value=pd.DataFrame(),
        ):
            with pytest.raises(ValueError, match="price data"):
                backtest_service.run_and_persist(
                    tickers=["AAPL"],
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 12, 31),
                    pipeline_config={"optimizer_type": "mean_risk", "run_cv": False},
                    session=session,
                    run_id=uuid.uuid4(),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_mock_portfolio() -> MagicMock:
    dates = pd.date_range("2022-01-03", periods=10, freq="B")
    portfolio = MagicMock()
    portfolio.cumulative_returns_df = pd.Series(
        [1.0 + i * 0.01 for i in range(10)], index=dates
    )
    portfolio.drawdowns_df = pd.Series([-0.01 * i for i in range(10)], index=dates)
    portfolio.returns_df = pd.Series([0.01] * 10, index=dates)
    portfolio.rolling_measure.return_value = pd.Series([1.0] * 10, index=dates)
    portfolio.summary.return_value = pd.Series(
        {"Annualized Sharpe Ratio": 1.2, "Max Drawdown": -0.05}
    )
    portfolio.max_drawdown = -0.05
    return portfolio


def _make_clean_mock_result(*, with_backtest: bool = True) -> MagicMock:
    result = MagicMock()
    result.portfolio = _make_clean_mock_portfolio()
    result.backtest = _make_clean_mock_portfolio() if with_backtest else None
    result.summary = {"annualized_sharpe_ratio": 1.2, "max_drawdown": -0.05}
    result.weight_history = None
    return result
