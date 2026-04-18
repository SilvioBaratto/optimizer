"""Unit tests for app/services/backtest_service.py.

Covers:
  - validate_prices raises descriptive ValueError on empty DataFrame
  - validate_prices raises descriptive ValueError when all tickers missing
  - extract_backtest_metrics returns all required BacktestRun keys
  - build_optimizer builds a valid skfolio optimizer
  - _ensure_datetime_index coerces a plain Index of datetime.date to DatetimeIndex
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestValidatePrices:
    """validate_prices raises ValueError with descriptive message on bad data."""

    def test_empty_dataframe_raises(self) -> None:
        from app.services.backtest_service import validate_prices

        with pytest.raises(ValueError, match="price data"):
            validate_prices(pd.DataFrame(), ["AAPL", "MSFT"])

    def test_single_missing_ticker_mentioned_in_error(self) -> None:
        from app.services.backtest_service import validate_prices

        prices = pd.DataFrame({"AAPL": [1.0, 2.0]})
        with pytest.raises(ValueError, match="MSFT"):
            validate_prices(prices, ["AAPL", "MSFT"])

    def test_valid_prices_does_not_raise(self) -> None:
        from app.services.backtest_service import validate_prices

        prices = pd.DataFrame({"AAPL": [1.0, 2.0], "MSFT": [3.0, 4.0]})
        validate_prices(prices, ["AAPL", "MSFT"])  # no exception


class TestBuildOptimizer:
    """build_optimizer maps pipeline_config to an optimizer instance."""

    def test_hrp_config_builds_optimizer(self) -> None:
        from app.services.backtest_service import build_optimizer

        opt = build_optimizer({"optimizer_type": "hrp"})
        assert opt is not None

    def test_default_type_when_missing(self) -> None:
        from app.services.backtest_service import build_optimizer

        opt = build_optimizer({})
        assert opt is not None

    def test_unknown_type_raises(self) -> None:
        from app.services.backtest_service import build_optimizer

        with pytest.raises(ValueError, match="Unknown"):
            build_optimizer({"optimizer_type": "unknown_xyz"})


class TestEnsureDatetimeIndex:
    """_ensure_datetime_index coerces ``prices.index`` to a tz-naive DatetimeIndex.

    Regression for issue #422: ``fetch_close_prices`` builds a DataFrame whose
    index is a plain ``Index[object]`` of ``datetime.date``. ``run_full_pipeline``
    requires a ``DatetimeIndex`` and crashes with ``TypeError: Only valid with
    DatetimeIndex, TimedeltaIndex or PeriodIndex`` otherwise.
    """

    def test_plain_date_index_is_coerced_to_datetimeindex(self) -> None:
        from app.services.backtest_service import _ensure_datetime_index

        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]},
            index=pd.Index([date(2020, 1, 1), date(2020, 1, 2)]),
        )
        assert not isinstance(prices.index, pd.DatetimeIndex)

        out = _ensure_datetime_index(prices)

        assert isinstance(out.index, pd.DatetimeIndex)

    def test_result_is_tz_naive(self) -> None:
        from app.services.backtest_service import _ensure_datetime_index

        prices = pd.DataFrame(
            {"AAPL": [100.0]},
            index=pd.Index([date(2020, 1, 1)]),
        )
        out = _ensure_datetime_index(prices)

        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is None

    def test_input_is_not_mutated(self) -> None:
        from app.services.backtest_service import _ensure_datetime_index

        prices = pd.DataFrame(
            {"AAPL": [100.0]},
            index=pd.Index([date(2020, 1, 1)]),
        )
        original_index_type = type(prices.index)

        _ensure_datetime_index(prices)

        assert type(prices.index) is original_index_type

    def test_existing_datetimeindex_is_preserved(self) -> None:
        from app.services.backtest_service import _ensure_datetime_index

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=dates)

        out = _ensure_datetime_index(prices)

        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is None

    def test_tz_aware_datetimeindex_is_normalised_to_naive(self) -> None:
        from app.services.backtest_service import _ensure_datetime_index

        dates = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=dates)

        out = _ensure_datetime_index(prices)

        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is None


class TestRunAndPersistCoercesPrices:
    """run_and_persist must coerce plain-Index prices before calling the pipeline.

    Regression for issue #422: the live bug surfaced inside ``run_and_persist``
    where ``fetch_close_prices`` returns a DataFrame with a plain ``Index`` of
    ``datetime.date``, and ``run_full_pipeline`` crashes on a non-DatetimeIndex.
    """

    def test_plain_date_index_does_not_raise_typeerror(self) -> None:
        """run_and_persist normalises the index before handing off to the pipeline."""
        from app.services import backtest_service

        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]},
            index=pd.Index([date(2020, 1, 1), date(2020, 1, 2)]),
        )
        captured: dict[str, pd.DataFrame] = {}

        def _fake_pipeline(**kwargs):  # type: ignore[no-untyped-def]
            captured["prices"] = kwargs["prices"]
            return _make_mock_result()

        repo = MagicMock()
        repo.update_backtest_status.return_value = MagicMock()
        session = MagicMock()

        with (
            patch.object(backtest_service, "fetch_close_prices", return_value=prices),
            patch.object(backtest_service, "run_full_pipeline", side_effect=_fake_pipeline),
            patch.object(backtest_service, "ExecutionRepository", return_value=repo),
            patch.object(backtest_service, "extract_backtest_metrics", return_value={}),
        ):
            backtest_service.run_and_persist(
                tickers=["AAPL", "MSFT"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 2),
                pipeline_config={"optimizer_type": "hrp", "run_cv": False},
                session=session,
                run_id=__import__("uuid").uuid4(),
            )

        assert isinstance(captured["prices"].index, pd.DatetimeIndex)


class TestExtractBacktestMetrics:
    """extract_backtest_metrics returns all keys required by BacktestRun model."""

    def test_required_keys_present(self) -> None:
        from app.services.backtest_service import extract_backtest_metrics

        result = _make_mock_result()
        metrics = extract_backtest_metrics(result)

        required = {
            "equity_curve",
            "drawdowns",
            "monthly_returns",
            "yearly_returns",
            "rolling_metrics",
            "turnover_history",
            "summary_stats",
        }
        assert required.issubset(metrics.keys())

    def test_equity_curve_is_dict(self) -> None:
        from app.services.backtest_service import extract_backtest_metrics

        metrics = extract_backtest_metrics(_make_mock_result())
        assert isinstance(metrics["equity_curve"], dict)

    def test_cv_fold_metrics_none_when_no_backtest(self) -> None:
        from app.services.backtest_service import extract_backtest_metrics

        result = _make_mock_result(with_backtest=False)
        metrics = extract_backtest_metrics(result)
        assert metrics.get("cv_fold_metrics") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_portfolio() -> MagicMock:
    """Return a mock skfolio Portfolio with required attributes."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    portfolio = MagicMock()
    portfolio.cumulative_returns_df = pd.Series(
        data=[1.0 + i * 0.01 for i in range(10)],
        index=dates,
        name="cumulative_returns",
    )
    portfolio.drawdowns_df = pd.Series(
        data=[-0.01 * i for i in range(10)],
        index=dates,
        name="drawdowns",
    )
    portfolio.returns_df = pd.Series(
        data=[0.01] * 10,
        index=dates,
        name="returns",
    )
    portfolio.rolling_measure.return_value = pd.Series(
        data=[1.0] * 10, index=dates
    )
    portfolio.summary.return_value = pd.Series(
        {"Annualized Sharpe Ratio": 1.2, "Max Drawdown": -0.05}
    )
    portfolio.max_drawdown = -0.05
    return portfolio


def _make_mock_result(with_backtest: bool = True) -> MagicMock:
    import pandas as pd

    result = MagicMock()
    result.portfolio = _make_mock_portfolio()
    result.backtest = _make_mock_portfolio() if with_backtest else None
    result.summary = {"annualized_sharpe_ratio": 1.2, "max_drawdown": -0.05}
    result.weight_history = None
    return result
