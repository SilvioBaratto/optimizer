"""Tests for pipeline orchestrator functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import EqualWeighted

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import (
    PortfolioResult,
    backtest,
    build_portfolio_pipeline,
    optimize,
    run_full_pipeline,
    run_full_pipeline_with_selection,
    tune_and_optimize,
)
from optimizer.rebalancing._config import (
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
)
from optimizer.validation import WalkForwardConfig


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 10 assets and 400 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 400, 10
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2022-01-01", periods=n_obs, freq="B"),
    )


@pytest.fixture()
def prices_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic price DataFrame built from returns."""
    prices = (1 + returns_df).cumprod() * 100
    return prices


@pytest.fixture()
def pipeline(returns_df: pd.DataFrame) -> object:
    """A simple portfolio pipeline."""
    return build_portfolio_pipeline(EqualWeighted())


class TestOptimize:
    def test_returns_portfolio_result(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        result = optimize(pipe, returns_df)
        assert isinstance(result, PortfolioResult)

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        result = optimize(pipe, returns_df)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_index_matches_tickers(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        result = optimize(pipe, returns_df)
        assert len(result.weights) == returns_df.shape[1]

    def test_summary_has_metrics(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        result = optimize(pipe, returns_df)
        assert "sharpe_ratio" in result.summary
        assert "max_drawdown" in result.summary
        assert "mean" in result.summary

    def test_pipeline_is_fitted(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        result = optimize(pipe, returns_df)
        assert result.pipeline is not None
        # Can predict on new data
        portfolio = result.pipeline.predict(returns_df)
        assert hasattr(portfolio, "weights")

    def test_with_mean_risk(self, returns_df: pd.DataFrame) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
        pipe = build_portfolio_pipeline(optimizer)
        result = optimize(pipe, returns_df)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-6 for w in result.weights)


class TestBacktest:
    def test_walk_forward(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        cv_cfg = WalkForwardConfig(test_size=21, train_size=100)
        bt = backtest(pipe, returns_df, cv_config=cv_cfg)
        assert hasattr(bt, "sharpe_ratio")

    def test_default_cv(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        bt = backtest(pipe, returns_df)
        assert hasattr(bt, "sharpe_ratio")


class TestTuneAndOptimize:
    def test_grid_search(self, returns_df: pd.DataFrame) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
        pipe = build_portfolio_pipeline(optimizer)
        from optimizer.tuning import GridSearchConfig

        cfg = GridSearchConfig(
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        result = tune_and_optimize(
            pipe,
            returns_df,
            param_grid={"optimizer__l2_coef": [0.0, 0.01]},
            tuning_config=cfg,
        )
        assert isinstance(result, PortfolioResult)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestRunFullPipeline:
    def test_end_to_end(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
        )
        assert isinstance(result, PortfolioResult)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.backtest is None  # no cv_config → no backtest

    def test_with_backtest(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.backtest is not None
        assert hasattr(result.backtest, "sharpe_ratio")

    def test_with_rebalancing(self, prices_df: pd.DataFrame) -> None:
        prev = np.full(prices_df.shape[1], 1.0 / prices_df.shape[1])
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
        )
        assert result.rebalance_needed is not None
        assert result.turnover is not None
        # Equal-weighted → equal-weighted: minimal turnover
        assert result.turnover == pytest.approx(0.0, abs=1e-6)

    def test_with_mean_risk_optimizer(self, prices_df: pd.DataFrame) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=optimizer,
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.backtest is not None
        assert "sharpe_ratio" in result.summary

    def test_rebalancing_detects_drift(self, prices_df: pd.DataFrame) -> None:
        # Previous weights heavily concentrated → rebalancing needed
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0  # 100% in first asset
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
        )
        assert result.rebalance_needed is True
        assert result.turnover is not None
        assert result.turnover > 0.3  # significant turnover


class TestRunFullPipelineRebalancing:
    """Rebalancing config paths through run_full_pipeline (issue #76)."""

    def test_threshold_absolute_breach(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0
        cfg = ThresholdRebalancingConfig.for_absolute(0.05)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
        )
        assert result.rebalance_needed is True

    def test_threshold_absolute_no_breach(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.full(n, 1.0 / n)
        cfg = ThresholdRebalancingConfig.for_absolute(0.05)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
        )
        assert result.rebalance_needed is False

    def test_threshold_relative_breach(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0
        cfg = ThresholdRebalancingConfig.for_relative(0.25)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
        )
        assert result.rebalance_needed is True

    def test_hybrid_elapsed_with_drift(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0
        cfg = HybridRebalancingConfig.for_monthly_with_5pct_threshold()
        current = pd.Timestamp("2023-10-01")
        last_review = current - pd.offsets.BDay(42)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
            current_date=current,
            last_review_date=last_review,
        )
        assert result.rebalance_needed is True

    def test_hybrid_not_elapsed_blocks(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0
        cfg = HybridRebalancingConfig.for_monthly_with_5pct_threshold()
        current = pd.Timestamp("2023-10-01")
        last_review = current - pd.offsets.BDay(3)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
            current_date=current,
            last_review_date=last_review,
        )
        assert result.rebalance_needed is False

    def test_hybrid_elapsed_no_drift(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.full(n, 1.0 / n)
        cfg = HybridRebalancingConfig.for_monthly_with_5pct_threshold()
        current = pd.Timestamp("2023-10-01")
        last_review = current - pd.offsets.BDay(42)
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
            current_date=current,
            last_review_date=last_review,
        )
        assert result.rebalance_needed is False

    def test_hybrid_default_last_review(self, prices_df: pd.DataFrame) -> None:
        n = prices_df.shape[1]
        prev = np.zeros(n)
        prev[0] = 1.0
        cfg = HybridRebalancingConfig.for_monthly_with_5pct_threshold()
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            previous_weights=prev,
            rebalancing_config=cfg,
        )
        # Default last_review is computed to be calendar.trading_days*2 before
        # current_date, so the calendar gate is always elapsed
        assert result.rebalance_needed is True

    def test_no_previous_weights(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
        )
        assert result.rebalance_needed is None
        assert result.turnover is None


class TestRunFullPipelineWithSelection:
    """End-to-end run_full_pipeline_with_selection tests (issue #77)."""

    def test_fundamentals_none_delegates(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline_with_selection(
            prices=prices_df,
            optimizer=EqualWeighted(),
            fundamentals=None,
        )
        assert isinstance(result, PortfolioResult)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_with_fundamentals_calls_all_steps(
        self, prices_df: pd.DataFrame
    ) -> None:
        from unittest.mock import patch

        tickers = list(prices_df.columns[:5])
        mock_investable = pd.Index(tickers)
        mock_factors = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (len(tickers), 3)),
            index=tickers,
            columns=["F1", "F2", "F3"],
        )
        mock_coverage = pd.Series(1.0, index=["F1", "F2", "F3"])
        mock_composite = pd.Series(
            np.random.default_rng(42).uniform(0, 1, len(tickers)),
            index=tickers,
        )
        mock_selected = pd.Index(tickers[:3])

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9] * len(tickers)}, index=tickers
        )

        with (
            patch(
                "optimizer.pipeline._orchestrator.screen_universe",
                return_value=mock_investable,
            ) as m_screen,
            patch(
                "optimizer.pipeline._orchestrator.compute_all_factors",
                return_value=mock_factors,
            ) as m_factors,
            patch(
                "optimizer.pipeline._orchestrator.standardize_all_factors",
                return_value=(mock_factors, mock_coverage),
            ) as m_std,
            patch(
                "optimizer.pipeline._orchestrator.compute_composite_score",
                return_value=mock_composite,
            ) as m_score,
            patch(
                "optimizer.pipeline._orchestrator.select_stocks",
                return_value=mock_selected,
            ) as m_select,
        ):
            result = run_full_pipeline_with_selection(
                prices=prices_df,
                optimizer=EqualWeighted(),
                fundamentals=fundamentals,
            )
            m_screen.assert_called_once()
            m_factors.assert_called_once()
            m_std.assert_called_once()
            m_score.assert_called_once()
            m_select.assert_called_once()
            assert isinstance(result, PortfolioResult)

    def test_selected_subset_used(self, prices_df: pd.DataFrame) -> None:
        from unittest.mock import patch

        tickers = list(prices_df.columns[:3])
        mock_investable = pd.Index(tickers)
        mock_factors = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (len(tickers), 2)),
            index=tickers,
            columns=["F1", "F2"],
        )
        mock_coverage = pd.Series(1.0, index=["F1", "F2"])
        mock_composite = pd.Series(
            np.random.default_rng(42).uniform(0, 1, len(tickers)),
            index=tickers,
        )
        mock_selected = pd.Index(tickers)

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9] * len(tickers)}, index=tickers
        )

        with (
            patch(
                "optimizer.pipeline._orchestrator.screen_universe",
                return_value=mock_investable,
            ),
            patch(
                "optimizer.pipeline._orchestrator.compute_all_factors",
                return_value=mock_factors,
            ),
            patch(
                "optimizer.pipeline._orchestrator.standardize_all_factors",
                return_value=(mock_factors, mock_coverage),
            ),
            patch(
                "optimizer.pipeline._orchestrator.compute_composite_score",
                return_value=mock_composite,
            ),
            patch(
                "optimizer.pipeline._orchestrator.select_stocks",
                return_value=mock_selected,
            ),
        ):
            result = run_full_pipeline_with_selection(
                prices=prices_df,
                optimizer=EqualWeighted(),
                fundamentals=fundamentals,
            )
            assert len(result.weights) <= 3

    def test_cv_config_forwarded(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline_with_selection(
            prices=prices_df,
            optimizer=EqualWeighted(),
            fundamentals=None,
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.backtest is not None

    def test_volume_history_none_uses_fallback(
        self, prices_df: pd.DataFrame
    ) -> None:
        from unittest.mock import patch

        tickers = list(prices_df.columns[:5])
        mock_investable = pd.Index(tickers)
        mock_factors = pd.DataFrame(
            np.random.default_rng(42).normal(0, 1, (len(tickers), 2)),
            index=tickers,
            columns=["F1", "F2"],
        )
        mock_coverage = pd.Series(1.0, index=["F1", "F2"])
        mock_composite = pd.Series(
            np.random.default_rng(42).uniform(0, 1, len(tickers)),
            index=tickers,
        )
        mock_selected = pd.Index(tickers[:3])

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9] * len(tickers)}, index=tickers
        )

        with (
            patch(
                "optimizer.pipeline._orchestrator.screen_universe",
                return_value=mock_investable,
            ) as m_screen,
            patch(
                "optimizer.pipeline._orchestrator.compute_all_factors",
                return_value=mock_factors,
            ),
            patch(
                "optimizer.pipeline._orchestrator.standardize_all_factors",
                return_value=(mock_factors, mock_coverage),
            ),
            patch(
                "optimizer.pipeline._orchestrator.compute_composite_score",
                return_value=mock_composite,
            ),
            patch(
                "optimizer.pipeline._orchestrator.select_stocks",
                return_value=mock_selected,
            ),
        ):
            run_full_pipeline_with_selection(
                prices=prices_df,
                optimizer=EqualWeighted(),
                fundamentals=fundamentals,
                volume_history=None,
            )
            # screen_universe should have been called with an empty DF for volume
            call_kwargs = m_screen.call_args
            vol_arg = call_kwargs.kwargs.get(
                "volume_history", call_kwargs[1].get("volume_history")
            ) if call_kwargs.kwargs else call_kwargs[1].get("volume_history")
            if vol_arg is None:
                # Positional arg
                vol_arg = call_kwargs[0][2]
            assert isinstance(vol_arg, pd.DataFrame)
            assert len(vol_arg) == 0
