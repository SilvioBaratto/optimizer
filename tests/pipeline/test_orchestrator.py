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
    compute_net_backtest_returns,
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

    def test_with_fundamentals_calls_all_steps(self, prices_df: pd.DataFrame) -> None:
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
        mock_selected = pd.Index(tickers[:5])

        fundamentals = pd.DataFrame({"market_cap": [1e9] * len(tickers)}, index=tickers)

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
        mock_selected = pd.Index(tickers)

        fundamentals = pd.DataFrame({"market_cap": [1e9] * len(tickers)}, index=tickers)

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
            assert len(result.weights) <= 5

    def test_cv_config_forwarded(self, prices_df: pd.DataFrame) -> None:
        result = run_full_pipeline_with_selection(
            prices=prices_df,
            optimizer=EqualWeighted(),
            fundamentals=None,
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.backtest is not None

    def test_volume_history_none_uses_fallback(self, prices_df: pd.DataFrame) -> None:
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
        mock_selected = pd.Index(tickers[:5])

        fundamentals = pd.DataFrame({"market_cap": [1e9] * len(tickers)}, index=tickers)

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
            vol_arg = (
                call_kwargs.kwargs.get(
                    "volume_history", call_kwargs[1].get("volume_history")
                )
                if call_kwargs.kwargs
                else call_kwargs[1].get("volume_history")
            )
            if vol_arg is None:
                # Positional arg
                vol_arg = call_kwargs[0][2]
            assert isinstance(vol_arg, pd.DataFrame)
            assert len(vol_arg) == 0


class TestEmptySelectionGuard:
    """Guard against empty or degenerate stock selection (issue #268)."""

    def _run_with_mocked_selection(
        self,
        prices_df: pd.DataFrame,
        mock_selected: pd.Index,
        n_nan_scores: int = 0,
    ):
        """Run run_full_pipeline_with_selection with all upstream mocks."""
        from unittest.mock import patch

        tickers = list(prices_df.columns[:5])
        mock_investable = pd.Index(tickers)
        mock_factors = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, (len(tickers), 2)),
            index=tickers,
            columns=["F1", "F2"],
        )
        mock_coverage = pd.Series(1.0, index=["F1", "F2"])

        scores = list(np.random.default_rng(0).uniform(0, 1, len(tickers)))
        for i in range(n_nan_scores):
            scores[i] = float("nan")
        mock_composite = pd.Series(scores, index=tickers)

        fundamentals = pd.DataFrame(
            {"market_cap": [1e9] * len(tickers)}, index=tickers
        )

        pfx = "optimizer.pipeline._orchestrator"
        with (
            patch(f"{pfx}.screen_universe", return_value=mock_investable),
            patch(f"{pfx}.compute_all_factors", return_value=mock_factors),
            patch(
                f"{pfx}.standardize_all_factors",
                return_value=(mock_factors, mock_coverage),
            ),
            patch(f"{pfx}.compute_composite_score", return_value=mock_composite),
            patch(f"{pfx}.select_stocks", return_value=mock_selected),
        ):
            return run_full_pipeline_with_selection(
                prices=prices_df,
                optimizer=EqualWeighted(),
                fundamentals=fundamentals,
            )

    def test_empty_selection_raises_data_error(
        self, prices_df: pd.DataFrame
    ) -> None:
        from optimizer.exceptions import DataError

        with pytest.raises(DataError, match="empty universe"):
            self._run_with_mocked_selection(
                prices_df, mock_selected=pd.Index([]), n_nan_scores=5
            )

    def test_below_minimum_raises_data_error(
        self, prices_df: pd.DataFrame
    ) -> None:
        from optimizer.exceptions import DataError

        with pytest.raises(DataError, match="below the minimum of 5"):
            self._run_with_mocked_selection(
                prices_df,
                mock_selected=pd.Index(list(prices_df.columns[:3])),
                n_nan_scores=2,
            )

    def test_error_message_includes_nan_count(
        self, prices_df: pd.DataFrame
    ) -> None:
        from optimizer.exceptions import DataError

        with pytest.raises(DataError, match="5 of 5"):
            self._run_with_mocked_selection(
                prices_df, mock_selected=pd.Index([]), n_nan_scores=5
            )

    def test_sufficient_selection_does_not_raise(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = self._run_with_mocked_selection(
            prices_df,
            mock_selected=pd.Index(list(prices_df.columns[:5])),
            n_nan_scores=0,
        )
        assert isinstance(result, PortfolioResult)


class TestTuneRandomizedSearch:
    """Tests for RandomizedSearchConfig in tune_and_optimize (issue #97)."""

    def test_tune_randomized_search(self, returns_df: pd.DataFrame) -> None:
        from optimizer.tuning import RandomizedSearchConfig

        optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
        pipe = build_portfolio_pipeline(optimizer)
        cfg = RandomizedSearchConfig(
            n_iter=2,
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        result = tune_and_optimize(
            pipe,
            returns_df,
            param_grid={"optimizer__l2_coef": [0.0, 0.01, 0.05]},
            tuning_config=cfg,
        )
        assert isinstance(result, PortfolioResult)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestComputeNetBacktestReturns:
    """Tests for compute_net_backtest_returns (issue #98)."""

    def test_zero_cost_equals_gross(self) -> None:
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        gross = pd.Series([0.01, 0.02, -0.01, 0.005, 0.03], index=dates)
        changes = pd.DataFrame(
            {"A": [0.1, 0.0, 0.0, 0.0, 0.0], "B": [-0.1, 0.0, 0.0, 0.0, 0.0]},
            index=dates,
        )
        net = compute_net_backtest_returns(gross, changes, cost_bps=0.0)
        pd.testing.assert_series_equal(net, gross)

    def test_full_turnover_deducts_cost(self) -> None:
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        gross = pd.Series([0.01, 0.02, 0.03], index=dates)
        # One-way turnover = sum(abs) / 2 = (0.5 + 0.5) / 2 = 0.5
        changes = pd.DataFrame(
            {"A": [0.5, 0.0, 0.0], "B": [-0.5, 0.0, 0.0]},
            index=dates,
        )
        net = compute_net_backtest_returns(gross, changes, cost_bps=10.0)
        # First date: 0.01 - 0.5 * 10/10000 = 0.01 - 0.0005 = 0.0095
        assert net.iloc[0] == pytest.approx(0.0095)
        # Other dates: no changes → same as gross
        assert net.iloc[1] == pytest.approx(0.02)
        assert net.iloc[2] == pytest.approx(0.03)

    def test_one_way_turnover_not_double_counted(self) -> None:
        """A shift of w from A to B costs w * cost_bps/10_000, not 2w."""
        dates = pd.date_range("2023-01-01", periods=2, freq="B")
        gross = pd.Series([0.0, 0.0], index=dates)
        # Move 20% from A to B: weight_change = [-0.2, +0.2]
        # One-way turnover = (0.2 + 0.2) / 2 = 0.2
        # Cost = 0.2 * 50/10_000 = 0.001
        changes = pd.DataFrame(
            {"A": [-0.2, 0.0], "B": [0.2, 0.0]},
            index=dates,
        )
        net = compute_net_backtest_returns(gross, changes, cost_bps=50.0)
        assert net.iloc[0] == pytest.approx(-0.001)
        assert net.iloc[1] == pytest.approx(0.0)

    def test_net_returns_consistent_with_compute_turnover(self) -> None:
        """compute_net_backtest_returns and compute_turnover agree."""
        from optimizer.rebalancing._rebalancer import compute_turnover

        dates = pd.date_range("2023-01-01", periods=1, freq="B")
        gross = pd.Series([0.05], index=dates)
        prev = np.array([0.6, 0.4])
        new = np.array([0.5, 0.5])
        changes = pd.DataFrame(
            {"A": [new[0] - prev[0]], "B": [new[1] - prev[1]]},
            index=dates,
        )
        cost_bps = 20.0
        net = compute_net_backtest_returns(gross, changes, cost_bps=cost_bps)
        expected_turnover = compute_turnover(prev, new)  # = 0.1
        expected_net = 0.05 - expected_turnover * cost_bps / 10_000
        assert net.iloc[0] == pytest.approx(expected_net)

    def test_no_weight_change_no_deduction(self) -> None:
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        gross = pd.Series([0.01, 0.02, 0.03], index=dates)
        changes = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
            index=dates,
        )
        net = compute_net_backtest_returns(gross, changes, cost_bps=10.0)
        pd.testing.assert_series_equal(net, gross)


class TestRiskFreeRatePropagation:
    """Risk-free rate propagation through the pipeline (issue #272)."""

    def test_rf_injected_into_mean_risk_optimizer(
        self, prices_df: pd.DataFrame
    ) -> None:
        """When risk_free_rate != 0, MeanRisk optimizer receives the value."""
        rf = 0.0002  # ~5% annualised daily
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        assert optimizer.risk_free_rate == 0.0  # default before injection

        result = run_full_pipeline(
            prices=prices_df,
            optimizer=optimizer,
            risk_free_rate=rf,
        )
        assert isinstance(result, PortfolioResult)
        assert result.risk_free_rate == rf
        # The fitted pipeline's optimizer should have the injected Rf
        fitted_opt = result.pipeline[-1]  # last step is the optimizer
        assert fitted_opt.risk_free_rate == pytest.approx(rf)

    def test_rf_zero_does_not_copy_optimizer(
        self, prices_df: pd.DataFrame
    ) -> None:
        """When risk_free_rate=0.0 (default), optimizer is not deepcopied."""
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

        result = run_full_pipeline(
            prices=prices_df,
            optimizer=optimizer,
            risk_free_rate=0.0,
        )
        assert result.risk_free_rate == 0.0
        assert "sharpe_ratio" in result.summary

    def test_rf_ignored_for_equal_weighted_with_warning(
        self, prices_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """EqualWeighted has no risk_free_rate attr; a warning is logged."""
        import logging

        log = "optimizer.pipeline._orchestrator"
        with caplog.at_level(logging.WARNING, logger=log):
            result = run_full_pipeline(
                prices=prices_df,
                optimizer=EqualWeighted(),
                risk_free_rate=0.0005,
            )
        assert result.risk_free_rate == 0.0005
        assert isinstance(result, PortfolioResult)
        assert any("risk_free_rate" in rec.message for rec in caplog.records)

    def test_summary_sharpe_uses_injected_rf(
        self, prices_df: pd.DataFrame
    ) -> None:
        """Portfolio Sharpe in summary uses the injected risk_free_rate."""
        rf = 0.0002
        # Run with rf=0 and rf=nonzero, Sharpe should differ
        result_zero = run_full_pipeline(
            prices=prices_df,
            optimizer=build_mean_risk(MeanRiskConfig.for_max_sharpe()),
            risk_free_rate=0.0,
        )
        result_with_rf = run_full_pipeline(
            prices=prices_df,
            optimizer=build_mean_risk(MeanRiskConfig.for_max_sharpe()),
            risk_free_rate=rf,
        )
        # With positive rf, Sharpe should be lower
        sharpe_rf = result_with_rf.summary["sharpe_ratio"]
        sharpe_0 = result_zero.summary["sharpe_ratio"]
        assert sharpe_rf < sharpe_0

    def test_original_optimizer_not_mutated(
        self, prices_df: pd.DataFrame
    ) -> None:
        """The caller's original optimizer object must not be modified."""
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        assert optimizer.risk_free_rate == 0.0

        run_full_pipeline(
            prices=prices_df,
            optimizer=optimizer,
            risk_free_rate=0.001,
        )
        # Original should still be 0.0 — deepcopy protects it
        assert optimizer.risk_free_rate == 0.0


class TestWeightHistoryExtraction:
    """Walk-forward weight history extraction in PortfolioResult (issue #285)."""

    def test_weight_history_none_without_backtest(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
        )
        assert result.weight_history is None

    def test_weight_history_populated_with_backtest(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        assert isinstance(result.weight_history, pd.DataFrame)
        # At least the initial allocation is present
        assert len(result.weight_history) >= 1

    def test_weight_history_multiple_events_with_mean_risk(
        self, prices_df: pd.DataFrame
    ) -> None:
        """MeanRisk produces different weights per window → multiple events."""
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=build_mean_risk(MeanRiskConfig.for_min_variance()),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        assert len(result.weight_history) > 1

    def test_weight_history_is_rebalancing_events_only(
        self, prices_df: pd.DataFrame
    ) -> None:
        """Rows in weight_history must be fewer than total observations."""
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        assert result.backtest is not None
        total_obs = len(result.backtest.returns_df)
        assert len(result.weight_history) < total_obs

    def test_weight_history_contains_absolute_weights(
        self, prices_df: pd.DataFrame
    ) -> None:
        """Each row must be absolute weights summing to ~1, not deltas."""
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        for _, row in result.weight_history.iterrows():
            assert row.sum() == pytest.approx(1.0, abs=1e-4)

    def test_weight_history_compatible_with_compute_net_alpha(
        self, prices_df: pd.DataFrame
    ) -> None:
        """weight_history can be passed directly to compute_net_alpha."""
        from optimizer.factors._integration import compute_net_alpha

        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        # Dummy IC series
        ic_series = pd.Series(
            np.random.default_rng(42).normal(0.03, 0.01, 10),
            index=pd.date_range("2023-01-01", periods=10, freq="ME"),
        )
        na_result = compute_net_alpha(
            ic_series=ic_series,
            weights_history=result.weight_history,
            cost_bps=10.0,
        )
        # With multiple rebalancing events, turnover should be > 0
        # (EqualWeighted may have near-zero turnover between windows,
        # but the initial allocation from zero counts)
        assert na_result.avg_turnover >= 0.0
        assert na_result.net_alpha <= na_result.gross_alpha

    def test_single_period_vs_full_history_net_alpha(
        self, prices_df: pd.DataFrame
    ) -> None:
        """Full-history net alpha must show higher cost drag than single-period."""
        from optimizer.factors._integration import compute_net_alpha

        result = run_full_pipeline(
            prices=prices_df,
            optimizer=build_mean_risk(MeanRiskConfig.for_min_variance()),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.weight_history is not None
        assert len(result.weight_history) > 1

        ic_series = pd.Series(
            np.random.default_rng(42).normal(0.05, 0.02, 10),
            index=pd.date_range("2023-01-01", periods=10, freq="ME"),
        )
        cost_bps = 50.0

        # Full history: multiple rebalancing events → non-zero turnover
        full_result = compute_net_alpha(
            ic_series=ic_series,
            weights_history=result.weight_history,
            cost_bps=cost_bps,
        )

        # Single-period snapshot (the old buggy approach)
        single_snapshot = pd.DataFrame(
            [result.weights], index=[result.weight_history.index[-1]]
        )
        single_result = compute_net_alpha(
            ic_series=ic_series,
            weights_history=single_snapshot,
            cost_bps=cost_bps,
        )

        # Single snapshot always has avg_turnover=0 (only one row)
        assert single_result.avg_turnover == 0.0
        # Full history should have non-zero turnover (MeanRisk weights change)
        assert full_result.avg_turnover > 0.0
        # Therefore full-history net alpha is lower (more cost drag)
        assert full_result.net_alpha < single_result.net_alpha


class TestNetBacktestReturns:
    """Net transaction cost wiring in run_full_pipeline (issue #284)."""

    def test_net_returns_none_without_backtest(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
        )
        assert result.net_returns is None
        assert result.net_sharpe_ratio is None

    def test_net_returns_populated_with_backtest(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
        )
        assert result.net_returns is not None
        assert isinstance(result.net_returns, pd.Series)
        assert result.net_sharpe_ratio is not None

    def test_net_sharpe_lower_than_gross(
        self, prices_df: pd.DataFrame
    ) -> None:
        from optimizer.pipeline._orchestrator import _compute_net_sharpe

        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
            cost_bps=50.0,
        )
        assert result.backtest is not None
        assert result.net_returns is not None
        assert result.net_sharpe_ratio is not None
        # Compare using same formula for both gross and net
        gross_sharpe = _compute_net_sharpe(result.backtest.returns_df)
        assert gross_sharpe is not None
        assert result.net_sharpe_ratio < gross_sharpe

    def test_net_returns_lower_by_cost_amount(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
            cost_bps=10.0,
        )
        assert result.net_returns is not None
        gross = result.backtest.returns_df
        diff = gross.sum() - result.net_returns.sum()
        # Net must be lower (costs deducted)
        assert diff > 0

    def test_zero_cost_bps_net_equals_gross(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline(
            prices=prices_df,
            optimizer=EqualWeighted(),
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
            cost_bps=0.0,
        )
        assert result.net_returns is not None
        pd.testing.assert_series_equal(
            result.net_returns, result.backtest.returns_df
        )

    def test_cost_bps_forwarded_through_selection(
        self, prices_df: pd.DataFrame
    ) -> None:
        result = run_full_pipeline_with_selection(
            prices=prices_df,
            optimizer=EqualWeighted(),
            fundamentals=None,
            cv_config=WalkForwardConfig(test_size=21, train_size=100),
            cost_bps=25.0,
        )
        assert result.net_sharpe_ratio is not None


class TestReturnsDfFallback:
    """Regression tests for issue #309: resilient bt.returns_df accessor.

    Verifies the fallback path (reconstructing gross returns from public
    ``bt.returns`` + ``bt.observations``) and that both branches correctly
    upgrade the silent debug log to a warning.
    """

    def _make_mock_bt(
        self,
        n_obs: int = 50,
        *,
        include_returns_df: bool,
    ) -> object:
        """Build a minimal backtest-like namespace.

        Uses ``types.SimpleNamespace`` so ``hasattr`` checks reflect only
        explicitly set attributes — unlike ``MagicMock`` which auto-creates.
        """
        import types

        rng = np.random.default_rng(0)
        obs_index = pd.date_range("2023-01-01", periods=n_obs, freq="B")
        raw_returns = rng.normal(0.001, 0.02, n_obs)

        bt = types.SimpleNamespace(
            weights_per_observation=pd.DataFrame(
                {"TICK_00": [0.5] * n_obs, "TICK_01": [0.5] * n_obs},
                index=obs_index,
            ),
            returns=raw_returns,
            observations=obs_index,
            sharpe_ratio=0.75,
        )
        if include_returns_df:
            bt.returns_df = pd.Series(raw_returns, index=obs_index, name="returns")
        return bt

    def test_fallback_reconstruction_when_returns_df_absent(
        self, prices_df: pd.DataFrame
    ) -> None:
        """Without returns_df, gross returns are reconstructed from public members."""
        from unittest.mock import patch

        mock_bt = self._make_mock_bt(include_returns_df=False)
        assert not hasattr(mock_bt, "returns_df")

        with patch(
            "optimizer.pipeline._orchestrator.backtest",
            return_value=mock_bt,
        ):
            result = run_full_pipeline(
                prices=prices_df,
                optimizer=EqualWeighted(),
                cv_config=WalkForwardConfig(test_size=21, train_size=100),
            )

        assert result.net_returns is not None
        assert isinstance(result.net_returns, pd.Series)
        assert len(result.net_returns) == 50

    def test_fallback_issues_warning(
        self, prices_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Absent returns_df must trigger a logger.warning, not silent degradation."""
        import logging
        from unittest.mock import patch

        mock_bt = self._make_mock_bt(include_returns_df=False)

        with caplog.at_level(
            logging.WARNING, logger="optimizer.pipeline._orchestrator"
        ), patch(
            "optimizer.pipeline._orchestrator.backtest",
            return_value=mock_bt,
        ):
            run_full_pipeline(
                prices=prices_df,
                optimizer=EqualWeighted(),
                cv_config=WalkForwardConfig(test_size=21, train_size=100),
            )

        assert any(
            "returns_df" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_returns_df_preferred_when_present(
        self, prices_df: pd.DataFrame
    ) -> None:
        """When bt has returns_df, it is used directly without fallback."""
        from unittest.mock import patch

        mock_bt = self._make_mock_bt(include_returns_df=True)
        assert hasattr(mock_bt, "returns_df")

        with patch(
            "optimizer.pipeline._orchestrator.backtest",
            return_value=mock_bt,
        ):
            result = run_full_pipeline(
                prices=prices_df,
                optimizer=EqualWeighted(),
                cv_config=WalkForwardConfig(test_size=21, train_size=100),
            )

        assert result.net_returns is not None
        assert isinstance(result.net_returns, pd.Series)

    def test_elif_branch_emits_warning_not_debug(
        self, prices_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """bt without weights_per_observation emits WARNING (issue #309, fix 2)."""
        import logging
        import types
        from unittest.mock import patch

        # A bt without weights_per_observation triggers the elif branch
        mock_bt = types.SimpleNamespace(sharpe_ratio=0.5)

        with caplog.at_level(
            logging.WARNING, logger="optimizer.pipeline._orchestrator"
        ), patch(
            "optimizer.pipeline._orchestrator.backtest",
            return_value=mock_bt,
        ):
            run_full_pipeline(
                prices=prices_df,
                optimizer=EqualWeighted(),
                cv_config=WalkForwardConfig(test_size=21, train_size=100),
            )

        assert any(
            "weights_per_observation" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )
