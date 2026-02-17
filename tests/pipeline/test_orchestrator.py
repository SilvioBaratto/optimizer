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
    tune_and_optimize,
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
