"""Tests for pipeline builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import EqualWeighted, HierarchicalRiskParity, MeanRisk
from sklearn.pipeline import Pipeline

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import build_portfolio_pipeline
from optimizer.pre_selection import PreSelectionConfig


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 15 assets and 300 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 300, 15
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2022-01-01", periods=n_obs, freq="B"),
    )


class TestBuildPortfolioPipeline:
    def test_returns_pipeline(self) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        assert isinstance(pipe, Pipeline)

    def test_has_optimizer_step(self) -> None:
        pipe = build_portfolio_pipeline(MeanRisk())
        step_names = [name for name, _ in pipe.steps]
        assert "optimizer" in step_names

    def test_has_preselection_steps(self) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        step_names = [name for name, _ in pipe.steps]
        assert "validate" in step_names
        assert "select_complete" in step_names

    def test_custom_preselection(self) -> None:
        cfg = PreSelectionConfig(correlation_threshold=0.80, top_k=10)
        pipe = build_portfolio_pipeline(EqualWeighted(), pre_selection_config=cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_k" in step_names

    def test_get_params_exposes_nested(self) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        pipe = build_portfolio_pipeline(optimizer)
        params = pipe.get_params()
        assert "optimizer__risk_aversion" in params
        assert "validate__max_abs_return" in params
        assert "drop_correlated__threshold" in params

    def test_fit_predict(self, returns_df: pd.DataFrame) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
        pipe = build_portfolio_pipeline(optimizer)
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert hasattr(portfolio, "weights")
        assert hasattr(portfolio, "sharpe_ratio")
        assert len(portfolio.weights) > 0

    def test_with_hrp(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(HierarchicalRiskParity())
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert portfolio.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sector_mapping(self, returns_df: pd.DataFrame) -> None:
        mapping = dict.fromkeys(returns_df.columns, "Tech")
        pipe = build_portfolio_pipeline(
            EqualWeighted(),
            sector_mapping=mapping,
        )
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert len(portfolio.weights) > 0
