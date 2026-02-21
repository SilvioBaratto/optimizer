"""Tests for build_preselection_pipeline factory and PreSelectionConfig."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 50 assets and 200 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 200, 50
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2023-01-01", periods=n_obs, freq="B"),
    )


class TestPreSelectionConfig:
    def test_default_config(self) -> None:
        cfg = PreSelectionConfig()
        assert cfg.max_abs_return == 10.0
        assert cfg.winsorize_threshold == 3.0
        assert cfg.correlation_threshold == 0.95
        assert cfg.top_k is None
        assert cfg.use_pareto is False

    def test_frozen(self) -> None:
        cfg = PreSelectionConfig()
        with pytest.raises(AttributeError):
            cfg.max_abs_return = 5.0  # type: ignore[misc]

    def test_for_daily_annual(self) -> None:
        cfg = PreSelectionConfig.for_daily_annual()
        assert cfg.max_abs_return == 10.0
        assert cfg.is_log_normal is True

    def test_for_conservative(self) -> None:
        cfg = PreSelectionConfig.for_conservative()
        assert cfg.max_abs_return == 5.0
        assert cfg.correlation_threshold == 0.85
        assert cfg.top_k == 50


class TestBuildPreselectionPipeline:
    def test_returns_pipeline(self) -> None:
        pipe = build_preselection_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_default_steps(self) -> None:
        pipe = build_preselection_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert step_names == [
            "validate",
            "outliers",
            "impute",
            "select_complete",
            "drop_zero_variance",
            "drop_correlated",
        ]

    def test_top_k_step_added(self) -> None:
        cfg = PreSelectionConfig(top_k=20)
        pipe = build_preselection_pipeline(cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_k" in step_names

    def test_pareto_step_added(self) -> None:
        cfg = PreSelectionConfig(use_pareto=True)
        pipe = build_preselection_pipeline(cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_pareto" in step_names

    def test_non_expiring_step_added(self) -> None:
        cfg = PreSelectionConfig(use_non_expiring=True, expiration_lookahead=90)
        pipe = build_preselection_pipeline(cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_non_expiring" in step_names

    def test_non_expiring_requires_lookahead(self) -> None:
        cfg = PreSelectionConfig(use_non_expiring=True)
        pipe = build_preselection_pipeline(cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_non_expiring" not in step_names

    def test_get_params_accessible(self) -> None:
        pipe = build_preselection_pipeline()
        params = pipe.get_params()
        assert "outliers__winsorize_threshold" in params
        assert "validate__max_abs_return" in params
        assert "drop_correlated__threshold" in params

    def test_set_params(self) -> None:
        pipe = build_preselection_pipeline()
        pipe.set_params(outliers__winsorize_threshold=2.5)
        assert pipe.named_steps["outliers"].winsorize_threshold == 2.5

    def test_fit_transform_reduces_or_preserves(self, returns_df: pd.DataFrame) -> None:
        pipe = build_preselection_pipeline()
        out = pipe.fit_transform(returns_df)
        assert isinstance(out, pd.DataFrame)
        # Output should have same or fewer columns
        assert out.shape[1] <= returns_df.shape[1]
        # Output should have same number of rows
        assert out.shape[0] == returns_df.shape[0]

    def test_with_sector_mapping(self, returns_df: pd.DataFrame) -> None:
        mapping = {col: f"Sector_{i % 5}" for i, col in enumerate(returns_df.columns)}
        pipe = build_preselection_pipeline(sector_mapping=mapping)
        out = pipe.fit_transform(returns_df)
        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == returns_df.shape[0]

    def test_with_correlated_assets(self) -> None:
        """Pipeline should drop one of two perfectly correlated assets."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.02, 200)
        df = pd.DataFrame(
            {
                "A": base,
                "B": base,  # perfect copy
                "C": rng.normal(0, 0.02, 200),
            },
            index=pd.date_range("2023-01-01", periods=200, freq="B"),
        )
        cfg = PreSelectionConfig(correlation_threshold=0.99)
        pipe = build_preselection_pipeline(cfg)
        out = pipe.fit_transform(df)
        # Should have dropped one of A/B
        assert out.shape[1] == 2

    def test_with_config_none(self, returns_df: pd.DataFrame) -> None:
        pipe = build_preselection_pipeline(config=None)
        out = pipe.fit_transform(returns_df)
        assert isinstance(out, pd.DataFrame)


class TestSkfolioIntegration:
    """Smoke tests with skfolio's built-in dataset."""

    def test_sp500_dataset(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)
        cfg = PreSelectionConfig(correlation_threshold=0.90)
        pipe = build_preselection_pipeline(cfg)
        clean = pipe.fit_transform(returns)

        assert isinstance(clean, pd.DataFrame)
        assert clean.shape[1] <= returns.shape[1]
        assert clean.shape[0] == returns.shape[0]
        assert not clean.isna().any().any()
