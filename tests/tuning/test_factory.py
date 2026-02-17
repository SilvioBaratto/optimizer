"""Tests for tuning factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skfolio.optimization import MeanRisk

from optimizer.tuning import (
    GridSearchConfig,
    RandomizedSearchConfig,
    build_grid_search_cv,
    build_randomized_search_cv,
)


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


class TestBuildGridSearchCV:
    def test_default_config(self) -> None:
        model = MeanRisk()
        grid = {"l2_coef": [0.0, 0.01, 0.1]}
        gs = build_grid_search_cv(model, grid)
        assert isinstance(gs, GridSearchCV)

    def test_custom_config(self) -> None:
        cfg = GridSearchConfig.for_quick_search()
        model = MeanRisk()
        grid = {"l2_coef": [0.0, 0.01]}
        gs = build_grid_search_cv(model, grid, config=cfg)
        assert isinstance(gs, GridSearchCV)

    def test_nested_params(self) -> None:
        model = MeanRisk()
        grid = {
            "l2_coef": [0.0, 0.01],
            "risk_aversion": [0.5, 1.0, 2.0],
        }
        gs = build_grid_search_cv(model, grid)
        assert isinstance(gs, GridSearchCV)


class TestBuildRandomizedSearchCV:
    def test_default_config(self) -> None:
        model = MeanRisk()
        dists = {"l2_coef": [0.0, 0.01, 0.1]}
        rs = build_randomized_search_cv(model, dists)
        assert isinstance(rs, RandomizedSearchCV)

    def test_custom_config(self) -> None:
        cfg = RandomizedSearchConfig.for_quick_search(n_iter=5)
        model = MeanRisk()
        dists = {"l2_coef": [0.0, 0.01, 0.1]}
        rs = build_randomized_search_cv(model, dists, config=cfg)
        assert isinstance(rs, RandomizedSearchCV)
