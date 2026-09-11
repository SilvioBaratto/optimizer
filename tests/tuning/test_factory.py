"""Tests for tuning factory functions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MeanRisk
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from optimizer.scoring import ScorerConfig
from optimizer.tuning import (
    GridSearchConfig,
    RandomizedSearchConfig,
    build_grid_search_cv,
    build_randomized_search_cv,
    search_results_dataframe,
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


class TestResilienceKnobsWiring:
    def test_grid_wires_error_score_refit_verbose(self) -> None:
        cfg = GridSearchConfig(error_score="raise", refit=False, verbose=3)
        gs = build_grid_search_cv(MeanRisk(), {"l2_coef": [0.0]}, config=cfg)
        assert gs.error_score == "raise"
        assert gs.refit is False
        assert gs.verbose == 3

    def test_grid_default_error_score_is_nan(self) -> None:
        gs = build_grid_search_cv(MeanRisk(), {"l2_coef": [0.0]})
        assert math.isnan(gs.error_score)

    def test_randomized_wires_error_score_refit_verbose(self) -> None:
        cfg = RandomizedSearchConfig(error_score="raise", refit=False, verbose=2)
        rs = build_randomized_search_cv(MeanRisk(), {"l2_coef": [0.0]}, config=cfg)
        assert rs.error_score == "raise"
        assert rs.refit is False
        assert rs.verbose == 2


class TestScorerForwarding:
    def test_grid_forwards_custom_score_func(self) -> None:
        # A custom scorer config carries no ratio measure; the callable must
        # be threaded through the tuning factory to build_scorer.
        cfg = GridSearchConfig(scorer_config=ScorerConfig.for_custom())

        def _score(portfolio: object) -> float:
            return float(getattr(portfolio, "mean", 0.0))

        gs = build_grid_search_cv(
            MeanRisk(), {"l2_coef": [0.0]}, config=cfg, score_func=_score
        )
        assert gs.scoring is not None

    def test_grid_custom_without_score_func_raises(self) -> None:
        cfg = GridSearchConfig(scorer_config=ScorerConfig.for_custom())
        with pytest.raises(Exception, match="score_func"):
            build_grid_search_cv(MeanRisk(), {"l2_coef": [0.0]}, config=cfg)

    def test_grid_forwards_benchmark_returns_for_information_ratio(self) -> None:
        cfg = GridSearchConfig(scorer_config=ScorerConfig.for_information_ratio())
        bm = pd.Series(
            np.zeros(50),
            index=pd.date_range("2022-01-01", periods=50, freq="B"),
        )
        gs = build_grid_search_cv(
            MeanRisk(), {"l2_coef": [0.0]}, config=cfg, benchmark_returns=bm
        )
        assert gs.scoring is not None

    def test_information_ratio_without_benchmark_raises(self) -> None:
        cfg = RandomizedSearchConfig(scorer_config=ScorerConfig.for_information_ratio())
        with pytest.raises(Exception, match="benchmark_returns"):
            build_randomized_search_cv(MeanRisk(), {"l2_coef": [0.0]}, config=cfg)


class TestSearchResultsDataframe:
    def test_raises_before_fit(self) -> None:
        gs = build_grid_search_cv(MeanRisk(), {"l2_coef": [0.0, 0.1]})
        with pytest.raises(AttributeError, match="cv_results_"):
            search_results_dataframe(gs)

    def test_sorted_after_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = GridSearchConfig.for_quick_search()
        gs = build_grid_search_cv(MeanRisk(), {"l2_coef": [0.0, 0.05, 0.2]}, config=cfg)
        gs.fit(returns_df)
        df = search_results_dataframe(gs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "rank_test_score" in df.columns
        # Rank-sorted ascending: first row is the best (rank 1).
        assert df["rank_test_score"].iloc[0] == df["rank_test_score"].min()
