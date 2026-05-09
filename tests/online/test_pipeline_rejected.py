"""Reject Pipeline inputs at the online wrapper boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.online import (
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
    build_online_grid_search,
    build_online_randomized_search,
    run_online_predict,
    run_online_score,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_obs, n_assets = 200, 5
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
        columns=cols,
    )


@pytest.fixture
def pipeline_estimator() -> Pipeline:
    return Pipeline(
        [
            (
                "mr",
                MeanRisk(
                    prior_estimator=EmpiricalPrior(
                        mu_estimator=EWMu(),
                        covariance_estimator=EWCovariance(),
                    )
                ),
            )
        ]
    )


def test_when_pipeline_passed_to_online_predict_then_raises(
    pipeline_estimator: Pipeline,
    returns: pd.DataFrame,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=100)
    with pytest.raises(ConfigurationError, match="partial_fit"):
        run_online_predict(pipeline_estimator, returns, None, config=cfg)


def test_when_pipeline_passed_to_online_score_then_raises(
    pipeline_estimator: Pipeline,
    returns: pd.DataFrame,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=100)
    with pytest.raises(ConfigurationError, match="partial_fit"):
        run_online_score(
            pipeline_estimator,
            returns,
            None,
            scorer=None,
            config=cfg,
        )


def test_when_pipeline_passed_to_online_grid_search_then_raises(
    pipeline_estimator: Pipeline,
) -> None:
    cfg = OnlineGridSearchConfig()
    with pytest.raises(ConfigurationError, match="partial_fit"):
        build_online_grid_search(cfg, pipeline_estimator, {"foo": [1]})


def test_when_pipeline_passed_to_online_randomized_search_then_raises(
    pipeline_estimator: Pipeline,
) -> None:
    cfg = OnlineRandomizedSearchConfig()
    with pytest.raises(ConfigurationError, match="partial_fit"):
        build_online_randomized_search(cfg, pipeline_estimator, {"foo": [1]})


def test_when_estimator_lacks_partial_fit_then_raises(
    returns: pd.DataFrame,
) -> None:
    class Dummy:
        pass

    cfg = OnlinePredictConfig(warmup_size=100)
    with pytest.raises(ConfigurationError, match="partial_fit"):
        run_online_predict(Dummy(), returns, None, config=cfg)
