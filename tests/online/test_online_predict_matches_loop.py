"""``online_predict`` produces a MultiPeriodPortfolio for warmup_size=252."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio.measures import RatioMeasure
from skfolio.model_selection import (
    OnlineGridSearch,
    OnlineRandomizedSearch,
)
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior

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
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices).tail(400)


@pytest.fixture
def estimator() -> MeanRisk:
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40.0),
            covariance_estimator=EWCovariance(half_life=40.0),
        )
    )


def test_when_warmup_size_set_then_online_predict_returns_portfolio(
    estimator: MeanRisk,
    returns,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=252)
    result = run_online_predict(estimator, returns, None, config=cfg)
    # MultiPeriodPortfolio exposes ``returns_`` of length ``n_obs - warmup_size``.
    assert hasattr(result, "returns")
    assert len(result.returns) == len(returns) - cfg.warmup_size


def test_when_run_twice_with_same_config_then_results_match(
    estimator: MeanRisk,
    returns,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=252)
    a = run_online_predict(estimator, returns, None, config=cfg)
    # Use a fresh estimator instance — online state mutates.
    fresh = MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40.0),
            covariance_estimator=EWCovariance(half_life=40.0),
        )
    )
    b = run_online_predict(fresh, returns, None, config=cfg)
    assert np.allclose(a.returns, b.returns, atol=1e-8)


def test_when_run_online_score_then_returns_score(
    estimator: MeanRisk,
    returns,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=252)
    score = run_online_score(
        estimator,
        returns,
        None,
        scorer=RatioMeasure.SHARPE_RATIO,
        config=cfg,
    )
    assert isinstance(score, float)


def test_when_build_online_grid_search_then_returns_skfolio_class(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineGridSearchConfig()
    cfg = OnlineGridSearchConfig(
        base=cfg.base, online=OnlinePredictConfig(warmup_size=252, n_jobs=1)
    )
    search = build_online_grid_search(
        cfg,
        estimator,
        {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]},
    )
    assert isinstance(search, OnlineGridSearch)
    assert search.warmup_size == 252


def test_when_build_online_randomized_search_then_returns_skfolio_class(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineRandomizedSearchConfig()
    search = build_online_randomized_search(
        cfg,
        estimator,
        {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]},
    )
    assert isinstance(search, OnlineRandomizedSearch)
