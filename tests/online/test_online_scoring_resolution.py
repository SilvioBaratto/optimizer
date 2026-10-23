"""Online searches must pass a BaseMeasure, not a make_scorer object.

skfolio 1.0.6 online portfolio evaluation rejects ``make_scorer`` results
with a ``TypeError`` at ``.fit()``. The wrappers therefore resolve the
``ScorerConfig`` to a skfolio ``BaseMeasure`` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import BaseMeasure, RatioMeasure
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior

from optimizer.exceptions import ConfigurationError
from optimizer.online import (
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
    build_online_grid_search,
    build_online_randomized_search,
)
from optimizer.online._factory import _resolve_measure
from optimizer.optimization._config import RatioMeasureType
from optimizer.scoring._config import ScorerConfig
from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(300, 5)),
        columns=[f"A{i:02d}" for i in range(5)],
    )


@pytest.fixture
def estimator() -> MeanRisk:
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40.0),
            covariance_estimator=EWCovariance(half_life=40.0),
        )
    )


def test_when_grid_search_built_then_scoring_is_base_measure(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineGridSearchConfig(
        base=OnlineGridSearchConfig().base,
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_grid_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert isinstance(search.scoring, BaseMeasure)


def test_when_grid_search_fit_then_succeeds_with_measure(
    estimator: MeanRisk,
    returns: pd.DataFrame,
) -> None:
    cfg = OnlineGridSearchConfig(
        base=OnlineGridSearchConfig().base,
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_grid_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    # Regression: previously raised TypeError (make_scorer) at fit time.
    search.fit(returns)
    assert search.best_params_["prior_estimator__mu_estimator__half_life"] in (
        20.0,
        40.0,
    )


def test_when_randomized_search_built_then_scoring_is_base_measure(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineRandomizedSearchConfig(
        base=OnlineRandomizedSearchConfig().base,
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_randomized_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert isinstance(search.scoring, BaseMeasure)


def test_when_resolve_sortino_then_returns_matching_measure() -> None:
    measure = _resolve_measure(ScorerConfig.for_sortino())
    assert measure == RatioMeasure.SORTINO_RATIO


def test_when_resolve_custom_none_ratio_then_raises() -> None:
    with pytest.raises(ConfigurationError, match="ratio measure"):
        _resolve_measure(ScorerConfig(ratio_measure=None))


def test_when_resolve_information_ratio_then_raises() -> None:
    with pytest.raises(ConfigurationError, match="Information Ratio"):
        _resolve_measure(ScorerConfig(ratio_measure=RatioMeasureType.INFORMATION_RATIO))


def _online_cfg(scorer_config: ScorerConfig) -> OnlineGridSearchConfig:
    return OnlineGridSearchConfig(
        base=GridSearchConfig(scorer_config=scorer_config),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )


def test_when_grid_scorer_has_rf_and_annualization_then_portfolio_params_set(
    estimator: MeanRisk,
) -> None:
    cfg = _online_cfg(
        ScorerConfig(
            ratio_measure=RatioMeasureType.ANNUALIZED_SHARPE_RATIO,
            risk_free_rate=0.01,
            annualization_factor=365.0,
        )
    )
    search = build_online_grid_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert search.portfolio_params == {
        "risk_free_rate": 0.01,
        "annualization_factor": 365.0,
    }


def test_when_randomized_scorer_has_rf_and_annualization_then_portfolio_params_set(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineRandomizedSearchConfig(
        base=RandomizedSearchConfig(
            scorer_config=ScorerConfig(risk_free_rate=0.01, annualization_factor=12.0)
        ),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_randomized_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert search.portfolio_params == {
        "risk_free_rate": 0.01,
        "annualization_factor": 12.0,
    }


def test_when_scorer_uses_defaults_then_portfolio_params_left_unset(
    estimator: MeanRisk,
) -> None:
    # Leaving portfolio_params unset lets skfolio inherit the optimizer's own
    # risk_free_rate instead of forcing 0.0 over it.
    search = build_online_grid_search(
        _online_cfg(ScorerConfig()),
        estimator,
        {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]},
    )
    assert search.portfolio_params is None


def test_when_scorer_sets_risk_free_rate_then_search_scores_change(
    estimator: MeanRisk,
    returns: pd.DataFrame,
) -> None:
    grid = {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    base = build_online_grid_search(_online_cfg(ScorerConfig()), estimator, grid)
    base.fit(returns)
    hurdle = build_online_grid_search(
        _online_cfg(ScorerConfig(risk_free_rate=0.01)), estimator, grid
    )
    hurdle.fit(returns)
    # A 1%-per-period hurdle dwarfs the ~0.05% mean return, so every Sharpe
    # score must drop.
    assert np.all(
        np.asarray(hurdle.cv_results_["mean_score"])
        < np.asarray(base.cv_results_["mean_score"])
    )
