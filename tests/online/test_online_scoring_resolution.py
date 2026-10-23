"""Online searches must pass a BaseMeasure, not a make_scorer object.

skfolio 1.0.6 online portfolio evaluation rejects ``make_scorer`` results
with a ``TypeError`` at ``.fit()``. The wrappers therefore resolve the
``ScorerConfig`` to a skfolio ``BaseMeasure`` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cvxpy.error import SolverError
from skfolio.measures import BaseMeasure, PerfMeasure, RatioMeasure, RiskMeasure
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
from optimizer.optimization._config import RatioMeasureType, RiskMeasureType
from optimizer.scoring._config import PerfMeasureType, ScorerConfig
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


def test_when_grid_base_sets_error_score_refit_verbose_then_forwarded(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineGridSearchConfig(
        base=GridSearchConfig(error_score="raise", refit=False, verbose=5),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_grid_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert search.error_score == "raise"
    assert search.refit is False
    assert search.verbose == 5


def test_when_randomized_base_sets_error_score_refit_verbose_then_forwarded(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineRandomizedSearchConfig(
        base=RandomizedSearchConfig(error_score="raise", refit=False, verbose=3),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_randomized_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert search.error_score == "raise"
    assert search.refit is False
    assert search.verbose == 3


def test_when_online_verbose_set_then_it_still_enables_verbosity(
    estimator: MeanRisk,
) -> None:
    cfg = OnlineGridSearchConfig(
        base=GridSearchConfig(),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1, verbose=True),
    )
    search = build_online_grid_search(
        cfg, estimator, {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    )
    assert search.verbose == 1


def test_when_error_score_raise_and_candidate_infeasible_then_fit_raises(
    estimator: MeanRisk,
    returns: pd.DataFrame,
) -> None:
    # 5 assets capped at 0.1 each can never sum to a fully-invested portfolio.
    grid = {"max_weights": [1.0, 0.1]}
    cfg = OnlineGridSearchConfig(
        base=GridSearchConfig(error_score="raise"),
        online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
    )
    search = build_online_grid_search(cfg, estimator, grid)
    with pytest.raises(SolverError):
        search.fit(returns)


def test_when_resolve_risk_measure_then_returns_matching_measure() -> None:
    measure = _resolve_measure(ScorerConfig.for_risk_measure(RiskMeasureType.CVAR))
    assert measure is RiskMeasure.CVAR


def test_when_resolve_perf_measure_then_returns_matching_measure() -> None:
    measure = _resolve_measure(
        ScorerConfig.for_perf_measure(PerfMeasureType.ANNUALIZED_MEAN)
    )
    assert measure is PerfMeasure.ANNUALIZED_MEAN


@pytest.mark.parametrize(
    ("scorer_config", "expected"),
    [
        (ScorerConfig.for_risk_measure(), RiskMeasure.VARIANCE),
        (ScorerConfig.for_perf_measure(), PerfMeasure.MEAN),
    ],
)
def test_when_scorer_is_perf_or_risk_family_then_searches_accept_it(
    estimator: MeanRisk, scorer_config: ScorerConfig, expected: object
) -> None:
    grid = {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]}
    grid_search = build_online_grid_search(_online_cfg(scorer_config), estimator, grid)
    randomized = build_online_randomized_search(
        OnlineRandomizedSearchConfig(
            base=RandomizedSearchConfig(scorer_config=scorer_config),
            online=OnlinePredictConfig(warmup_size=150, n_jobs=1),
        ),
        estimator,
        grid,
    )
    assert grid_search.scoring is expected
    assert randomized.scoring is expected


def test_when_risk_measure_scorer_then_online_search_fits(
    estimator: MeanRisk,
    returns: pd.DataFrame,
) -> None:
    search = build_online_grid_search(
        _online_cfg(ScorerConfig.for_risk_measure()),
        estimator,
        {"prior_estimator__mu_estimator__half_life": [20.0, 40.0]},
    )
    search.fit(returns)
    # skfolio negates risk measures so that a higher score is always better.
    assert np.all(np.asarray(search.cv_results_["mean_score"]) <= 0.0)
