"""Online searches must not crown a candidate nobody scored, nor skew the rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior

from optimizer.exceptions import DataError, OptimizationError
from optimizer.online import (
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
    build_online_grid_search,
    build_online_randomized_search,
    run_online_predict,
    run_online_score,
)
from optimizer.tuning._config import RandomizedSearchConfig

_ONLINE = OnlinePredictConfig(warmup_size=252, test_size=21, n_jobs=1)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(400, 8)),
        columns=[f"T{i}" for i in range(8)],
        index=pd.bdate_range("2023-01-02", periods=400),
    )


def _estimator(*, raise_on_failure: bool = True, risk_free_rate: float = 0.0):
    return MeanRisk(
        risk_free_rate=risk_free_rate,
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40),
            covariance_estimator=EWCovariance(half_life=40),
        ),
        raise_on_failure=raise_on_failure,
    )


# 8 assets with a 20% floor each is infeasible, so with raise_on_failure=False
# every candidate fails on every fold.
_INFEASIBLE_GRID = {"min_weights": [0.2, 0.3]}


class TestEveryCandidateScoredNaN:
    def test_grid_search_raises_instead_of_picking_the_first_candidate(
        self, returns: pd.DataFrame
    ) -> None:
        search = build_online_grid_search(
            OnlineGridSearchConfig(online=_ONLINE),
            _estimator(raise_on_failure=False),
            _INFEASIBLE_GRID,
        )

        with pytest.raises(OptimizationError, match="every candidate scored NaN"):
            search.fit(returns)

    def test_randomized_search_raises_instead_of_picking_the_first_candidate(
        self, returns: pd.DataFrame
    ) -> None:
        cfg = OnlineRandomizedSearchConfig(
            base=RandomizedSearchConfig(n_iter=2, random_state=0),
            online=_ONLINE,
        )
        search = build_online_randomized_search(
            cfg, _estimator(raise_on_failure=False), _INFEASIBLE_GRID
        )

        with pytest.raises(OptimizationError, match="every candidate scored NaN"):
            search.fit(returns)

    def test_a_single_scored_candidate_still_wins(self, returns: pd.DataFrame) -> None:
        # 0.2 per asset is infeasible over 8 assets, 0.05 is not.
        search = build_online_grid_search(
            OnlineGridSearchConfig(online=_ONLINE),
            _estimator(raise_on_failure=False),
            {"min_weights": [0.2, 0.05]},
        )

        search.fit(returns)

        assert search.best_params_ == {"min_weights": 0.05}


class TestScorerRiskFreeRateAppliesToEveryCandidate:
    def test_candidates_with_identical_weights_score_identically(
        self, returns: pd.DataFrame
    ) -> None:
        # A minimum-risk objective ignores the optimizer's risk_free_rate, so the
        # two candidates hold the same weights. The scorer's rate (0.0) must be
        # applied to both, not forced only when it is non-zero.
        search = build_online_grid_search(
            OnlineGridSearchConfig(online=_ONLINE),
            _estimator(),
            {"risk_free_rate": [0.0, 0.0004]},
        )

        search.fit(returns)

        scores = np.asarray(search.cv_results_["mean_score"])
        assert scores[0] == pytest.approx(scores[1])


class TestBenchmarkIsPairedByDate:
    def test_predict_rejects_a_benchmark_on_another_row_order(
        self, returns: pd.DataFrame
    ) -> None:
        benchmark = returns.mean(axis=1).sample(frac=1.0, random_state=0)

        with pytest.raises(DataError, match="y"):
            run_online_predict(_estimator(), returns, benchmark, config=_ONLINE)

    def test_score_rejects_a_benchmark_on_another_row_order(
        self, returns: pd.DataFrame
    ) -> None:
        from skfolio.measures import RatioMeasure

        benchmark = returns.mean(axis=1).sample(frac=1.0, random_state=0)

        with pytest.raises(DataError, match="y"):
            run_online_score(
                _estimator(),
                returns,
                benchmark,
                scorer=RatioMeasure.SHARPE_RATIO,
                config=_ONLINE,
            )
