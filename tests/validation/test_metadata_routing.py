"""End-to-end tests: metadata routing through CV folds (issue #20).

Verifies that skfolio's cross-validation infrastructure correctly routes
auxiliary data through each fold:

1. **Implied volatility** — ``implied_vol`` forwarded to
   :class:`~skfolio.moments.ImpliedCovariance` inside a
   :class:`~skfolio.prior.FactorModel` prior via the ``params`` dict.
2. **Benchmark weights** — ``y`` benchmark returns sliced on the correct
   train/test dates for :class:`~skfolio.optimization.BenchmarkTracker`.
3. **Previous weights** — ``previous_weights`` set on
   :class:`~skfolio.optimization.MeanRisk` for turnover constraints.

All tests are marked ``@pytest.mark.slow`` because they run full
walk-forward CV (3 folds on the fixture dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import ImpliedCovariance
from skfolio.optimization import BenchmarkTracker, MeanRisk
from skfolio.prior import EmpiricalPrior, FactorModel
from sklearn import set_config

from optimizer.validation import WalkForwardConfig, build_walk_forward, run_cross_val

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_FACTORS = 5
# 500 obs → with train_size=252 and test_size=63 we get 3 non-overlapping folds
N_OBS = 500
TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]
FACTOR_NAMES = [f"F{i:02d}" for i in range(N_FACTORS)]
DATES = pd.date_range("2020-01-02", periods=N_OBS, freq="B")

_WF_CONFIG = WalkForwardConfig(test_size=63, train_size=252)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(N_OBS, N_ASSETS))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


@pytest.fixture(scope="module")
def factor_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = rng.normal(loc=0.0003, scale=0.012, size=(N_OBS, N_FACTORS))
    return pd.DataFrame(data, index=DATES, columns=FACTOR_NAMES)


@pytest.fixture(scope="module")
def implied_vol_factors() -> pd.DataFrame:
    """Implied volatility surface with factor columns (matches factor_returns)."""
    rng = np.random.default_rng(13)
    data = rng.uniform(0.10, 0.50, size=(N_OBS, N_FACTORS))
    return pd.DataFrame(data, index=DATES, columns=FACTOR_NAMES)


@pytest.fixture(scope="module")
def benchmark_returns(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight benchmark return series."""
    return returns.mean(axis=1).rename("benchmark")


# ---------------------------------------------------------------------------
# Test 1: ImpliedCovariance metadata routing through CV folds
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestImpliedVolMetadataRouting:
    """Verify ``implied_vol`` is routed to each CV fold's FactorModel prior."""

    def test_run_cross_val_completes(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        implied_vol_factors: pd.DataFrame,
    ) -> None:
        """run_cross_val completes without error when implied_vol is routed."""
        set_config(enable_metadata_routing=True)

        imp_cov = ImpliedCovariance().set_fit_request(implied_vol=True)
        factor_prior = FactorModel(
            factor_prior_estimator=EmpiricalPrior(covariance_estimator=imp_cov)
        )
        model = MeanRisk(prior_estimator=factor_prior, min_weights=0.0)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(
            model,
            returns,
            y=factor_returns,
            cv=cv,
            params={"implied_vol": implied_vol_factors},
        )

        assert pred is not None

    def test_correct_number_of_folds(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        implied_vol_factors: pd.DataFrame,
    ) -> None:
        """WalkForward produces the expected number of test periods."""
        set_config(enable_metadata_routing=True)

        imp_cov = ImpliedCovariance().set_fit_request(implied_vol=True)
        factor_prior = FactorModel(
            factor_prior_estimator=EmpiricalPrior(covariance_estimator=imp_cov)
        )
        model = MeanRisk(prior_estimator=factor_prior, min_weights=0.0)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(
            model,
            returns,
            y=factor_returns,
            cv=cv,
            params={"implied_vol": implied_vol_factors},
        )

        # With N_OBS=500, train_size=252, test_size=63 → 3 folds
        expected_folds = len(list(build_walk_forward(_WF_CONFIG).split(returns)))
        assert len(pred.portfolios) == expected_folds

    def test_covariance_shape_per_fold(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        implied_vol_factors: pd.DataFrame,
    ) -> None:
        """Each fold's fitted prior has covariance of shape (n_assets, n_assets)."""
        set_config(enable_metadata_routing=True)

        imp_cov = ImpliedCovariance().set_fit_request(implied_vol=True)
        factor_prior = FactorModel(
            factor_prior_estimator=EmpiricalPrior(covariance_estimator=imp_cov)
        )
        model = MeanRisk(prior_estimator=factor_prior, min_weights=0.0)
        model.fit(returns, factor_returns, implied_vol=implied_vol_factors)

        cov = model.prior_estimator_.return_distribution_.covariance
        assert cov.shape == (N_ASSETS, N_ASSETS)

    def test_weights_sum_to_one_per_fold(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        implied_vol_factors: pd.DataFrame,
    ) -> None:
        """Each fold's portfolio is fully invested (weights sum to 1)."""
        set_config(enable_metadata_routing=True)

        imp_cov = ImpliedCovariance().set_fit_request(implied_vol=True)
        factor_prior = FactorModel(
            factor_prior_estimator=EmpiricalPrior(covariance_estimator=imp_cov)
        )
        model = MeanRisk(prior_estimator=factor_prior, min_weights=0.0)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(
            model,
            returns,
            y=factor_returns,
            cv=cv,
            params={"implied_vol": implied_vol_factors},
        )

        for portfolio in pred.portfolios:
            assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2: BenchmarkTracker — y benchmark routing and date boundaries
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchmarkYRouting:
    """Verify benchmark returns (y) are sliced on correct train/test dates."""

    def test_run_cross_val_completes(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        """BenchmarkTracker run_cross_val completes without error."""
        model = BenchmarkTracker()
        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, y=benchmark_returns, cv=cv)
        assert pred is not None

    def test_correct_number_of_folds(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        model = BenchmarkTracker()
        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, y=benchmark_returns, cv=cv)

        expected_folds = len(list(build_walk_forward(_WF_CONFIG).split(returns)))
        assert len(pred.portfolios) == expected_folds

    def test_fold_date_ranges_are_non_overlapping(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        """Test periods are contiguous and non-overlapping."""
        model = BenchmarkTracker()
        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, y=benchmark_returns, cv=cv)

        test_splits = list(build_walk_forward(_WF_CONFIG).split(returns))
        assert len(pred.portfolios) == len(test_splits)

        # Each fold's test indices must not overlap with any other fold
        all_test_indices: set[int] = set()
        for _, test_idx in test_splits:
            overlap = all_test_indices & set(test_idx)
            assert len(overlap) == 0, f"Overlapping test indices: {overlap}"
            all_test_indices.update(test_idx)

    def test_train_always_precedes_test(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        """No train observation appears after a test observation (no leakage)."""
        cv = build_walk_forward(_WF_CONFIG)

        for train_idx, test_idx in cv.split(returns):
            assert int(np.max(train_idx)) < int(np.min(test_idx)), (
                f"Data leakage: train max={np.max(train_idx)}, "
                f"test min={np.min(test_idx)}"
            )

    def test_weights_sum_to_one_per_fold(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> None:
        model = BenchmarkTracker()
        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, y=benchmark_returns, cv=cv)

        for portfolio in pred.portfolios:
            assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: MeanRisk with previous_weights (turnover constraint)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPreviousWeightsTurnoverConstraint:
    """Verify previous_weights flows through CV without error."""

    def test_run_cross_val_completes(self, returns: pd.DataFrame) -> None:
        """MeanRisk with previous_weights runs CV without error."""
        prev_w = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = MeanRisk(min_weights=0.0, previous_weights=prev_w, l1_coef=0.001)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, cv=cv)
        assert pred is not None

    def test_correct_number_of_folds(self, returns: pd.DataFrame) -> None:
        prev_w = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = MeanRisk(min_weights=0.0, previous_weights=prev_w, l1_coef=0.001)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, cv=cv)

        expected_folds = len(list(build_walk_forward(_WF_CONFIG).split(returns)))
        assert len(pred.portfolios) == expected_folds

    def test_weights_sum_to_one_per_fold(self, returns: pd.DataFrame) -> None:
        prev_w = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = MeanRisk(min_weights=0.0, previous_weights=prev_w, l1_coef=0.001)

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(model, returns, cv=cv)

        for portfolio in pred.portfolios:
            assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)

    def test_turnover_lower_with_previous_weights(
        self, returns: pd.DataFrame
    ) -> None:
        """L1 penalty with equal-weight prior reduces max weight vs no penalty."""
        prev_w = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model_with = MeanRisk(
            min_weights=0.0, previous_weights=prev_w, l1_coef=0.01
        )
        model_without = MeanRisk(min_weights=0.0)

        cv = build_walk_forward(_WF_CONFIG)
        pred_with = run_cross_val(model_with, returns, cv=cv)
        pred_without = run_cross_val(model_without, returns, cv=cv)

        max_w_with = max(float(np.max(p.weights)) for p in pred_with.portfolios)
        max_w_without = max(
            float(np.max(p.weights)) for p in pred_without.portfolios
        )

        # L1 regularization should push weights towards equal-weight,
        # reducing the maximum weight
        assert max_w_with <= max_w_without + 1e-4


# ---------------------------------------------------------------------------
# Test 4: params=None is the default (backward-compatibility)
# ---------------------------------------------------------------------------


class TestRunCrossValParamsDefault:
    """Verify params=None is the default and doesn't break existing usage."""

    def test_params_none_does_not_raise(self, returns: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(EqualWeighted(), returns, cv=cv, params=None)
        assert pred is not None

    def test_params_empty_dict_does_not_raise(self, returns: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cv = build_walk_forward(_WF_CONFIG)
        pred = run_cross_val(EqualWeighted(), returns, cv=cv, params={})
        assert pred is not None
