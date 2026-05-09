"""Smoke tests for partial_fit on EW priors and regime estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import (
    EWCovariance,
    EWMu,
    RegimeAdjustedEWCovariance,
)
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_obs, n_assets = 250, 5
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
        columns=cols,
    )


def test_when_meanrisk_with_ew_prior_partial_fit_two_batches_then_fits(
    returns: pd.DataFrame,
) -> None:
    prior = EmpiricalPrior(
        mu_estimator=EWMu(half_life=20.0),
        covariance_estimator=EWCovariance(half_life=20.0),
    )
    est = MeanRisk(prior_estimator=prior)
    est.partial_fit(returns.iloc[:120])
    est.partial_fit(returns.iloc[120:])
    assert est.weights_.shape == (returns.shape[1],)
    assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)


def test_when_meanrisk_with_regime_adjusted_partial_fit_two_batches_then_fits(
    returns: pd.DataFrame,
) -> None:
    prior = EmpiricalPrior(
        mu_estimator=EWMu(half_life=20.0),
        covariance_estimator=RegimeAdjustedEWCovariance(half_life=20.0),
    )
    est = MeanRisk(prior_estimator=prior)
    est.partial_fit(returns.iloc[:120])
    est.partial_fit(returns.iloc[120:])
    assert est.weights_.shape == (returns.shape[1],)
    assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)
