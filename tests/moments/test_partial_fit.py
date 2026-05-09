"""Round-trip equivalence tests for ``partial_fit`` on EW estimators.

For exponential-weighted estimators, two consecutive ``partial_fit``
calls must produce numerically identical state to a single full
``fit`` on the concatenated data. Smoke-only testing would hide
state-accumulation bugs that surface in production multi-batch usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import (
    EWCovariance,
    EWMu,
    EWVariance,
    RegimeAdjustedEWCovariance,
    RegimeAdjustedEWVariance,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_obs, n_assets = 200, 4
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
        columns=cols,
    )


@pytest.mark.parametrize(
    ("estimator_factory", "fitted_attr"),
    [
        (EWMu, "mu_"),
        (EWCovariance, "covariance_"),
        (EWVariance, "variance_"),
        (RegimeAdjustedEWVariance, "variance_"),
        (RegimeAdjustedEWCovariance, "covariance_"),
    ],
    ids=[
        "EWMu",
        "EWCovariance",
        "EWVariance",
        "RegimeAdjustedEWVariance",
        "RegimeAdjustedEWCovariance",
    ],
)
def test_when_two_partial_fits_then_matches_full_fit(
    estimator_factory: type,
    fitted_attr: str,
    returns: pd.DataFrame,
) -> None:
    """``partial_fit(X[:k]).partial_fit(X[k:])`` ≈ ``fit(X)`` within atol=1e-8."""
    n_obs = len(returns)
    k = n_obs // 2

    full = estimator_factory()
    full.fit(returns)

    incremental = estimator_factory()
    incremental.partial_fit(returns.iloc[:k])
    incremental.partial_fit(returns.iloc[k:])

    full_value = getattr(full, fitted_attr)
    incremental_value = getattr(incremental, fitted_attr)
    assert np.allclose(full_value, incremental_value, atol=1e-8)
