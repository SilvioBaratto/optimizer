"""Tests for online covariance forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import CovarianceForecastComparison
from skfolio.moments import (
    EWCovariance,
    LedoitWolf,
    RegimeAdjustedEWCovariance,
)

from optimizer.exceptions import ConfigurationError
from optimizer.validation import (
    OnlineCovarianceForecastConfig,
    run_online_covariance_forecast_evaluation,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_obs, n_assets = 600, 5
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
        columns=cols,
        index=pd.bdate_range("2020-01-01", periods=n_obs, freq="B"),
    )


def test_when_partial_fit_estimators_then_comparison_returned(returns) -> None:
    estimators = [
        ("ew_40", EWCovariance(half_life=40)),
        ("regime_ew", RegimeAdjustedEWCovariance(half_life=23, corr_half_life=50)),
    ]
    cfg = OnlineCovarianceForecastConfig(warmup_size=252, test_size=21)
    result = run_online_covariance_forecast_evaluation(estimators, returns, config=cfg)
    assert isinstance(result, CovarianceForecastComparison)
    assert set(result.names) == {"ew_40", "regime_ew"}


def test_when_estimator_lacks_partial_fit_then_raises(returns) -> None:
    estimators = [
        ("ew_40", EWCovariance(half_life=40)),
        ("ledoit_wolf", LedoitWolf()),
    ]
    cfg = OnlineCovarianceForecastConfig(warmup_size=252, test_size=21)
    with pytest.raises(ConfigurationError, match="ledoit_wolf"):
        run_online_covariance_forecast_evaluation(estimators, returns, config=cfg)


def test_when_default_config_then_warmup_252(returns) -> None:
    cfg = OnlineCovarianceForecastConfig()
    assert cfg.warmup_size == 252
    assert cfg.test_size == 1


def test_when_config_none_then_default_used(returns) -> None:
    estimators = [("ew_40", EWCovariance(half_life=40))]
    result = run_online_covariance_forecast_evaluation(estimators, returns, config=None)
    assert isinstance(result, CovarianceForecastComparison)
