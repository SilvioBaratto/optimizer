"""Tests for offline covariance forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import CovarianceForecastComparison
from skfolio.moments import EmpiricalCovariance, EWCovariance, LedoitWolf

from optimizer.validation import (
    CovarianceForecastConfig,
    run_covariance_forecast_evaluation,
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


def test_when_three_estimators_then_comparison_returned(returns) -> None:
    estimators = [
        ("empirical", EmpiricalCovariance()),
        ("ledoit_wolf", LedoitWolf()),
        ("ew_40", EWCovariance(half_life=40)),
    ]
    cfg = CovarianceForecastConfig(train_size=252, test_size=21)
    result = run_covariance_forecast_evaluation(estimators, returns, config=cfg)
    assert isinstance(result, CovarianceForecastComparison)
    assert set(result.names) == {"empirical", "ledoit_wolf", "ew_40"}


def test_when_summary_then_ranks_estimators(returns) -> None:
    estimators = [
        ("empirical", EmpiricalCovariance()),
        ("ledoit_wolf", LedoitWolf()),
    ]
    cfg = CovarianceForecastConfig(train_size=252, test_size=21)
    result = run_covariance_forecast_evaluation(estimators, returns, config=cfg)
    summary = result.summary()
    assert summary is not None
    assert len(summary) > 0


def test_when_run_twice_with_same_config_then_results_match(returns) -> None:
    estimators = [
        ("ledoit_wolf", LedoitWolf()),
    ]
    cfg = CovarianceForecastConfig(train_size=252, test_size=21)
    a = run_covariance_forecast_evaluation(estimators, returns, config=cfg)
    b = run_covariance_forecast_evaluation(
        [("ledoit_wolf", LedoitWolf())], returns, config=cfg
    )
    # Determinism: same input + same config → same names and bias_statistic.
    assert a.names == b.names
    np.testing.assert_allclose(
        a.bias_statistic_summary().values,
        b.bias_statistic_summary().values,
    )


def test_when_default_config_then_train_size_252(returns) -> None:
    cfg = CovarianceForecastConfig()
    assert cfg.train_size == 252
    assert cfg.test_size == 1
    assert cfg.expand_train is False


def test_when_config_none_then_default_used(returns) -> None:
    estimators = [("ledoit_wolf", LedoitWolf())]
    result = run_covariance_forecast_evaluation(estimators, returns, config=None)
    assert isinstance(result, CovarianceForecastComparison)
