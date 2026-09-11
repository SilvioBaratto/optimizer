"""Covariance-forecast evaluation wrappers (optimizer-independent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import CovarianceForecastEvaluation
from skfolio.moments import EWCovariance, RegimeAdjustedEWCovariance
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.online import (
    CovarianceForecastConfig,
    build_covariance_forecast_comparison,
    run_covariance_forecast_evaluation,
    run_online_covariance_forecast_evaluation,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(400, 4)),
        columns=list("ABCD"),
        index=idx,
    )


def test_when_walk_forward_evaluation_then_returns_evaluation(
    returns: pd.DataFrame,
) -> None:
    cfg = CovarianceForecastConfig(train_size=252)
    ev = run_covariance_forecast_evaluation(
        EWCovariance(half_life=40.0), returns, config=cfg
    )
    assert isinstance(ev, CovarianceForecastEvaluation)
    summary = ev.summary()
    assert summary is not None


def test_when_walk_forward_default_config_then_runs(returns: pd.DataFrame) -> None:
    ev = run_covariance_forecast_evaluation(EWCovariance(half_life=40.0), returns)
    assert isinstance(ev, CovarianceForecastEvaluation)


def test_when_online_evaluation_then_returns_evaluation(
    returns: pd.DataFrame,
) -> None:
    cfg = CovarianceForecastConfig(train_size=252)
    ev = run_online_covariance_forecast_evaluation(
        RegimeAdjustedEWCovariance(half_life=40.0), returns, config=cfg
    )
    assert isinstance(ev, CovarianceForecastEvaluation)


def test_when_online_evaluation_with_pipeline_then_raises(
    returns: pd.DataFrame,
) -> None:
    pipe = Pipeline([("cov", EWCovariance(half_life=40.0))])
    with pytest.raises(ConfigurationError, match="partial_fit"):
        run_online_covariance_forecast_evaluation(pipe, returns)


def test_when_comparison_built_then_summarises(returns: pd.DataFrame) -> None:
    ew = run_covariance_forecast_evaluation(EWCovariance(half_life=40.0), returns)
    reg = run_online_covariance_forecast_evaluation(
        RegimeAdjustedEWCovariance(half_life=40.0), returns
    )
    comp = build_covariance_forecast_comparison([ew, reg], names=["EW", "RegimeEW"])
    assert comp.summary() is not None


def test_when_comparison_empty_then_raises() -> None:
    with pytest.raises(ConfigurationError, match="at least one"):
        build_covariance_forecast_comparison([])


def test_when_comparison_names_mismatch_then_raises(returns: pd.DataFrame) -> None:
    ew = run_covariance_forecast_evaluation(EWCovariance(half_life=40.0), returns)
    with pytest.raises(ConfigurationError, match="names length"):
        build_covariance_forecast_comparison([ew], names=["a", "b"])
