"""End-to-end integration: uncertainty_set Configs feed RobustMeanRisk."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio.optimization import MeanRisk
from skfolio.uncertainty_set import (
    BootstrapMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)

from optimizer.optimization import RobustMeanRiskConfig, build_robust_mean_risk
from optimizer.uncertainty_set import (
    CovarianceUncertaintySetConfig,
    MuUncertaintySetConfig,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


def test_when_empirical_mu_uncertainty_then_estimator_built() -> None:
    cfg = RobustMeanRiskConfig(
        mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(
            confidence_level=0.95
        )
    )
    est = build_robust_mean_risk(cfg)
    assert isinstance(est, MeanRisk)
    assert isinstance(est.mu_uncertainty_set_estimator, EmpiricalMuUncertaintySet)
    assert est.mu_uncertainty_set_estimator.confidence_level == 0.95


def test_when_bootstrap_mu_uncertainty_then_estimator_built() -> None:
    cfg = RobustMeanRiskConfig(
        mu_uncertainty_set_config=MuUncertaintySetConfig.for_bootstrap(
            n_bootstrap_samples=50, random_state=0
        )
    )
    est = build_robust_mean_risk(cfg)
    assert isinstance(est.mu_uncertainty_set_estimator, BootstrapMuUncertaintySet)


def test_when_empirical_cov_uncertainty_then_estimator_built() -> None:
    cfg = RobustMeanRiskConfig(
        covariance_uncertainty_set_config=CovarianceUncertaintySetConfig.for_empirical()
    )
    est = build_robust_mean_risk(cfg)
    assert isinstance(
        est.covariance_uncertainty_set_estimator, EmpiricalCovarianceUncertaintySet
    )


def test_when_both_uncertainty_configs_then_both_estimators_present(
    returns,
) -> None:
    cfg = RobustMeanRiskConfig(
        mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(),
        covariance_uncertainty_set_config=CovarianceUncertaintySetConfig.for_empirical(),
    )
    est = build_robust_mean_risk(cfg)
    est.fit(returns)
    assert est.weights_.shape == (returns.shape[1],)
    assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)
