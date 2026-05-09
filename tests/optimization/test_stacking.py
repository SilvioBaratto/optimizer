"""Tests for the StackingOptimization factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import (
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    StackingOptimization,
)

from optimizer.exceptions import ConfigurationError
from optimizer.optimization import (
    StackingConfig,
    build_equal_weighted,
    build_inverse_volatility,
    build_stacking,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


@pytest.fixture
def base_estimators():
    return [
        ("ew", build_equal_weighted()),
        ("inv_vol", build_inverse_volatility()),
    ]


class TestStackingConfig:
    def test_when_default_then_no_prior_config(self) -> None:
        assert StackingConfig().prior_config is None

    def test_when_default_then_quantile_half(self) -> None:
        assert StackingConfig().quantile == 0.5

    def test_when_constructed_then_frozen(self) -> None:
        cfg = StackingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.quantile = 0.25  # type: ignore[misc]


class TestPresets:
    def test_when_for_min_variance_final_then_returns_config(self) -> None:
        cfg = StackingConfig.for_min_variance_final()
        assert isinstance(cfg, StackingConfig)

    def test_when_for_max_sharpe_final_then_returns_config(self) -> None:
        cfg = StackingConfig.for_max_sharpe_final()
        assert isinstance(cfg, StackingConfig)


class TestBuildStacking:
    def test_when_built_then_returns_skfolio_class(self, base_estimators) -> None:
        est = build_stacking(StackingConfig(), estimators=base_estimators)
        assert isinstance(est, StackingOptimization)

    def test_when_estimators_missing_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="estimators"):
            build_stacking(StackingConfig())

    def test_when_estimators_passed_then_forwarded(self, base_estimators) -> None:
        est = build_stacking(StackingConfig(), estimators=base_estimators)
        assert est.estimators is base_estimators

    def test_when_for_min_variance_final_then_meanrisk_final_built(
        self, base_estimators
    ) -> None:
        est = build_stacking(
            StackingConfig.for_min_variance_final(), estimators=base_estimators
        )
        assert isinstance(est.final_estimator, MeanRisk)

    def test_when_for_max_sharpe_final_then_meanrisk_final_built(
        self, base_estimators
    ) -> None:
        from skfolio.optimization.convex._base import ObjectiveFunction

        est = build_stacking(
            StackingConfig.for_max_sharpe_final(), estimators=base_estimators
        )
        assert isinstance(est.final_estimator, MeanRisk)
        assert (
            est.final_estimator.objective_function == ObjectiveFunction.MAXIMIZE_RATIO
        )

    def test_when_explicit_final_estimator_then_overrides_preset(
        self, base_estimators
    ) -> None:
        custom = build_equal_weighted()
        est = build_stacking(
            StackingConfig.for_min_variance_final(),
            estimators=base_estimators,
            final_estimator=custom,
        )
        assert est.final_estimator is custom

    def test_when_quantile_set_then_forwarded(self, base_estimators) -> None:
        cfg = StackingConfig(quantile=0.3)
        est = build_stacking(cfg, estimators=base_estimators)
        assert est.quantile == 0.3

    def test_when_kwargs_passed_then_forwarded(self, base_estimators) -> None:
        est = build_stacking(
            StackingConfig(),
            estimators=base_estimators,
            raise_on_failure=False,
        )
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self, base_estimators) -> None:
        est = build_stacking(None, estimators=base_estimators)
        assert isinstance(est, StackingOptimization)

    def test_when_prior_config_then_inner_prior_built(self, base_estimators) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = StackingConfig(
            prior_config=MomentEstimationConfig(),
            final_default=StackingConfig.for_min_variance_final().final_default,
        )
        est = build_stacking(cfg, estimators=base_estimators)
        assert isinstance(est.final_estimator, MeanRisk)
        assert isinstance(est.final_estimator.prior_estimator, EmpiricalPrior)


class TestIntegration:
    def test_when_two_base_estimators_then_fits(self, returns) -> None:
        estimators = [
            ("ew", EqualWeighted()),
            ("inv_vol", InverseVolatility()),
        ]
        est = build_stacking(
            StackingConfig.for_min_variance_final(), estimators=estimators
        )
        est.fit(returns)
        n = returns.shape[1]
        assert est.weights_.shape == (n,)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)

    def test_when_three_base_estimators_then_fits(self, returns) -> None:
        estimators = [
            ("ew", EqualWeighted()),
            ("inv_vol", InverseVolatility()),
            ("ew2", EqualWeighted()),
        ]
        est = build_stacking(
            StackingConfig.for_max_sharpe_final(), estimators=estimators
        )
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
