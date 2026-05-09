"""Tests for naive benchmark optimizer factories."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.moments import EWCovariance
from skfolio.optimization import EqualWeighted, InverseVolatility, Random
from skfolio.prior import EmpiricalPrior

from optimizer.optimization import (
    EqualWeightedConfig,
    InverseVolatilityConfig,
    RandomConfig,
    build_equal_weighted,
    build_inverse_volatility,
    build_random,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestEqualWeightedConfig:
    def test_when_default_then_constructs_without_error(self) -> None:
        cfg = EqualWeightedConfig()
        assert isinstance(cfg, EqualWeightedConfig)

    def test_when_constructed_then_frozen(self) -> None:
        cfg = EqualWeightedConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.previous_weights = 0.0  # type: ignore[misc]


class TestInverseVolatilityConfig:
    def test_when_default_then_no_prior_config(self) -> None:
        assert InverseVolatilityConfig().prior_config is None

    def test_when_constructed_then_frozen(self) -> None:
        cfg = InverseVolatilityConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.prior_config = None  # type: ignore[misc]


class TestRandomConfig:
    def test_when_default_then_n_portfolios_set(self) -> None:
        # Spec default is 100 (reserved field; skfolio 0.20.1 ignores).
        assert RandomConfig().n_portfolios == 100

    def test_when_default_then_random_state_none(self) -> None:
        assert RandomConfig().random_state is None

    def test_when_constructed_then_frozen(self) -> None:
        cfg = RandomConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.random_state = 42  # type: ignore[misc]


class TestPresets:
    def test_when_inv_for_default_then_no_prior(self) -> None:
        cfg = InverseVolatilityConfig.for_default()
        assert cfg.prior_config is None

    def test_when_random_for_n_portfolios_then_fields_set(self) -> None:
        cfg = RandomConfig.for_n_portfolios(n_portfolios=50, random_state=7)
        assert cfg.n_portfolios == 50
        assert cfg.random_state == 7


class TestBuildEqualWeighted:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_equal_weighted(EqualWeightedConfig())
        assert isinstance(est, EqualWeighted)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_equal_weighted(EqualWeightedConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_equal_weighted(None)
        assert isinstance(est, EqualWeighted)


class TestBuildInverseVolatility:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_inverse_volatility(InverseVolatilityConfig())
        assert isinstance(est, InverseVolatility)

    def test_when_explicit_prior_estimator_then_forwarded(self) -> None:
        prior = EmpiricalPrior()
        est = build_inverse_volatility(InverseVolatilityConfig(), prior_estimator=prior)
        assert est.prior_estimator is prior

    def test_when_for_ew_covariance_preset_then_ew_prior_built(self) -> None:
        est = build_inverse_volatility(
            InverseVolatilityConfig.for_ew_covariance(half_life=63)
        )
        assert isinstance(est.prior_estimator, EmpiricalPrior)
        assert isinstance(est.prior_estimator.covariance_estimator, EWCovariance)
        assert est.prior_estimator.covariance_estimator.half_life == 63

    def test_when_prior_config_set_then_inner_prior_built(self) -> None:
        from optimizer.moments import MomentEstimationConfig

        cfg = InverseVolatilityConfig(prior_config=MomentEstimationConfig())
        est = build_inverse_volatility(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)

    def test_when_config_none_then_default(self) -> None:
        est = build_inverse_volatility(None)
        assert isinstance(est, InverseVolatility)


class TestBuildRandom:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_random(RandomConfig())
        assert isinstance(est, Random)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_random(RandomConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_random(None)
        assert isinstance(est, Random)


class TestIntegration:
    def test_when_equal_weighted_fits_then_uniform_weights(self, returns) -> None:
        est = build_equal_weighted(EqualWeightedConfig())
        est.fit(returns)
        n = returns.shape[1]
        assert est.weights_.shape == (n,)
        assert np.allclose(est.weights_, 1.0 / n)

    def test_when_inverse_volatility_fits_then_sums_to_one(self, returns) -> None:
        est = build_inverse_volatility(InverseVolatilityConfig.for_default())
        est.fit(returns)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_inverse_volatility_ew_fits(self, returns) -> None:
        est = build_inverse_volatility(
            InverseVolatilityConfig.for_ew_covariance(half_life=63)
        )
        est.fit(returns)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_random_fits_then_dirichlet_weights(self, returns) -> None:
        est = build_random(RandomConfig.for_n_portfolios())
        est.fit(returns)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)
        assert (est.weights_ >= 0).all()
