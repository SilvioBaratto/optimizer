"""Tests for MaximumDiversification factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import MaximumDiversification

from optimizer.optimization import (
    MaxDiversificationConfig,
    build_max_diversification,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestMaxDiversificationConfig:
    def test_when_default_then_long_only(self) -> None:
        # Diversification ratio is undefined for short positions.
        cfg = MaxDiversificationConfig()
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0

    def test_when_default_then_no_prior_config(self) -> None:
        assert MaxDiversificationConfig().prior_config is None

    def test_when_constructed_then_frozen(self) -> None:
        cfg = MaxDiversificationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_weights = -0.5  # type: ignore[misc]


class TestPresets:
    def test_when_for_default_then_long_only(self) -> None:
        cfg = MaxDiversificationConfig.for_default()
        assert cfg.min_weights == 0.0

    def test_when_for_long_only_capped_then_max_weight_set(self) -> None:
        cfg = MaxDiversificationConfig.for_long_only_capped(max_weight=0.10)
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 0.10

    def test_when_for_long_only_capped_default_cap(self) -> None:
        cfg = MaxDiversificationConfig.for_long_only_capped()
        assert cfg.max_weights == 0.10


class TestBuildMaxDiversification:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_max_diversification(MaxDiversificationConfig())
        assert isinstance(est, MaximumDiversification)

    def test_when_min_weights_set_then_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(min_weights=0.01)
        est = build_max_diversification(cfg)
        assert est.min_weights == 0.01

    def test_when_max_weights_set_then_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(max_weights=0.10)
        est = build_max_diversification(cfg)
        assert est.max_weights == 0.10

    def test_when_transaction_costs_set_then_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(transaction_costs=0.001)
        est = build_max_diversification(cfg)
        assert est.transaction_costs == 0.001

    def test_when_explicit_prior_estimator_then_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_max_diversification(
            MaxDiversificationConfig(), prior_estimator=prior
        )
        assert est.prior_estimator is prior

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = MaxDiversificationConfig(prior_config=MomentEstimationConfig())
        est = build_max_diversification(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_max_diversification(
            MaxDiversificationConfig(), raise_on_failure=False
        )
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_max_diversification(None)
        assert isinstance(est, MaximumDiversification)


class TestIntegration:
    def test_when_default_then_fits_and_sums_to_one(self, returns) -> None:
        est = build_max_diversification(MaxDiversificationConfig.for_default())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_long_only_capped_then_weights_below_cap(self, returns) -> None:
        cfg = MaxDiversificationConfig.for_long_only_capped(max_weight=0.15)
        est = build_max_diversification(cfg)
        est.fit(returns)
        assert (est.weights_ <= 0.15 + 1e-6).all()
