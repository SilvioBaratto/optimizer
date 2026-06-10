"""Tests for variance estimator dispatch and presets."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import (
    EmpiricalVariance,
    EWVariance,
    RegimeAdjustedEWVariance,
)
from skfolio.moments.variance._base import BaseVariance

from optimizer.exceptions import ConfigurationError
from optimizer.moments import (
    CovEstimatorType,
    MomentEstimationConfig,
    VarianceEstimatorType,
    build_variance_estimator,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_obs, n_assets = 250, 5
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
        columns=cols,
    )


class TestVarianceEstimatorType:
    def test_when_listed_then_three_members(self) -> None:
        assert {m.name for m in VarianceEstimatorType} == {
            "EMPIRICAL",
            "EW",
            "REGIME_ADJUSTED_EW",
        }


class TestBuildVarianceEstimator:
    @pytest.mark.parametrize(
        ("variance_type", "expected_class"),
        [
            (VarianceEstimatorType.EMPIRICAL, EmpiricalVariance),
            (VarianceEstimatorType.EW, EWVariance),
            (VarianceEstimatorType.REGIME_ADJUSTED_EW, RegimeAdjustedEWVariance),
        ],
    )
    def test_when_dispatched_then_correct_class(
        self,
        variance_type: VarianceEstimatorType,
        expected_class: type[BaseVariance],
    ) -> None:
        cfg = MomentEstimationConfig(variance_estimator=variance_type)
        est = build_variance_estimator(cfg)
        assert isinstance(est, expected_class)
        assert isinstance(est, BaseVariance)

    def test_when_variance_estimator_none_then_raises(self) -> None:
        cfg = MomentEstimationConfig(variance_estimator=None)
        with pytest.raises(ConfigurationError, match="variance_estimator"):
            build_variance_estimator(cfg)

    def test_when_ew_then_half_life_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            variance_estimator=VarianceEstimatorType.EW,
            variance_half_life=23.0,
        )
        est = build_variance_estimator(cfg)
        assert isinstance(est, EWVariance)
        assert est.half_life == 23.0

    def test_when_regime_adjusted_ew_then_kwargs_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            variance_estimator=VarianceEstimatorType.REGIME_ADJUSTED_EW,
            variance_half_life=23.0,
            hac_lags=4,
        )
        est = build_variance_estimator(cfg)
        assert isinstance(est, RegimeAdjustedEWVariance)
        assert est.half_life == 23.0
        assert est.hac_lags == 4


class TestVarianceFitShape:
    @pytest.mark.parametrize(
        "variance_type",
        list(VarianceEstimatorType),
    )
    def test_when_fitted_then_variance_attribute_is_one_dim(
        self,
        variance_type: VarianceEstimatorType,
        returns: pd.DataFrame,
    ) -> None:
        cfg = MomentEstimationConfig(variance_estimator=variance_type)
        est = build_variance_estimator(cfg)
        est.fit(returns)
        n_assets = returns.shape[1]
        assert est.variance_.shape == (n_assets,)


class TestVarianceVsCovarianceGotcha:
    """Variance estimators expose 1-D ``variance_``, never 2-D ``covariance_``."""

    @pytest.mark.parametrize("variance_type", list(VarianceEstimatorType))
    def test_when_fitted_then_variance_1d_and_covariance_absent(
        self,
        variance_type: VarianceEstimatorType,
        returns: pd.DataFrame,
    ) -> None:
        cfg = MomentEstimationConfig(variance_estimator=variance_type)
        est = build_variance_estimator(cfg)
        est.fit(returns)
        # 1-D variance_ of shape (n_assets,) — asserted after fit (not vacuous).
        assert est.variance_.ndim == 1
        assert est.variance_.shape == (returns.shape[1],)
        # Variance estimators must NOT expose a 2-D covariance_ attribute.
        assert not hasattr(est, "covariance_")


class TestRegimeAdjustedVarianceForwarding:
    """REGIME_ADJUSTED_EW forwards the regime-specific kwargs."""

    def test_when_regime_adjusted_ew_then_regime_kwargs_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            variance_estimator=VarianceEstimatorType.REGIME_ADJUSTED_EW,
            regime_half_life=15.0,
            regime_multiplier_clip=(0.8, 1.4),
            min_observations=120,
        )
        est = build_variance_estimator(cfg)
        assert isinstance(est, RegimeAdjustedEWVariance)
        assert est.regime_half_life == 15.0
        assert est.regime_multiplier_clip == (0.8, 1.4)
        assert est.min_observations == 120


class TestUnsupportedVarianceEstimatorRaises:
    """An unknown (non-enum) variance type hits the defensive default."""

    def test_unsupported_variance_estimator_raises(self) -> None:
        cfg = MomentEstimationConfig(
            variance_estimator=cast(VarianceEstimatorType, "not_an_estimator")
        )
        with pytest.raises(ConfigurationError, match="variance_estimator"):
            build_variance_estimator(cfg)


class TestRegimeAdjustedEWPreset:
    def test_when_for_regime_adjusted_ew_then_estimator_is_set(self) -> None:
        cfg = MomentEstimationConfig.for_regime_adjusted_ew()
        assert cfg.variance_estimator == VarianceEstimatorType.REGIME_ADJUSTED_EW
        assert cfg.cov_estimator == CovEstimatorType.REGIME_ADJUSTED_EW

    def test_when_for_regime_adjusted_ew_then_half_lives_set(self) -> None:
        cfg = MomentEstimationConfig.for_regime_adjusted_ew()
        assert cfg.variance_half_life == 23.0
        assert cfg.corr_half_life == 50.0

    def test_when_for_regime_adjusted_ew_then_round_trip_fits(
        self,
        returns: pd.DataFrame,
    ) -> None:
        cfg = MomentEstimationConfig.for_regime_adjusted_ew()
        var_est = build_variance_estimator(cfg)
        var_est.fit(returns)
        assert var_est.variance_.shape == (returns.shape[1],)
