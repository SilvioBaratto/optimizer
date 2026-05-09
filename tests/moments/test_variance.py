"""Tests for variance estimator dispatch and presets."""

from __future__ import annotations

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
