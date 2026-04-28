"""Tests for MomentEstimationConfig and related enums."""

from __future__ import annotations

import pytest

from optimizer.moments import (
    CovEstimatorType,
    MomentEstimationConfig,
    MuEstimatorType,
    ShrinkageMethod,
)


class TestEnums:
    def test_mu_estimator_members(self) -> None:
        assert set(MuEstimatorType) == {
            MuEstimatorType.EMPIRICAL,
            MuEstimatorType.SHRUNK,
            MuEstimatorType.EW,
            MuEstimatorType.EQUILIBRIUM,
        }

    def test_cov_estimator_members(self) -> None:
        assert set(CovEstimatorType) == {
            CovEstimatorType.EMPIRICAL,
            CovEstimatorType.LEDOIT_WOLF,
            CovEstimatorType.OAS,
            CovEstimatorType.SHRUNK,
            CovEstimatorType.EW,
            CovEstimatorType.GERBER,
            CovEstimatorType.GRAPHICAL_LASSO_CV,
            CovEstimatorType.DENOISE,
            CovEstimatorType.DETONE,
            CovEstimatorType.IMPLIED,
        }

    def test_shrinkage_method_members(self) -> None:
        assert set(ShrinkageMethod) == {
            ShrinkageMethod.JAMES_STEIN,
            ShrinkageMethod.BAYES_STEIN,
            ShrinkageMethod.BODNAR_OKHRIN,
        }

    def test_str_serialization(self) -> None:
        assert MuEstimatorType.EMPIRICAL.value == "empirical"
        assert CovEstimatorType.LEDOIT_WOLF.value == "ledoit_wolf"
        assert ShrinkageMethod.JAMES_STEIN.value == "james_stein"


class TestMomentEstimationConfig:
    def test_default_values(self) -> None:
        cfg = MomentEstimationConfig()
        assert cfg.mu_estimator == MuEstimatorType.EMPIRICAL
        assert cfg.cov_estimator == CovEstimatorType.LEDOIT_WOLF
        assert cfg.shrinkage_method == ShrinkageMethod.JAMES_STEIN
        assert cfg.ew_mu_alpha == 0.2
        assert cfg.risk_aversion == 1.0
        assert cfg.ew_cov_alpha == 0.2
        assert cfg.shrunk_cov_shrinkage == 0.1
        assert cfg.gerber_threshold == 0.5
        assert cfg.is_log_normal is False
        assert cfg.investment_horizon is None
        assert cfg.use_factor_model is False
        assert cfg.residual_variance is True

    def test_frozen(self) -> None:
        cfg = MomentEstimationConfig()
        with pytest.raises(AttributeError):
            cfg.mu_estimator = MuEstimatorType.SHRUNK  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EW,
            ew_mu_alpha=0.5,
            cov_estimator=CovEstimatorType.GERBER,
            gerber_threshold=0.3,
            is_log_normal=True,
            investment_horizon=252.0,
        )
        assert cfg.mu_estimator == MuEstimatorType.EW
        assert cfg.ew_mu_alpha == 0.5
        assert cfg.cov_estimator == CovEstimatorType.GERBER
        assert cfg.gerber_threshold == 0.3
        assert cfg.is_log_normal is True
        assert cfg.investment_horizon == 252.0


class TestFactoryMethods:
    def test_for_equilibrium_ledoitwolf(self) -> None:
        cfg = MomentEstimationConfig.for_equilibrium_ledoitwolf()
        assert cfg.mu_estimator == MuEstimatorType.EQUILIBRIUM
        assert cfg.cov_estimator == CovEstimatorType.LEDOIT_WOLF

    def test_for_shrunk_denoised(self) -> None:
        cfg = MomentEstimationConfig.for_shrunk_denoised()
        assert cfg.mu_estimator == MuEstimatorType.SHRUNK
        assert cfg.shrinkage_method == ShrinkageMethod.JAMES_STEIN
        assert cfg.cov_estimator == CovEstimatorType.DENOISE

    def test_for_adaptive(self) -> None:
        cfg = MomentEstimationConfig.for_adaptive()
        assert cfg.mu_estimator == MuEstimatorType.EW
        assert cfg.cov_estimator == CovEstimatorType.EW
