"""Tests for uncertainty-set factory functions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)

from optimizer.exceptions import ConfigurationError
from optimizer.uncertainty_set import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
    build_covariance_uncertainty_set,
    build_mu_uncertainty_set,
)


class TestEnums:
    def test_when_listed_then_mu_enum_has_two_members(self) -> None:
        assert {m.name for m in MuUncertaintySetType} == {"EMPIRICAL", "BOOTSTRAP"}

    def test_when_listed_then_covariance_enum_has_two_members(self) -> None:
        assert {m.name for m in CovarianceUncertaintySetType} == {
            "EMPIRICAL",
            "BOOTSTRAP",
        }


class TestMuUncertaintySetConfig:
    def test_when_default_then_empirical_kind(self) -> None:
        cfg = MuUncertaintySetConfig()
        assert cfg.kind == MuUncertaintySetType.EMPIRICAL

    def test_when_default_then_confidence_level_0_95(self) -> None:
        cfg = MuUncertaintySetConfig()
        assert cfg.confidence_level == 0.95

    def test_when_default_then_n_bootstrap_samples_1000(self) -> None:
        cfg = MuUncertaintySetConfig()
        assert cfg.n_bootstrap_samples == 1000

    def test_when_default_then_block_size_none(self) -> None:
        cfg = MuUncertaintySetConfig()
        assert cfg.block_size is None

    def test_when_default_then_random_state_none(self) -> None:
        cfg = MuUncertaintySetConfig()
        assert cfg.random_state is None

    def test_when_constructed_then_frozen(self) -> None:
        cfg = MuUncertaintySetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.kind = MuUncertaintySetType.BOOTSTRAP  # type: ignore[misc]

    def test_when_empirical_with_block_size_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="BOOTSTRAP"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.EMPIRICAL,
                block_size=5.0,
            )

    def test_when_empirical_with_random_state_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="BOOTSTRAP"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.EMPIRICAL,
                random_state=42,
            )

    def test_when_empirical_with_non_default_n_bootstrap_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_bootstrap_samples"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.EMPIRICAL,
                n_bootstrap_samples=500,
            )


class TestCovarianceUncertaintySetConfig:
    def test_when_default_then_empirical_kind(self) -> None:
        cfg = CovarianceUncertaintySetConfig()
        assert cfg.kind == CovarianceUncertaintySetType.EMPIRICAL

    def test_when_constructed_then_frozen(self) -> None:
        cfg = CovarianceUncertaintySetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.kind = CovarianceUncertaintySetType.BOOTSTRAP  # type: ignore[misc]

    def test_when_empirical_with_block_size_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="BOOTSTRAP"):
            CovarianceUncertaintySetConfig(
                kind=CovarianceUncertaintySetType.EMPIRICAL,
                block_size=5.0,
            )


class TestPresets:
    def test_when_mu_for_empirical_then_empirical_kind(self) -> None:
        cfg = MuUncertaintySetConfig.for_empirical()
        assert cfg.kind == MuUncertaintySetType.EMPIRICAL

    def test_when_mu_for_bootstrap_then_bootstrap_kind(self) -> None:
        cfg = MuUncertaintySetConfig.for_bootstrap(confidence_level=0.99)
        assert cfg.kind == MuUncertaintySetType.BOOTSTRAP
        assert cfg.confidence_level == 0.99

    def test_when_cov_for_empirical_then_empirical_kind(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_empirical()
        assert cfg.kind == CovarianceUncertaintySetType.EMPIRICAL

    def test_when_cov_for_bootstrap_then_bootstrap_kind(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_bootstrap(
            confidence_level=0.99,
            n_bootstrap_samples=500,
            block_size=10.0,
            random_state=7,
        )
        assert cfg.kind == CovarianceUncertaintySetType.BOOTSTRAP
        assert cfg.n_bootstrap_samples == 500
        assert cfg.block_size == 10.0
        assert cfg.random_state == 7


class TestBuildMuUncertaintySet:
    def test_when_empirical_then_empirical_class_returned(self) -> None:
        cfg = MuUncertaintySetConfig.for_empirical()
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, EmpiricalMuUncertaintySet)
        assert isinstance(est, BaseMuUncertaintySet)

    def test_when_bootstrap_then_bootstrap_class_returned(self) -> None:
        cfg = MuUncertaintySetConfig.for_bootstrap()
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, BootstrapMuUncertaintySet)
        assert isinstance(est, BaseMuUncertaintySet)

    def test_when_confidence_level_set_then_forwarded(self) -> None:
        cfg = MuUncertaintySetConfig.for_empirical(confidence_level=0.99)
        est = build_mu_uncertainty_set(cfg)
        assert est.confidence_level == 0.99

    def test_when_bootstrap_then_n_bootstrap_samples_forwarded(self) -> None:
        cfg = MuUncertaintySetConfig.for_bootstrap(n_bootstrap_samples=250)
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, BootstrapMuUncertaintySet)
        assert est.n_bootstrap_samples == 250

    def test_when_bootstrap_then_random_state_forwarded_as_seed(self) -> None:
        cfg = MuUncertaintySetConfig.for_bootstrap(random_state=11)
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, BootstrapMuUncertaintySet)
        assert est.seed == 11


class TestBuildCovarianceUncertaintySet:
    def test_when_empirical_then_empirical_class_returned(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_empirical()
        est = build_covariance_uncertainty_set(cfg)
        assert isinstance(est, EmpiricalCovarianceUncertaintySet)
        assert isinstance(est, BaseCovarianceUncertaintySet)

    def test_when_bootstrap_then_bootstrap_class_returned(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_bootstrap()
        est = build_covariance_uncertainty_set(cfg)
        assert isinstance(est, BootstrapCovarianceUncertaintySet)
        assert isinstance(est, BaseCovarianceUncertaintySet)

    def test_when_bootstrap_then_block_size_forwarded(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_bootstrap(block_size=8.0)
        est = build_covariance_uncertainty_set(cfg)
        assert isinstance(est, BootstrapCovarianceUncertaintySet)
        assert est.block_size == 8.0


class TestIntegration:
    @pytest.fixture(scope="class")
    def returns(self) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        n_obs, n_assets = 200, 5
        cols = [f"A{i:02d}" for i in range(n_assets)]
        return pd.DataFrame(
            rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
            columns=cols,
        )

    def test_when_empirical_mu_fits_then_uncertainty_set_set(
        self,
        returns: pd.DataFrame,
    ) -> None:
        est = build_mu_uncertainty_set(MuUncertaintySetConfig.for_empirical())
        est.fit(returns)
        assert est.uncertainty_set_ is not None

    def test_when_bootstrap_cov_fits_then_uncertainty_set_set(
        self,
        returns: pd.DataFrame,
    ) -> None:
        cfg = CovarianceUncertaintySetConfig.for_bootstrap(
            n_bootstrap_samples=50,
            random_state=0,
        )
        est = build_covariance_uncertainty_set(cfg)
        est.fit(returns)
        assert est.uncertainty_set_ is not None
