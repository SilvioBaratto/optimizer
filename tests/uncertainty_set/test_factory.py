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
    OrthogonalCovarianceUncertaintySet,
    OrthogonalMuUncertaintySet,
)
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)
from skfolio.uncertainty_set._orthogonal import CSWeighting

from optimizer.exceptions import ConfigurationError
from optimizer.uncertainty_set import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    CrossSectionalWeighting,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
    OrthogonalUncertaintyShape,
    build_covariance_uncertainty_set,
    build_mu_uncertainty_set,
)


class TestEnums:
    def test_when_listed_then_mu_enum_has_three_members(self) -> None:
        assert {m.name for m in MuUncertaintySetType} == {
            "EMPIRICAL",
            "BOOTSTRAP",
            "ORTHOGONAL",
        }

    def test_when_listed_then_covariance_enum_has_three_members(self) -> None:
        assert {m.name for m in CovarianceUncertaintySetType} == {
            "EMPIRICAL",
            "BOOTSTRAP",
            "ORTHOGONAL",
        }

    def test_when_listed_then_cs_weighting_enum_mirrors_skfolio(self) -> None:
        assert {m.value for m in CrossSectionalWeighting} == {
            m.value for m in CSWeighting
        }

    def test_when_listed_then_uncertainty_shape_enum_has_two_members(self) -> None:
        assert {m.value for m in OrthogonalUncertaintyShape} == {
            "identity",
            "idio_variance",
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


class TestNewSharedFields:
    def test_when_default_then_diagonal_true(self) -> None:
        assert MuUncertaintySetConfig().diagonal is True
        assert CovarianceUncertaintySetConfig().diagonal is True

    def test_when_default_then_n_eff_none(self) -> None:
        assert MuUncertaintySetConfig().n_eff is None
        assert CovarianceUncertaintySetConfig().n_eff is None

    def test_when_default_then_cs_weighting_inverse_idio(self) -> None:
        assert (
            MuUncertaintySetConfig().cs_weighting
            == CrossSectionalWeighting.INVERSE_IDIO_VARIANCE
        )

    def test_when_empirical_diagonal_false_then_forwarded(self) -> None:
        cfg = MuUncertaintySetConfig.for_empirical(diagonal=False)
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, EmpiricalMuUncertaintySet)
        assert est.diagonal is False

    def test_when_empirical_n_eff_then_forwarded(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_empirical(n_eff=120.0)
        est = build_covariance_uncertainty_set(cfg)
        assert isinstance(est, EmpiricalCovarianceUncertaintySet)
        assert est.n_eff == 120.0

    def test_when_bootstrap_diagonal_false_then_forwarded(self) -> None:
        cfg = MuUncertaintySetConfig.for_bootstrap(diagonal=False)
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, BootstrapMuUncertaintySet)
        assert est.diagonal is False

    def test_when_n_eff_on_bootstrap_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_eff"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.BOOTSTRAP,
                n_eff=100.0,
            )

    def test_when_n_eff_on_orthogonal_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_eff"):
            CovarianceUncertaintySetConfig(
                kind=CovarianceUncertaintySetType.ORTHOGONAL,
                n_eff=100.0,
            )

    def test_when_diagonal_false_on_orthogonal_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="diagonal"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.ORTHOGONAL,
                diagonal=False,
            )

    def test_when_cs_weighting_on_empirical_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="cs_weighting"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.EMPIRICAL,
                cs_weighting=CrossSectionalWeighting.IDENTITY,
            )


class TestOrthogonalMuConfig:
    def test_when_uncertainty_shape_on_empirical_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="uncertainty_shape"):
            MuUncertaintySetConfig(
                kind=MuUncertaintySetType.EMPIRICAL,
                uncertainty_shape=OrthogonalUncertaintyShape.IDIO_VARIANCE,
            )

    def test_when_for_orthogonal_then_orthogonal_kind(self) -> None:
        cfg = MuUncertaintySetConfig.for_orthogonal(
            confidence_level=0.9,
            cs_weighting=CrossSectionalWeighting.IDENTITY,
            uncertainty_shape=OrthogonalUncertaintyShape.IDIO_VARIANCE,
        )
        assert cfg.kind == MuUncertaintySetType.ORTHOGONAL
        assert cfg.confidence_level == 0.9
        assert cfg.cs_weighting == CrossSectionalWeighting.IDENTITY
        assert cfg.uncertainty_shape == OrthogonalUncertaintyShape.IDIO_VARIANCE

    def test_when_orthogonal_then_orthogonal_class_returned(self) -> None:
        est = build_mu_uncertainty_set(MuUncertaintySetConfig.for_orthogonal())
        assert isinstance(est, OrthogonalMuUncertaintySet)
        assert isinstance(est, BaseMuUncertaintySet)

    def test_when_orthogonal_then_cs_weighting_forwarded_as_skfolio_enum(self) -> None:
        cfg = MuUncertaintySetConfig.for_orthogonal(
            cs_weighting=CrossSectionalWeighting.IDENTITY,
            uncertainty_shape=OrthogonalUncertaintyShape.IDIO_VARIANCE,
        )
        est = build_mu_uncertainty_set(cfg)
        assert isinstance(est, OrthogonalMuUncertaintySet)
        assert est.cs_weighting is CSWeighting.IDENTITY
        assert est.uncertainty_shape == "idio_variance"
        assert est.confidence_level == 0.95

    def test_when_orthogonal_with_prior_estimator_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="prior_estimator"):
            build_mu_uncertainty_set(
                MuUncertaintySetConfig.for_orthogonal(),
                prior_estimator=EmpiricalMuUncertaintySet(),
            )


class TestOrthogonalCovConfig:
    def test_when_radius_on_empirical_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="radius"):
            CovarianceUncertaintySetConfig(
                kind=CovarianceUncertaintySetType.EMPIRICAL,
                radius=2.0,
            )

    def test_when_confidence_level_on_orthogonal_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="confidence_level"):
            CovarianceUncertaintySetConfig(
                kind=CovarianceUncertaintySetType.ORTHOGONAL,
                confidence_level=0.9,
            )

    def test_when_for_orthogonal_then_orthogonal_kind(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_orthogonal(
            radius=2.5,
            cs_weighting=CrossSectionalWeighting.REGRESSION,
        )
        assert cfg.kind == CovarianceUncertaintySetType.ORTHOGONAL
        assert cfg.radius == 2.5
        assert cfg.cs_weighting == CrossSectionalWeighting.REGRESSION

    def test_when_orthogonal_then_orthogonal_class_returned(self) -> None:
        cfg = CovarianceUncertaintySetConfig.for_orthogonal(radius=3.0)
        est = build_covariance_uncertainty_set(cfg)
        assert isinstance(est, OrthogonalCovarianceUncertaintySet)
        assert isinstance(est, BaseCovarianceUncertaintySet)
        assert est.radius == 3.0

    def test_when_orthogonal_with_prior_estimator_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="prior_estimator"):
            build_covariance_uncertainty_set(
                CovarianceUncertaintySetConfig.for_orthogonal(),
                prior_estimator=EmpiricalCovarianceUncertaintySet(),
            )


class TestPriorEstimatorForwarding:
    def test_when_prior_estimator_then_forwarded_to_empirical_mu(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_mu_uncertainty_set(
            MuUncertaintySetConfig.for_empirical(),
            prior_estimator=prior,
        )
        assert est.prior_estimator is prior

    def test_when_prior_estimator_then_forwarded_to_bootstrap_cov(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_covariance_uncertainty_set(
            CovarianceUncertaintySetConfig.for_bootstrap(),
            prior_estimator=prior,
        )
        assert est.prior_estimator is prior


class TestOrthogonalIntegration:
    @pytest.fixture(scope="class")
    def factor_return_distribution(self):
        from skfolio.prior import TimeSeriesFactorModel

        rng = np.random.default_rng(11)
        n_obs, n_factors, n_assets = 250, 3, 4
        factors = pd.DataFrame(
            rng.normal(0.0, 0.01, size=(n_obs, n_factors)),
            columns=[f"F{i}" for i in range(n_factors)],
        )
        loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
        assets = pd.DataFrame(
            factors.to_numpy() @ loadings.T
            + rng.normal(0.0, 0.005, size=(n_obs, n_assets)),
            columns=[f"A{i}" for i in range(n_assets)],
        )
        model = TimeSeriesFactorModel().fit(assets, factors=factors)
        return assets, model.return_distribution_

    def test_when_orthogonal_mu_fits_then_uncertainty_set_set(
        self,
        factor_return_distribution,
    ) -> None:
        assets, rd = factor_return_distribution
        est = build_mu_uncertainty_set(MuUncertaintySetConfig.for_orthogonal())
        est.fit(assets, return_distribution=rd)
        assert est.uncertainty_set_ is not None
        assert est.uncertainty_set_.geometry.shape[0] == assets.shape[1]
        assert est.uncertainty_set_.norm == 2

    def test_when_orthogonal_cov_fits_then_uncertainty_set_set(
        self,
        factor_return_distribution,
    ) -> None:
        assets, rd = factor_return_distribution
        cfg = CovarianceUncertaintySetConfig.for_orthogonal(radius=1.5)
        est = build_covariance_uncertainty_set(cfg)
        est.fit(assets, return_distribution=rd)
        assert est.uncertainty_set_ is not None
        assert est.uncertainty_set_.radius == 1.5
