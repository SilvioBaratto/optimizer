"""Tests for moment estimation factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import (
    DenoiseCovariance,
    DetoneCovariance,
    EmpiricalCovariance,
    EmpiricalMu,
    EquilibriumMu,
    EWCovariance,
    EWMu,
    GerberCovariance,
    GraphicalLassoCV,
    ImpliedCovariance,
    LedoitWolf,
    OAS,
    ShrunkCovariance,
    ShrunkMu,
)
from skfolio.prior import EmpiricalPrior, FactorModel

from optimizer.moments import (
    CovEstimatorType,
    MomentEstimationConfig,
    MuEstimatorType,
    ShrinkageMethod,
    build_cov_estimator,
    build_mu_estimator,
    build_prior,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 20 assets and 200 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 200, 20
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2023-01-01", periods=n_obs, freq="B"),
    )


class TestBuildMuEstimator:
    @pytest.mark.parametrize(
        ("mu_type", "expected_class"),
        [
            (MuEstimatorType.EMPIRICAL, EmpiricalMu),
            (MuEstimatorType.SHRUNK, ShrunkMu),
            (MuEstimatorType.EW, EWMu),
            (MuEstimatorType.EQUILIBRIUM, EquilibriumMu),
        ],
    )
    def test_each_type_produces_correct_class(
        self,
        mu_type: MuEstimatorType,
        expected_class: type,
    ) -> None:
        cfg = MomentEstimationConfig(mu_estimator=mu_type)
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, expected_class)

    def test_shrunk_method_forwarded(self) -> None:
        from skfolio.moments.expected_returns._shrunk_mu import ShrunkMuMethods

        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.SHRUNK,
            shrinkage_method=ShrinkageMethod.BAYES_STEIN,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, ShrunkMu)
        assert estimator.method == ShrunkMuMethods.BAYES_STEIN

    def test_ew_alpha_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EW,
            ew_mu_alpha=0.5,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EWMu)
        assert estimator.alpha == 0.5

    def test_equilibrium_risk_aversion_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            risk_aversion=2.5,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EquilibriumMu)
        assert estimator.risk_aversion == 2.5


class TestBuildCovEstimator:
    @pytest.mark.parametrize(
        ("cov_type", "expected_class"),
        [
            (CovEstimatorType.EMPIRICAL, EmpiricalCovariance),
            (CovEstimatorType.LEDOIT_WOLF, LedoitWolf),
            (CovEstimatorType.OAS, OAS),
            (CovEstimatorType.SHRUNK, ShrunkCovariance),
            (CovEstimatorType.EW, EWCovariance),
            (CovEstimatorType.GERBER, GerberCovariance),
            (CovEstimatorType.GRAPHICAL_LASSO_CV, GraphicalLassoCV),
            (CovEstimatorType.DENOISE, DenoiseCovariance),
            (CovEstimatorType.DETONE, DetoneCovariance),
            (CovEstimatorType.IMPLIED, ImpliedCovariance),
        ],
    )
    def test_each_type_produces_correct_class(
        self,
        cov_type: CovEstimatorType,
        expected_class: type,
    ) -> None:
        cfg = MomentEstimationConfig(cov_estimator=cov_type)
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, expected_class)

    def test_shrunk_shrinkage_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.SHRUNK,
            shrunk_cov_shrinkage=0.5,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, ShrunkCovariance)
        assert estimator.shrinkage == 0.5  # type: ignore[comparison-overlap]

    def test_ew_alpha_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.EW,
            ew_cov_alpha=0.3,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, EWCovariance)
        assert estimator.alpha == 0.3

    def test_gerber_threshold_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.GERBER,
            gerber_threshold=0.7,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, GerberCovariance)
        assert estimator.threshold == 0.7

    def test_denoise_nests_inner_covariance(self) -> None:
        cfg = MomentEstimationConfig(cov_estimator=CovEstimatorType.DENOISE)
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, DenoiseCovariance)
        assert isinstance(estimator.covariance_estimator, EmpiricalCovariance)

    def test_detone_nests_inner_covariance(self) -> None:
        cfg = MomentEstimationConfig(cov_estimator=CovEstimatorType.DETONE)
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, DetoneCovariance)
        assert isinstance(estimator.covariance_estimator, EmpiricalCovariance)


class TestBuildPrior:
    def test_default_returns_empirical_prior(self) -> None:
        prior = build_prior()
        assert isinstance(prior, EmpiricalPrior)

    def test_none_config_returns_empirical_prior(self) -> None:
        prior = build_prior(config=None)
        assert isinstance(prior, EmpiricalPrior)

    def test_factor_model_when_enabled(self) -> None:
        cfg = MomentEstimationConfig(use_factor_model=True)
        prior = build_prior(cfg)
        assert isinstance(prior, FactorModel)

    def test_factor_model_residual_variance(self) -> None:
        cfg = MomentEstimationConfig(
            use_factor_model=True,
            residual_variance=False,
        )
        prior = build_prior(cfg)
        assert isinstance(prior, FactorModel)
        assert prior.residual_variance is False

    def test_is_log_normal_forwarded(self) -> None:
        cfg = MomentEstimationConfig(is_log_normal=True)
        prior = build_prior(cfg)
        assert isinstance(prior, EmpiricalPrior)
        assert prior.is_log_normal is True

    def test_investment_horizon_forwarded(self) -> None:
        cfg = MomentEstimationConfig(investment_horizon=252.0)
        prior = build_prior(cfg)
        assert isinstance(prior, EmpiricalPrior)
        assert prior.investment_horizon == 252.0

    def test_mu_and_cov_composed(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.SHRUNK,
            cov_estimator=CovEstimatorType.OAS,
        )
        prior = build_prior(cfg)
        assert isinstance(prior, EmpiricalPrior)
        assert isinstance(prior.mu_estimator, ShrunkMu)
        assert isinstance(prior.covariance_estimator, OAS)


class TestIntegration:
    """Integration tests using real skfolio fit/predict."""

    def test_default_prior_fit(self, returns_df: pd.DataFrame) -> None:
        prior = build_prior()
        prior.fit(returns_df)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None
        assert rd.mu.shape == (returns_df.shape[1],)
        assert rd.covariance.shape == (
            returns_df.shape[1],
            returns_df.shape[1],
        )

    def test_equilibrium_prior_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MomentEstimationConfig.for_equilibrium_ledoitwolf()
        prior = build_prior(cfg)
        prior.fit(returns_df)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_shrunk_denoised_prior_fit(
        self, returns_df: pd.DataFrame
    ) -> None:
        cfg = MomentEstimationConfig.for_shrunk_denoised()
        prior = build_prior(cfg)
        prior.fit(returns_df)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_adaptive_prior_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MomentEstimationConfig.for_adaptive()
        prior = build_prior(cfg)
        prior.fit(returns_df)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_sp500_dataset(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)
        prior = build_prior()
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu.shape == (returns.shape[1],)
        assert rd.covariance.shape == (
            returns.shape[1],
            returns.shape[1],
        )

    def test_prior_composes_with_meanrisk(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio.optimization import MeanRisk

        prior = build_prior()
        model = MeanRisk(prior_estimator=prior)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) > 0

    def test_factor_model_fit(self) -> None:
        from skfolio.datasets import load_factors_dataset, load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        factor_prices = load_factors_dataset()

        # Align date ranges before converting to returns
        common_idx = prices.index.intersection(factor_prices.index)
        X = prices_to_returns(prices.loc[common_idx])
        y = prices_to_returns(factor_prices.loc[common_idx])

        cfg = MomentEstimationConfig(use_factor_model=True)
        prior = build_prior(cfg)
        prior.fit(X, y=y)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None
