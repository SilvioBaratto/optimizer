"""Tests for moment estimation factory functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import (
    OAS,
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
    RegimeAdjustedEWCovariance,
    ShrunkCovariance,
    ShrunkMu,
)
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel

from optimizer.exceptions import ConfigurationError
from optimizer.moments import (
    CovEstimatorType,
    FactorModelType,
    MomentEstimationConfig,
    MuEstimatorType,
    ShrinkageMethod,
    build_characteristics_factor_model,
    build_cov_estimator,
    build_mu_estimator,
    build_prior,
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

    def test_ew_half_life_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EW,
            ew_mu_half_life=20.0,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EWMu)
        assert estimator.half_life == 20.0

    def test_equilibrium_risk_aversion_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            risk_aversion=2.5,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EquilibriumMu)
        assert estimator.risk_aversion == 2.5

    def test_ew_min_observations_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EW,
            min_observations=25,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EWMu)
        assert estimator.min_observations == 25


class TestEquilibriumMuWiring:
    """EquilibriumMu is reverse-optimised against the configured covariance and
    (optionally) DB market-cap weights, not skfolio's silent defaults."""

    def test_covariance_estimator_matches_config(self) -> None:
        # Without this wiring EquilibriumMu would silently use its own default
        # EmpiricalCovariance, diverging from the prior's LedoitWolf/OAS/... .
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            cov_estimator=CovEstimatorType.OAS,
        )
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EquilibriumMu)
        assert isinstance(estimator.covariance_estimator, OAS)

    def test_market_weights_default_none(self) -> None:
        cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.EQUILIBRIUM)
        estimator = build_mu_estimator(cfg)
        assert isinstance(estimator, EquilibriumMu)
        assert estimator.weights is None

    def test_market_weights_forwarded(self) -> None:
        cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.EQUILIBRIUM)
        weights = np.array([0.5, 0.3, 0.2])
        estimator = build_mu_estimator(cfg, market_weights=weights)
        assert isinstance(estimator, EquilibriumMu)
        assert np.array_equal(estimator.weights, weights)

    def test_market_weights_ignored_for_non_equilibrium(self) -> None:
        cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.EMPIRICAL)
        estimator = build_mu_estimator(cfg, market_weights=np.array([0.5, 0.5]))
        assert isinstance(estimator, EmpiricalMu)

    def test_build_prior_forwards_market_weights(self) -> None:
        cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.EQUILIBRIUM)
        weights = np.array([0.6, 0.25, 0.15])
        prior = build_prior(cfg, market_weights=weights)
        assert isinstance(prior, EmpiricalPrior)
        assert isinstance(prior.mu_estimator, EquilibriumMu)
        assert np.array_equal(prior.mu_estimator.weights, weights)

    def test_cap_weighted_equilibrium_prior_fits(self) -> None:
        rng = np.random.default_rng(0)
        cols = [f"A{i}" for i in range(4)]
        returns = pd.DataFrame(
            rng.normal(0.001, 0.02, (200, 4)), columns=cols
        )
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            cov_estimator=CovEstimatorType.LEDOIT_WOLF,
        )
        prior = build_prior(cfg, market_weights=weights)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu.shape == (4,)
        assert rd.covariance.shape == (4, 4)


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
            (CovEstimatorType.REGIME_ADJUSTED_EW, RegimeAdjustedEWCovariance),
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

    def test_ew_half_life_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.EW,
            ew_cov_half_life=15.0,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, EWCovariance)
        assert estimator.half_life == 15.0

    def test_gerber_threshold_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.GERBER,
            gerber_threshold=0.7,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, GerberCovariance)
        assert estimator.threshold == 0.7

    def test_regime_adjusted_ew_kwargs_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.REGIME_ADJUSTED_EW,
            variance_half_life=23.0,
            corr_half_life=50.0,
            hac_lags=4,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, RegimeAdjustedEWCovariance)
        assert estimator.half_life == 23.0
        assert estimator.corr_half_life == 50.0
        assert estimator.hac_lags == 4

    def test_ew_min_observations_forwarded(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.EW,
            min_observations=30,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, EWCovariance)
        assert estimator.min_observations == 30

    def test_ew_min_observations_defaults_none(self) -> None:
        cfg = MomentEstimationConfig(cov_estimator=CovEstimatorType.EW)
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, EWCovariance)
        assert estimator.min_observations is None

    def test_implied_uses_new_annualization_factor_name(self) -> None:
        # skfolio 1.0 renamed annualized_factor -> annualization_factor;
        # the factory must set the NEW name and leave the deprecated one None.
        cfg = MomentEstimationConfig(
            cov_estimator=CovEstimatorType.IMPLIED,
            implied_annualization_factor=252.0,
            implied_window_size=15,
        )
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, ImpliedCovariance)
        assert estimator.annualization_factor == 252.0
        assert estimator.window_size == 15
        assert estimator.annualized_factor is None

    def test_implied_annualization_factor_defaults_none(self) -> None:
        # config default is None; skfolio resolves None to its 252.0 default
        # inside __init__, so the built estimator carries 252.0 (never the
        # deprecated annualized_factor).
        cfg = MomentEstimationConfig(cov_estimator=CovEstimatorType.IMPLIED)
        estimator = build_cov_estimator(cfg)
        assert isinstance(estimator, ImpliedCovariance)
        assert estimator.annualization_factor == 252.0
        assert estimator.annualized_factor is None
        assert estimator.window_size == 20


class TestUnsupportedEstimatorRaises:
    """The match dispatch rejects unknown estimator values (defensive guard)."""

    def test_unsupported_mu_estimator_raises(self) -> None:
        cfg = MomentEstimationConfig(
            mu_estimator=cast(MuEstimatorType, "not_an_estimator")
        )
        with pytest.raises(ConfigurationError, match="mu_estimator"):
            build_mu_estimator(cfg)

    def test_unsupported_cov_estimator_raises(self) -> None:
        cfg = MomentEstimationConfig(
            cov_estimator=cast(CovEstimatorType, "not_an_estimator")
        )
        with pytest.raises(ConfigurationError, match="cov_estimator"):
            build_cov_estimator(cfg)


class TestPresetRegression:
    """Existing presets must survive the new field additions."""

    def test_when_for_equilibrium_ledoitwolf_then_unchanged(self) -> None:
        cfg = MomentEstimationConfig.for_equilibrium_ledoitwolf()
        assert cfg.mu_estimator == MuEstimatorType.EQUILIBRIUM
        assert cfg.cov_estimator == CovEstimatorType.LEDOIT_WOLF

    def test_when_for_shrunk_denoised_then_unchanged(self) -> None:
        cfg = MomentEstimationConfig.for_shrunk_denoised()
        assert cfg.mu_estimator == MuEstimatorType.SHRUNK
        assert cfg.shrinkage_method == ShrinkageMethod.JAMES_STEIN
        assert cfg.cov_estimator == CovEstimatorType.DENOISE

    def test_when_for_adaptive_then_unchanged(self) -> None:
        cfg = MomentEstimationConfig.for_adaptive()
        assert cfg.mu_estimator == MuEstimatorType.EW
        assert cfg.cov_estimator == CovEstimatorType.EW


class TestBuildCovEstimatorNested:
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
        assert isinstance(prior, TimeSeriesFactorModel)

    def test_time_series_factor_model_type_explicit(self) -> None:
        cfg = MomentEstimationConfig(
            use_factor_model=True,
            factor_model_type=FactorModelType.TIME_SERIES,
        )
        prior = build_prior(cfg)
        assert isinstance(prior, TimeSeriesFactorModel)

    def test_characteristics_via_build_prior_raises(self) -> None:
        cfg = MomentEstimationConfig(
            use_factor_model=True,
            factor_model_type=FactorModelType.CHARACTERISTICS,
        )
        with pytest.raises(ConfigurationError, match="CharacteristicsFactorModel"):
            build_prior(cfg)

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

    def test_shrunk_denoised_prior_fit(self, returns_df: pd.DataFrame) -> None:
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

    def test_prior_composes_with_meanrisk(self, returns_df: pd.DataFrame) -> None:
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
        factors = prices_to_returns(factor_prices.loc[common_idx])

        cfg = MomentEstimationConfig(use_factor_model=True)
        prior = build_prior(cfg)
        prior.fit(X, factors=factors)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None


@pytest.fixture()
def asset_panel():  # type: ignore[no-untyped-def]
    """Minimal cross-sectional AssetPanel for CharacteristicsFactorModel.

    Enough assets (12) so the cross-sectional regression is well posed
    once ``min_regression_assets`` is lowered from the skfolio default.
    """
    import numpy as np
    from skfolio.containers import AssetPanel, Field2D

    n_obs, n_assets = 160, 12
    rng = np.random.default_rng(0)
    return AssetPanel(
        fields={
            "returns": Field2D(rng.normal(0.0005, 0.01, (n_obs, n_assets))),
            "value": Field2D(rng.normal(0.0, 1.0, (n_obs, n_assets))),
            "market_cap": Field2D(np.abs(rng.normal(1e6, 1e5, (n_obs, n_assets)))),
        },
        observations=pd.bdate_range("2020-01-01", periods=n_obs),
        asset_names=[f"A{i}" for i in range(n_assets)],
    )


def _make_factors():  # type: ignore[no-untyped-def]
    from skfolio.descriptor import Passthrough
    from skfolio.factor_exposure import FixedWeightedFactor, GlobalFactor

    return [
        ("market", GlobalFactor(family="market")),
        (
            "value",
            FixedWeightedFactor(
                descriptors=[("value", Passthrough("value"))],
                family="value",
            ),
        ),
    ]


class TestBuildCharacteristicsFactorModel:
    def test_returns_characteristics_factor_model(self) -> None:
        from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior

        cfg = MomentEstimationConfig(
            exposure_lag=2,
            min_regression_assets=3,
            mu_estimator=MuEstimatorType.SHRUNK,
        )
        model = build_characteristics_factor_model(cfg, factors=_make_factors())
        assert isinstance(model, CharacteristicsFactorModel)
        assert model.exposure_lag == 2
        assert model.min_regression_assets == 3
        assert isinstance(model.factor_prior_estimator, EmpiricalPrior)
        assert isinstance(model.factor_prior_estimator.mu_estimator, ShrunkMu)

    def test_none_config_defaults(self) -> None:
        from skfolio.prior import CharacteristicsFactorModel

        model = build_characteristics_factor_model(factors=_make_factors())
        assert isinstance(model, CharacteristicsFactorModel)
        assert model.exposure_lag == 1

    def test_neutralize_against_forwarded(self) -> None:
        model = build_characteristics_factor_model(
            factors=_make_factors(),
            neutralize_against={"value": ["market"]},
        )
        assert model.neutralize_against == {"value": ["market"]}

    def test_market_weights_forwarded_to_factor_prior(self) -> None:
        weights = np.array([0.7, 0.3])
        cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.EQUILIBRIUM)
        model = build_characteristics_factor_model(
            cfg, factors=_make_factors(), market_weights=weights
        )
        assert isinstance(model.factor_prior_estimator, EmpiricalPrior)
        assert isinstance(model.factor_prior_estimator.mu_estimator, EquilibriumMu)
        assert np.array_equal(
            model.factor_prior_estimator.mu_estimator.weights, weights
        )

    def test_fit_produces_full_universe_moments(self, asset_panel) -> None:  # type: ignore[no-untyped-def]
        cfg = MomentEstimationConfig(min_regression_assets=3)
        model = build_characteristics_factor_model(cfg, factors=_make_factors())
        model.fit(characteristics=asset_panel)
        rd = model.return_distribution_
        assert rd.mu.shape == (12,)
        assert rd.covariance.shape == (12, 12)
