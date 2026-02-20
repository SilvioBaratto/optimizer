"""Tests for view integration factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.prior import (
    BlackLitterman,
    EntropyPooling,
    FactorModel,
    OpinionPooling,
)
from skfolio.prior._base import BasePrior

from optimizer.moments import MomentEstimationConfig, MuEstimatorType
from optimizer.views import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
    build_black_litterman,
    build_entropy_pooling,
    build_opinion_pooling,
)
from optimizer.views._factory import _EmpiricalOmegaBlackLitterman


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


class TestBuildBlackLitterman:
    def test_produces_black_litterman_instance(self) -> None:
        cfg = BlackLittermanConfig(views=("TICK_00 == 0.05",))
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)

    def test_tau_forwarded(self) -> None:
        cfg = BlackLittermanConfig(views=("TICK_00 == 0.05",), tau=0.1)
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert prior.tau == 0.1

    def test_risk_free_rate_forwarded(self) -> None:
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",), risk_free_rate=0.02
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert prior.risk_free_rate == 0.02

    def test_he_litterman_no_view_confidences(self) -> None:
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",),
            uncertainty_method=ViewUncertaintyMethod.HE_LITTERMAN,
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert prior.view_confidences is None

    def test_idzorek_view_confidences_forwarded(self) -> None:
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05", "TICK_01 == 0.03"),
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
            view_confidences=(0.8, 0.6),
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert prior.view_confidences == [0.8, 0.6]

    def test_groups_forwarded(self) -> None:
        groups = {"group_a": ["TICK_00", "TICK_01"]}
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",), groups=groups
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert prior.groups == groups

    def test_custom_prior_config_forwarded(self) -> None:
        prior_cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.SHRUNK
        )
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",), prior_config=prior_cfg
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        from skfolio.prior import EmpiricalPrior

        assert isinstance(prior.prior_estimator, EmpiricalPrior)
        from skfolio.moments import ShrunkMu

        assert isinstance(prior.prior_estimator.mu_estimator, ShrunkMu)

    def test_factor_model_variant(self) -> None:
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",), use_factor_model=True
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, FactorModel)

    def test_factor_model_residual_variance(self) -> None:
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",),
            use_factor_model=True,
            residual_variance=False,
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, FactorModel)
        assert prior.residual_variance is False

    def test_views_converted_to_list(self) -> None:
        cfg = BlackLittermanConfig(views=("TICK_00 == 0.05",))
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        assert isinstance(prior.views, list)


class TestBuildEntropyPooling:
    def test_produces_entropy_pooling_instance(self) -> None:
        cfg = EntropyPoolingConfig(mean_views=("TICK_00 == 0.05",))
        prior = build_entropy_pooling(cfg)
        assert isinstance(prior, EntropyPooling)

    def test_mean_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(mean_views=("TICK_00 == 0.05",))
        prior = build_entropy_pooling(cfg)
        assert prior.mean_views == ["TICK_00 == 0.05"]

    def test_variance_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(variance_views=("TICK_00 == 0.04",))
        prior = build_entropy_pooling(cfg)
        assert prior.variance_views == ["TICK_00 == 0.04"]

    def test_correlation_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(
            correlation_views=("TICK_00; TICK_01 == 0.5",)
        )
        prior = build_entropy_pooling(cfg)
        assert prior.correlation_views == ["TICK_00; TICK_01 == 0.5"]

    def test_skew_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(skew_views=("TICK_00 == -0.5",))
        prior = build_entropy_pooling(cfg)
        assert prior.skew_views == ["TICK_00 == -0.5"]

    def test_kurtosis_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(kurtosis_views=("TICK_00 == 3.0",))
        prior = build_entropy_pooling(cfg)
        assert prior.kurtosis_views == ["TICK_00 == 3.0"]

    def test_cvar_views_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(cvar_views=("TICK_00 <= -0.05",))
        prior = build_entropy_pooling(cfg)
        assert prior.cvar_views == ["TICK_00 <= -0.05"]

    def test_cvar_beta_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(
            cvar_views=("TICK_00 <= -0.05",), cvar_beta=0.99
        )
        prior = build_entropy_pooling(cfg)
        assert prior.cvar_beta == 0.99

    def test_solver_forwarded(self) -> None:
        cfg = EntropyPoolingConfig(
            mean_views=("TICK_00 == 0.05",), solver="SLSQP"
        )
        prior = build_entropy_pooling(cfg)
        assert prior.solver == "SLSQP"

    def test_groups_forwarded(self) -> None:
        groups = {"group_a": ["TICK_00", "TICK_01"]}
        cfg = EntropyPoolingConfig(
            mean_views=("TICK_00 == 0.05",), groups=groups
        )
        prior = build_entropy_pooling(cfg)
        assert prior.groups == groups

    def test_custom_prior_config_forwarded(self) -> None:
        prior_cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.SHRUNK
        )
        cfg = EntropyPoolingConfig(
            mean_views=("TICK_00 == 0.05",), prior_config=prior_cfg
        )
        prior = build_entropy_pooling(cfg)
        from skfolio.prior import EmpiricalPrior

        assert isinstance(prior.prior_estimator, EmpiricalPrior)
        from skfolio.moments import ShrunkMu

        assert isinstance(prior.prior_estimator.mu_estimator, ShrunkMu)

    def test_none_views_stay_none(self) -> None:
        cfg = EntropyPoolingConfig(mean_views=("TICK_00 == 0.05",))
        prior = build_entropy_pooling(cfg)
        assert prior.variance_views is None
        assert prior.correlation_views is None
        assert prior.skew_views is None
        assert prior.kurtosis_views is None
        assert prior.cvar_views is None


class TestBuildOpinionPooling:
    def test_produces_opinion_pooling_instance(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        prior = build_opinion_pooling(estimators)
        assert isinstance(prior, OpinionPooling)

    def test_opinion_probabilities_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        cfg = OpinionPoolingConfig(opinion_probabilities=(0.6, 0.4))
        prior = build_opinion_pooling(estimators, cfg)
        assert prior.opinion_probabilities == [0.6, 0.4]

    def test_is_linear_pooling_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        cfg = OpinionPoolingConfig(is_linear_pooling=False)
        prior = build_opinion_pooling(estimators, cfg)
        assert prior.is_linear_pooling is False

    def test_divergence_penalty_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        cfg = OpinionPoolingConfig(divergence_penalty=0.1)
        prior = build_opinion_pooling(estimators, cfg)
        assert prior.divergence_penalty == 0.1

    def test_common_prior_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        prior_cfg = MomentEstimationConfig()
        cfg = OpinionPoolingConfig(prior_config=prior_cfg)
        prior = build_opinion_pooling(estimators, cfg)
        assert prior.prior_estimator is not None
        assert isinstance(prior.prior_estimator, EmpiricalPrior)

    def test_n_jobs_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        cfg = OpinionPoolingConfig(n_jobs=2)
        prior = build_opinion_pooling(estimators, cfg)
        assert prior.n_jobs == 2

    def test_default_config_when_none(self) -> None:
        from skfolio.prior import EmpiricalPrior

        estimators = [
            ("expert_1", EmpiricalPrior()),
            ("expert_2", EmpiricalPrior()),
        ]
        prior = build_opinion_pooling(estimators, config=None)
        assert isinstance(prior, OpinionPooling)
        assert prior.is_linear_pooling is True
        assert prior.divergence_penalty == 0.0


class TestIntegration:
    """Integration tests using real skfolio fit/predict."""

    def test_bl_fit_absolute_views(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = BlackLittermanConfig.for_equilibrium(
            views=("AAPL == 0.05", "JPM == 0.03")
        )
        prior = build_black_litterman(cfg)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None
        assert rd.mu.shape == (returns.shape[1],)

    def test_bl_fit_relative_views(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = BlackLittermanConfig.for_equilibrium(
            views=("AAPL - MSFT == 0.02",)
        )
        prior = build_black_litterman(cfg)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None

    def test_ep_fit_mean_views(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = EntropyPoolingConfig.for_mean_views(
            mean_views=("AAPL == 0.05",)
        )
        prior = build_entropy_pooling(cfg)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_ep_fit_stress_test(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = EntropyPoolingConfig.for_stress_test(
            variance_views=("AAPL == 0.04",),
            correlation_views=("(AAPL, JPM) == 0.5",),
        )
        prior = build_entropy_pooling(cfg)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_op_fit_combining_experts(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        # OpinionPooling requires individual experts have prior_estimator=None
        expert_1 = EntropyPooling(
            mean_views=["AAPL == 0.05"],
        )
        expert_2 = EntropyPooling(
            mean_views=["JPM == 0.03"],
        )
        estimators: list[tuple[str, BasePrior]] = [
            ("ep_1", expert_1),
            ("ep_2", expert_2),
        ]
        cfg = OpinionPoolingConfig(opinion_probabilities=(0.6, 0.4))
        prior = build_opinion_pooling(estimators, cfg)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_bl_factor_model_fit(self) -> None:
        from skfolio.datasets import load_factors_dataset, load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        factor_prices = load_factors_dataset()

        common_idx = prices.index.intersection(factor_prices.index)
        asset_returns = prices_to_returns(prices.loc[common_idx])
        factor_returns = prices_to_returns(factor_prices.loc[common_idx])

        # Views must reference factor names when wrapped in FactorModel
        cfg = BlackLittermanConfig.for_factor_model(
            views=("MTUM == 0.05",)
        )
        prior = build_black_litterman(cfg)
        prior.fit(asset_returns, y=factor_returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_empirical_omega_via_histories(self) -> None:
        """build_black_litterman with view/return histories produces valid posterior."""
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        n_obs = 30
        dates = returns.index[-n_obs:]
        # Simulate 1 absolute view with forecast history
        view_h = pd.DataFrame(
            {"view_0": [0.001] * n_obs}, index=dates, dtype=float
        )
        ret_h = pd.DataFrame(
            returns.loc[dates, "AAPL"].rename("view_0")
        )
        cfg = BlackLittermanConfig(
            views=("AAPL == 0.05",),
            uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
        )
        prior = build_black_litterman(cfg, view_history=view_h, return_history=ret_h)
        assert isinstance(prior, _EmpiricalOmegaBlackLitterman)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.mu.shape == (returns.shape[1],)
        assert np.all(np.isfinite(rd.mu))

    def test_empirical_omega_via_precomputed_array(self) -> None:
        """build_black_litterman accepts a pre-computed omega directly."""
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        omega = np.diag([1e-4])
        cfg = BlackLittermanConfig(
            views=("AAPL == 0.05",),
            uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
        )
        prior = build_black_litterman(cfg, omega=omega)
        assert isinstance(prior, _EmpiricalOmegaBlackLitterman)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.covariance is not None

    def test_empirical_omega_raises_without_data(self) -> None:
        """EMPIRICAL_TRACK_RECORD without omega or histories raises ValueError."""
        cfg = BlackLittermanConfig(
            views=("TICK_00 == 0.05",),
            uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
        )
        with pytest.raises(ValueError, match="EMPIRICAL_TRACK_RECORD"):
            build_black_litterman(cfg)

    def test_empirical_omega_covariance_is_psd(self) -> None:
        """Posterior covariance from empirical omega is positive semi-definite."""
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        omega = np.diag([5e-5])
        cfg = BlackLittermanConfig(
            views=("AAPL == 0.05",),
            uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
        )
        prior = build_black_litterman(cfg, omega=omega)
        prior.fit(returns)
        eigvals = np.linalg.eigvalsh(prior.return_distribution_.covariance)
        assert np.all(eigvals >= -1e-10)

    def test_bl_composes_with_meanrisk(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.optimization import MeanRisk
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = BlackLittermanConfig.for_equilibrium(
            views=("AAPL == 0.05",)
        )
        prior = build_black_litterman(cfg)
        model = MeanRisk(prior_estimator=prior)
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) > 0
