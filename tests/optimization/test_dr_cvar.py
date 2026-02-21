"""Unit tests for distributionally robust CVaR optimization (issue #19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import DistributionallyRobustCVaR, MeanRisk

from optimizer.optimization._config import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
)
from optimizer.optimization._dr_cvar import DRCVaRConfig, build_dr_cvar
from optimizer.optimization._factory import build_mean_risk

# ---------------------------------------------------------------------------
# Shared fixtures — returns provided by optimization/conftest.py
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_OBS = 252
TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]
DATES = pd.date_range("2020-01-02", periods=N_OBS, freq="B")


@pytest.fixture(scope="module")
def returns(returns_10a_252: pd.DataFrame) -> pd.DataFrame:
    return returns_10a_252


# ---------------------------------------------------------------------------
# TestDRCVaRConfig
# ---------------------------------------------------------------------------


class TestDRCVaRConfig:
    def test_is_frozen(self) -> None:
        cfg = DRCVaRConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.epsilon = 99.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        cfg = DRCVaRConfig(epsilon=0.005)
        assert isinstance(hash(cfg), int)

    def test_usable_in_set(self) -> None:
        s = {
            DRCVaRConfig(epsilon=0.001),
            DRCVaRConfig(epsilon=0.01),
            DRCVaRConfig(epsilon=0.001),
        }
        assert len(s) == 2

    def test_default_epsilon(self) -> None:
        assert DRCVaRConfig().epsilon == pytest.approx(0.001)

    def test_default_alpha(self) -> None:
        assert DRCVaRConfig().alpha == pytest.approx(0.95)

    def test_default_risk_aversion(self) -> None:
        assert DRCVaRConfig().risk_aversion == pytest.approx(1.0)

    def test_default_norm(self) -> None:
        assert DRCVaRConfig().norm == 1

    def test_default_min_weights(self) -> None:
        assert DRCVaRConfig().min_weights == pytest.approx(0.0)

    def test_default_max_weights(self) -> None:
        assert DRCVaRConfig().max_weights == pytest.approx(1.0)

    def test_default_budget(self) -> None:
        assert DRCVaRConfig().budget == pytest.approx(1.0)

    def test_for_conservative_epsilon(self) -> None:
        assert DRCVaRConfig.for_conservative().epsilon == pytest.approx(0.01)

    def test_for_standard_epsilon(self) -> None:
        assert DRCVaRConfig.for_standard().epsilon == pytest.approx(0.001)

    def test_prior_config_field(self) -> None:
        assert DRCVaRConfig().prior_config is None

    def test_usable_in_param_grid(self) -> None:
        grid = [
            {"epsilon": [0.0, 0.001, 0.01]},
            {"config": [DRCVaRConfig.for_standard(), DRCVaRConfig.for_conservative()]},
        ]
        assert len(grid) == 2


# ---------------------------------------------------------------------------
# TestBuildDrCvar — dispatcher behaviour
# ---------------------------------------------------------------------------


class TestBuildDrCvarDispatch:
    def test_epsilon_zero_returns_mean_risk(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
        assert isinstance(model, MeanRisk)

    def test_epsilon_positive_returns_dr_cvar(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.001))
        assert isinstance(model, DistributionallyRobustCVaR)

    def test_default_config_returns_dr_cvar(self) -> None:
        model = build_dr_cvar()
        assert isinstance(model, DistributionallyRobustCVaR)

    def test_negative_epsilon_raises(self) -> None:
        """Negative epsilon raises ValueError (issue #73)."""
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            build_dr_cvar(DRCVaRConfig(epsilon=-0.01))

    def test_none_config_uses_defaults(self) -> None:
        model = build_dr_cvar(None)
        assert isinstance(model, DistributionallyRobustCVaR)

    def test_epsilon_zero_mean_risk_objective_is_minimize_risk(self) -> None:
        from skfolio.optimization.convex._base import ObjectiveFunction

        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
        assert isinstance(model, MeanRisk)
        assert model.objective_function == ObjectiveFunction.MINIMIZE_RISK

    def test_epsilon_zero_mean_risk_risk_measure_is_cvar(self) -> None:
        from skfolio.measures import RiskMeasure

        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
        assert isinstance(model, MeanRisk)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_alpha_forwarded_to_mean_risk(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0, alpha=0.99))
        assert isinstance(model, MeanRisk)
        assert model.cvar_beta == pytest.approx(0.99)

    def test_alpha_forwarded_to_dr_cvar(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.005, alpha=0.99))
        assert isinstance(model, DistributionallyRobustCVaR)
        assert model.cvar_beta == pytest.approx(0.99)

    def test_epsilon_forwarded_as_wasserstein_radius(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.007))
        assert isinstance(model, DistributionallyRobustCVaR)
        assert model.wasserstein_ball_radius == pytest.approx(0.007)

    def test_risk_aversion_forwarded(self) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.005, risk_aversion=2.0))
        assert isinstance(model, DistributionallyRobustCVaR)
        assert model.risk_aversion == pytest.approx(2.0)

    def test_kwargs_forwarded_to_mean_risk(self) -> None:
        prev = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0), previous_weights=prev)
        assert isinstance(model, MeanRisk)
        np.testing.assert_array_equal(model.previous_weights, prev)

    def test_kwargs_forwarded_to_dr_cvar(self) -> None:
        prev = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.001), previous_weights=prev)
        assert isinstance(model, DistributionallyRobustCVaR)
        np.testing.assert_array_equal(model.previous_weights, prev)


# ---------------------------------------------------------------------------
# TestBuildDrCvar — fit & predict
# ---------------------------------------------------------------------------


class TestBuildDrCvarFit:
    def test_epsilon_zero_fits_and_predicts(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == N_ASSETS

    def test_epsilon_positive_fits_and_predicts(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.001))
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == N_ASSETS

    def test_weights_sum_to_one_epsilon_zero(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
        model.fit(returns)
        w = model.predict(returns).weights
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)

    def test_weights_sum_to_one_epsilon_positive(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.001))
        model.fit(returns)
        w = model.predict(returns).weights
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)

    def test_long_only_epsilon_zero(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.0, min_weights=0.0))
        model.fit(returns)
        w = model.predict(returns).weights
        assert np.all(w >= -1e-8)

    def test_long_only_epsilon_positive(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig(epsilon=0.001, min_weights=0.0))
        model.fit(returns)
        w = model.predict(returns).weights
        assert np.all(w >= -1e-8)

    # -- ε=0 acceptance criterion -----------------------------------------

    def test_epsilon_zero_matches_standard_cvar(self, returns: pd.DataFrame) -> None:
        """ε=0 → identical to build_mean_risk with MINIMIZE_RISK + CVAR."""
        standard = build_mean_risk(
            MeanRiskConfig(
                objective=ObjectiveFunctionType.MINIMIZE_RISK,
                risk_measure=RiskMeasureType.CVAR,
                cvar_beta=0.95,
            )
        )
        dr_zero = build_dr_cvar(DRCVaRConfig(epsilon=0.0, alpha=0.95))

        standard.fit(returns)
        dr_zero.fit(returns)

        w_std = standard.predict(returns).weights
        w_dr = dr_zero.predict(returns).weights

        np.testing.assert_allclose(w_dr, w_std, atol=1e-6)

    # -- ε monotonicity: larger ε → more diversified ----------------------

    def test_larger_epsilon_more_diversified(self, returns: pd.DataFrame) -> None:
        """Larger ε produces a more diversified portfolio (lower max weight)."""
        epsilons = [0.001, 0.005, 0.05]
        max_weights: list[float] = []

        for eps in epsilons:
            model = build_dr_cvar(DRCVaRConfig(epsilon=eps))
            model.fit(returns)
            w = model.predict(returns).weights
            max_weights.append(float(np.max(w)))

        assert max_weights[0] >= max_weights[1] - 1e-4
        assert max_weights[1] >= max_weights[2] - 1e-4

    # -- presets -----------------------------------------------------------

    def test_conservative_preset_fits(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig.for_conservative())
        model.fit(returns)
        w = model.predict(returns).weights
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)

    def test_standard_preset_fits(self, returns: pd.DataFrame) -> None:
        model = build_dr_cvar(DRCVaRConfig.for_standard())
        model.fit(returns)
        w = model.predict(returns).weights
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
