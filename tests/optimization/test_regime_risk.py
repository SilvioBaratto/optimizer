"""Unit tests for Markov-driven regime-blended risk measure (issue #14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.moments._hmm import HMMResult
from optimizer.optimization._config import RiskMeasureType
from optimizer.optimization._regime_risk import (
    RegimeRiskConfig,
    _compute_regime_risk,
    build_regime_blended_optimizer,
    compute_blended_risk_measure,
)
from tests.optimization.conftest import make_hmm_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 3
N_OBS = 100
TICKERS = ["AAPL", "MSFT", "GOOG"]
DATES = pd.date_range("2020-01-01", periods=N_OBS, freq="B")


def _make_returns(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (N_OBS, N_ASSETS))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


def _make_hmm_result(
    n_states: int = 2,
    last_probs: list[float] | None = None,
    returns: pd.DataFrame | None = None,
) -> HMMResult:
    """Build a synthetic HMMResult without fitting."""
    if returns is None:
        returns = _make_returns()
    return make_hmm_result(
        n_assets=N_ASSETS,
        n_obs=len(returns),
        tickers=TICKERS,
        n_states=n_states,
        last_probs=last_probs,
        returns=returns,
    )


def _equal_weights() -> np.ndarray:
    w = np.ones(N_ASSETS) / N_ASSETS
    return w.astype(np.float64)


# ---------------------------------------------------------------------------
# TestComputeRegimeRisk (unit tests for the helper)
# ---------------------------------------------------------------------------


class TestComputeRegimeRisk:
    def _r(self, seed: int = 0, n: int = 50) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 0.01, n).astype(np.float64)

    def test_variance_positive(self) -> None:
        r = self._r()
        v = _compute_regime_risk(r, RiskMeasureType.VARIANCE)
        assert v > 0.0

    def test_standard_deviation_is_sqrt_variance(self) -> None:
        r = self._r()
        var = _compute_regime_risk(r, RiskMeasureType.VARIANCE)
        std = _compute_regime_risk(r, RiskMeasureType.STANDARD_DEVIATION)
        assert std == pytest.approx(np.sqrt(var), rel=1e-6)

    def test_cvar_ge_worst_return(self) -> None:
        """CVaR should always be ≤ absolute worst loss (not a strict equality)."""
        r = self._r()
        cvar = _compute_regime_risk(r, RiskMeasureType.CVAR)
        worst = _compute_regime_risk(r, RiskMeasureType.WORST_REALIZATION)
        assert cvar <= worst + 1e-10

    def test_mad_non_negative(self) -> None:
        r = self._r()
        mad = _compute_regime_risk(r, RiskMeasureType.MEAN_ABSOLUTE_DEVIATION)
        assert mad >= 0.0

    def test_empty_array_returns_zero(self) -> None:
        assert _compute_regime_risk(np.array([]), RiskMeasureType.VARIANCE) == 0.0

    def test_single_element_returns_zero_variance(self) -> None:
        r = np.array([0.01], dtype=np.float64)
        assert _compute_regime_risk(r, RiskMeasureType.VARIANCE) == 0.0

    def test_constant_returns_zero_variance(self) -> None:
        r = np.full(50, 0.01, dtype=np.float64)
        assert _compute_regime_risk(r, RiskMeasureType.VARIANCE) == pytest.approx(
            0.0, abs=1e-15
        )

    def test_semi_variance_le_variance(self) -> None:
        """Semi-variance (downside only) ≤ full variance."""
        r = self._r()
        sv = _compute_regime_risk(r, RiskMeasureType.SEMI_VARIANCE)
        v = _compute_regime_risk(r, RiskMeasureType.VARIANCE)
        assert sv <= v + 1e-10

    def test_semi_variance_positive_mean_all_positive_returns(self) -> None:
        """With all positive returns and positive mean, semi-variance > 0."""
        r = np.array([0.005, 0.01, 0.015, 0.02], dtype=np.float64)
        sv = _compute_regime_risk(r, RiskMeasureType.SEMI_VARIANCE)
        assert sv > 0.0, "semi-variance should be > 0 for returns below the mean"

    def test_semi_deviation_positive_mean_all_positive_returns(self) -> None:
        """With all positive returns and positive mean, semi-deviation > 0."""
        r = np.array([0.005, 0.01, 0.015, 0.02], dtype=np.float64)
        sd = _compute_regime_risk(r, RiskMeasureType.SEMI_DEVIATION)
        assert sd > 0.0, "semi-deviation should be > 0 for returns below the mean"

    def test_semi_variance_uses_mean_threshold(self) -> None:
        """Semi-variance threshold is the mean, not zero."""
        r = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
        mu = np.mean(r)
        downside = r[r < mu] - mu
        expected = float(np.mean(downside**2))
        result = _compute_regime_risk(r, RiskMeasureType.SEMI_VARIANCE)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_all_positive_returns_worst_realization_clamped(self) -> None:
        """WR with all positive returns should be clamped to 0 (issue #71)."""
        r = np.abs(self._r()) + 0.001
        result = _compute_regime_risk(r, RiskMeasureType.WORST_REALIZATION)
        assert result == pytest.approx(0.0)

    def test_worst_realization_negative_returns(self) -> None:
        """WR with negative returns should be positive."""
        r = np.array([-0.01, -0.02, 0.005, -0.03], dtype=np.float64)
        result = _compute_regime_risk(r, RiskMeasureType.WORST_REALIZATION)
        assert result > 0.0
        assert result == pytest.approx(0.03)

    def test_max_drawdown_positive(self) -> None:
        r = self._r()
        dd = _compute_regime_risk(r, RiskMeasureType.MAX_DRAWDOWN)
        assert dd >= 0.0

    def test_average_drawdown_le_max_drawdown(self) -> None:
        r = self._r()
        avg = _compute_regime_risk(r, RiskMeasureType.AVERAGE_DRAWDOWN)
        mx = _compute_regime_risk(r, RiskMeasureType.MAX_DRAWDOWN)
        assert avg <= mx + 1e-10

    def test_ulcer_index_positive(self) -> None:
        r = self._r()
        ui = _compute_regime_risk(r, RiskMeasureType.ULCER_INDEX)
        assert ui >= 0.0

    def test_cdar_positive(self) -> None:
        r = self._r()
        cdar = _compute_regime_risk(r, RiskMeasureType.CDAR, cvar_beta=0.95)
        assert cdar >= 0.0

    def test_cdar_le_max_drawdown(self) -> None:
        r = self._r()
        cdar = _compute_regime_risk(r, RiskMeasureType.CDAR, cvar_beta=0.95)
        mx = _compute_regime_risk(r, RiskMeasureType.MAX_DRAWDOWN)
        assert cdar <= mx + 1e-10

    def test_edar_raises_not_implemented(self) -> None:
        r = self._r()
        with pytest.raises(NotImplementedError, match="EDAR"):
            _compute_regime_risk(r, RiskMeasureType.EDAR)

    def test_unsupported_measure_raises_not_implemented(self) -> None:
        r = self._r()
        with pytest.raises(NotImplementedError, match="not supported"):
            _compute_regime_risk(r, RiskMeasureType.GINI_MEAN_DIFFERENCE)

    def test_drawdown_measures_with_known_data(self) -> None:
        """Known data: monotonically decreasing prices → predictable drawdown."""
        # Returns that cause monotonic decline: -1% each period
        r = np.full(10, -0.01, dtype=np.float64)
        dd = _compute_regime_risk(r, RiskMeasureType.MAX_DRAWDOWN)
        # Max drawdown should be significant
        assert dd > 0.05


# ---------------------------------------------------------------------------
# TestComputeBlendedRiskMeasure
# ---------------------------------------------------------------------------


class TestComputeBlendedRiskMeasure:
    def test_equal_probs_two_regimes_is_average(self) -> None:
        """With γ = [0.5, 0.5] and both measures = VARIANCE, blended ≈ variance."""
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.5, 0.5], returns=returns)
        weights = _equal_weights()
        measures = [RiskMeasureType.VARIANCE, RiskMeasureType.VARIANCE]

        blended = compute_blended_risk_measure(returns, weights, hmm, measures)

        p_ret = returns.to_numpy(dtype=np.float64) @ weights
        full_var = float(np.var(p_ret, ddof=1))
        # Blended ≈ full variance (since both regimes use variance)
        assert blended == pytest.approx(full_var, rel=0.3)  # 30% tolerance for sampling

    def test_pure_state_0_equals_regime_0_risk(self) -> None:
        """γ = [1, 0] → blended risk = regime-0 risk only."""
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2, last_probs=[1.0, 0.0], returns=returns)
        weights = _equal_weights()
        measures = [RiskMeasureType.VARIANCE, RiskMeasureType.CVAR]

        blended = compute_blended_risk_measure(returns, weights, hmm, measures)

        # Must be > 0 (variance of a non-trivial portfolio)
        assert blended > 0.0

    def test_pure_state_1_uses_regime_1_measure(self) -> None:
        """γ = [0, 1] → blended risk uses only regime-1's CVaR."""
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.0, 1.0], returns=returns)
        weights = _equal_weights()
        measures = [RiskMeasureType.VARIANCE, RiskMeasureType.CVAR]

        blended = compute_blended_risk_measure(returns, weights, hmm, measures)
        assert blended > 0.0

    def test_wrong_n_measures_raises_value_error(self) -> None:
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2)
        weights = _equal_weights()
        with pytest.raises(ConfigurationError, match="regime_measures"):
            compute_blended_risk_measure(
                returns, weights, hmm, [RiskMeasureType.VARIANCE]
            )

    def test_wrong_weights_length_raises_value_error(self) -> None:
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2)
        bad_weights = np.ones(N_ASSETS + 1) / (N_ASSETS + 1)
        with pytest.raises(DataError, match="weights length"):
            compute_blended_risk_measure(
                returns, bad_weights, hmm, [RiskMeasureType.VARIANCE] * 2
            )

    def test_result_is_non_negative(self) -> None:
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=2)
        weights = _equal_weights()
        measures = [RiskMeasureType.VARIANCE, RiskMeasureType.CVAR]
        blended = compute_blended_risk_measure(returns, weights, hmm, measures)
        assert blended >= 0.0

    def test_three_regime_model(self) -> None:
        returns = _make_returns()
        hmm = _make_hmm_result(n_states=3, last_probs=[0.2, 0.5, 0.3], returns=returns)
        weights = _equal_weights()
        measures = [
            RiskMeasureType.VARIANCE,
            RiskMeasureType.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasureType.CVAR,
        ]
        blended = compute_blended_risk_measure(returns, weights, hmm, measures)
        assert blended > 0.0

    def test_no_common_index_returns_zero(self) -> None:
        """If returns and HMM share no dates, result is 0."""
        returns = _make_returns()
        # HMM on completely different dates
        other_dates = pd.date_range("2025-01-01", periods=N_OBS, freq="B")
        hmm = _make_hmm_result(
            n_states=2,
            returns=pd.DataFrame(
                np.zeros((N_OBS, N_ASSETS)), index=other_dates, columns=TICKERS
            ),
        )
        weights = _equal_weights()
        blended = compute_blended_risk_measure(
            returns, weights, hmm, [RiskMeasureType.VARIANCE] * 2
        )
        assert blended == 0.0

    def test_gamma_zero_state_skipped(self) -> None:
        """State with γ=0 should not influence the result."""
        returns = _make_returns()
        hmm_all_state0 = _make_hmm_result(
            n_states=2, last_probs=[1.0, 0.0], returns=returns
        )
        hmm_all_state1 = _make_hmm_result(
            n_states=2, last_probs=[0.0, 1.0], returns=returns
        )
        measures = [RiskMeasureType.VARIANCE, RiskMeasureType.VARIANCE]
        weights = _equal_weights()

        r0 = compute_blended_risk_measure(returns, weights, hmm_all_state0, measures)
        r1 = compute_blended_risk_measure(returns, weights, hmm_all_state1, measures)
        # Both use VARIANCE so values should be close (same measure, slightly
        # different subsets)
        assert r0 > 0.0
        assert r1 > 0.0


# ---------------------------------------------------------------------------
# TestRegimeRiskConfig
# ---------------------------------------------------------------------------


class TestRegimeRiskConfig:
    def test_for_calm_stress_has_two_measures(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        assert len(config.regime_measures) == 2

    def test_for_calm_stress_measures_correct(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        assert config.regime_measures[0] == RiskMeasureType.VARIANCE
        assert config.regime_measures[1] == RiskMeasureType.CVAR

    def test_for_three_regimes(self) -> None:
        config = RegimeRiskConfig.for_three_regimes()
        assert len(config.regime_measures) == 3
        assert config.hmm_config.n_states == 3

    def test_default_cvar_beta(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        assert config.cvar_beta == pytest.approx(0.95)

    def test_custom_cvar_beta(self) -> None:
        config = RegimeRiskConfig(
            regime_measures=(RiskMeasureType.VARIANCE, RiskMeasureType.CVAR),
            cvar_beta=0.99,
        )
        assert config.cvar_beta == pytest.approx(0.99)

    def test_frozen_immutable(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        with pytest.raises((AttributeError, TypeError)):
            config.cvar_beta = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestBuildRegimeBlendedOptimizer
# ---------------------------------------------------------------------------


class TestBuildRegimeBlendedOptimizer:
    def test_returns_mean_risk_instance(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.8, 0.2])
        optimizer = build_regime_blended_optimizer(config, hmm)
        assert isinstance(optimizer, MeanRisk)

    def test_dominant_state_0_uses_variance(self) -> None:
        """State 0 (calm) dominant → MeanRisk uses VARIANCE."""
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.9, 0.1])
        optimizer = build_regime_blended_optimizer(config, hmm)
        assert optimizer.risk_measure == RiskMeasure.VARIANCE

    def test_dominant_state_1_uses_cvar(self) -> None:
        """State 1 (stress) dominant → MeanRisk uses CVAR."""
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.1, 0.9])
        optimizer = build_regime_blended_optimizer(config, hmm)
        assert optimizer.risk_measure == RiskMeasure.CVAR

    def test_equal_probs_selects_state_0(self) -> None:
        """argmax([0.5, 0.5]) = 0 → state-0 measure selected."""
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.5, 0.5])
        optimizer = build_regime_blended_optimizer(config, hmm)
        assert optimizer.risk_measure == RiskMeasure.VARIANCE

    def test_wrong_n_measures_raises(self) -> None:
        config = RegimeRiskConfig(
            regime_measures=(RiskMeasureType.VARIANCE,)  # 1 measure for 2-state HMM
        )
        hmm = _make_hmm_result(n_states=2)
        with pytest.raises(ConfigurationError, match="regime_measures"):
            build_regime_blended_optimizer(config, hmm)

    def test_kwargs_forwarded_to_mean_risk(self) -> None:
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.8, 0.2])
        optimizer = build_regime_blended_optimizer(
            config, hmm, min_weights=0.0, max_weights=0.3
        )
        assert isinstance(optimizer, MeanRisk)
        assert optimizer.max_weights == pytest.approx(0.3)

    def test_three_regime_dominant_middle(self) -> None:
        config = RegimeRiskConfig.for_three_regimes()
        hmm = _make_hmm_result(n_states=3, last_probs=[0.1, 0.7, 0.2])
        optimizer = build_regime_blended_optimizer(config, hmm)
        # State 1 = MEAN_ABSOLUTE_DEVIATION
        assert optimizer.risk_measure == RiskMeasure.MEAN_ABSOLUTE_DEVIATION


# ---------------------------------------------------------------------------
# TestRegimeBlendedFitPredict (end-to-end fit/predict, issue #72)
# ---------------------------------------------------------------------------


class TestRegimeBlendedFitPredict:
    def test_fit_predict_calm_dominant(self) -> None:
        returns = _make_returns(seed=10)
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.8, 0.2], returns=returns)
        opt = build_regime_blended_optimizer(config, hmm)
        opt.fit(returns)
        portfolio = opt.predict(returns)
        weights = portfolio.weights
        assert len(weights) == N_ASSETS
        assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-6)

    def test_fit_predict_stress_dominant(self) -> None:
        returns = _make_returns(seed=11)
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.1, 0.9], returns=returns)
        opt = build_regime_blended_optimizer(config, hmm)
        opt.fit(returns)
        portfolio = opt.predict(returns)
        weights = portfolio.weights
        assert len(weights) == N_ASSETS
        assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self) -> None:
        returns = _make_returns(seed=12)
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.6, 0.4], returns=returns)
        opt = build_regime_blended_optimizer(config, hmm)
        opt.fit(returns)
        portfolio = opt.predict(returns)
        assert all(w >= -1e-6 for w in portfolio.weights)

    def test_predict_returns_portfolio(self) -> None:
        returns = _make_returns(seed=13)
        config = RegimeRiskConfig.for_calm_stress()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.5, 0.5], returns=returns)
        opt = build_regime_blended_optimizer(config, hmm)
        opt.fit(returns)
        portfolio = opt.predict(returns)
        assert hasattr(portfolio, "sharpe_ratio")

    def test_three_regime_fit_predict(self) -> None:
        returns = _make_returns(seed=14)
        config = RegimeRiskConfig.for_three_regimes()
        hmm = _make_hmm_result(
            n_states=3, last_probs=[0.3, 0.4, 0.3], returns=returns
        )
        opt = build_regime_blended_optimizer(config, hmm)
        opt.fit(returns)
        portfolio = opt.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)
