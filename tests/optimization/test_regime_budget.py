"""Unit tests for regime-conditional risk budgets (issue #15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import RiskBudgeting

from optimizer.exceptions import ConfigurationError
from optimizer.moments._hmm import HMMResult
from optimizer.optimization._config import RiskBudgetingConfig
from optimizer.optimization._regime_risk import (
    build_regime_risk_budgeting,
    compute_regime_budget,
)
from tests.optimization.conftest import make_hmm_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 4
N_OBS = 80
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN"]
DATES = pd.date_range("2020-01-01", periods=N_OBS, freq="B")


def _make_hmm_result(
    n_states: int = 2,
    last_probs: list[float] | None = None,
) -> HMMResult:
    return make_hmm_result(
        n_assets=N_ASSETS,
        n_obs=N_OBS,
        tickers=TICKERS,
        n_states=n_states,
        last_probs=last_probs,
    )


def _uniform_budget() -> np.ndarray:
    return np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)


# ---------------------------------------------------------------------------
# TestComputeRegimeBudget
# ---------------------------------------------------------------------------


class TestComputeRegimeBudget:
    def test_equal_probs_returns_average(self) -> None:
        """γ = [0.5, 0.5] → blended = 0.5*b0 + 0.5*b1."""
        b0 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        b1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        gamma = np.array([0.5, 0.5], dtype=np.float64)

        result = compute_regime_budget([b0, b1], gamma)
        expected = (b0 + b1) / 2.0

        np.testing.assert_allclose(result, expected / expected.sum(), atol=1e-12)

    def test_pure_state_0_returns_b0(self) -> None:
        """γ = [1, 0] → blended = b0 (after normalisation)."""
        b0 = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
        b1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        gamma = np.array([1.0, 0.0], dtype=np.float64)

        result = compute_regime_budget([b0, b1], gamma)
        np.testing.assert_allclose(result, b0 / b0.sum(), atol=1e-12)

    def test_pure_state_1_returns_b1(self) -> None:
        """γ = [0, 1] → blended = b1 (after normalisation)."""
        b0 = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
        b1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        gamma = np.array([0.0, 1.0], dtype=np.float64)

        result = compute_regime_budget([b0, b1], gamma)
        np.testing.assert_allclose(result, b1 / b1.sum(), atol=1e-12)

    def test_sums_to_one(self) -> None:
        """Blended budget always sums to 1.0."""
        rng = np.random.default_rng(0)
        b0 = rng.dirichlet(np.ones(N_ASSETS))
        b1 = rng.dirichlet(np.ones(N_ASSETS))
        gamma = np.array([0.3, 0.7], dtype=np.float64)

        result = compute_regime_budget(
            [b0.astype(np.float64), b1.astype(np.float64)], gamma
        )
        assert float(result.sum()) == pytest.approx(1.0, abs=1e-8)

    def test_all_values_non_negative(self) -> None:
        """No negative budget weights."""
        rng = np.random.default_rng(1)
        b0 = rng.dirichlet(np.ones(N_ASSETS)).astype(np.float64)
        b1 = rng.dirichlet(np.ones(N_ASSETS)).astype(np.float64)
        gamma = np.array([0.6, 0.4], dtype=np.float64)

        result = compute_regime_budget([b0, b1], gamma)
        assert np.all(result >= 0.0)

    def test_three_regimes_weighted_average(self) -> None:
        """Three-state γ = [0.2, 0.5, 0.3] → correct weighted sum."""
        b0 = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float64)
        b1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        b2 = np.array([0.1, 0.1, 0.3, 0.5], dtype=np.float64)
        gamma = np.array([0.2, 0.5, 0.3], dtype=np.float64)

        result = compute_regime_budget([b0, b1, b2], gamma)
        raw = 0.2 * b0 + 0.5 * b1 + 0.3 * b2
        expected = raw / raw.sum()

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_wrong_n_budgets_raises_value_error(self) -> None:
        b0 = _uniform_budget()
        gamma = np.array([0.5, 0.5], dtype=np.float64)
        with pytest.raises(ConfigurationError, match="regime_budgets"):
            compute_regime_budget([b0], gamma)  # 1 budget for 2-state probs

    def test_shape_preserved(self) -> None:
        """Output shape equals budget vector length."""
        b0 = _uniform_budget()
        b1 = _uniform_budget()
        gamma = np.array([0.4, 0.6], dtype=np.float64)
        result = compute_regime_budget([b0, b1], gamma)
        assert result.shape == (N_ASSETS,)

    def test_uniform_budgets_stay_uniform(self) -> None:
        """If all regime budgets are uniform, blended budget is uniform."""
        b0 = _uniform_budget()
        b1 = _uniform_budget()
        gamma = np.array([0.3, 0.7], dtype=np.float64)
        result = compute_regime_budget([b0, b1], gamma)
        np.testing.assert_allclose(result, _uniform_budget(), atol=1e-12)

    def test_integer_budget_arrays_accepted(self) -> None:
        """Budget arrays with integer dtype are cast to float64."""
        b0 = np.array([1, 1, 1, 1])  # integer dtype
        b1 = np.array([2, 2, 1, 1])
        gamma = np.array([1.0, 0.0], dtype=np.float64)
        result = compute_regime_budget([b0, b1], gamma)  # type: ignore[arg-type]
        assert result.dtype == np.float64
        assert float(result.sum()) == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# TestBuildRegimeRiskBudgeting
# ---------------------------------------------------------------------------


class TestBuildRegimeRiskBudgeting:
    def test_returns_risk_budgeting_instance(self) -> None:
        hmm = _make_hmm_result(n_states=2, last_probs=[0.7, 0.3])
        b0 = _uniform_budget()
        b1 = _uniform_budget()
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(config, hmm, [b0, b1])
        assert isinstance(optimizer, RiskBudgeting)

    def test_pure_state_0_budget_equals_b0(self) -> None:
        """γ = [1, 0] → risk_budget in optimizer == b0 (normalised)."""
        b0 = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
        b1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        hmm = _make_hmm_result(n_states=2, last_probs=[1.0, 0.0])
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(config, hmm, [b0, b1])
        expected = b0 / b0.sum()
        np.testing.assert_allclose(optimizer.risk_budget, expected, atol=1e-10)

    def test_pure_state_1_budget_equals_b1(self) -> None:
        """γ = [0, 1] → risk_budget in optimizer == b1 (normalised)."""
        b0 = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
        b1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        hmm = _make_hmm_result(n_states=2, last_probs=[0.0, 1.0])
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(config, hmm, [b0, b1])
        expected = b1 / b1.sum()
        np.testing.assert_allclose(optimizer.risk_budget, expected, atol=1e-10)

    def test_budget_sums_to_one(self) -> None:
        b0 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        b1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        hmm = _make_hmm_result(n_states=2, last_probs=[0.6, 0.4])
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(config, hmm, [b0, b1])
        assert float(optimizer.risk_budget.sum()) == pytest.approx(1.0, abs=1e-8)

    def test_wrong_n_budgets_raises_value_error(self) -> None:
        hmm = _make_hmm_result(n_states=2)
        config = RiskBudgetingConfig.for_risk_parity()
        with pytest.raises(ConfigurationError, match="regime_budgets"):
            build_regime_risk_budgeting(config, hmm, [_uniform_budget()])

    def test_three_regime_model(self) -> None:
        b0 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        b1 = _uniform_budget()
        b2 = np.array([0.1, 0.1, 0.3, 0.5], dtype=np.float64)
        hmm = _make_hmm_result(n_states=3, last_probs=[0.2, 0.5, 0.3])
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(config, hmm, [b0, b1, b2])
        assert isinstance(optimizer, RiskBudgeting)
        assert float(optimizer.risk_budget.sum()) == pytest.approx(1.0, abs=1e-8)

    def test_kwargs_forwarded(self) -> None:
        """Extra kwargs (e.g. transaction_costs) are forwarded to RiskBudgeting."""
        b0 = _uniform_budget()
        b1 = _uniform_budget()
        hmm = _make_hmm_result(n_states=2, last_probs=[0.5, 0.5])
        config = RiskBudgetingConfig.for_risk_parity()
        optimizer = build_regime_risk_budgeting(
            config, hmm, [b0, b1], transaction_costs=0.001
        )
        assert optimizer.transaction_costs == pytest.approx(0.001)
