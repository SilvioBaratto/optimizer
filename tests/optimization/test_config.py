"""Tests for optimization configs and enums."""

from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.optimization import (
    BenchmarkTrackerConfig,
    FallbackPolicy,
    HERCConfig,
    HRPConfig,
    MeanRiskConfig,
    ObjectiveFunctionType,
    RatioMeasureType,
    RiskBudgetingConfig,
    RiskMeasureType,
    SchurComplementaryConfig,
    build_benchmark_tracker,
    build_herc,
    build_hrp,
    build_risk_budgeting,
    build_schur_complementary,
)


class TestObjectiveFunctionType:
    def test_members(self) -> None:
        assert set(ObjectiveFunctionType) == {
            ObjectiveFunctionType.MINIMIZE_RISK,
            ObjectiveFunctionType.MAXIMIZE_RETURN,
            ObjectiveFunctionType.MAXIMIZE_UTILITY,
            ObjectiveFunctionType.MAXIMIZE_RATIO,
        }

    def test_str_serialization(self) -> None:
        assert ObjectiveFunctionType.MINIMIZE_RISK.value == "minimize_risk"
        assert ObjectiveFunctionType.MAXIMIZE_RATIO.value == "maximize_ratio"


class TestRiskMeasureType:
    def test_members(self) -> None:
        assert len(RiskMeasureType) == 15

    def test_str_serialization(self) -> None:
        assert RiskMeasureType.VARIANCE.value == "variance"
        assert RiskMeasureType.CVAR.value == "cvar"
        assert RiskMeasureType.MAX_DRAWDOWN.value == "max_drawdown"
        assert RiskMeasureType.GINI_MEAN_DIFFERENCE.value == "gini_mean_difference"


class TestRatioMeasureType:
    def test_members(self) -> None:
        assert len(RatioMeasureType) == 19

    def test_str_serialization(self) -> None:
        assert RatioMeasureType.SHARPE_RATIO.value == "sharpe_ratio"
        assert RatioMeasureType.SORTINO_RATIO.value == "sortino_ratio"
        assert RatioMeasureType.CALMAR_RATIO.value == "calmar_ratio"
        assert RatioMeasureType.CVAR_RATIO.value == "cvar_ratio"


class TestFallbackPolicy:
    def test_members(self) -> None:
        assert set(FallbackPolicy) == {
            FallbackPolicy.NONE,
            FallbackPolicy.PREVIOUS_WEIGHTS,
        }

    def test_str_serialization(self) -> None:
        assert FallbackPolicy.NONE.value == "none"
        assert FallbackPolicy.PREVIOUS_WEIGHTS.value == "previous_weights"


class TestMeanRiskConfig:
    def test_default_values(self) -> None:
        cfg = MeanRiskConfig()
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.risk_aversion == 1.0
        assert cfg.efficient_frontier_size is None
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.budget == 1.0
        assert cfg.max_short is None
        assert cfg.max_long is None
        assert cfg.cardinality is None
        assert cfg.transaction_costs == 0.0
        assert cfg.management_fees == 0.0
        assert cfg.max_tracking_error is None
        assert cfg.l1_coef == 0.0
        assert cfg.l2_coef == 0.0
        assert cfg.risk_free_rate == 0.0
        assert cfg.cvar_beta == 0.95
        assert cfg.solver == "CLARABEL"
        assert cfg.solver_params is None
        assert cfg.prior_config is None
        assert cfg.max_sector_weight is None
        assert cfg.raise_on_failure is True
        assert cfg.fallback_policy == FallbackPolicy.NONE

    def test_frozen(self) -> None:
        cfg = MeanRiskConfig()
        with pytest.raises(AttributeError):
            cfg.risk_aversion = 2.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.CVAR,
            risk_aversion=2.5,
            cvar_beta=0.99,
            l1_coef=0.01,
        )
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.CVAR
        assert cfg.risk_aversion == 2.5
        assert cfg.cvar_beta == 0.99
        assert cfg.l1_coef == 0.01

    def test_transaction_costs_and_fees(self) -> None:
        cfg = MeanRiskConfig(
            transaction_costs=0.001,
            management_fees=0.002,
        )
        assert cfg.transaction_costs == 0.001
        assert cfg.management_fees == 0.002

    def test_max_tracking_error(self) -> None:
        cfg = MeanRiskConfig(max_tracking_error=0.02)
        assert cfg.max_tracking_error == 0.02

    def test_for_min_variance(self) -> None:
        cfg = MeanRiskConfig.for_min_variance()
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_max_sharpe(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe()
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_max_utility(self) -> None:
        cfg = MeanRiskConfig.for_max_utility(risk_aversion=3.0)
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_UTILITY
        assert cfg.risk_aversion == 3.0

    def test_for_min_cvar(self) -> None:
        cfg = MeanRiskConfig.for_min_cvar(beta=0.99)
        assert cfg.risk_measure == RiskMeasureType.CVAR
        assert cfg.cvar_beta == 0.99

    def test_for_efficient_frontier(self) -> None:
        cfg = MeanRiskConfig.for_efficient_frontier(size=30)
        assert cfg.efficient_frontier_size == 30
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK

    def test_max_sector_weight_roundtrip(self) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.25)
        assert cfg.max_sector_weight == 0.25

    def test_for_max_sharpe_sector_constrained_defaults(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained()
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.max_weights == 0.10
        assert 0.0 < cfg.l2_coef <= 1e-2
        assert cfg.max_sector_weight == 0.25
        assert cfg.prior_config is not None

    def test_for_max_sharpe_sector_constrained_custom_cap(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained(max_sector_weight=0.30)
        assert cfg.max_sector_weight == 0.30

    def test_for_max_sharpe_sector_constrained_is_frozen(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained()
        with pytest.raises(AttributeError):
            cfg.max_sector_weight = 0.5  # type: ignore[misc]

    def test_resilience_fields_custom(self) -> None:
        cfg = MeanRiskConfig(
            raise_on_failure=False,
            fallback_policy=FallbackPolicy.PREVIOUS_WEIGHTS,
        )
        assert cfg.raise_on_failure is False
        assert cfg.fallback_policy == FallbackPolicy.PREVIOUS_WEIGHTS

    def test_for_tracking_error_defaults(self) -> None:
        cfg = MeanRiskConfig.for_tracking_error()
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.max_tracking_error == pytest.approx(0.05 / math.sqrt(252))

    def test_for_tracking_error_custom(self) -> None:
        cfg = MeanRiskConfig.for_tracking_error(max_tracking_error=0.12)
        assert cfg.max_tracking_error == pytest.approx(0.12 / math.sqrt(252))

    def test_for_tracking_error_converts_with_periods_per_year(self) -> None:
        cfg = MeanRiskConfig.for_tracking_error(
            max_tracking_error=0.06, periods_per_year=12
        )
        assert cfg.max_tracking_error == pytest.approx(0.06 / math.sqrt(12))

    @pytest.mark.parametrize("periods", [0, -252])
    def test_for_tracking_error_rejects_non_positive_periods(
        self, periods: int
    ) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            MeanRiskConfig.for_tracking_error(periods_per_year=periods)

    def test_for_resilient_walk_forward(self) -> None:
        cfg = MeanRiskConfig.for_resilient_walk_forward()
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.raise_on_failure is False
        assert cfg.fallback_policy == FallbackPolicy.PREVIOUS_WEIGHTS


# Optimisers whose skfolio implementation never reads the cost terms while
# solving: the fields only feed the cost accounting of the resulting Portfolio.
_REPORTING_ONLY_COST_BUILDERS = [
    (HRPConfig, build_hrp),
    (HERCConfig, build_herc),
    (SchurComplementaryConfig, build_schur_complementary),
    (RiskBudgetingConfig, build_risk_budgeting),
    (BenchmarkTrackerConfig, build_benchmark_tracker),
]


def _attribute_doc(config_cls: type, name: str) -> str:
    """Return the whitespace-normalised ``Attributes:`` entry for ``name``."""
    doc = inspect.getdoc(config_cls) or ""
    match = re.search(rf"^    {name}:(.*?)(?=^    \w+:|^\w|\Z)", doc, re.M | re.S)
    assert match is not None, f"{config_cls.__name__} does not document {name}"
    return " ".join(match.group(1).split())


class TestCostFieldsAreReportingOnly:
    @pytest.fixture(scope="class")
    def returns(self):
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        return prices_to_returns(load_sp500_dataset()).iloc[:, :8]

    @pytest.mark.parametrize(("config_cls", "build"), _REPORTING_ONLY_COST_BUILDERS)
    def test_when_costs_set_then_weights_do_not_change(
        self, returns, config_cls: type, build
    ) -> None:
        previous = np.array([1.0] + [0.0] * 7)
        benchmark = returns.mean(axis=1)

        def fitted_weights(**costs: float) -> np.ndarray:
            model = build(config_cls(**costs), previous_weights=previous)
            model.fit(returns, benchmark)
            return model.weights_

        free = fitted_weights()
        charged = fitted_weights(transaction_costs=0.05, management_fees=0.05)
        assert np.allclose(free, charged)

    @pytest.mark.parametrize(
        ("config_cls", "build"),
        _REPORTING_ONLY_COST_BUILDERS,
        ids=[cls.__name__ for cls, _ in _REPORTING_ONLY_COST_BUILDERS],
    )
    @pytest.mark.parametrize("field", ["transaction_costs", "management_fees"])
    def test_when_documented_then_says_weights_are_unchanged(
        self, config_cls: type, build, field: str
    ) -> None:
        doc = _attribute_doc(config_cls, field)
        assert "does not change the optimised weights" in doc
        assert "penalis" not in doc
