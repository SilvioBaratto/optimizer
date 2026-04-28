"""Tests for optimization configs and enums."""

from __future__ import annotations

import pytest

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RatioMeasureType,
    RiskMeasureType,
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
        assert cfg.l2_coef == 0.05
        assert cfg.max_sector_weight == 0.25
        assert cfg.prior_config is not None

    def test_for_max_sharpe_sector_constrained_custom_cap(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained(max_sector_weight=0.30)
        assert cfg.max_sector_weight == 0.30

    def test_for_max_sharpe_sector_constrained_is_frozen(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained()
        with pytest.raises(AttributeError):
            cfg.max_sector_weight = 0.5  # type: ignore[misc]
