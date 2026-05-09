"""Tests for optimization factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex._base import ObjectiveFunction

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
    build_sector_constraints,
)

# ---------------------------------------------------------------------------
# MeanRisk
# ---------------------------------------------------------------------------


class TestBuildMeanRisk:
    def test_default_produces_mean_risk(self) -> None:
        model = build_mean_risk()
        assert isinstance(model, MeanRisk)

    def test_objective_forwarded(self) -> None:
        cfg = MeanRiskConfig(objective=ObjectiveFunctionType.MAXIMIZE_RATIO)
        model = build_mean_risk(cfg)
        assert model.objective_function == ObjectiveFunction.MAXIMIZE_RATIO

    def test_risk_measure_forwarded(self) -> None:
        cfg = MeanRiskConfig(risk_measure=RiskMeasureType.CVAR)
        model = build_mean_risk(cfg)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_risk_aversion_forwarded(self) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
            risk_aversion=2.5,
        )
        model = build_mean_risk(cfg)
        assert model.risk_aversion == 2.5

    def test_efficient_frontier_size_forwarded(self) -> None:
        cfg = MeanRiskConfig(efficient_frontier_size=30)
        model = build_mean_risk(cfg)
        assert model.efficient_frontier_size == 30

    def test_weight_bounds_forwarded(self) -> None:
        cfg = MeanRiskConfig(min_weights=-0.1, max_weights=0.5)
        model = build_mean_risk(cfg)
        assert model.min_weights == -0.1
        assert model.max_weights == 0.5

    def test_budget_forwarded(self) -> None:
        cfg = MeanRiskConfig(budget=0.5)
        model = build_mean_risk(cfg)
        assert model.budget == 0.5

    def test_cardinality_forwarded(self) -> None:
        cfg = MeanRiskConfig(cardinality=10)
        model = build_mean_risk(cfg)
        assert model.cardinality == 10

    def test_transaction_costs_forwarded(self) -> None:
        cfg = MeanRiskConfig(transaction_costs=0.001)
        model = build_mean_risk(cfg)
        assert model.transaction_costs == 0.001

    def test_management_fees_forwarded(self) -> None:
        cfg = MeanRiskConfig(management_fees=0.002)
        model = build_mean_risk(cfg)
        assert model.management_fees == 0.002

    def test_max_tracking_error_forwarded(self) -> None:
        cfg = MeanRiskConfig(max_tracking_error=0.02)
        model = build_mean_risk(cfg)
        assert model.max_tracking_error == 0.02

    def test_regularisation_forwarded(self) -> None:
        cfg = MeanRiskConfig(l1_coef=0.01, l2_coef=0.02)
        model = build_mean_risk(cfg)
        assert model.l1_coef == 0.01
        assert model.l2_coef == 0.02

    def test_solver_forwarded(self) -> None:
        cfg = MeanRiskConfig(solver="SCS")
        model = build_mean_risk(cfg)
        assert model.solver == "SCS"

    def test_beta_params_forwarded(self) -> None:
        cfg = MeanRiskConfig(
            cvar_beta=0.99,
            evar_beta=0.90,
            cdar_beta=0.85,
            edar_beta=0.80,
        )
        model = build_mean_risk(cfg)
        assert model.cvar_beta == 0.99
        assert model.evar_beta == 0.90
        assert model.cdar_beta == 0.85
        assert model.edar_beta == 0.80

    def test_prior_config_builds_prior(self) -> None:
        from skfolio.moments import ShrunkMu
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig, MuEstimatorType

        prior_cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.SHRUNK)
        cfg = MeanRiskConfig(prior_config=prior_cfg)
        model = build_mean_risk(cfg)
        assert isinstance(model.prior_estimator, EmpiricalPrior)
        assert isinstance(model.prior_estimator.mu_estimator, ShrunkMu)

    def test_explicit_prior_estimator_overrides_config(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig, MuEstimatorType

        explicit_prior = EmpiricalPrior()
        cfg = MeanRiskConfig(
            prior_config=MomentEstimationConfig(
                mu_estimator=MuEstimatorType.SHRUNK,
            ),
        )
        model = build_mean_risk(cfg, prior_estimator=explicit_prior)
        assert model.prior_estimator is explicit_prior

    def test_kwargs_forwarded(self) -> None:
        groups = {"tech": ["TICK_00", "TICK_01"]}
        model = build_mean_risk(groups=groups)
        assert model.groups == groups


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests using real skfolio fit/predict."""

    def test_mean_risk_min_variance_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_min_variance()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_mean_risk_max_sharpe_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_max_sharpe()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_max_utility_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_max_utility(risk_aversion=2.0)
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_min_cvar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_min_cvar()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_with_transaction_costs(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            transaction_costs=0.001,
            management_fees=0.0005,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_sp500_mean_risk(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = MeanRiskConfig.for_min_variance()
        model = build_mean_risk(cfg)
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns.shape[1]

    def test_efficient_frontier(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_efficient_frontier(size=5)
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        population = model.predict(returns_df)
        assert len(population) == 5


# ---------------------------------------------------------------------------
# Partial-investment budget (issue #107)
# ---------------------------------------------------------------------------


class TestPartialInvestmentBudget:
    def test_budget_0_8_weights_sum(self, returns_df: pd.DataFrame) -> None:
        """Partial budget=0.8 → weights sum to 0.8."""
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            budget=0.8,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert float(np.sum(portfolio.weights)) == pytest.approx(0.8, abs=1e-6)

    def test_budget_0_weights_sum(self, returns_df: pd.DataFrame) -> None:
        """Zero budget=0.0 → weights sum to 0.0."""
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            budget=0.0,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert float(np.sum(portfolio.weights)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Diversification ratio >= 1 (issue #106)
# ---------------------------------------------------------------------------


class TestDiversificationRatioGeOne:
    def test_min_variance_dr_ge_one(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_variance())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.diversification >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# CVaR >= VaR (issue #106)
# ---------------------------------------------------------------------------


class TestCVaRGeVaR:
    def test_min_cvar_portfolio(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_cvar())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.cvar >= portfolio.value_at_risk - 1e-8

    def test_min_variance_portfolio(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_variance())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.cvar >= portfolio.value_at_risk - 1e-8


# ---------------------------------------------------------------------------
# Sector constraints (issue #271)
# ---------------------------------------------------------------------------


class TestBuildSectorConstraints:
    def test_basic_grouping(self) -> None:
        mapping = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
        groups, constraints = build_sector_constraints(mapping, 0.25)
        # groups is ticker->sector (passthrough of sector_mapping)
        assert groups["AAPL"] == "Technology"
        assert groups["JPM"] == "Financials"
        assert "Financials <= 0.25" in constraints
        assert "Technology <= 0.25" in constraints
        assert len(constraints) == 2

    def test_single_sector(self) -> None:
        mapping = {"A": "Tech", "B": "Tech"}
        groups, constraints = build_sector_constraints(mapping, 0.40)
        assert groups is mapping
        assert constraints == ["Tech <= 0.4"]

    def test_cap_formatting(self) -> None:
        _, constraints = build_sector_constraints({"A": "S"}, 1 / 3)
        assert "0.333333" in constraints[0]

    def test_no_sector_mapping_warning_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = MeanRiskConfig(max_sector_weight=0.25)
        with caplog.at_level(logging.WARNING, logger="optimizer.optimization._factory"):
            model = build_mean_risk(cfg)
        assert isinstance(model, MeanRisk)
        assert "sector_mapping" in caplog.text

    def test_explicit_groups_kwarg_takes_precedence(self) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.25)
        custom_groups = {"TICK_00": "MyGroup"}
        mapping = {"TICK_00": "Technology"}
        model = build_mean_risk(cfg, sector_mapping=mapping, groups=custom_groups)
        assert model.groups == custom_groups


class TestSectorConstraintIntegration:
    """Fit-level tests verifying sector caps are enforced after optimization."""

    def test_sector_weights_below_cap(
        self,
        returns_df: pd.DataFrame,
        sector_mapping: dict[str, str],
    ) -> None:
        cap = 0.25
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained(max_sector_weight=cap)
        model = build_mean_risk(cfg, sector_mapping=sector_mapping)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)

        weights = pd.Series(portfolio.weights, index=returns_df.columns)
        for sector in set(sector_mapping.values()):
            members = [t for t, s in sector_mapping.items() if s == sector]
            sector_weight = weights[members].sum()
            assert sector_weight <= cap + 1e-6, (
                f"Sector {sector!r} weight {sector_weight:.4f} exceeds cap {cap}"
            )

    def test_sector_weights_below_tighter_cap(
        self,
        returns_df: pd.DataFrame,
        sector_mapping: dict[str, str],
    ) -> None:
        # 4 sectors, cap=0.30 => 4*0.30=1.20 > budget=1.0 so feasible
        cap = 0.30
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained(max_sector_weight=cap)
        model = build_mean_risk(cfg, sector_mapping=sector_mapping)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)

        weights = pd.Series(portfolio.weights, index=returns_df.columns)
        for sector in set(sector_mapping.values()):
            members = [t for t, s in sector_mapping.items() if s == sector]
            assert weights[members].sum() <= cap + 1e-6

    def test_no_sector_mapping_still_fits(
        self,
        returns_df: pd.DataFrame,
    ) -> None:
        cfg = MeanRiskConfig.for_max_sharpe_sector_constrained()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert abs(sum(portfolio.weights) - 1.0) < 1e-5

    def test_groups_and_constraints_injected(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.30)
        model = build_mean_risk(cfg, sector_mapping=sector_mapping)
        assert model.groups is not None
        # groups dict maps ticker -> sector label
        assert model.groups == sector_mapping
        assert model.linear_constraints is not None
        sectors = set(sector_mapping.values())
        assert len(model.linear_constraints) == len(sectors)


# ---------------------------------------------------------------------------
# Sector floor constraints (issue #532)
# ---------------------------------------------------------------------------


class TestMinSectorWeights:
    def test_when_floors_supplied_linear_constraints_contain_ge_rows(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig()
        floors = {"Healthcare": 0.08, "Technology": 0.10}
        model = build_mean_risk(
            cfg,
            sector_mapping=sector_mapping,
            min_sector_weights=floors,
        )
        assert model.linear_constraints is not None
        assert "Healthcare >= 0.08" in model.linear_constraints
        assert "Technology >= 0.1" in model.linear_constraints

    def test_when_floors_supplied_groups_are_injected(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig()
        floors = {"Technology": 0.10}
        model = build_mean_risk(
            cfg,
            sector_mapping=sector_mapping,
            min_sector_weights=floors,
        )
        assert model.groups == sector_mapping

    def test_when_floors_without_sector_mapping_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="sector_mapping"):
            build_mean_risk(
                MeanRiskConfig(),
                min_sector_weights={"Technology": 0.10},
            )

    def test_when_empty_floors_dict_then_no_error_no_constraints(self) -> None:
        cfg = MeanRiskConfig()
        model = build_mean_risk(cfg, min_sector_weights={})
        assert model.linear_constraints is None

    def test_when_floors_and_caps_combined_constraints_coexist(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.30)
        floors = {"Healthcare": 0.08, "Technology": 0.10}
        model = build_mean_risk(
            cfg,
            sector_mapping=sector_mapping,
            min_sector_weights=floors,
        )
        assert model.linear_constraints is not None
        constraints = list(model.linear_constraints)
        sectors = set(sector_mapping.values())
        for sector in sectors:
            assert f"{sector} <= 0.3" in constraints
        assert "Healthcare >= 0.08" in constraints
        assert "Technology >= 0.1" in constraints

    def test_when_explicit_linear_constraints_kwarg_then_takes_precedence(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.30)
        custom = ["Technology <= 0.5"]
        model = build_mean_risk(
            cfg,
            sector_mapping=sector_mapping,
            min_sector_weights={"Healthcare": 0.08},
            linear_constraints=custom,
        )
        assert model.linear_constraints == custom

    def test_when_floors_supplied_input_dict_not_mutated(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        floors = {"Healthcare": 0.08, "Technology": 0.10}
        snapshot = dict(floors)
        build_mean_risk(
            MeanRiskConfig(),
            sector_mapping=sector_mapping,
            min_sector_weights=floors,
        )
        assert floors == snapshot

    def test_when_floor_sum_exceeds_one_then_value_error(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        floors = {"Healthcare": 0.6, "Technology": 0.6}
        with pytest.raises(ValueError, match="sum"):
            build_mean_risk(
                MeanRiskConfig(),
                sector_mapping=sector_mapping,
                min_sector_weights=floors,
            )

    def test_when_floor_exceeds_cap_then_value_error(
        self,
        sector_mapping: dict[str, str],
    ) -> None:
        cfg = MeanRiskConfig(max_sector_weight=0.10)
        floors = {"Technology": 0.20}
        with pytest.raises(ValueError, match="exceeds"):
            build_mean_risk(
                cfg,
                sector_mapping=sector_mapping,
                min_sector_weights=floors,
            )


class TestMinSectorWeightsIntegration:
    """Fit-level tests verifying floors are respected after optimization."""

    def test_when_floors_set_sector_weights_at_or_above_floor(
        self,
        returns_df: pd.DataFrame,
        sector_mapping: dict[str, str],
    ) -> None:
        floors = {"Healthcare": 0.08, "Technology": 0.10}
        cfg = MeanRiskConfig.for_min_variance()
        model = build_mean_risk(
            cfg,
            sector_mapping=sector_mapping,
            min_sector_weights=floors,
        )
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        weights = pd.Series(portfolio.weights, index=returns_df.columns)
        for sector, floor in floors.items():
            members = [t for t, s in sector_mapping.items() if s == sector]
            sector_weight = weights[members].sum()
            assert sector_weight >= floor - 1e-6, (
                f"Sector {sector!r} weight {sector_weight:.4f} below floor {floor}"
            )
