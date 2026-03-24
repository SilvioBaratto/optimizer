"""Tests for cli/portfolio.py — Strategy enum and _build_optimizer."""

from __future__ import annotations

import pytest

from cli.portfolio import Strategy, _build_optimizer

# ---------------------------------------------------------------------------
# TestStrategy
# ---------------------------------------------------------------------------


class TestStrategy:
    def test_all_eleven_strategies_exist(self) -> None:
        expected = {
            "max-sharpe",
            "min-variance",
            "min-cvar",
            "max-utility",
            "risk-parity",
            "cvar-parity",
            "hrp",
            "herc",
            "max-diversification",
            "equal-weight",
            "inverse-vol",
        }
        actual = {s.value for s in Strategy}
        assert actual == expected

    def test_strategy_is_str_subclass(self) -> None:
        assert isinstance(Strategy.MAX_SHARPE, str)

    def test_round_trip_from_string(self) -> None:
        assert Strategy("max-sharpe") is Strategy.MAX_SHARPE

    def test_all_enum_values_are_strings(self) -> None:
        for s in Strategy:
            assert isinstance(s.value, str)

    def test_eleven_members_total(self) -> None:
        assert len(Strategy) == 11

    @pytest.mark.parametrize(
        "value,member",
        [
            ("max-sharpe", "MAX_SHARPE"),
            ("min-variance", "MIN_VARIANCE"),
            ("min-cvar", "MIN_CVAR"),
            ("max-utility", "MAX_UTILITY"),
            ("risk-parity", "RISK_PARITY"),
            ("cvar-parity", "CVAR_PARITY"),
            ("hrp", "HRP"),
            ("herc", "HERC"),
            ("max-diversification", "MAX_DIVERSIFICATION"),
            ("equal-weight", "EQUAL_WEIGHT"),
            ("inverse-vol", "INVERSE_VOL"),
        ],
    )
    def test_value_to_member_lookup(self, value: str, member: str) -> None:
        assert Strategy(value).name == member


# ---------------------------------------------------------------------------
# TestBuildOptimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    @pytest.mark.parametrize("strategy", list(Strategy))
    def test_all_strategies_return_non_none(self, strategy: Strategy) -> None:
        result = _build_optimizer(strategy)
        assert result is not None

    def test_max_sharpe_applies_rf_daily(self) -> None:
        rf = 0.001
        result = _build_optimizer(Strategy.MAX_SHARPE, rf_daily=rf)
        assert result.risk_free_rate == pytest.approx(rf)

    def test_max_utility_applies_rf_daily(self) -> None:
        rf = 0.0002
        result = _build_optimizer(Strategy.MAX_UTILITY, rf_daily=rf)
        assert result.risk_free_rate == pytest.approx(rf)

    def test_max_sharpe_zero_rf_is_valid(self) -> None:
        result = _build_optimizer(Strategy.MAX_SHARPE, rf_daily=0.0)
        assert result is not None
        assert result.risk_free_rate == pytest.approx(0.0)

    def test_min_variance_returns_non_none(self) -> None:
        result = _build_optimizer(Strategy.MIN_VARIANCE, rf_daily=0.005)
        assert result is not None

    def test_hrp_returns_hierarchical_risk_parity(self) -> None:
        from skfolio.optimization import HierarchicalRiskParity

        result = _build_optimizer(Strategy.HRP)
        assert isinstance(result, HierarchicalRiskParity)

    def test_herc_returns_herc_type(self) -> None:
        from skfolio.optimization import HierarchicalEqualRiskContribution

        result = _build_optimizer(Strategy.HERC)
        assert isinstance(result, HierarchicalEqualRiskContribution)

    def test_equal_weight_returns_equal_weighted_type(self) -> None:
        from skfolio.optimization import EqualWeighted

        result = _build_optimizer(Strategy.EQUAL_WEIGHT)
        assert isinstance(result, EqualWeighted)

    def test_inverse_vol_returns_inverse_volatility_type(self) -> None:
        from skfolio.optimization import InverseVolatility

        result = _build_optimizer(Strategy.INVERSE_VOL)
        assert isinstance(result, InverseVolatility)

    def test_risk_parity_returns_risk_budgeting_type(self) -> None:
        from skfolio.optimization import RiskBudgeting

        result = _build_optimizer(Strategy.RISK_PARITY)
        assert isinstance(result, RiskBudgeting)

    def test_cvar_parity_returns_risk_budgeting_type(self) -> None:
        from skfolio.optimization import RiskBudgeting

        result = _build_optimizer(Strategy.CVAR_PARITY)
        assert isinstance(result, RiskBudgeting)

    def test_max_diversification_returns_correct_type(self) -> None:
        from skfolio.optimization import MaximumDiversification

        result = _build_optimizer(Strategy.MAX_DIVERSIFICATION)
        assert isinstance(result, MaximumDiversification)

    def test_max_sharpe_returns_mean_risk_type(self) -> None:
        from skfolio.optimization import MeanRisk

        result = _build_optimizer(Strategy.MAX_SHARPE)
        assert isinstance(result, MeanRisk)

    def test_min_variance_returns_mean_risk_type(self) -> None:
        from skfolio.optimization import MeanRisk

        result = _build_optimizer(Strategy.MIN_VARIANCE)
        assert isinstance(result, MeanRisk)
