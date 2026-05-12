"""Tests for research/strategies/_runner.py — issue #668.

Covers Strategy enum, _build_optimizer dispatch for every member,
and cross-layer import constraints.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("typer")

from research.strategies._runner import Strategy, _build_optimizer

# ---------------------------------------------------------------------------
# TestStrategy
# ---------------------------------------------------------------------------


class TestStrategy:
    def test_when_all_members_checked_then_eleven_present(self) -> None:
        assert len(Strategy) == 11

    def test_when_member_values_checked_then_all_eleven_present(self) -> None:
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
        assert {s.value for s in Strategy} == expected

    def test_when_strategy_type_checked_then_is_str_subclass(self) -> None:
        assert issubclass(Strategy, str)

    def test_when_member_used_as_string_then_value_returned(self) -> None:
        assert Strategy.MAX_SHARPE.value == "max-sharpe"

    def test_when_string_passed_then_round_trip_returns_same_member(self) -> None:
        assert Strategy("max-sharpe") is Strategy.MAX_SHARPE

    @pytest.mark.parametrize(
        "value, member_name",
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
    def test_when_value_looked_up_then_correct_member_name_returned(
        self, value: str, member_name: str
    ) -> None:
        assert Strategy(value).name == member_name


# ---------------------------------------------------------------------------
# TestBuildOptimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    @pytest.mark.parametrize("strategy", list(Strategy))
    def test_when_strategy_dispatched_then_non_none_optimizer_returned(
        self, strategy: Strategy
    ) -> None:
        assert _build_optimizer(strategy) is not None

    @pytest.mark.parametrize(
        "strategy, expected_cls_name",
        [
            (Strategy.MAX_SHARPE, "MeanRisk"),
            (Strategy.MIN_VARIANCE, "MeanRisk"),
            (Strategy.MIN_CVAR, "MeanRisk"),
            (Strategy.MAX_UTILITY, "MeanRisk"),
            (Strategy.RISK_PARITY, "RiskBudgeting"),
            (Strategy.CVAR_PARITY, "RiskBudgeting"),
            (Strategy.HRP, "HierarchicalRiskParity"),
            (Strategy.HERC, "HierarchicalEqualRiskContribution"),
            (Strategy.MAX_DIVERSIFICATION, "MaximumDiversification"),
            (Strategy.EQUAL_WEIGHT, "EqualWeighted"),
            (Strategy.INVERSE_VOL, "InverseVolatility"),
        ],
    )
    def test_when_strategy_dispatched_then_correct_optimizer_type_returned(
        self, strategy: Strategy, expected_cls_name: str
    ) -> None:
        result = _build_optimizer(strategy)
        assert type(result).__name__ == expected_cls_name

    def test_when_max_sharpe_dispatched_with_rf_daily_then_rate_set_on_optimizer(
        self,
    ) -> None:
        rf = 0.001
        result = _build_optimizer(Strategy.MAX_SHARPE, rf_daily=rf)
        assert result.risk_free_rate == pytest.approx(rf)

    def test_when_max_utility_dispatched_with_rf_daily_then_rate_set_on_optimizer(
        self,
    ) -> None:
        rf = 0.0005
        result = _build_optimizer(Strategy.MAX_UTILITY, rf_daily=rf)
        assert result.risk_free_rate == pytest.approx(rf)

    def test_when_min_variance_dispatched_with_rf_daily_then_rate_ignored(
        self,
    ) -> None:
        result_default = _build_optimizer(Strategy.MIN_VARIANCE)
        result_rf = _build_optimizer(Strategy.MIN_VARIANCE, rf_daily=0.005)
        assert type(result_default).__name__ == type(result_rf).__name__


# ---------------------------------------------------------------------------
# TestRunnerModuleConstraints
# ---------------------------------------------------------------------------


class TestRunnerModuleConstraints:
    """_runner.py must not import cross-layer research modules at module level."""

    def _top_level_import_modules(self) -> list[str]:
        import sys

        _m = sys.modules["research.strategies._runner"]
        assert _m.__file__ is not None
        source = Path(_m.__file__).read_text()
        tree = ast.parse(source)
        modules: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
        return modules

    def test_when_top_level_imports_checked_then_no_research_reporting_import(
        self,
    ) -> None:
        bad = [
            m
            for m in self._top_level_import_modules()
            if m.startswith("research.reporting")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"

    def test_when_top_level_imports_checked_then_no_research_pipeline_import(
        self,
    ) -> None:
        bad = [
            m
            for m in self._top_level_import_modules()
            if m.startswith("research.pipeline")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"

    def test_when_top_level_imports_checked_then_no_research_cli_import(
        self,
    ) -> None:
        bad = [
            m for m in self._top_level_import_modules() if m.startswith("research.cli")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"
