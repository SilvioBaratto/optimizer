"""Tests for research/strategies/_inspect.py — issue #669.

Covers data_summary(DataAssembly) -> str, strategies() -> dict,
and cross-layer import constraints.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from research.data._container import DataAssembly
from research.strategies._inspect import data_summary, strategies

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_assembly(n_tickers: int = 3, n_days: int = 10) -> DataAssembly:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    prices = pd.DataFrame(
        np.ones((n_days, n_tickers)) * 100.0,
        index=dates,
        columns=tickers,
    )
    return DataAssembly(
        prices=prices,
        volumes=pd.DataFrame(np.ones_like(prices.values), index=dates, columns=tickers),
        fundamentals=pd.DataFrame(index=tickers),
        sector_mapping=dict.fromkeys(tickers, "Tech"),
        financial_statements=pd.DataFrame(),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        macro_data=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# TestDataSummary
# ---------------------------------------------------------------------------


class TestDataSummary:
    def test_when_assembly_passed_then_returns_string(self) -> None:
        data = _make_assembly()
        result = data_summary(data)
        assert isinstance(result, str)

    def test_when_assembly_passed_then_result_not_empty(self) -> None:
        data = _make_assembly()
        result = data_summary(data)
        assert len(result.strip()) > 0

    def test_when_assembly_passed_then_ticker_count_in_output(self) -> None:
        data = _make_assembly(n_tickers=3)
        result = data_summary(data)
        assert "3" in result

    def test_when_assembly_passed_then_trading_days_in_output(self) -> None:
        data = _make_assembly(n_days=10)
        result = data_summary(data)
        assert "10" in result

    def test_when_assembly_passed_then_tickers_key_in_output(self) -> None:
        data = _make_assembly()
        result = data_summary(data)
        assert "tickers" in result.lower()

    def test_when_assembly_passed_then_trading_days_key_in_output(self) -> None:
        data = _make_assembly()
        result = data_summary(data)
        assert "trading_days" in result.lower()

    def test_when_assembly_with_prices_then_date_range_in_output(self) -> None:
        data = _make_assembly(n_days=5)
        result = data_summary(data)
        assert "2024" in result

    def test_when_snapshot_compared_then_output_stable(self) -> None:
        data = _make_assembly(n_tickers=2, n_days=5)
        result = data_summary(data)
        expected_fragment = "tickers"
        assert expected_fragment in result.lower()
        expected_fragment2 = "trading_days"
        assert expected_fragment2 in result.lower()

    def test_when_empty_prices_then_returns_string_without_date_range(self) -> None:
        data = _make_assembly(n_tickers=0, n_days=0)
        result = data_summary(data)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestStrategies
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_when_called_then_returns_dict(self) -> None:
        result = strategies()
        assert isinstance(result, dict)

    def test_when_called_then_all_eleven_strategies_present(self) -> None:
        result = strategies()
        assert len(result) == 11

    def test_when_called_then_keys_are_all_valid_strategy_values(self) -> None:
        from research.strategies._runner import Strategy

        result = strategies()
        expected_values = {s.value for s in Strategy}
        assert set(result.keys()) == expected_values

    def test_when_called_then_each_description_is_non_empty_string(self) -> None:
        result = strategies()
        for key, desc in result.items():
            assert isinstance(desc, str), f"{key!r} description not str"
            assert len(desc.strip()) > 0, f"{key!r} has empty description"

    def test_when_max_sharpe_key_then_sharpe_in_description(self) -> None:
        result = strategies()
        assert "sharpe" in result["max-sharpe"].lower()

    def test_when_equal_weight_key_then_description_contains_equal(self) -> None:
        result = strategies()
        assert "equal" in result["equal-weight"].lower()


# ---------------------------------------------------------------------------
# TestInspectModuleConstraints
# ---------------------------------------------------------------------------


class TestInspectModuleConstraints:
    """_inspect.py must not import cross-layer research modules at module level."""

    def _top_level_import_modules(self) -> list[str]:
        import sys

        _m = sys.modules["research.strategies._inspect"]
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

    def test_when_top_level_imports_checked_then_no_research_reporting(
        self,
    ) -> None:
        bad = [
            m
            for m in self._top_level_import_modules()
            if m.startswith("research.reporting")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"

    def test_when_top_level_imports_checked_then_no_research_pipeline(
        self,
    ) -> None:
        bad = [
            m
            for m in self._top_level_import_modules()
            if m.startswith("research.pipeline")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"

    def test_when_top_level_imports_checked_then_no_research_cli(
        self,
    ) -> None:
        bad = [
            m for m in self._top_level_import_modules() if m.startswith("research.cli")
        ]
        assert bad == [], f"Forbidden top-level import(s): {bad}"

    def test_when_line_count_checked_then_at_most_200_lines(self) -> None:
        import sys

        _m = sys.modules["research.strategies._inspect"]
        assert _m.__file__ is not None
        lines = Path(_m.__file__).read_text().splitlines()
        assert len(lines) <= 200, f"File has {len(lines)} lines (limit: 200)"
