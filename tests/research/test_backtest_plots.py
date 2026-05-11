"""Tests for ``research/_backtest_plots.py`` — issue #548.

Covers two new charts wired into ``generate_backtest_plots``:

- ``plot_country_allocation`` — stacked-area country exposure over time.
- ``plot_factor_ic`` — grouped IS / OOS mean-IC bars per factor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorOOSResult,
    FactorValidationReport,
)
from optimizer.factors._validation import ICResult
from research.reporting._plots import (
    generate_backtest_plots,
    plot_country_allocation,
    plot_factor_ic,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def weight_history() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=4, freq="QE")
    return pd.DataFrame(
        [
            {"AAPL": 0.5, "JPM": 0.3, "ORA": 0.2},
            {"AAPL": 0.4, "JPM": 0.4, "ORA": 0.2},
            {"AAPL": 0.3, "JPM": 0.5, "ORA": 0.2},
            {"AAPL": 0.6, "JPM": 0.2, "ORA": 0.2},
        ],
        index=dates,
    )


@pytest.fixture()
def country_map() -> dict[str, str]:
    return {"AAPL": "US", "JPM": "US", "ORA": "FR"}


@pytest.fixture()
def sector_mapping() -> dict[str, str]:
    return {"AAPL": "Tech", "JPM": "Financials", "ORA": "Comms"}


@pytest.fixture()
def portfolio_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=300)
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)


@pytest.fixture()
def validation_report() -> FactorValidationReport:
    return FactorValidationReport(
        ic_results=[
            ICResult(
                factor_name="value",
                mean_ic=0.04,
                ic_std=0.05,
                t_stat=2.5,
                p_value=0.01,
                significant=True,
            ),
            ICResult(
                factor_name="momentum",
                mean_ic=0.06,
                ic_std=0.04,
                t_stat=3.1,
                p_value=0.001,
                significant=True,
            ),
            ICResult(
                factor_name="quality",
                mean_ic=0.02,
                ic_std=0.05,
                t_stat=1.0,
                p_value=0.32,
                significant=False,
            ),
        ],
    )


@pytest.fixture()
def oos_per_fold_ic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [0.03, 0.025, 0.04],
            "momentum": [0.05, 0.07, 0.04],
            "quality": [0.01, -0.005, 0.015],
        }
    )


@pytest.fixture()
def oos_result(oos_per_fold_ic: pd.DataFrame) -> FactorOOSResult:
    return FactorOOSResult(
        per_fold_ic=oos_per_fold_ic,
        per_fold_spread=pd.DataFrame(),
        mean_oos_ic=oos_per_fold_ic.mean(axis=0),
        mean_oos_icir=oos_per_fold_ic.mean(axis=0) / oos_per_fold_ic.std(axis=0),
        n_folds=3,
    )


# ---------------------------------------------------------------------------
# plot_country_allocation
# ---------------------------------------------------------------------------


class TestPlotCountryAllocation:
    def test_when_weight_history_none_returns_none(
        self, country_map: dict[str, str], tmp_path: Path
    ) -> None:
        result = plot_country_allocation(None, country_map, output_dir=tmp_path)
        assert result is None

    def test_when_weight_history_empty_returns_none(
        self, country_map: dict[str, str], tmp_path: Path
    ) -> None:
        result = plot_country_allocation(
            pd.DataFrame(),
            country_map,
            output_dir=tmp_path,
        )
        assert result is None

    def test_when_called_returns_path_to_country_allocation_png(
        self,
        weight_history: pd.DataFrame,
        country_map: dict[str, str],
        tmp_path: Path,
    ) -> None:
        result = plot_country_allocation(
            weight_history,
            country_map,
            output_dir=tmp_path,
        )
        assert result == tmp_path / "country_allocation.png"

    def test_when_called_creates_non_empty_png_file(
        self,
        weight_history: pd.DataFrame,
        country_map: dict[str, str],
        tmp_path: Path,
    ) -> None:
        result = plot_country_allocation(
            weight_history,
            country_map,
            output_dir=tmp_path,
        )
        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 0

    def test_when_country_missing_uses_unknown_bucket(
        self,
        weight_history: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        partial_map = {"AAPL": "US"}  # JPM and ORA unmapped
        result = plot_country_allocation(
            weight_history,
            partial_map,
            output_dir=tmp_path,
        )
        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# plot_factor_ic
# ---------------------------------------------------------------------------


class TestPlotFactorIC:
    def test_when_called_returns_path_to_factor_ic_png(
        self,
        validation_report: FactorValidationReport,
        oos_per_fold_ic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        result = plot_factor_ic(
            validation_report,
            oos_per_fold_ic,
            output_dir=tmp_path,
        )
        assert result == tmp_path / "factor_ic.png"

    def test_when_called_creates_non_empty_png_file(
        self,
        validation_report: FactorValidationReport,
        oos_per_fold_ic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        result = plot_factor_ic(
            validation_report,
            oos_per_fold_ic,
            output_dir=tmp_path,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_when_factors_overlap_only_chart_uses_intersection(
        self,
        validation_report: FactorValidationReport,
        tmp_path: Path,
    ) -> None:
        partial_oos = pd.DataFrame({"value": [0.03, 0.025], "momentum": [0.05, 0.06]})
        result = plot_factor_ic(
            validation_report,
            partial_oos,
            output_dir=tmp_path,
        )
        assert result.exists()


# ---------------------------------------------------------------------------
# generate_backtest_plots wiring
# ---------------------------------------------------------------------------


class TestGenerateBacktestPlots:
    def test_when_only_existing_inputs_returns_four_paths(
        self,
        portfolio_returns: pd.Series,
        weight_history: pd.DataFrame,
        sector_mapping: dict[str, str],
        tmp_path: Path,
    ) -> None:
        paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=weight_history,
            sector_mapping=sector_mapping,
            output_dir=tmp_path,
        )
        assert len(paths) == 4

    def test_when_all_inputs_supplied_returns_six_paths(
        self,
        portfolio_returns: pd.Series,
        weight_history: pd.DataFrame,
        sector_mapping: dict[str, str],
        country_map: dict[str, str],
        validation_report: FactorValidationReport,
        oos_per_fold_ic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=weight_history,
            sector_mapping=sector_mapping,
            country_map=country_map,
            validation_report=validation_report,
            oos_per_fold_ic=oos_per_fold_ic,
            output_dir=tmp_path,
        )
        assert len(paths) == 6
        names = {p.name for p in paths}
        assert "country_allocation.png" in names
        assert "factor_ic.png" in names

    def test_when_country_map_missing_country_chart_skipped(
        self,
        portfolio_returns: pd.Series,
        weight_history: pd.DataFrame,
        sector_mapping: dict[str, str],
        validation_report: FactorValidationReport,
        oos_per_fold_ic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=weight_history,
            sector_mapping=sector_mapping,
            validation_report=validation_report,
            oos_per_fold_ic=oos_per_fold_ic,
            output_dir=tmp_path,
        )
        names = {p.name for p in paths}
        assert "country_allocation.png" not in names
        assert "factor_ic.png" in names

    def test_when_validation_report_missing_factor_ic_chart_skipped(
        self,
        portfolio_returns: pd.Series,
        weight_history: pd.DataFrame,
        sector_mapping: dict[str, str],
        country_map: dict[str, str],
        oos_per_fold_ic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=weight_history,
            sector_mapping=sector_mapping,
            country_map=country_map,
            oos_per_fold_ic=oos_per_fold_ic,
            output_dir=tmp_path,
        )
        names = {p.name for p in paths}
        assert "country_allocation.png" in names
        assert "factor_ic.png" not in names

    def test_when_oos_ic_missing_factor_ic_chart_skipped(
        self,
        portfolio_returns: pd.Series,
        weight_history: pd.DataFrame,
        sector_mapping: dict[str, str],
        validation_report: FactorValidationReport,
        tmp_path: Path,
    ) -> None:
        paths = generate_backtest_plots(
            portfolio_returns=portfolio_returns,
            weight_history=weight_history,
            sector_mapping=sector_mapping,
            validation_report=validation_report,
            output_dir=tmp_path,
        )
        names = {p.name for p in paths}
        assert "factor_ic.png" not in names
