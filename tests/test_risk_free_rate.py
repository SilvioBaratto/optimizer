"""Tests for DataAssembly risk-free rate properties."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("typer") is None,
    reason="typer not available in CI (cli dependency)",
)


def _make_assembly(**kwargs):
    """Build a minimal DataAssembly with defaults for required fields."""
    from cli.data_assembly import DataAssembly

    defaults: dict = {
        "prices": pd.DataFrame(),
        "volumes": pd.DataFrame(),
        "fundamentals": pd.DataFrame(),
        "sector_mapping": {},
        "financial_statements": pd.DataFrame(),
        "analyst_data": pd.DataFrame(),
        "insider_data": pd.DataFrame(),
        "macro_data": pd.DataFrame(),
    }
    defaults.update(kwargs)
    return DataAssembly(**defaults)


class TestRiskFreeRateSeries:
    def test_empty_fred_data(self) -> None:
        data = _make_assembly()
        series = data.risk_free_rate_series
        assert isinstance(series, pd.Series)
        assert series.empty

    def test_fred_data_without_dgs3mo(self) -> None:
        fred = pd.DataFrame({"VIXCLS": [20.0, 21.0]})
        data = _make_assembly(fred_data=fred)
        assert data.risk_free_rate_series.empty

    def test_conversion_formula(self) -> None:
        """DGS3MO=5.0% annual -> (1.05)^(1/252) - 1."""
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        fred = pd.DataFrame({"DGS3MO": [5.0, 5.0, 5.0]}, index=dates)
        data = _make_assembly(fred_data=fred)

        series = data.risk_free_rate_series
        expected = (1.05) ** (1.0 / 252) - 1
        assert len(series) == 3
        np.testing.assert_allclose(series.values, expected, rtol=1e-10)

    def test_nan_values_dropped(self) -> None:
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        fred = pd.DataFrame({"DGS3MO": [5.0, np.nan, 5.25, np.nan]}, index=dates)
        data = _make_assembly(fred_data=fred)
        assert len(data.risk_free_rate_series) == 2

    def test_series_name(self) -> None:
        dates = pd.date_range("2024-01-02", periods=1, freq="B")
        fred = pd.DataFrame({"DGS3MO": [4.5]}, index=dates)
        data = _make_assembly(fred_data=fred)
        assert data.risk_free_rate_series.name == "risk_free_rate"


class TestRiskFreeRateScalar:
    def test_returns_zero_when_missing(self) -> None:
        data = _make_assembly()
        assert data.risk_free_rate == 0.0

    def test_returns_latest_value(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        fred = pd.DataFrame({"DGS3MO": [4.0, 4.5, 5.0]}, index=dates)
        data = _make_assembly(fred_data=fred)

        expected = (1.05) ** (1.0 / 252) - 1
        assert data.risk_free_rate == pytest.approx(expected, rel=1e-10)


class TestSummaryRiskFreeRate:
    def test_summary_includes_rf_fields(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        fred = pd.DataFrame({"DGS3MO": [5.0, 5.25]}, index=dates)
        data = _make_assembly(fred_data=fred)

        s = data.summary()
        assert "risk_free_rate_pct" in s
        assert "risk_free_rate_obs" in s
        assert s["risk_free_rate_obs"] == 2
        assert s["risk_free_rate_pct"] is not None

    def test_summary_rf_none_when_missing(self) -> None:
        data = _make_assembly()
        s = data.summary()
        assert s["risk_free_rate_pct"] is None
        assert s["risk_free_rate_obs"] == 0
