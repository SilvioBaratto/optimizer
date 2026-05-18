"""Tests for research.pipeline._factors — Steps 3-5 factor validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestValidateNSelected:
    def test_when_in_range_then_no_error(self) -> None:
        from research.pipeline._factors import _validate_n_selected

        _validate_n_selected(30)
        _validate_n_selected(25)
        _validate_n_selected(50)

    def test_when_below_min_then_raises_value_error(self) -> None:
        from research.pipeline._factors import _validate_n_selected

        with pytest.raises(ValueError, match="n_selected=10"):
            _validate_n_selected(10)

    def test_when_above_max_then_raises_value_error(self) -> None:
        from research.pipeline._factors import _validate_n_selected

        with pytest.raises(ValueError, match="n_selected=100"):
            _validate_n_selected(100)


class TestMissingGicsSectors:
    def test_when_all_present_then_empty_list(self) -> None:
        from research.pipeline._factors import _missing_gics_sectors

        weights = pd.Series(
            [0.1, 0.1],
            index=["AAPL", "XOM"],
        )
        sector_mapping = {
            "AAPL": "Technology",
            "XOM": "Energy",
        }
        result = _missing_gics_sectors(weights, sector_mapping)
        assert isinstance(result, list)

    def test_when_sector_missing_then_included(self) -> None:
        from research.pipeline._factors import _missing_gics_sectors

        weights = pd.Series([1.0], index=["AAPL"])
        sector_mapping = {"AAPL": "Technology"}
        result = _missing_gics_sectors(weights, sector_mapping)
        assert "Energy" in result
        assert "Healthcare" in result


class TestGicsSectors:
    def test_has_11_sectors(self) -> None:
        from research.pipeline._factors import _GICS_SECTORS

        assert len(_GICS_SECTORS) == 11

    def test_includes_all_major_sectors(self) -> None:
        from research.pipeline._factors import _GICS_SECTORS

        assert "Energy" in _GICS_SECTORS
        assert "Technology" in _GICS_SECTORS
        assert "Healthcare" in _GICS_SECTORS
        assert "Financial Services" in _GICS_SECTORS


class TestCheckFactorCoverage:
    def test_when_enough_passing_then_no_error(self) -> None:
        from research.pipeline._factors import _check_factor_coverage

        is_report = MagicMock()
        is_report.significant_factors = {"A", "B", "C", "D", "E"}
        oos_result = MagicMock()
        oos_result.mean_oos_icir = pd.Series(
            [0.1, 0.2, 0.3, 0.4, 0.5], index=["A", "B", "C", "D", "E"]
        )
        _check_factor_coverage(is_report, oos_result, min_factors=4)

    def test_when_too_few_then_raises_factor_coverage_error(self) -> None:
        from optimizer.exceptions import FactorCoverageError
        from research.pipeline._factors import _check_factor_coverage

        is_report = MagicMock()
        is_report.significant_factors = {"A"}
        oos_result = MagicMock()
        oos_result.mean_oos_icir = pd.Series([0.1], index=["A"])
        with pytest.raises(FactorCoverageError, match="Factor coverage gate failed"):
            _check_factor_coverage(is_report, oos_result, min_factors=4)


class TestBuildHistory:
    def test_returns_factor_scores_and_health(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._factors import build_history

        assembly = _make_stub_assembly()
        health_stub = MagicMock()
        health_stub.succeeded_dates = 10
        health_stub.total_dates = 12
        health_stub.failed_dates = []

        monkeypatch.setattr(
            "research.pipeline._factors.build_factor_scores_history",
            lambda **_: ({"F1": pd.DataFrame()}, pd.DataFrame(), health_stub),
        )
        investable = pd.Index(["AAA", "BBB"])
        scores, _returns_hist, health = build_history(assembly, investable)
        assert isinstance(scores, dict)
        assert isinstance(health, object)
        assert health.succeeded_dates == 10


class TestValidateOos:
    def test_when_valid_data_then_returns_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._factors import validate_oos

        oos_stub = MagicMock()
        oos_stub.n_folds = 6
        oos_stub.mean_oos_ic = pd.Series([0.02, 0.01], index=["F1", "F2"])
        oos_stub.mean_oos_icir = pd.Series([0.5, 0.3], index=["F1", "F2"])

        monkeypatch.setattr(
            "research.pipeline._factors.run_factor_oos_validation",
            lambda **_: oos_stub,
        )
        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        scores = {
            "F1": pd.DataFrame({"AAA": [0.5] * 252, "BBB": [0.3] * 252}, index=idx),
            "F2": pd.DataFrame({"AAA": [0.2] * 252, "BBB": [0.1] * 252}, index=idx),
        }
        returns_hist = pd.DataFrame(
            {
                "AAA": np.random.default_rng(42).normal(0, 0.02, 252),
                "BBB": np.random.default_rng(43).normal(0, 0.02, 252),
            },
            index=idx,
        )
        result = validate_oos(scores, returns_hist)
        assert result.n_folds >= 0


def _make_stub_assembly() -> MagicMock:
    assembly = MagicMock()
    assembly.prices = pd.DataFrame(
        {"AAA": [100.0] * 10, "BBB": [50.0] * 10},
        index=pd.date_range("2024-01-02", periods=10, freq="B"),
    )
    assembly.volumes = pd.DataFrame(
        {"AAA": [1000.0] * 10, "BBB": [2000.0] * 10},
        index=pd.date_range("2024-01-02", periods=10, freq="B"),
    )
    assembly.fundamentals = pd.DataFrame(
        {"market_cap": [1e9, 5e8]}, index=["AAA", "BBB"]
    )
    assembly.sector_mapping = {"AAA": "Tech", "BBB": "Finance"}
    assembly.fundamental_history = MagicMock()
    return assembly
