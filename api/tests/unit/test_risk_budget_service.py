"""Unit tests for LLM-driven risk budget calibration service (issue #17)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from app.services.risk_budget_service import (
    _expand_sector_to_asset_budgets,
    _normalise,
    _to_budget_array,
    calibrate_risk_budget,
)
from baml_client.types import RiskBudgetOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SECTORS = ["Technology", "Healthcare", "Energy", "Financials"]

ASSET_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JNJ": "Healthcare",
    "XOM": "Energy",
    "JPM": "Financials",
}

OUTLOOK = (
    "Overweight Technology and Healthcare; underweight Energy; neutral Financials."
)


def _make_output(
    sector_budgets: dict[str, float] | None = None,
    asset_budgets: dict[str, float] | None = None,
    rationale: str = "Test rationale.",
) -> RiskBudgetOutput:
    if sector_budgets is None:
        sector_budgets = {
            "Technology": 0.40,
            "Healthcare": 0.30,
            "Energy": 0.10,
            "Financials": 0.20,
        }
    if asset_budgets is None:
        asset_budgets = {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "JNJ": 0.30,
            "XOM": 0.10,
            "JPM": 0.20,
        }
    return RiskBudgetOutput(
        sector_budgets=sector_budgets,
        asset_budgets=asset_budgets,
        rationale=rationale,
    )


# ===========================================================================
# TestExpandSectorToAssetBudgets
# ===========================================================================


class TestExpandSectorToAssetBudgets:
    def test_equal_split_within_sector(self) -> None:
        """Two assets in Technology → each gets half of sector budget."""
        sector_budgets = {"Technology": 0.40, "Healthcare": 0.30, "Energy": 0.30}
        asset_map = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}
        result = _expand_sector_to_asset_budgets(sector_budgets, asset_map)
        assert result["AAPL"] == pytest.approx(0.20)
        assert result["MSFT"] == pytest.approx(0.20)
        assert result["JNJ"] == pytest.approx(0.30)

    def test_single_asset_sector_gets_full_budget(self) -> None:
        sector_budgets = {"Energy": 0.50}
        result = _expand_sector_to_asset_budgets(sector_budgets, {"XOM": "Energy"})
        assert result["XOM"] == pytest.approx(0.50)

    def test_missing_sector_gets_zero(self) -> None:
        """Asset whose sector is not in sector_budgets receives 0."""
        result = _expand_sector_to_asset_budgets(
            {"Technology": 0.60}, {"AAPL": "Technology", "JNJ": "Healthcare"}
        )
        assert result["JNJ"] == pytest.approx(0.0)

    def test_all_assets_present_in_output(self) -> None:
        sector_budgets = {"Technology": 0.40, "Healthcare": 0.60}
        result = _expand_sector_to_asset_budgets(sector_budgets, ASSET_SECTOR_MAP)
        assert set(result.keys()) == set(ASSET_SECTOR_MAP.keys())

    def test_negative_sector_budget_clamped_to_zero(self) -> None:
        result = _expand_sector_to_asset_budgets(
            {"Technology": -0.10}, {"AAPL": "Technology"}
        )
        assert result["AAPL"] == pytest.approx(0.0)


# ===========================================================================
# TestNormalise
# ===========================================================================


class TestNormalise:
    def test_values_sum_to_one(self) -> None:
        d = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        result = _normalise(d)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-10)

    def test_already_normalised_unchanged(self) -> None:
        d = {"a": 0.5, "b": 0.5}
        result = _normalise(d)
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_zero_total_falls_back_to_equal(self) -> None:
        d = {"a": 0.0, "b": 0.0}
        result = _normalise(d)
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_all_values_non_negative(self) -> None:
        d = {"a": 3.0, "b": 1.0, "c": 0.5}
        result = _normalise(d)
        assert all(v >= 0.0 for v in result.values())


# ===========================================================================
# TestToBudgetArray
# ===========================================================================


class TestToBudgetArray:
    def test_correct_order(self) -> None:
        budgets = {"AAPL": 0.5, "MSFT": 0.3, "JNJ": 0.2}
        arr = _to_budget_array(budgets, ["JNJ", "AAPL", "MSFT"])
        assert arr[0] == pytest.approx(0.2)
        assert arr[1] == pytest.approx(0.5)
        assert arr[2] == pytest.approx(0.3)

    def test_missing_asset_defaults_to_zero(self) -> None:
        budgets = {"AAPL": 1.0}
        arr = _to_budget_array(budgets, ["AAPL", "MSFT"])
        assert arr[1] == pytest.approx(0.0)

    def test_dtype_float64(self) -> None:
        arr = _to_budget_array({"AAPL": 1.0}, ["AAPL"])
        assert arr.dtype == np.float64

    def test_shape_matches_asset_list(self) -> None:
        arr = _to_budget_array({"a": 0.5, "b": 0.5}, ["a", "b", "c"])
        assert arr.shape == (3,)


# ===========================================================================
# TestCalibrateRiskBudget
# ===========================================================================


class TestCalibrateRiskBudget:
    def test_returns_numpy_array(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output()
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_n_assets(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output()
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assert result.shape == (len(ASSET_SECTOR_MAP),)

    def test_sums_to_one(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output()
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assert float(result.sum()) == pytest.approx(1.0, abs=1e-8)

    def test_all_values_non_negative(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output()
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assert np.all(result >= 0.0)

    def test_order_matches_asset_sector_map(self) -> None:
        """Budget vector entries follow dict insertion order of asset_sector_map."""
        asset_budgets = {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "JNJ": 0.30,
            "XOM": 0.10,
            "JPM": 0.20,
        }
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output(
                asset_budgets=asset_budgets
            )
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assets = list(ASSET_SECTOR_MAP.keys())
        for i, asset in enumerate(assets):
            expected = asset_budgets[asset] / sum(asset_budgets.values())
            assert result[i] == pytest.approx(expected, abs=1e-8)

    def test_missing_asset_in_llm_output_falls_back_to_sector_expansion(self) -> None:
        """If LLM omits an asset, service falls back to sector expansion."""
        incomplete = {"AAPL": 0.30, "MSFT": 0.30, "JNJ": 0.40}  # XOM, JPM missing
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output(
                asset_budgets=incomplete
            )
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assert result.shape == (len(ASSET_SECTOR_MAP),)
        assert float(result.sum()) == pytest.approx(1.0, abs=1e-8)

    def test_baml_args_forwarded_correctly(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output()
            calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        call_kwargs = mock_b.CalibrateRiskBudget.call_args.kwargs
        assert call_kwargs["sector_outlook"] == OUTLOOK
        assert set(call_kwargs["sector_universe"]) == set(SECTORS)
        assert call_kwargs["asset_sector_map"] == ASSET_SECTOR_MAP

    def test_empty_sector_universe_raises(self) -> None:
        with pytest.raises(ValueError, match="sector_universe"):
            calibrate_risk_budget(OUTLOOK, [], ASSET_SECTOR_MAP)

    def test_empty_asset_sector_map_raises(self) -> None:
        with pytest.raises(ValueError, match="asset_sector_map"):
            calibrate_risk_budget(OUTLOOK, SECTORS, {})

    def test_baml_failure_raises_runtime_error(self) -> None:
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.side_effect = Exception("LLM error")
            with pytest.raises(RuntimeError, match="LLM risk budget"):
                calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)

    def test_sector_expansion_overweight_tech(self) -> None:
        """Overweight Technology → AAPL + MSFT combined budget > single sector baseline."""
        sector_budgets = {
            "Technology": 0.50,
            "Healthcare": 0.20,
            "Energy": 0.10,
            "Financials": 0.20,
        }
        asset_budgets = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "JNJ": 0.20,
            "XOM": 0.10,
            "JPM": 0.20,
        }
        with patch("app.services.risk_budget_service.b") as mock_b:
            mock_b.CalibrateRiskBudget.return_value = _make_output(
                sector_budgets=sector_budgets, asset_budgets=asset_budgets
            )
            result = calibrate_risk_budget(OUTLOOK, SECTORS, ASSET_SECTOR_MAP)
        assets = list(ASSET_SECTOR_MAP.keys())
        aapl_idx = assets.index("AAPL")
        msft_idx = assets.index("MSFT")
        xom_idx = assets.index("XOM")
        # Tech combined > energy (overweight tech)
        assert result[aapl_idx] + result[msft_idx] > result[xom_idx]
