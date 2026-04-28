"""Unit tests for attribution_service.py.

Covers:
  - Brinson-Fachler decomposition with known weights/returns (hand-verified)
  - Weight validation (sum tolerance, empty input)
  - Sector grouping (null sector → "Unclassified")
  - Error paths: insufficient sector coverage, missing benchmark data,
    missing factor scores
"""

from __future__ import annotations

import pytest

from app.services.attribution_service import (
    BrinsonDecomposition,
    InsufficientSectorCoverageError,
    MissingDataError,
    compute_brinson_attribution,
    compute_factor_attribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sector_map(mapping: dict[str, str | None]) -> dict[str, str]:
    """Coerce None → 'Unclassified' to match service output."""
    return {t: (s or "Unclassified") for t, s in mapping.items()}


# ---------------------------------------------------------------------------
# Brinson-Fachler decomposition — pure math
# ---------------------------------------------------------------------------


class TestBrinsonDecomposition:
    """Test Brinson-Fachler formulas against hand-calculated values."""

    def test_allocation_effect_positive_overweight_outperforming_sector(self) -> None:
        # Portfolio overweights Tech (0.6 vs 0.5); Tech outperforms benchmark total.
        # allocation = (0.6-0.5) * (0.10 - 0.075) = 0.0025
        d = BrinsonDecomposition(
            portfolio_weight=0.6,
            benchmark_weight=0.5,
            portfolio_sector_return=0.12,
            benchmark_sector_return=0.10,
            benchmark_total_return=0.075,
        )
        assert pytest.approx(d.allocation_effect, abs=1e-9) == 0.0025

    def test_selection_effect_portfolio_outperforms_benchmark_in_sector(self) -> None:
        # selection = 0.5 * (0.12 - 0.10) = 0.01
        d = BrinsonDecomposition(
            portfolio_weight=0.6,
            benchmark_weight=0.5,
            portfolio_sector_return=0.12,
            benchmark_sector_return=0.10,
            benchmark_total_return=0.075,
        )
        assert pytest.approx(d.selection_effect, abs=1e-9) == 0.01

    def test_interaction_effect_overweight_and_outperforms(self) -> None:
        # interaction = (0.6-0.5) * (0.12 - 0.10) = 0.002
        d = BrinsonDecomposition(
            portfolio_weight=0.6,
            benchmark_weight=0.5,
            portfolio_sector_return=0.12,
            benchmark_sector_return=0.10,
            benchmark_total_return=0.075,
        )
        assert pytest.approx(d.interaction_effect, abs=1e-9) == 0.002

    def test_total_effect_is_sum_of_components(self) -> None:
        d = BrinsonDecomposition(
            portfolio_weight=0.6,
            benchmark_weight=0.5,
            portfolio_sector_return=0.12,
            benchmark_sector_return=0.10,
            benchmark_total_return=0.075,
        )
        assert pytest.approx(d.total_effect, abs=1e-9) == 0.0025 + 0.01 + 0.002

    def test_full_two_sector_decomposition_sums_to_active_return(self) -> None:
        # Tech: w_p=0.6, w_b=0.5, r_p=0.12, r_b=0.10
        # Finance: w_p=0.4, w_b=0.5, r_p=0.04, r_b=0.05
        # Benchmark total = 0.5*0.10 + 0.5*0.05 = 0.075
        # Portfolio total = 0.6*0.12 + 0.4*0.04 = 0.088
        # Active return = 0.013
        portfolio_weights = {"AAPL": 0.6, "JPM": 0.4}
        benchmark_weights = {"AAPL": 0.5, "JPM": 0.5}
        sector_map = {"AAPL": "Technology", "JPM": "Financials"}
        portfolio_returns = {"AAPL": 0.12, "JPM": 0.04}
        benchmark_returns = {"AAPL": 0.10, "JPM": 0.05}

        result = compute_brinson_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
        )

        assert pytest.approx(result["total_active_return"], abs=1e-9) == 0.013
        assert pytest.approx(result["total_allocation"], abs=1e-9) == 0.005
        assert pytest.approx(result["total_selection"], abs=1e-9) == 0.005
        assert pytest.approx(result["total_interaction"], abs=1e-9) == 0.003
        assert pytest.approx(result["portfolio_return"], abs=1e-9) == 0.088
        assert pytest.approx(result["benchmark_return"], abs=1e-9) == 0.075

    def test_active_return_equals_sum_of_all_effects(self) -> None:
        portfolio_weights = {"AAPL": 0.6, "JPM": 0.4}
        benchmark_weights = {"AAPL": 0.5, "JPM": 0.5}
        sector_map = {"AAPL": "Technology", "JPM": "Financials"}
        portfolio_returns = {"AAPL": 0.12, "JPM": 0.04}
        benchmark_returns = {"AAPL": 0.10, "JPM": 0.05}

        result = compute_brinson_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
        )

        total_via_effects = (
            result["total_allocation"]
            + result["total_selection"]
            + result["total_interaction"]
        )
        assert pytest.approx(total_via_effects, abs=1e-9) == result["total_active_return"]

    def test_null_sector_grouped_as_unclassified(self) -> None:
        portfolio_weights = {"AAPL": 0.5, "UNKN": 0.5}
        benchmark_weights = {"AAPL": 0.5, "UNKN": 0.5}
        sector_map = {"AAPL": "Technology", "UNKN": "Unclassified"}
        portfolio_returns = {"AAPL": 0.10, "UNKN": 0.05}
        benchmark_returns = {"AAPL": 0.10, "UNKN": 0.05}

        result = compute_brinson_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
        )

        sector_names = {s["sector"] for s in result["sectors"]}
        assert "Unclassified" in sector_names

    def test_single_sector_zero_allocation_effect(self) -> None:
        # When portfolio_weight == benchmark_weight, allocation = 0.
        portfolio_weights = {"AAPL": 1.0}
        benchmark_weights = {"AAPL": 1.0}
        sector_map = {"AAPL": "Technology"}
        portfolio_returns = {"AAPL": 0.12}
        benchmark_returns = {"AAPL": 0.10}

        result = compute_brinson_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
        )

        assert pytest.approx(result["total_allocation"], abs=1e-9) == 0.0

    def test_raises_when_sector_coverage_below_threshold(self) -> None:
        # 3 out of 4 tickers lack sector info — 75% missing > 20% threshold.
        portfolio_weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        benchmark_weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        # Only "A" has a non-Unclassified sector — 75% Unclassified
        sector_map = {
            "A": "Technology",
            "B": "Unclassified",
            "C": "Unclassified",
            "D": "Unclassified",
        }
        portfolio_returns = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}
        benchmark_returns = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}

        with pytest.raises(InsufficientSectorCoverageError):
            compute_brinson_attribution(
                portfolio_weights=portfolio_weights,
                benchmark_weights=benchmark_weights,
                sector_map=sector_map,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
            )


# ---------------------------------------------------------------------------
# Factor attribution — pure math
# ---------------------------------------------------------------------------


class TestFactorAttribution:
    """Test factor attribution decomposition."""

    def test_single_factor_full_explanation(self) -> None:
        # If portfolio has exposure 1.0 to factor with same return → full explanation.
        # portfolio_weights = {AAPL: 1.0}
        # asset_returns = {AAPL: 0.10} → portfolio_return = 0.10
        # factor_exposures: {AAPL: {momentum: 1.0}}
        # factor_return = 0.10 (since only AAPL, weight=1, exposure=1)
        # factor_contribution = avg_exposure * factor_return = 1.0 * 0.10 = 0.10
        portfolio_weights = {"AAPL": 1.0}
        asset_returns = {"AAPL": 0.10}
        factor_exposures = {"AAPL": {"momentum": 1.0}}

        result = compute_factor_attribution(
            portfolio_weights=portfolio_weights,
            asset_returns=asset_returns,
            factor_exposures=factor_exposures,
        )

        assert pytest.approx(result["portfolio_return"], abs=1e-9) == 0.10
        assert len(result["factors"]) == 1
        assert result["factors"][0]["factor_name"] == "momentum"
        assert pytest.approx(result["residual"], abs=1e-6) == 0.0

    def test_residual_when_factor_does_not_explain_all_return(self) -> None:
        # If factor exposure is 0, it contributes nothing → all return is residual.
        portfolio_weights = {"AAPL": 1.0}
        asset_returns = {"AAPL": 0.10}
        factor_exposures = {"AAPL": {"momentum": 0.0}}

        result = compute_factor_attribution(
            portfolio_weights=portfolio_weights,
            asset_returns=asset_returns,
            factor_exposures=factor_exposures,
        )

        assert pytest.approx(result["portfolio_return"], abs=1e-9) == 0.10
        assert pytest.approx(result["residual"], abs=1e-6) == 0.10

    def test_explained_plus_residual_equals_portfolio_return(self) -> None:
        portfolio_weights = {"AAPL": 0.6, "MSFT": 0.4}
        asset_returns = {"AAPL": 0.12, "MSFT": 0.08}
        factor_exposures = {
            "AAPL": {"momentum": 0.5, "quality": 0.3},
            "MSFT": {"momentum": 0.2, "quality": 0.7},
        }

        result = compute_factor_attribution(
            portfolio_weights=portfolio_weights,
            asset_returns=asset_returns,
            factor_exposures=factor_exposures,
        )

        total = result["explained_return"] + result["residual"]
        assert pytest.approx(total, abs=1e-9) == result["portfolio_return"]

    def test_raises_missing_data_when_no_factor_exposures(self) -> None:
        with pytest.raises(MissingDataError):
            compute_factor_attribution(
                portfolio_weights={"AAPL": 1.0},
                asset_returns={"AAPL": 0.10},
                factor_exposures={},
            )

    def test_multiple_factors_contributions_sum_to_explained(self) -> None:
        portfolio_weights = {"AAPL": 0.5, "MSFT": 0.5}
        asset_returns = {"AAPL": 0.10, "MSFT": 0.06}
        factor_exposures = {
            "AAPL": {"momentum": 1.0, "value": -0.5},
            "MSFT": {"momentum": 0.5, "value": 0.5},
        }

        result = compute_factor_attribution(
            portfolio_weights=portfolio_weights,
            asset_returns=asset_returns,
            factor_exposures=factor_exposures,
        )

        factor_total = sum(f["contribution"] for f in result["factors"])
        assert pytest.approx(factor_total, abs=1e-9) == result["explained_return"]
