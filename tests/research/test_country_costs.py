"""Tests for ``compute_weighted_cost_bps`` (Cycle 4 §9.1) — issue #554.

Per-country round-trip transaction-cost rates (``COUNTRY_COSTS_BPS``)
must weight-sum correctly and the ``_DEFAULT_COSTS`` fallback (≈ 18 bps)
must trigger when a ticker's country is unmapped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.stock_selection_pipeline import (
    _DEFAULT_COSTS,
    COUNTRY_COSTS_BPS,
    compute_weighted_cost_bps,
)


def _country_total(country: str) -> float:
    return float(sum(COUNTRY_COSTS_BPS[country].values()))


_DEFAULT_TOTAL = float(sum(_DEFAULT_COSTS.values()))


# ---------------------------------------------------------------------------
# Single-country portfolios
# ---------------------------------------------------------------------------


class TestSingleCountryWeights:
    def test_when_full_uk_weight_then_cost_equals_uk_total(self) -> None:
        weights = pd.Series({"BARC": 1.0})
        country_map = {"BARC": "United Kingdom"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            _country_total("United Kingdom")
        )

    def test_when_full_us_weight_then_cost_equals_us_total(self) -> None:
        weights = pd.Series({"AAPL": 1.0})
        country_map = {"AAPL": "United States"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            _country_total("United States")
        )

    def test_when_full_italy_weight_then_cost_equals_italy_total(self) -> None:
        weights = pd.Series({"ENI": 1.0})
        country_map = {"ENI": "Italy"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            _country_total("Italy")
        )


# ---------------------------------------------------------------------------
# Multi-country portfolios
# ---------------------------------------------------------------------------


class TestMultiCountryWeights:
    def test_when_50_50_uk_us_then_arithmetic_mean_of_totals(self) -> None:
        weights = pd.Series({"BARC": 0.5, "AAPL": 0.5})
        country_map = {"BARC": "United Kingdom", "AAPL": "United States"}
        expected = 0.5 * _country_total("United Kingdom") + 0.5 * _country_total(
            "United States"
        )
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            expected
        )

    def test_when_three_countries_then_weighted_sum(self) -> None:
        weights = pd.Series({"BARC": 0.4, "AAPL": 0.3, "ENI": 0.3})
        country_map = {
            "BARC": "United Kingdom",
            "AAPL": "United States",
            "ENI": "Italy",
        }
        expected = (
            0.4 * _country_total("United Kingdom")
            + 0.3 * _country_total("United States")
            + 0.3 * _country_total("Italy")
        )
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            expected
        )


# ---------------------------------------------------------------------------
# Default fallback
# ---------------------------------------------------------------------------


class TestDefaultFallback:
    def test_when_country_unmapped_then_uses_default_total(self) -> None:
        weights = pd.Series({"XXX": 1.0})
        country_map: dict[str, str] = {}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            _DEFAULT_TOTAL
        )

    def test_when_country_value_missing_from_costs_dict_then_default_used(
        self,
    ) -> None:
        weights = pd.Series({"GAZP": 1.0})
        country_map = {"GAZP": "Russia"}  # not in COUNTRY_COSTS_BPS
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            _DEFAULT_TOTAL
        )

    def test_when_default_total_then_equals_eighteen_bps(self) -> None:
        # Sentinel: stamp(0) + spread(6) + fx(12) = 18 bps.
        assert pytest.approx(18.0) == _DEFAULT_TOTAL

    def test_when_mix_of_known_and_unknown_then_weighted_sum_with_default(
        self,
    ) -> None:
        weights = pd.Series({"AAPL": 0.6, "XXX": 0.4})
        country_map = {"AAPL": "United States"}
        expected = 0.6 * _country_total("United States") + 0.4 * _DEFAULT_TOTAL
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            expected
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_when_weights_empty_then_zero_returned(self) -> None:
        assert compute_weighted_cost_bps(pd.Series(dtype=float), {}) == 0.0

    def test_when_weights_all_nan_then_zero_returned(self) -> None:
        weights = pd.Series({"AAPL": float("nan"), "MSFT": float("nan")})
        country_map = {"AAPL": "United States", "MSFT": "United States"}
        assert compute_weighted_cost_bps(weights, country_map) == 0.0

    def test_when_negative_weights_passed_then_signed_sum_returned(self) -> None:
        # Function does not normalise — caller responsibility.
        weights = pd.Series({"AAPL": -1.0})
        country_map = {"AAPL": "United States"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            -_country_total("United States")
        )

    def test_when_weights_do_not_sum_to_one_then_no_normalisation(self) -> None:
        weights = pd.Series({"AAPL": 0.50})  # only half allocated
        country_map = {"AAPL": "United States"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            0.50 * _country_total("United States")
        )

    def test_when_partial_nan_then_dropna_applied(self) -> None:
        weights = pd.Series({"AAPL": 0.6, "GHOST": float("nan")})
        country_map = {"AAPL": "United States"}
        assert compute_weighted_cost_bps(weights, country_map) == pytest.approx(
            0.6 * _country_total("United States")
        )


# ---------------------------------------------------------------------------
# Constants — sentinel values
# ---------------------------------------------------------------------------


class TestCountryCostsConstants:
    def test_when_uk_costs_then_total_is_58_bps(self) -> None:
        assert _country_total("United Kingdom") == pytest.approx(58.0)

    def test_when_france_costs_then_total_is_48_bps(self) -> None:
        assert _country_total("France") == pytest.approx(48.0)

    def test_when_italy_costs_then_total_is_30_bps(self) -> None:
        assert _country_total("Italy") == pytest.approx(30.0)

    def test_when_us_costs_then_total_is_15_bps(self) -> None:
        assert _country_total("United States") == pytest.approx(15.0)

    def test_when_default_costs_then_keys_match_country_costs_dict(self) -> None:
        # Default schema must align with per-country schema.
        any_country = next(iter(COUNTRY_COSTS_BPS.values()))
        assert set(_DEFAULT_COSTS.keys()) == set(any_country.keys())


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_when_called_then_returns_python_float(self) -> None:
        weights = pd.Series({"AAPL": 1.0})
        country_map = {"AAPL": "United States"}
        result = compute_weighted_cost_bps(weights, country_map)
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)
