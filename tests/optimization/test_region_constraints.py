"""Tests for region linear-constraint helper (issue #533)."""

from __future__ import annotations

import pytest

from optimizer.optimization import build_region_linear_constraints


class TestBuildRegionLinearConstraints:
    def test_when_empty_country_map_then_empty_outputs(self) -> None:
        groups, constraints = build_region_linear_constraints({}, {})
        assert groups == {}
        assert constraints == []

    def test_when_basic_inputs_then_groups_map_ticker_to_region(self) -> None:
        country_map = {"AAPL": "United States", "TSM": "Taiwan", "SAP": "Germany"}
        region_map = {
            "United States": "Americas",
            "Taiwan": "AsiaPacific",
            "Germany": "Europe",
        }
        groups, _ = build_region_linear_constraints(country_map, region_map)
        assert groups == {
            "AAPL": "Americas",
            "TSM": "AsiaPacific",
            "SAP": "Europe",
        }

    def test_when_basic_inputs_then_constraint_rows_per_region(self) -> None:
        country_map = {"AAPL": "United States", "TSM": "Taiwan", "SAP": "Germany"}
        region_map = {
            "United States": "Americas",
            "Taiwan": "AsiaPacific",
            "Germany": "Europe",
        }
        _, constraints = build_region_linear_constraints(country_map, region_map)
        assert "Americas <= 0.6" in constraints
        assert "AsiaPacific <= 0.6" in constraints
        assert "Europe <= 0.6" in constraints
        assert len(constraints) == 3

    def test_when_country_missing_from_region_map_then_other_fallback(self) -> None:
        country_map = {"AAPL": "United States", "MYS": "Malaysia"}
        region_map = {"United States": "Americas"}
        groups, constraints = build_region_linear_constraints(country_map, region_map)
        assert groups["MYS"] == "Other"
        assert "Other <= 0.6" in constraints

    def test_when_custom_cap_then_constraint_uses_custom_cap(self) -> None:
        country_map = {"AAPL": "United States"}
        region_map = {"United States": "Americas"}
        _, constraints = build_region_linear_constraints(
            country_map, region_map, max_region_weight=0.45
        )
        assert constraints == ["Americas <= 0.45"]

    def test_when_multiple_tickers_same_region_then_constraints_deduped(self) -> None:
        country_map = {
            "AAPL": "United States",
            "MSFT": "United States",
            "RY": "Canada",
        }
        region_map = {"United States": "Americas", "Canada": "Americas"}
        _, constraints = build_region_linear_constraints(country_map, region_map)
        assert constraints == ["Americas <= 0.6"]

    def test_when_called_then_inputs_not_mutated(self) -> None:
        country_map = {"AAPL": "United States"}
        region_map = {"United States": "Americas"}
        country_snapshot = dict(country_map)
        region_snapshot = dict(region_map)
        build_region_linear_constraints(country_map, region_map)
        assert country_map == country_snapshot
        assert region_map == region_snapshot

    def test_when_cap_is_irrational_then_rounded_to_six_decimals(self) -> None:
        country_map = {"AAPL": "United States"}
        region_map = {"United States": "Americas"}
        _, constraints = build_region_linear_constraints(
            country_map, region_map, max_region_weight=1 / 3
        )
        assert "0.333333" in constraints[0]

    def test_when_returned_then_constraint_rows_are_sorted(self) -> None:
        country_map = {"A": "DE", "B": "US", "C": "JP"}
        region_map = {"DE": "Europe", "US": "Americas", "JP": "AsiaPacific"}
        _, constraints = build_region_linear_constraints(country_map, region_map)
        assert constraints == [
            "Americas <= 0.6",
            "AsiaPacific <= 0.6",
            "Europe <= 0.6",
        ]

    def test_when_invalid_cap_then_value_error(self) -> None:
        country_map = {"AAPL": "United States"}
        region_map = {"United States": "Americas"}
        with pytest.raises(ValueError, match="max_region_weight"):
            build_region_linear_constraints(
                country_map, region_map, max_region_weight=0.0
            )
        with pytest.raises(ValueError, match="max_region_weight"):
            build_region_linear_constraints(
                country_map, region_map, max_region_weight=1.5
            )
