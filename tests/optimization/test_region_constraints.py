"""Tests for region linear-constraint helper (issue #533)."""

from __future__ import annotations

import pytest

from optimizer.optimization import build_region_linear_constraints, sanitize_group_token


class TestSanitizeGroupToken:
    """Tests for the skfolio DSL token sanitizer."""

    def test_hyphen_replaced_with_space(self) -> None:
        assert sanitize_group_token("Asia-Pacific") == "Asia Pacific"

    def test_no_hyphen_unchanged(self) -> None:
        assert sanitize_group_token("Americas") == "Americas"
        assert sanitize_group_token("Middle East & Africa") == "Middle East & Africa"

    def test_multiple_hyphens_all_replaced(self) -> None:
        assert sanitize_group_token("A-B-C") == "A B C"

    def test_empty_string(self) -> None:
        assert sanitize_group_token("") == ""

    def test_idempotent(self) -> None:
        result = sanitize_group_token("Asia-Pacific")
        assert sanitize_group_token(result) == result


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

    def test_when_country_missing_from_region_map_then_other_fallback_in_groups(
        self,
    ) -> None:
        """Tickers with unknown countries fall back to 'Other' in groups.

        The 'Other' catch-all bucket is intentionally excluded from constraint
        emission — constraining an uncategorised fallback is semantically
        meaningless and skfolio warns when a constrained group is absent from
        the fitted universe.
        """
        country_map = {"AAPL": "United States", "MYS": "Malaysia"}
        region_map = {"United States": "Americas"}
        groups, constraints = build_region_linear_constraints(country_map, region_map)
        assert groups["MYS"] == "Other"
        # "Other" must NOT appear in constraints (it is a catch-all, not a
        # cap-able group).
        assert "Other <= 0.6" not in constraints
        # Only "Americas" should be constrained.
        assert constraints == ["Americas <= 0.6"]

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

    def test_when_hyphenated_region_label_then_sanitized_in_groups_and_constraints(
        self,
    ) -> None:
        """Hyphenated region labels are sanitized to DSL-safe space-separated tokens.

        skfolio's equations parser treats '-' as arithmetic minus.  A region
        label 'Asia-Pacific' would be split into tokens 'Asia' and 'Pacific',
        triggering a 'missing from the groups' UserWarning.  After sanitization
        the emitted groups value and constraint string both use 'Asia Pacific'.
        """
        import warnings

        import numpy as np
        from skfolio.utils.equations import equations_to_matrix

        country_map = {"TSM": "Taiwan", "AAPL": "United States"}
        region_map = {"Taiwan": "Asia-Pacific", "United States": "Americas"}
        groups, constraints = build_region_linear_constraints(country_map, region_map)

        # groups values must be sanitized (hyphen → space)
        assert groups["TSM"] == "Asia Pacific"
        assert groups["AAPL"] == "Americas"

        # constraint strings must use the sanitized token
        assert "Asia Pacific <= 0.6" in constraints
        assert "Asia-Pacific" not in " ".join(constraints)

        # Build the 2-D groups array as skfolio does at fit time and confirm
        # that equations_to_matrix emits zero 'missing from the groups' warnings.
        tickers = list(groups.keys())
        groups_2d = np.array(
            [
                tickers,
                [groups[t] for t in tickers],
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            equations_to_matrix(groups_2d, constraints)

        missing_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "missing from the groups" in str(w.message)
        ]
        assert missing_warnings == [], (
            f"Unexpected 'missing from the groups' warnings: "
            f"{[str(w.message) for w in missing_warnings]}"
        )

    def test_when_hyphenated_region_on_meanrisk_fit_then_no_missing_groups_warning(
        self,
    ) -> None:
        """MeanRisk.fit on a mixed-region universe with hyphenated region labels.

        Verifies end-to-end that sanitized tokens produce zero UserWarning
        mentions of 'missing from the groups' during an actual MeanRisk.fit.
        """
        import warnings

        import numpy as np
        import pandas as pd
        from skfolio.optimization import MeanRisk
        from skfolio.preprocessing import prices_to_returns

        country_map = {
            "AAPL": "United States",
            "MSFT": "United States",
            "TSM": "Taiwan",
            "9984": "Japan",
        }
        region_map = {
            "United States": "Americas",
            "Taiwan": "Asia-Pacific",
            "Japan": "Asia-Pacific",
        }
        groups, constraints = build_region_linear_constraints(country_map, region_map)

        np.random.seed(42)
        tickers = list(country_map.keys())
        n = len(tickers)
        dates = pd.date_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.normal(0.0005, 0.01, (300, n)), axis=0),
            index=dates,
            columns=tickers,
        )
        X = prices_to_returns(prices)

        opt = MeanRisk(groups=groups, linear_constraints=constraints)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt.fit(X)

        missing_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "missing from the groups" in str(w.message)
        ]
        assert missing_warnings == [], (
            f"MeanRisk.fit emitted unexpected warnings: "
            f"{[str(w.message) for w in missing_warnings]}"
        )
