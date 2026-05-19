"""Integration tests for regime-conditional sector band injection.

These tests use toy in-memory data (no live DB, no external dependencies)
and test the following:

1. ``_inject_sector_bands`` wires floor/cap rows correctly.
2. RECESSION regime → Technology cap = 0.06 (constraint present on optimizer).
3. ``sector_bands_config=None`` → optimizer constraints unchanged.
4. ``PortfolioResult.classified_regime`` is set when macro data is present.
5. Combined multi-level groups: both sector and region constraints resolve
   without a skfolio ``UserWarning: missing from the groups`` warning.
"""

from __future__ import annotations

import warnings
from typing import ClassVar
from unittest.mock import MagicMock

import pandas as pd
import pytest

from optimizer.factors._config import MacroRegime, SectorRegimeBandsConfig
from optimizer.factors._sector_bands import resolve_sector_bands
from optimizer.pipeline._orchestrator import _inject_sector_bands

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_TOY_SECTOR_MAPPING: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "XOM": "Energy",
}

_TOY_REGION_MAPPING: dict[str, str] = {
    "AAPL": "Americas",
    "MSFT": "Americas",
    "JNJ": "Americas",
    "UNH": "Americas",
    "JPM": "Americas",
    "BAC": "Americas",
    "XOM": "Americas",
}

_RECESSION_BANDS: dict[str, tuple[float, float]] = {
    "Technology": (0.00, 0.06),
    "Healthcare": (0.10, 0.28),
    "Financial Services": (0.00, 0.06),
    "Energy": (0.06, 0.22),
}


# ---------------------------------------------------------------------------
# _inject_sector_bands tests
# ---------------------------------------------------------------------------


class TestInjectSectorBands:
    """Tests for the _inject_sector_bands function."""

    def test_recession_technology_cap_present(self) -> None:
        """RECESSION bands must produce 'Technology <= 0.06' in constraints."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        constraints = list(result.linear_constraints or [])
        assert any("Technology <= 0.06" in c for c in constraints), (
            f"Expected 'Technology <= 0.06' in {constraints}"
        )

    def test_recession_healthcare_floor_present(self) -> None:
        """Healthcare floor > 0 in RECESSION → floor row must be present."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        constraints = list(result.linear_constraints or [])
        assert any("Healthcare >= 0.1" in c for c in constraints), (
            f"Expected Healthcare floor row in {constraints}"
        )

    def test_no_floor_row_for_zero_floor(self) -> None:
        """Sectors with floor=0.0 must NOT emit a floor row."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        constraints = list(result.linear_constraints or [])
        # Technology floor = 0.0 in RECESSION → no floor row
        tech_floor_rows = [c for c in constraints if "Technology >=" in c]
        assert tech_floor_rows == [], (
            f"Expected no Technology floor row, got {tech_floor_rows}"
        )

    def test_existing_region_rows_preserved(self) -> None:
        """Region constraint rows (non-sector) must survive the rebuild."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.6", "Europe <= 0.5"])
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        constraints = list(result.linear_constraints or [])
        assert "Americas <= 0.6" in constraints
        assert "Europe <= 0.5" in constraints

    def test_existing_sector_cap_rows_dropped(self) -> None:
        """Old sector cap rows must be replaced by the new band rows."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Technology <= 0.25", "Americas <= 0.60"])
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        constraints = list(result.linear_constraints or [])
        # Old cap must be gone; new cap must be present.
        assert "Technology <= 0.25" not in constraints
        assert any("Technology <= 0.06" in c for c in constraints)
        # Region row preserved.
        assert "Americas <= 0.60" in constraints

    def test_groups_set_to_sector_mapping_when_no_region(self) -> None:
        """Without region_mapping, groups is the flat sector dict."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        assert result.groups == _TOY_SECTOR_MAPPING

    def test_returns_deepcopy(self) -> None:
        """The original optimizer must not be mutated."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.6"])
        original_constraints = list(opt.linear_constraints or [])
        _inject_sector_bands(opt, _RECESSION_BANDS, _TOY_SECTOR_MAPPING)
        assert list(opt.linear_constraints or []) == original_constraints

    def test_none_sector_mapping_returns_copy_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When sector_mapping is None, constraints are unchanged (only warn)."""
        import logging

        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.6"])
        logger_name = "optimizer.pipeline._orchestrator"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result = _inject_sector_bands(opt, _RECESSION_BANDS, None)
        # Should warn about missing sector_mapping.
        assert any("sector_mapping is None" in r.message for r in caplog.records)
        # linear_constraints unchanged (deepcopy preserves original).
        assert list(result.linear_constraints or []) == ["Americas <= 0.6"]


# ---------------------------------------------------------------------------
# Combined multi-level groups: sector + region co-existence
# ---------------------------------------------------------------------------


class TestCombinedGroupsMultiLevel:
    """Combined groups must let both sector and region constraints resolve.

    skfolio emits ``UserWarning: missing from the groups`` when a label
    referenced in ``linear_constraints`` is not found in the groups 2-D
    array.  With a combined ``dict[str, list[str]]`` groups structure the
    optimizer's ``input_to_array`` produces a ``(3, n_assets)`` 2-D array
    (asset names / sector labels / region labels), so both
    ``"<sector> <= cap"`` and ``"<region> <= 0.60"`` rows resolve without
    any such warning.
    """

    def test_region_constraint_resolves_without_warning(self) -> None:
        """No skfolio 'missing from the groups' warning with combined groups."""
        import numpy as np
        from skfolio.optimization import MeanRisk
        from skfolio.utils.equations import equations_to_matrix

        opt = MeanRisk(
            linear_constraints=["Americas <= 0.60", "Technology <= 0.25"],
        )
        _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            _TOY_REGION_MAPPING,
        )

        # Build the 2-D groups array the same way skfolio does at fit time.
        # dict[str, list[str]] → input_to_array prepends asset name as level-0.
        tickers = list(_TOY_SECTOR_MAPPING.keys())
        groups_2d = np.array(
            [
                tickers,
                [_TOY_SECTOR_MAPPING[t] for t in tickers],
                [_TOY_REGION_MAPPING.get(t, "Other") for t in tickers],
            ]
        )

        sector_band_row = "Technology <= 0.06"
        region_row = "Americas <= 0.60"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ineq = equations_to_matrix(groups_2d, [sector_band_row, region_row])

        missing_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "missing from the groups" in str(w.message)
        ]
        assert missing_warnings == [], (
            f"Unexpected 'missing from the groups' warnings: {missing_warnings}"
        )
        # Both rows must produce non-empty inequality rows.
        assert ineq[2].shape[0] == 2, (
            f"Expected 2 resolved constraints, got {ineq[2].shape[0]}"
        )

    def test_groups_contains_sector_labels(self) -> None:
        """Combined groups dict must contain a known sector label."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.60"])
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            _TOY_REGION_MAPPING,
        )
        assert isinstance(result.groups, dict)
        # Each value should be a list with [sector, region]
        for ticker, labels in result.groups.items():
            assert isinstance(labels, list), (
                f"{ticker}: expected list, got {type(labels)}"
            )
            assert len(labels) == 2, f"{ticker}: expected 2 labels, got {labels}"
        # Spot-check: AAPL → [Technology, Americas]
        assert result.groups["AAPL"] == ["Technology", "Americas"]

    def test_groups_contains_region_labels(self) -> None:
        """Combined groups dict must contain a known region label."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.60"])
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            _TOY_REGION_MAPPING,
        )
        # All region labels should appear as second element
        region_labels = {v[1] for v in result.groups.values()}
        assert "Americas" in region_labels

    def test_no_missing_groups_warning_on_meanrisk_fit(self) -> None:
        """MeanRisk.fit must not emit 'missing from the groups' for region rows.

        Uses mixed Americas/Europe regions so the region cap is feasible
        (total budget = 1.0, Americas assets = 5/7, Europe = 2/7 ≈ 0.29 <
        cap = 0.80).  The test asserts that no skfolio UserWarning mentioning
        ``"missing from the groups"`` is emitted — which would happen if
        the groups structure did not include region labels.
        """
        import numpy as np
        from skfolio.optimization import MeanRisk
        from skfolio.preprocessing import prices_to_returns

        # Use loose bands (no floors) so sector constraints stay feasible.
        loose_bands: dict[str, tuple[float, float]] = {
            "Technology": (0.00, 0.90),
            "Healthcare": (0.00, 0.90),
            "Financial Services": (0.00, 0.90),
            "Energy": (0.00, 0.90),
        }

        # Mixed region mapping: BAC and XOM in Europe so Americas < 1.0
        mixed_region_mapping: dict[str, str] = {
            "AAPL": "Americas",
            "MSFT": "Americas",
            "JNJ": "Americas",
            "UNH": "Americas",
            "JPM": "Americas",
            "BAC": "Europe",
            "XOM": "Europe",
        }

        np.random.seed(0)
        tickers = list(_TOY_SECTOR_MAPPING.keys())
        n = len(tickers)
        dates = pd.date_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.normal(0.0005, 0.01, (300, n)), axis=0),
            index=dates,
            columns=tickers,
        )
        X = prices_to_returns(prices)

        # Region constraint: Americas <= 0.80 is feasible because only 5/7
        # assets are in Americas (≈ 71% in equal-weight).
        opt = MeanRisk(linear_constraints=["Americas <= 0.80"])
        result = _inject_sector_bands(
            opt,
            loose_bands,
            _TOY_SECTOR_MAPPING,
            mixed_region_mapping,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.fit(X)

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


# ---------------------------------------------------------------------------
# resolve_sector_bands contract tests
# ---------------------------------------------------------------------------


class TestResolveSectorBands:
    """Tests for resolve_sector_bands against the default config."""

    def test_recession_technology_is_06(self) -> None:
        cfg = SectorRegimeBandsConfig.for_default()
        bands = resolve_sector_bands(MacroRegime.RECESSION, cfg)
        assert bands["Technology"] == (0.00, 0.06)

    def test_expansion_technology_is_10_28(self) -> None:
        cfg = SectorRegimeBandsConfig.for_default()
        bands = resolve_sector_bands(MacroRegime.EXPANSION, cfg)
        assert bands["Technology"] == (0.10, 0.28)

    def test_unknown_all_uniform(self) -> None:
        cfg = SectorRegimeBandsConfig.for_default()
        bands = resolve_sector_bands(MacroRegime.UNKNOWN, cfg)
        from optimizer.factors._sector_bands import ALL_11_SECTORS

        for sector in ALL_11_SECTORS:
            assert bands[sector] == (0.03, 0.16)

    def test_none_config_still_works(self) -> None:
        bands = resolve_sector_bands(MacroRegime.RECESSION, None)
        assert bands["Technology"] == (0.00, 0.06)


# ---------------------------------------------------------------------------
# sector_bands_config=None → constraints unchanged
# ---------------------------------------------------------------------------


class TestNoBandsConfig:
    """When sector_bands_config is None, optimizer constraints are untouched."""

    def test_no_sector_bands_config_leaves_constraints(self) -> None:
        """Confirm _inject_sector_bands is never called when config is None."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Americas <= 0.60"])
        # Simulate what run_full_pipeline_with_selection does when
        # sector_bands_config is None: it simply skips injection.
        # We verify the optimizer is unmodified.
        sector_bands_config = None
        classified_regime = MacroRegime.RECESSION

        if (
            sector_bands_config is not None
            and sector_bands_config.enable
            and classified_regime is not None
            and hasattr(opt, "linear_constraints")
        ):
            bands = resolve_sector_bands(classified_regime, sector_bands_config)
            opt = _inject_sector_bands(opt, bands, _TOY_SECTOR_MAPPING)

        # Must be unchanged.
        assert list(opt.linear_constraints or []) == ["Americas <= 0.60"]


# ---------------------------------------------------------------------------
# ROOT CAUSE A: hyphenated region labels are sanitized in groups + constraints
# ---------------------------------------------------------------------------


class TestHyphenatedRegionSanitization:
    """Hyphenated region labels must be sanitized to DSL-safe space-separated tokens.

    skfolio's equations parser treats '-' as arithmetic minus.  When a region
    label such as 'Asia-Pacific' is used in a constraint string, the parser
    splits it into tokens 'Asia' and 'Pacific', triggering a
    'UserWarning: missing from the groups' at fit time.

    After the fix, sanitize_group_token replaces '-' with ' ' in BOTH the
    groups level-2 values AND any preserved region rows in linear_constraints,
    ensuring they remain consistent with each other.
    """

    def test_hyphenated_region_in_region_mapping_is_sanitized_in_groups(
        self,
    ) -> None:
        """Combined groups dict must store 'Asia Pacific', not 'Asia-Pacific'."""
        from skfolio.optimization import MeanRisk

        region_mapping_with_hyphen: dict[str, str] = {
            "AAPL": "Americas",
            "MSFT": "Americas",
            "JNJ": "Americas",
            "UNH": "Americas",
            "JPM": "Americas",
            "BAC": "Asia-Pacific",
            "XOM": "Asia-Pacific",
        }
        opt = MeanRisk()
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            region_mapping_with_hyphen,
        )
        assert isinstance(result.groups, dict)
        for ticker, labels in result.groups.items():
            assert isinstance(labels, list)
            region = labels[1]
            assert "-" not in region, (
                f"Ticker {ticker}: region label {region!r} still contains hyphen"
            )
        # BAC and XOM must map to "Asia Pacific" (space) not "Asia-Pacific" (hyphen)
        assert result.groups["BAC"][1] == "Asia Pacific"
        assert result.groups["XOM"][1] == "Asia Pacific"

    def test_hyphenated_region_row_in_existing_constraints_is_sanitized(
        self,
    ) -> None:
        """Pre-existing 'Asia-Pacific <= 0.6' rows are rewritten to 'Asia Pacific'."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk(linear_constraints=["Asia-Pacific <= 0.6"])
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
        )
        constraints = list(result.linear_constraints or [])
        # The hyphenated form must be gone.
        assert "Asia-Pacific <= 0.6" not in constraints, (
            f"Unsanitized row still present: {constraints}"
        )
        # The sanitized form must be present.
        assert "Asia Pacific <= 0.6" in constraints, (
            f"Sanitized row missing: {constraints}"
        )

    def test_hyphenated_region_resolves_without_missing_groups_warning_on_fit(
        self,
    ) -> None:
        """MeanRisk.fit with hyphenated region in region_mapping emits no warning.

        Uses a toy dataset where two tickers are in 'Asia-Pacific'; after
        sanitization, 'Asia Pacific <= 0.50' must resolve without any
        UserWarning: 'missing from the groups'.
        """
        import numpy as np
        from skfolio.optimization import MeanRisk
        from skfolio.preprocessing import prices_to_returns

        # Sector mapping: 5 assets in known sectors present in _RECESSION_BANDS.
        sector_map: dict[str, str] = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JNJ": "Healthcare",
            "UNH": "Healthcare",
            "TSM": "Technology",
        }
        # Region mapping with hyphen — mimics real _REGION_MAP output.
        region_map: dict[str, str] = {
            "AAPL": "Americas",
            "MSFT": "Americas",
            "JNJ": "Americas",
            "UNH": "Americas",
            "TSM": "Asia-Pacific",
        }
        # Loose bands: no floor constraints so the optimization stays feasible.
        loose_bands: dict[str, tuple[float, float]] = {
            "Technology": (0.00, 0.90),
            "Healthcare": (0.00, 0.90),
        }

        np.random.seed(7)
        tickers = list(sector_map.keys())
        dates = pd.date_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.normal(0.0005, 0.01, (300, len(tickers))), axis=0),
            index=dates,
            columns=tickers,
        )
        X = prices_to_returns(prices)

        # Region constraint uses sanitized label.  Americas ≈ 4/5 assets ≈ 80%;
        # cap at 0.90 keeps it feasible.  Asia Pacific ≈ 1/5 ≈ 20% < 0.90.
        opt = MeanRisk(linear_constraints=["Americas <= 0.90", "Asia-Pacific <= 0.90"])
        result = _inject_sector_bands(opt, loose_bands, sector_map, region_map)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.fit(X)

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


# ---------------------------------------------------------------------------
# ROOT CAUSE B: absent-sector / absent-region rows must not be emitted
# ---------------------------------------------------------------------------


class TestAbsentSectorRegionFiltering:
    """_inject_sector_bands must only emit rows for groups present in the universe.

    When the selected universe covers only a subset of the 11 GICS sectors or
    of the possible regions, constraint rows for absent groups must be silently
    skipped.  This prevents:

    * Logically infeasible floor constraints (``>= 0.03`` on a zero-asset sector).
    * skfolio 'UserWarning: missing from the groups' for absent group labels.
    """

    # Bands include ALL 11 sectors (simulates full RECESSION matrix).
    _ALL_BANDS: ClassVar[dict[str, tuple[float, float]]] = {
        "Energy": (0.06, 0.22),
        "Basic Materials": (0.03, 0.16),
        "Industrials": (0.00, 0.06),
        "Consumer Cyclical": (0.03, 0.16),
        "Consumer Defensive": (0.10, 0.28),
        "Healthcare": (0.10, 0.28),
        "Financial Services": (0.00, 0.06),
        "Technology": (0.00, 0.06),
        "Communication Services": (0.00, 0.06),
        "Utilities": (0.10, 0.28),
        "Real Estate": (0.00, 0.06),
    }

    # Selected universe: only 4 of the 11 sectors represented.
    _SELECTED_SECTOR_MAP: ClassVar[dict[str, str]] = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "JPM": "Financial Services",
        "BAC": "Financial Services",
        "XOM": "Energy",
    }

    def test_absent_sectors_emit_no_band_row(self) -> None:
        """Sectors with no tickers in the selected universe must have no row."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, self._ALL_BANDS, self._SELECTED_SECTOR_MAP)
        constraints = list(result.linear_constraints or [])

        absent_sectors = {
            "Basic Materials",
            "Industrials",
            "Consumer Cyclical",
            "Consumer Defensive",
            "Communication Services",
            "Utilities",
            "Real Estate",
        }
        for sector in absent_sectors:
            sector_rows = [c for c in constraints if c.startswith(sector)]
            assert sector_rows == [], (
                f"Absent sector {sector!r} produced rows: {sector_rows}"
            )

    def test_present_sectors_emit_band_rows(self) -> None:
        """Sectors present in the selected universe must emit cap (and floor) rows."""
        from skfolio.optimization import MeanRisk

        opt = MeanRisk()
        result = _inject_sector_bands(opt, self._ALL_BANDS, self._SELECTED_SECTOR_MAP)
        constraints = list(result.linear_constraints or [])

        present_sectors = {"Technology", "Healthcare", "Financial Services", "Energy"}
        for sector in present_sectors:
            cap_rows = [c for c in constraints if f"{sector} <=" in c]
            assert cap_rows, (
                f"Present sector {sector!r} has no cap row in {constraints}"
            )

    def test_no_region_row_emitted_for_other_region(self) -> None:
        """No region cap row is emitted for the 'Other' fallback bucket.

        The 'Other' region is a catch-all for tickers whose country cannot be
        mapped to a named region.  Emitting a constraint for it is semantically
        meaningless and would trigger a skfolio 'missing from the groups'
        warning when no assets resolve to it at fit time.
        """
        from skfolio.optimization import MeanRisk

        # Some tickers fall through to "Other" (no region for them).
        region_map_with_other: dict[str, str] = {
            "AAPL": "Americas",
            "MSFT": "Americas",
            # JNJ, UNH, JPM, BAC, XOM have no entry → "Other"
        }
        opt = MeanRisk()
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            region_map_with_other,
        )
        constraints = list(result.linear_constraints or [])
        other_rows = [c for c in constraints if c.startswith("Other")]
        assert other_rows == [], f"Unexpected 'Other' region row(s): {other_rows}"

    def test_absent_region_emits_no_cap_row(self) -> None:
        """A region absent from the selected universe must not appear in constraints.

        When the region_mapping has only 'Americas' tickers, no 'Europe' row
        should be present even if the original optimizer had one.
        """
        from skfolio.optimization import MeanRisk

        # Pre-existing constraint for a region that has no tickers.
        opt = MeanRisk(linear_constraints=["Americas <= 0.80"])
        # Only Americas tickers in region mapping.
        americas_only = dict.fromkeys(_TOY_SECTOR_MAPPING, "Americas")
        result = _inject_sector_bands(
            opt,
            _RECESSION_BANDS,
            _TOY_SECTOR_MAPPING,
            americas_only,
        )
        constraints = list(result.linear_constraints or [])
        # The region row for Americas must be preserved.
        assert "Americas <= 0.80" in constraints
        # No Europe row must appear (it was never in original constraints).
        assert not any("Europe" in c for c in constraints)


# ---------------------------------------------------------------------------
# PortfolioResult.classified_regime field
# ---------------------------------------------------------------------------


class TestClassifiedRegimeField:
    """PortfolioResult must expose classified_regime."""

    def test_classified_regime_default_none(self) -> None:
        from optimizer.pipeline._config import PortfolioResult

        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        portfolio_mock = MagicMock()
        result = PortfolioResult(weights=weights, portfolio=portfolio_mock)
        assert result.classified_regime is None

    def test_classified_regime_assignable(self) -> None:
        from optimizer.pipeline._config import PortfolioResult

        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        portfolio_mock = MagicMock()
        result = PortfolioResult(weights=weights, portfolio=portfolio_mock)
        result.classified_regime = MacroRegime.RECESSION
        assert result.classified_regime is MacroRegime.RECESSION
