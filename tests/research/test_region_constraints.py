"""Research-side wiring tests for ``build_region_linear_constraints`` (issue #552).

Optimizer-side coverage for the helper itself lives in
``tests/optimization/test_region_constraints.py``.  This module verifies the
research-pipeline wiring: ``_make_builder`` and ``build_research_optimizer``
must propagate ``country_map`` + ``_REGION_MAP`` + ``_MAX_REGION_WEIGHT`` into
the produced ``MeanRisk.linear_constraints`` list, and the
``compute_binding_constraints`` helper (issue #549) must agree with the
linear-system definition on slack vs tight weight vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimizer.optimization import sanitize_group_token
from research.optimization._config import (
    _MAX_REGION_WEIGHT,
    _REGION_MAP,
    _make_builder,
    _make_opt_config,
    build_research_optimizer,
)
from research.reporting._report import compute_binding_constraints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TICKERS = ["AAPL", "MSFT", "SAP", "ORA", "TSM"]
_COUNTRY_MAP = {
    "AAPL": "United States",
    "MSFT": "United States",
    "SAP": "Germany",
    "ORA": "France",
    "TSM": "Taiwan",
}
_SECTOR_MAPPING = dict.fromkeys(_TICKERS, "Tech")


def _build_opt(country_map: dict[str, str] | None):
    config = _make_opt_config(
        n_survivors=len(_TICKERS),
        target_count=len(_TICKERS),
        cost_bps=10.0,
    )
    builder = _make_builder(
        sector_mapping=_SECTOR_MAPPING,
        country_map=country_map,
        previous_weights=None,
    )
    return builder(config)


# Sanitized region token set used for matching against emitted constraint rows.
# Hyphens are replaced with spaces to match the DSL-safe tokens that
# build_region_linear_constraints now emits (e.g. "Asia Pacific" not "Asia-Pacific").
_REGIONS = {sanitize_group_token(r) for r in _REGION_MAP.values()} | {"Other"}


def _region_rows(opt) -> list[str]:
    rows = []
    for row in opt.linear_constraints or []:
        if "<=" not in row:
            continue
        lhs = row.split(" <= ")[0]
        if lhs in _REGIONS:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Wiring — country_map propagation
# ---------------------------------------------------------------------------


class TestRegionConstraintsWiring:
    def test_when_country_map_none_then_no_region_rows_attached(self) -> None:
        opt = _build_opt(country_map=None)
        for row in opt.linear_constraints or []:
            assert "Europe" not in row
            assert "Americas" not in row
            assert "Asia-Pacific" not in row

    def test_when_country_map_empty_then_no_region_rows_attached(self) -> None:
        opt = _build_opt(country_map={})
        for row in opt.linear_constraints or []:
            assert "Europe" not in row
            assert "Americas" not in row
            assert "Asia-Pacific" not in row

    def test_when_country_map_supplied_then_region_rows_present(self) -> None:
        """Region rows are emitted with DSL-safe tokens (hyphens → spaces)."""
        opt = _build_opt(country_map=_COUNTRY_MAP)
        rows = _region_rows(opt)
        regions = {row.split(" <= ")[0] for row in rows}
        assert "Americas" in regions
        assert "Europe" in regions
        # "Asia-Pacific" is sanitized to "Asia Pacific" to avoid the skfolio
        # DSL parser treating "-" as arithmetic minus.
        assert "Asia Pacific" in regions
        assert "Asia-Pacific" not in regions

    def test_when_country_map_supplied_then_cap_is_max_region_weight(self) -> None:
        opt = _build_opt(country_map=_COUNTRY_MAP)
        rows = _region_rows(opt)
        for row in rows:
            cap = float(row.split(" <= ")[1])
            assert cap == _MAX_REGION_WEIGHT

    def test_when_called_then_only_regions_in_input_universe_appear(self) -> None:
        # Universe: US + EU only — Asia-Pacific must be absent.
        country_map = {"AAPL": "United States", "SAP": "Germany"}
        opt = _build_opt(country_map=country_map)
        rows = _region_rows(opt)
        regions = {row.split(" <= ")[0] for row in rows}
        assert regions == {"Americas", "Europe"}

    def test_when_country_unknown_to_region_map_then_other_not_in_constraints(
        self,
    ) -> None:
        """The 'Other' catch-all bucket is excluded from constraint emission.

        Tickers whose country is absent from the region map fall back to
        ``"Other"`` in the groups dict, but no ``"Other <= cap"`` constraint row
        is emitted.  Constraining an uncategorised catch-all is semantically
        meaningless, and skfolio warns when a constrained group label is absent
        from the fitted universe's groups array.
        """
        country_map = {"AAPL": "United States", "MYS": "Malaysia"}
        opt = _build_opt(country_map=country_map)
        rows = _region_rows(opt)
        regions = {row.split(" <= ")[0] for row in rows}
        assert "Other" not in regions
        # Only the known regions for the input universe are constrained.
        assert regions == {"Americas"}

    def test_when_max_region_weight_default_then_60_percent(self) -> None:
        assert _MAX_REGION_WEIGHT == 0.60

    def test_when_region_map_loaded_then_contains_required_regions(self) -> None:
        regions = set(_REGION_MAP.values())
        assert {"Americas", "Europe", "Asia-Pacific"} <= regions


# ---------------------------------------------------------------------------
# Wiring — build_research_optimizer entry point
# ---------------------------------------------------------------------------


class TestBuildResearchOptimizer:
    def test_when_called_then_attaches_region_rows(self) -> None:
        opt = build_research_optimizer(
            sector_mapping=_SECTOR_MAPPING,
            country_map=_COUNTRY_MAP,
            n_survivors=len(_TICKERS),
            target_count=len(_TICKERS),
            cost_bps=10.0,
        )
        rows = _region_rows(opt)
        assert any("Americas" in row for row in rows)

    def test_when_country_map_none_then_no_region_rows(self) -> None:
        opt = build_research_optimizer(
            sector_mapping=_SECTOR_MAPPING,
            country_map=None,
            n_survivors=len(_TICKERS),
            target_count=len(_TICKERS),
            cost_bps=10.0,
        )
        rows = _region_rows(opt)
        assert rows == [] or all(
            row.split(" <= ")[0] not in _REGION_MAP.values() for row in rows
        )


# ---------------------------------------------------------------------------
# Linear-system semantics — slack vs binding under compute_binding_constraints
# ---------------------------------------------------------------------------


def _build_region_matrix(
    country_map: dict[str, str],
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (A, b, labels) for region-cap rows on the given ticker order."""
    regions = sorted({_REGION_MAP.get(country_map[t], "Other") for t in tickers})
    a_mat = np.zeros((len(regions), len(tickers)))
    for j, ticker in enumerate(tickers):
        region = _REGION_MAP.get(country_map[ticker], "Other")
        a_mat[regions.index(region), j] = 1.0
    b = np.full(len(regions), _MAX_REGION_WEIGHT)
    labels = [f"region:{r} <= {_MAX_REGION_WEIGHT}" for r in regions]
    return a_mat, b, labels


class TestRegionBindingDetection:
    @pytest.mark.parametrize(
        "weights",
        [
            np.array([0.10, 0.15, 0.15, 0.20, 0.40]),  # all under 0.60
            np.array([0.05, 0.10, 0.20, 0.20, 0.45]),
        ],
    )
    def test_when_weights_under_cap_then_no_binding(self, weights: np.ndarray) -> None:
        a, b, labels = _build_region_matrix(_COUNTRY_MAP, _TICKERS)
        binding = compute_binding_constraints(a, b, weights, labels=labels)
        assert binding == []

    def test_when_americas_over_cap_then_americas_row_binds(self) -> None:
        # AAPL=0.40, MSFT=0.30 → Americas=0.70 > 0.60.
        weights = np.array([0.40, 0.30, 0.10, 0.10, 0.10])
        a, b, labels = _build_region_matrix(_COUNTRY_MAP, _TICKERS)
        binding = compute_binding_constraints(a, b, weights, labels=labels)
        assert any("Americas" in row for row in binding)
        assert not any("Asia-Pacific" in row for row in binding)

    def test_when_weights_exactly_at_cap_then_row_binds(self) -> None:
        # Europe = SAP + ORA = 0.30 + 0.30 = 0.60.
        weights = np.array([0.10, 0.10, 0.30, 0.30, 0.20])
        a, b, labels = _build_region_matrix(_COUNTRY_MAP, _TICKERS)
        binding = compute_binding_constraints(a, b, weights, labels=labels)
        assert any("Europe" in row for row in binding)


# ---------------------------------------------------------------------------
# Mutation isolation — _make_builder defensive copy (issue #670)
# ---------------------------------------------------------------------------


class TestMakeBuilderMutationIsolation:
    """_make_builder must snapshot mutable inputs at closure creation time.

    The retighten loop calls builder(config) *after* the caller may have
    modified the original dicts/arrays.  All three mutable parameters
    (previous_weights, sector_mapping, country_map) must be copied
    defensively so later mutations are invisible to the closure.
    """

    def _base_config(self) -> _make_opt_config:  # type: ignore[name-defined]
        from research.optimization._config import _make_opt_config

        return _make_opt_config(n_survivors=5, target_count=5, cost_bps=10.0)

    # -- previous_weights isolation -------------------------------------------

    def test_when_previous_weights_mutated_after_builder_creation_then_optimizer_uses_original(  # noqa: E501
        self,
    ) -> None:
        """Mutating previous_weights array after builder creation must not bleed in."""
        original = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prev = original.copy()

        builder = _make_builder(
            sector_mapping=dict.fromkeys(_TICKERS, "Tech"),
            country_map=None,
            previous_weights=prev,
        )

        # Mutate *after* closure creation
        prev[:] = 99.0

        opt = builder(self._base_config())

        # The optimizer must carry the original weights, not the mutated ones
        assert opt.previous_weights is not None
        np.testing.assert_array_equal(opt.previous_weights, original)

    def test_when_previous_weights_none_then_builder_still_works(self) -> None:
        builder = _make_builder(
            sector_mapping=dict.fromkeys(_TICKERS, "Tech"),
            country_map=None,
            previous_weights=None,
        )
        opt = builder(self._base_config())
        assert opt.previous_weights is None

    # -- sector_mapping isolation ---------------------------------------------

    def test_when_sector_mapping_mutated_after_builder_creation_then_optimizer_uses_original(  # noqa: E501
        self,
    ) -> None:
        """Mutating sector_mapping dict after builder creation must not bleed in."""
        sector_mapping = {"AAPL": "Tech", "MSFT": "Tech"}

        builder = _make_builder(
            sector_mapping=sector_mapping,
            country_map=None,
            previous_weights=None,
        )

        # Mutate *after* closure creation — add a new ticker with a new sector
        sector_mapping["JPM"] = "Financials"

        opt = builder(self._base_config())

        # The groups/constraints must reflect the original two-ticker mapping
        # not the mutated three-ticker one: "Financials" must not appear
        constraints = opt.linear_constraints or []
        assert not any("Financials" in c for c in constraints), (
            "sector_mapping mutation leaked into builder: 'Financials' found "
            f"in constraints {constraints!r}"
        )

    # -- country_map isolation ------------------------------------------------

    def test_when_country_map_mutated_after_builder_creation_then_optimizer_uses_original(  # noqa: E501
        self,
    ) -> None:
        """Mutating country_map dict after builder creation must not bleed in."""
        country_map = {
            "AAPL": "United States",
            "MSFT": "United States",
        }

        builder = _make_builder(
            sector_mapping=dict.fromkeys(["AAPL", "MSFT"], "Tech"),
            country_map=country_map,
            previous_weights=None,
        )

        # Mutate *after* closure creation — inject a European ticker
        country_map["SAP"] = "Germany"

        opt = builder(self._base_config())

        # Only "Americas" should appear; "Europe" must be absent because SAP was
        # added after builder creation.
        region_constraints = [
            row for row in (opt.linear_constraints or []) if "<=" in row
        ]
        regions_in_constraints = {row.split(" <= ")[0] for row in region_constraints}
        assert "Europe" not in regions_in_constraints, (
            "country_map mutation leaked into builder: 'Europe' found "
            f"in constraints {region_constraints!r}"
        )

    def test_when_country_map_none_mutated_to_dict_then_builder_sees_none(
        self,
    ) -> None:
        """None country_map at creation time means no region rows, period."""
        country_map: dict[str, str] | None = None

        builder = _make_builder(
            sector_mapping=dict.fromkeys(_TICKERS, "Tech"),
            country_map=country_map,
            previous_weights=None,
        )

        # Can't mutate None in-place; this tests that the closure captured None
        opt = builder(self._base_config())
        region_rows = [row for row in (opt.linear_constraints or []) if "<=" in row]
        assert all(
            row.split(" <= ")[0] not in set(_REGION_MAP.values()) for row in region_rows
        )
