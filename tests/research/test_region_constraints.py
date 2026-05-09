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

from research._optimization import (
    _MAX_REGION_WEIGHT,
    _REGION_MAP,
    _make_builder,
    _make_opt_config,
    build_research_optimizer,
)
from research._report import compute_binding_constraints

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


_REGIONS = set(_REGION_MAP.values()) | {"Other"}


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
        opt = _build_opt(country_map=_COUNTRY_MAP)
        rows = _region_rows(opt)
        regions = {row.split(" <= ")[0] for row in rows}
        assert "Americas" in regions
        assert "Europe" in regions
        assert "Asia-Pacific" in regions

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

    def test_when_country_unknown_to_region_map_then_other_bucket(self) -> None:
        country_map = {"AAPL": "United States", "MYS": "Malaysia"}
        opt = _build_opt(country_map=country_map)
        rows = _region_rows(opt)
        regions = {row.split(" <= ")[0] for row in rows}
        assert "Other" in regions

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
