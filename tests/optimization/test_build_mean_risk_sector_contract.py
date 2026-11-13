"""Pin how ``build_mean_risk`` turns the sector inputs into skfolio arguments.

Each case fixes, for one combination of a sector cap, a sector mapping, sector
floors, regime bands and explicit ``groups`` / ``linear_constraints`` keywords,
the exact ``groups`` object handed to skfolio, the exact constraint rows in
their order, every warning logged, or the exact error raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from optimizer.optimization import MeanRiskConfig, build_mean_risk

_LOGGER = "optimizer.optimization._factory"

_MAPPING = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"}
_EXPLICIT_GROUPS = {"A": "Tech", "B": "Other", "C": "Health"}
_EXPLICIT_ROWS = ["Tech <= 0.5"]
_CAP_ROWS = ["Energy <= 0.3", "Health <= 0.3", "Tech <= 0.3"]
_FLOORS = {"Tech": 0.1, "Health": 0.05}
_FLOOR_ROWS = ["Health >= 0.05", "Tech >= 0.1"]
_BANDS = {"Tech": (0.1, 0.5), "Health": (0.0, 0.4)}
_BAND_ROWS = ["Tech >= 0.1", "Health <= 0.4", "Tech <= 0.5"]

# Markers standing for "the very object passed in", and for a keyword left out.
_FROM_MAPPING = "the sector_mapping argument"
_FROM_KEYWORD = "the explicit keyword"
_NOT_PASSED = "not passed"


@dataclass(frozen=True)
class Case:
    """One combination of sector inputs and what ``build_mean_risk`` makes of it."""

    config: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, Any] | None = None
    floors: dict[str, float] | None = None
    bands: dict[str, tuple[float, float]] | None = None
    groups: object = _NOT_PASSED
    rows: object = _NOT_PASSED
    expect_groups: object = None
    expect_rows: object = None
    expect_logs: tuple[str, ...] = ()


_BUILT: dict[str, Case] = {
    "no sector input": Case(),
    "empty floors and bands without a mapping": Case(floors={}, bands={}),
    "cap without a mapping only warns": Case(
        config={"max_sector_weight": 0.3},
        expect_logs=(
            "MeanRiskConfig.max_sector_weight=0.3000 is set but no "
            "sector_mapping was provided; sector constraints are skipped.",
        ),
    ),
    "cap with a mapping": Case(
        config={"max_sector_weight": 0.3},
        mapping=_MAPPING,
        expect_groups=_FROM_MAPPING,
        expect_rows=_CAP_ROWS,
    ),
    "cap rows come from the mapping even under explicit groups": Case(
        config={"max_sector_weight": 0.3},
        mapping=_MAPPING,
        groups=_EXPLICIT_GROUPS,
        expect_groups=_FROM_KEYWORD,
        expect_rows=_CAP_ROWS,
    ),
    "explicit rows win over cap rows": Case(
        config={"max_sector_weight": 0.3},
        mapping=_MAPPING,
        rows=_EXPLICIT_ROWS,
        expect_groups=_FROM_MAPPING,
        expect_rows=_FROM_KEYWORD,
    ),
    "cap over a mapping without any sector label": Case(
        config={"max_sector_weight": 0.3},
        mapping={"A": None, "B": ""},
        expect_groups=_FROM_MAPPING,
    ),
    "floors with a mapping": Case(
        mapping=_MAPPING,
        floors=_FLOORS,
        expect_groups=_FROM_MAPPING,
        expect_rows=_FLOOR_ROWS,
    ),
    "zero floor on an absent sector is dropped": Case(
        mapping=_MAPPING,
        floors={"Absent": 0.0, "Tech": 0.1},
        expect_groups=_FROM_MAPPING,
        expect_rows=["Tech >= 0.1"],
    ),
    "floors are checked against explicit groups": Case(
        mapping=_MAPPING,
        floors={"Tech": 0.1, "Energy": 0.0},
        groups=_EXPLICIT_GROUPS,
        expect_groups=_FROM_KEYWORD,
        expect_rows=["Tech >= 0.1"],
    ),
    "floors pass through unchecked under explicit groups=None": Case(
        mapping=_MAPPING,
        floors={"Absent": 0.2, "Tech": 0.1},
        groups=None,
        expect_groups=None,
        expect_rows=["Absent >= 0.2", "Tech >= 0.1"],
    ),
    "bands with a mapping": Case(
        mapping=_MAPPING,
        bands=_BANDS,
        expect_groups=_FROM_MAPPING,
        expect_rows=_BAND_ROWS,
    ),
    "bands with explicit groups and no mapping": Case(
        bands=_BANDS,
        groups=_EXPLICIT_GROUPS,
        expect_groups=_FROM_KEYWORD,
        expect_rows=_BAND_ROWS,
    ),
    "bands pass through unchecked under explicit groups=None": Case(
        mapping=_MAPPING,
        bands={"Absent": (0.1, 0.2)},
        groups=None,
        expect_groups=None,
        expect_rows=["Absent >= 0.1", "Absent <= 0.2"],
    ),
    "bands of absent sectors are skipped with a warning": Case(
        mapping=_MAPPING,
        bands={"Tech": (0.0, 0.6), "Absent": (0.1, 0.2)},
        expect_groups=_FROM_MAPPING,
        expect_rows=["Tech <= 0.6"],
        expect_logs=(
            "sector_bands for sectors absent from the universe are skipped: "
            "['Absent'].",
        ),
    ),
    "caps under the budget over a fully capped mapping warn": Case(
        mapping=_MAPPING,
        bands={"Tech": (0.0, 0.2), "Health": (0.0, 0.2), "Energy": (0.0, 0.2)},
        expect_groups=_FROM_MAPPING,
        expect_rows=["Energy <= 0.2", "Health <= 0.2", "Tech <= 0.2"],
        expect_logs=(
            "sector_bands caps of the sectors in the mapping sum to 0.6000 < "
            "budget 1.0000; the fit is infeasible unless assets missing from "
            "the mapping absorb the remainder.",
        ),
    ),
    "no budget means no cap-sum warning": Case(
        config={"budget": None},
        mapping=_MAPPING,
        bands={"Tech": (0.0, 0.2), "Health": (0.0, 0.2), "Energy": (0.0, 0.2)},
        expect_groups=_FROM_MAPPING,
        expect_rows=["Energy <= 0.2", "Health <= 0.2", "Tech <= 0.2"],
    ),
    "explicit rows win over band rows": Case(
        mapping=_MAPPING,
        bands=_BANDS,
        rows=_EXPLICIT_ROWS,
        expect_groups=_FROM_MAPPING,
        expect_rows=_FROM_KEYWORD,
    ),
    "cap, floor and band rows in that order": Case(
        config={"max_sector_weight": 0.3},
        mapping=_MAPPING,
        floors=_FLOORS,
        bands=_BANDS,
        expect_groups=_FROM_MAPPING,
        expect_rows=[*_CAP_ROWS, *_FLOOR_ROWS, *_BAND_ROWS],
    ),
    "cap warning and band rows from explicit groups": Case(
        config={"max_sector_weight": 0.3},
        bands=_BANDS,
        groups=_EXPLICIT_GROUPS,
        expect_groups=_FROM_KEYWORD,
        expect_rows=_BAND_ROWS,
        expect_logs=(
            "MeanRiskConfig.max_sector_weight=0.3000 is set but no "
            "sector_mapping was provided; sector constraints are skipped.",
        ),
    ),
}

_REJECTED: dict[str, tuple[Case, str]] = {
    "floors without a mapping": (
        Case(floors=_FLOORS),
        "min_sector_weights requires sector_mapping to enforce sector floor "
        "constraints (no group mapping available).",
    ),
    "floors without a mapping despite explicit groups": (
        Case(floors=_FLOORS, groups=_EXPLICIT_GROUPS),
        "min_sector_weights requires sector_mapping to enforce sector floor "
        "constraints (no group mapping available).",
    ),
    "floors summing above one": (
        Case(mapping=_MAPPING, floors={"Tech": 0.6, "Health": 0.6}),
        "min_sector_weights sum to 1.2000 > 1.0; floors are infeasible.",
    ),
    "floor above the cap": (
        Case(
            config={"max_sector_weight": 0.3},
            mapping=_MAPPING,
            floors={"Tech": 0.4},
        ),
        "min_sector_weights['Tech']=0.4000 exceeds max_sector_weight=0.3000.",
    ),
    "positive floor on an absent sector": (
        Case(mapping=_MAPPING, floors={"Absent": 0.1, "Tech": 0.1}),
        "min_sector_weights sector(s) ['Absent'] do not occur in the sector "
        "mapping (sectors present: ['Energy', 'Health', 'Tech']); check for a "
        "naming mismatch.",
    ),
    "bands without a mapping or groups": (
        Case(bands=_BANDS),
        "sector_bands requires sector_mapping or an explicit groups kwarg to "
        "enforce sector constraints (no group mapping available).",
    ),
    "bands without a mapping under explicit groups=None": (
        Case(bands=_BANDS, groups=None),
        "sector_bands requires sector_mapping or an explicit groups kwarg to "
        "enforce sector constraints (no group mapping available).",
    ),
    "band floor above its cap": (
        Case(mapping=_MAPPING, bands={"Tech": (0.6, 0.5)}),
        "sector_bands['Tech'] floor 0.6000 exceeds cap 0.5000.",
    ),
    "no band sector in the mapping": (
        Case(mapping=_MAPPING, bands={"Absent": (0.1, 0.2)}),
        "No sector_bands sector ['Absent'] occurs in the sector mapping "
        "(sectors present: ['Energy', 'Health', 'Tech']); check for a naming "
        "mismatch.",
    ),
    "band floors summing above one": (
        Case(mapping=_MAPPING, bands={"Tech": (0.6, 0.7), "Health": (0.6, 0.7)}),
        "sector_bands floors of the present sectors sum to 1.2000 > 1.0; floors "
        "are infeasible.",
    ),
}


def _keywords(case: Case) -> dict[str, Any]:
    """Fresh copies of the explicit keywords, so identity checks are meaningful."""
    keywords: dict[str, Any] = {}
    if case.groups != _NOT_PASSED:
        groups = case.groups
        keywords["groups"] = dict(groups) if isinstance(groups, dict) else None
    if isinstance(case.rows, list):
        keywords["linear_constraints"] = list(case.rows)
    return keywords


def _build(case: Case) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    mapping = None if case.mapping is None else dict(case.mapping)
    keywords = _keywords(case)
    model = build_mean_risk(
        MeanRiskConfig(**case.config),
        sector_mapping=mapping,
        min_sector_weights=case.floors,
        sector_bands=case.bands,
        **keywords,
    )
    return model, mapping, keywords


def _passed_in(expected: object, mapping: object, keyword: object) -> object:
    if expected == _FROM_MAPPING:
        return mapping
    if expected == _FROM_KEYWORD:
        return keyword
    return expected


def _logged(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    ours = [record for record in caplog.records if record.name == _LOGGER]
    assert all(record.levelno == logging.WARNING for record in ours)
    return tuple(record.getMessage() for record in ours)


@pytest.mark.parametrize("case", _BUILT.values(), ids=_BUILT.keys())
def test_sector_inputs_become_these_groups_rows_and_warnings(
    case: Case, caplog: pytest.LogCaptureFixture
) -> None:
    """The groups object, the rows in order and the warnings are exact."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        model, mapping, keywords = _build(case)

    groups = _passed_in(case.expect_groups, mapping, keywords.get("groups"))
    assert model.groups is groups
    if case.expect_rows == _FROM_KEYWORD:
        assert model.linear_constraints is keywords["linear_constraints"]
    else:
        assert model.linear_constraints == case.expect_rows
    assert _logged(caplog) == case.expect_logs


@pytest.mark.parametrize(("case", "message"), _REJECTED.values(), ids=_REJECTED.keys())
def test_infeasible_or_unmapped_sector_inputs_raise_this_message(
    case: Case, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Each rejected combination raises ValueError with the exact message."""
    with (
        caplog.at_level(logging.DEBUG, logger=_LOGGER),
        pytest.raises(ValueError) as raised,
    ):
        _build(case)

    assert str(raised.value) == message
    assert _logged(caplog) == ()


def test_input_dicts_are_not_mutated() -> None:
    """The mapping, floors and bands come back exactly as they went in."""
    mapping = dict(_MAPPING)
    floors = dict(_FLOORS)
    bands = dict(_BANDS)

    build_mean_risk(
        MeanRiskConfig(max_sector_weight=0.3),
        sector_mapping=mapping,
        min_sector_weights=floors,
        sector_bands=bands,
    )

    assert mapping == _MAPPING
    assert floors == _FLOORS
    assert bands == _BANDS


def test_rows_are_a_new_list_each_build() -> None:
    """Two builds never share one constraint list."""
    first = build_mean_risk(
        MeanRiskConfig(max_sector_weight=0.3), sector_mapping=dict(_MAPPING)
    )
    second = build_mean_risk(
        MeanRiskConfig(max_sector_weight=0.3), sector_mapping=dict(_MAPPING)
    )

    assert first.linear_constraints == second.linear_constraints == _CAP_ROWS
    assert first.linear_constraints is not second.linear_constraints
