"""Task 1 — ``fund.schemas.enums`` pins the MiFID-facing ``str, Enum`` vocabulary.

These are the fund-side vocabulary shared by the Phase-4 schemas. They map onto
the ``optimizer.optimization`` enums only inside the schema methods (Tasks 3-4);
this module itself imports nothing from ``optimizer`` / ``deepagents`` / ``app``.
Each test asserts the exact member set so an accidental rename fails loud.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from fund.schemas import enums

_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

_EXPECTED: dict[type[Enum], dict[str, str]] = {
    enums.ObjectiveChoice: {
        "PROTECTION": "protection",
        "INCOME": "income",
        "GROWTH": "growth",
        "MAX": "max",
    },
    enums.RiskMeasureChoice: {
        "VARIANCE": "variance",
        "SEMI_VARIANCE": "semi_variance",
        "CVAR": "cvar",
        "CDAR": "cdar",
        "MAX_DRAWDOWN": "max_drawdown",
    },
    enums.Horizon: {
        "SHORT": "short",
        "MEDIUM": "medium",
        "LONG": "long",
    },
    enums.GicsSector: {
        "ENERGY": "energy",
        "MATERIALS": "materials",
        "INDUSTRIALS": "industrials",
        "CONSUMER_DISCRETIONARY": "consumer_discretionary",
        "CONSUMER_STAPLES": "consumer_staples",
        "HEALTH_CARE": "health_care",
        "FINANCIALS": "financials",
        "INFORMATION_TECHNOLOGY": "information_technology",
        "COMMUNICATION_SERVICES": "communication_services",
        "UTILITIES": "utilities",
        "REAL_ESTATE": "real_estate",
    },
    enums.MomentsEstimator: {
        "LEDOIT_WOLF": "ledoit_wolf",
        "EMPIRICAL": "empirical",
        "EW": "ew",
    },
    enums.UncertaintyLevel: {
        "NONE": "none",
        "LOW": "low",
        "HIGH": "high",
    },
}


def test_every_enum_is_a_str_enum():
    for enum_cls in _EXPECTED:
        assert issubclass(enum_cls, str)
        assert issubclass(enum_cls, Enum)


def test_member_sets_are_exact():
    for enum_cls, members in _EXPECTED.items():
        got = {m.name: m.value for m in enum_cls}
        assert got == members, f"{enum_cls.__name__} member set drifted"


def test_gics_sector_has_exactly_eleven_members():
    assert len(enums.GicsSector) == 11


def test_values_are_lowercase_snake_case():
    for enum_cls in _EXPECTED:
        for member in enum_cls:
            assert _SNAKE.match(member.value), f"{enum_cls.__name__}.{member.name}"


def test_ledoit_wolf_is_the_first_moments_estimator():
    # D23 default is ledoit_wolf; first member documents the intended default.
    assert next(iter(enums.MomentsEstimator)) is enums.MomentsEstimator.LEDOIT_WOLF


def test_all_lists_exactly_the_six_public_enums():
    assert set(enums.__all__) == {
        "ObjectiveChoice",
        "RiskMeasureChoice",
        "Horizon",
        "GicsSector",
        "MomentsEstimator",
        "UncertaintyLevel",
    }


def test_module_imports_nothing_forbidden():
    src = Path(enums.__file__).read_text(encoding="utf-8")
    for forbidden in ("optimizer", "deepagents", "langgraph", "app", "skfolio"):
        assert f"import {forbidden}" not in src
        assert f"from {forbidden}" not in src
