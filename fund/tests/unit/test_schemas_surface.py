"""``fund.schemas`` public surface tests.

The package ``__init__`` re-exports the four schemas (plus their nested models),
the shared MiFID enums, and the ``structured_call`` helper (with
``StructuredOutputError`` / ``SupportsStructuredOutput``) so callers import from
one place. These tests assert the documented surface resolves, that ``__all__``
has no dangling names (every listed name is a real, exported attribute), that it
covers every sub-module's own ``__all__``, and that it stays sorted +
duplicate-free.
"""

from __future__ import annotations

import importlib

import pytest

import fund.schemas as schemas

# Sub-modules whose public names roll up into the package surface.
_SUBMODULES = (
    "candidates",
    "enums",
    "mandate",
    "constraint_set",
    "views",
    "decision",
    "structured",
    "questionnaire",
)


@pytest.mark.criterion("REQ-FUND-043")
def test_documented_surface_resolves():
    """Validates fund.schemas.__all__: every documented name, the candidate-set
    models and ``CANDIDATE_IDS`` included, imports from the package itself."""
    from fund.schemas import (
        CANDIDATE_IDS,
        AllocDecision,
        CandidateSet,
        ConstraintSet,
        GicsSector,
        Horizon,
        MiFIDAnswers,
        MomentsEstimator,
        ObjectiveChoice,
        PortfolioMandate,
        RiskMeasureChoice,
        StructuredOutputError,
        SuitabilityAssessment,
        UncertaintyLevel,
        ViewSet,
        structured_call,
    )

    documented = (
        CANDIDATE_IDS,
        AllocDecision,
        CandidateSet,
        ConstraintSet,
        GicsSector,
        Horizon,
        MiFIDAnswers,
        MomentsEstimator,
        ObjectiveChoice,
        PortfolioMandate,
        RiskMeasureChoice,
        StructuredOutputError,
        SuitabilityAssessment,
        UncertaintyLevel,
        ViewSet,
        structured_call,
    )
    assert all(obj is not None for obj in documented)


def test_all_has_no_dangling_names():
    for name in schemas.__all__:
        assert hasattr(schemas, name), f"__all__ lists {name!r} but it is not exported"


@pytest.mark.criterion("REQ-FUND-043")
def test_all_covers_every_submodule_surface():
    """Validates fund.schemas.__all__: it lists every name each submodule exports,
    ``fund.schemas.candidates`` included, so no schema is reachable only by path."""
    for modname in _SUBMODULES:
        mod = importlib.import_module(f"fund.schemas.{modname}")
        for name in mod.__all__:
            assert name in schemas.__all__, (
                f"{name!r} (from {modname}.py) is missing from fund.schemas.__all__"
            )


def test_all_is_sorted_and_unique():
    assert schemas.__all__ == sorted(schemas.__all__)
    assert len(schemas.__all__) == len(set(schemas.__all__))


def test_reexports_are_identical_objects():
    # A re-export must be the same object, not a copy/shadow.
    from fund.schemas import constraint_set as constraint_set_mod

    assert schemas.ConstraintSet is constraint_set_mod.ConstraintSet
