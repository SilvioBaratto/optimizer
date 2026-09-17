"""Task 7 — ``fund.schemas`` public surface.

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

import fund.schemas as schemas

# Sub-modules whose public names roll up into the package surface.
_SUBMODULES = (
    "enums",
    "mandate",
    "constraint_set",
    "views",
    "decision",
    "structured",
)


def test_documented_surface_resolves():
    # The names §7 promises callers can import from ``fund.schemas`` directly.
    from fund.schemas import (
        AllocDecision,
        ConstraintSet,
        GicsSector,
        Horizon,
        MomentsEstimator,
        ObjectiveChoice,
        PortfolioMandate,
        RiskMeasureChoice,
        StructuredOutputError,
        UncertaintyLevel,
        ViewSet,
        structured_call,
    )

    documented = (
        AllocDecision,
        ConstraintSet,
        GicsSector,
        Horizon,
        MomentsEstimator,
        ObjectiveChoice,
        PortfolioMandate,
        RiskMeasureChoice,
        StructuredOutputError,
        UncertaintyLevel,
        ViewSet,
        structured_call,
    )
    assert all(obj is not None for obj in documented)


def test_all_has_no_dangling_names():
    # Every name advertised in ``__all__`` must be a real attribute.
    for name in schemas.__all__:
        assert hasattr(schemas, name), f"__all__ lists {name!r} but it is not exported"


def test_all_covers_every_submodule_surface():
    # The package surface is the union of the sub-modules' own ``__all__``.
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
