"""Task 5 — ``fund.schemas.decision`` pins the allocator's structured output.

``AllocDecision`` is the allocator's committed decision: it *references* the
persisted risk profile by key (``constraint_set_ref``) and *embeds* the per-run
``ViewSet`` (Q1), but carries **no weight field** — the no-weights invariant is
structural, not merely a convention (weights are computed downstream by the
optimizer, never chosen by the LLM). These tests assert validation, the
reference-not-embed boundary (a ``ConstraintSetRef``, never an inline
``ConstraintSet``), that the embedded ``ViewSet`` survives a JSON round-trip,
and the load-bearing no-weights invariant (``model_fields`` carries no
``weight``/``weights`` key).
"""

from __future__ import annotations

import pydantic
import pytest

from fund.schemas.decision import AllocDecision, ConstraintSetRef
from fund.schemas.enums import ObjectiveChoice
from fund.schemas.views import View, ViewSet


def _view_set(**overrides: object) -> ViewSet:
    kwargs: dict[str, object] = {
        "views": (View(target="AAPL", expected_return=0.0123, confidence=0.5),),
    }
    kwargs.update(overrides)
    return ViewSet(**kwargs)  # type: ignore[arg-type]


def _ref(**overrides: object) -> ConstraintSetRef:
    kwargs: dict[str, object] = {
        "portfolio_id": "pf-1",
        "store_key": "cs-2026-09-17",
    }
    kwargs.update(overrides)
    return ConstraintSetRef(**kwargs)  # type: ignore[arg-type]


def _decision(**overrides: object) -> AllocDecision:
    kwargs: dict[str, object] = {
        "portfolio_id": "pf-1",
        "universe": ("AAPL", "MSFT", "GOOG"),
        "constraint_set_ref": _ref(),
        "objective": ObjectiveChoice.GROWTH,
        "rationale": "Overweight tech on momentum.",
        "views": _view_set(),
    }
    kwargs.update(overrides)
    return AllocDecision(**kwargs)  # type: ignore[arg-type]


# -- validation --------------------------------------------------------------


def test_valid_decision_is_accepted():
    d = _decision()
    assert d.portfolio_id == "pf-1"
    assert d.universe == ("AAPL", "MSFT", "GOOG")
    assert d.objective is ObjectiveChoice.GROWTH
    assert d.rationale == "Overweight tech on momentum."


def test_constraint_set_ref_is_a_reference_not_an_embed():
    # The profile is referenced by (portfolio_id, store_key), never copied in.
    d = _decision()
    assert isinstance(d.constraint_set_ref, ConstraintSetRef)
    assert d.constraint_set_ref.portfolio_id == "pf-1"
    assert d.constraint_set_ref.store_key == "cs-2026-09-17"


def test_views_are_embedded():
    d = _decision()
    assert isinstance(d.views, ViewSet)
    assert d.views is not None
    assert d.views.views[0].target == "AAPL"


def test_views_default_to_none():
    d = _decision(views=None)
    assert d.views is None


def test_decision_and_ref_are_frozen():
    d = _decision()
    with pytest.raises(pydantic.ValidationError):
        d.rationale = "changed"  # type: ignore[misc]
    ref = _ref()
    with pytest.raises(pydantic.ValidationError):
        ref.store_key = "other"  # type: ignore[misc]


def test_models_are_hashable():
    # frozen=True keeps the schema hashable (matches the optimizer ethos).
    assert hash(_decision()) == hash(_decision())


def test_rejects_unknown_objective():
    with pytest.raises(pydantic.ValidationError):
        _decision(objective="speculation")


# -- JSON round-trip ---------------------------------------------------------


def test_json_round_trip_with_embedded_views():
    d = _decision()
    assert AllocDecision.model_validate(d.model_dump(mode="json")) == d


def test_json_round_trip_without_views():
    d = _decision(views=None)
    assert d.views is None
    assert AllocDecision.model_validate(d.model_dump(mode="json")) == d


# -- no-weights invariant (load-bearing) -------------------------------------


def test_carries_no_weight_field():
    # The allocator commits structured *inputs*, never weights; weights are
    # computed downstream by the optimizer. The invariant is structural.
    assert all("weight" not in name for name in AllocDecision.model_fields)


def test_embeds_no_inline_constraint_set():
    # Only a reference is carried — no field is typed as a full ConstraintSet.
    assert "constraint_set" not in AllocDecision.model_fields
    assert "constraint_set_ref" in AllocDecision.model_fields
