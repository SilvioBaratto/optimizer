"""Allocator's committed structured decision — never weights.

``AllocDecision`` is what the allocator agent emits: the filtered ``universe``,
a *reference* to the persisted risk profile (``constraint_set_ref`` = portfolio
id + store key, resolved at decision time — never an inline copy, so a decision
cannot drift from the profile) and an *embedded* per-run ``ViewSet``, plus the
chosen ``objective`` and a free-text ``rationale``.

It deliberately carries **no weight field**: the no-weights invariant (weights
are computed downstream by the optimizer, never chosen by the LLM) is enforced
structurally, not by convention. Pure serialisable, frozen (hashable)
pydantic-v2 data — constructing one imports **no** ``optimizer`` / ``deepagents``
/ ``app`` code (the embedded ``ViewSet`` keeps its optimizer mapping method-local).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from fund.schemas.enums import ObjectiveChoice
from fund.schemas.views import ViewSet

__all__ = ["AllocDecision", "ConstraintSetRef"]


class ConstraintSetRef(BaseModel):
    """Reference to a persisted ``ConstraintSet`` (portfolio id + store key).

    Referencing — not copying — keeps a decision from drifting away from the
    persisted profile.
    """

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    store_key: str


class AllocDecision(BaseModel):
    """The allocator's committed structured inputs — never weights.

    ``universe`` is the post-filter ticker set; ``constraint_set_ref`` references
    the persisted profile; ``views`` embeds the per-run ``ViewSet`` (``None`` when
    no views are active). No ``weight``/``weights`` field exists — the
    load-bearing invariant is structural, not a convention.
    """

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    universe: tuple[str, ...]
    constraint_set_ref: ConstraintSetRef
    objective: ObjectiveChoice
    rationale: str
    views: ViewSet | None = None
