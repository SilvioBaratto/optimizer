"""Allocator's committed structured decision (Task 5) — never weights.

``AllocDecision`` is what the allocator agent emits: the filtered ``universe``,
a *reference* to the persisted risk profile (``constraint_set_ref`` = portfolio
id + store key, resolved in Fase 7 — never an inline copy, so a decision cannot
drift from the profile) and an *embedded* per-run ``ViewSet`` (Q1), plus the
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

    The risk profile is persisted once per portfolio (Fase 5) and looked up by
    ``store_key`` at decision time (Fase 7). Referencing — not copying — keeps a
    decision from drifting away from the persisted profile.
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
    universe: tuple[str, ...]  # filtered ticker set
    constraint_set_ref: ConstraintSetRef  # reference to the persisted profile
    objective: ObjectiveChoice
    rationale: str
    views: ViewSet | None = None  # embedded per-run views (None → none active)
