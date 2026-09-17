"""MiFID II suitability questionnaire I/O (Fase 5, runtime step 0).

Three pure, frozen, serialisable pydantic-v2 shapes the profiler consumes and
produces — building any of them imports **no** ``optimizer`` / ``deepagents``
code (the deterministic MiFID→knob mapping lives in ``agents/profiler.py``, and
the LLM only ever emits typed answers, never a knob):

* ``MiFIDQuestion`` (+ the ``QUESTION_BANK`` constant) — the questions the adviser
  asks, one per row, tagged by ESMA pillar.
* ``MiFIDAnswers`` — the four ESMA pillars as nested frozen models: knowledge &
  experience (``KnowledgeAnswers``), financial / loss-capacity
  (``CapacityAnswers``), investment objectives incl. the risk-tolerance Likert +
  horizon + reaction-to-loss (``ObjectivesAnswers``), and ESG (``EsgAnswers``).
* ``SuitabilityAssessment`` — the structured MiFID record: the per-pillar answer
  snapshot, the two appetite scores (tolerance / capacity, kept disjoint), the
  binding ``a_gamma`` + its named ``RiskToleranceBand``, the derived ESG
  exclusions, ``inconsistency_flags`` and a free-text ``rationale``.

Validation matches SPEC §5: an out-of-range Likert response, a non-ISO currency,
and an unknown GICS sector are all rejected at construction. Everything is
``frozen=True`` (hashable) and JSON round-trips as a tested invariant.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
    RiskToleranceBand,
)

# ``QUESTION_BANK`` is a public module-level constant (accessed via
# ``fund.schemas.questionnaire.QUESTION_BANK``); it is deliberately not in
# ``__all__`` — the re-export surface lists the schema *types* only.
__all__ = [
    "CapacityAnswers",
    "EsgAnswers",
    "KnowledgeAnswers",
    "MiFIDAnswers",
    "MiFIDQuestion",
    "ObjectivesAnswers",
    "SuitabilityAssessment",
]

# A single Likert response on the 7-point agreement scale. SPEC §8.2 scores the
# tolerance appetite as ``mean(likert_items)`` rescaled ``(x - 1) / 6``, so the
# valid range is [1, 7] (1 → appetite 0.0, 7 → appetite 1.0).
LikertScore = Annotated[int, Field(ge=1, le=7)]

_Pillar = Literal["knowledge", "capacity", "objectives", "esg"]
_QuestionKind = Literal["single_choice", "multi_choice", "likert", "numeric"]


class MiFIDQuestion(BaseModel):
    """One questionnaire item, tagged by the ESMA pillar it assesses.

    Presentation-only metadata — the mapping never reads a question, only the
    typed ``MiFIDAnswers`` the adviser records. ``options`` lists the allowed
    responses for a choice question (empty for ``likert`` / ``numeric``).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    pillar: _Pillar
    text: str
    kind: _QuestionKind
    options: tuple[str, ...] = ()


class KnowledgeAnswers(BaseModel):
    """Pillar 1 — knowledge & experience. Low ``level`` (``none`` / ``basic``)
    drives the ``UniverseFilters`` restrictions in Fase 7."""

    model_config = ConfigDict(frozen=True)

    level: KnowledgeLevel


class CapacityAnswers(BaseModel):
    """Pillar 2 — financial situation / loss capacity (attitude-independent).

    ``max_1yr_loss_pct`` is the largest one-year loss the client can absorb as a
    fraction of the invested capital; ``buffer_months`` is the emergency cash
    buffer (months of essential expenses) held **outside** this portfolio. Both
    feed ``_appetite_from_capacity`` (SPEC §8.2) — never fused with tolerance.
    """

    model_config = ConfigDict(frozen=True)

    max_1yr_loss_pct: float = Field(ge=0.0, le=1.0)  # fraction of capital
    buffer_months: float = Field(ge=0.0)  # months of expenses covered


class ObjectivesAnswers(BaseModel):
    """Pillar 3 — investment objectives, horizon, and attitudinal risk tolerance.

    ``likert_items`` are the attitudinal risk-tolerance responses (the appetite
    driver); ``loss_reaction`` is the reaction to an extreme drawdown scenario
    (the ``risk_measure`` / ``beta`` driver) — the two are deliberately separate.
    """

    model_config = ConfigDict(frozen=True)

    goal: ObjectiveChoice
    horizon: Horizon
    likert_items: tuple[LikertScore, ...] = Field(min_length=1)
    loss_reaction: LossReaction


class EsgAnswers(BaseModel):
    """Pillar 4 — ESG preferences. ``exclusions`` are GICS sectors the client
    refuses to hold; the profiler turns them into a HARD ``EsgPolicy`` block
    (D16, exclusions-only now). Defaults to no stated preference."""

    model_config = ConfigDict(frozen=True)

    exclusions: tuple[GicsSector, ...] = ()


class MiFIDAnswers(BaseModel):
    """The four ESMA pillars a suitability assessment consumes.

    ``base_currency`` is the client's reporting currency (ISO-4217 alpha, D8).
    The ``portfolio_id`` is **not** carried here — it is a persistence concern
    supplied to ``build_constraint_set`` / ``run_profiler`` at mapping time.
    """

    model_config = ConfigDict(frozen=True)

    base_currency: str = Field(pattern=r"^[A-Z]{3}$")  # ISO-4217 alpha (D8)
    knowledge: KnowledgeAnswers
    capacity: CapacityAnswers
    objectives: ObjectivesAnswers
    esg: EsgAnswers = EsgAnswers()


class SuitabilityAssessment(BaseModel):
    """The structured MiFID suitability record (no rendered document this phase).

    Retains the per-pillar ``answers`` snapshot, the two disjoint appetite scores
    (``appetite_from_tolerance`` / ``appetite_from_capacity`` ∈ [0, 1]), the
    binding ``a_gamma`` and its named ``band``, the derived ESG ``esg_exclusions``
    (the HARD block), any ``inconsistency_flags`` surfaced at the HITL gate, and a
    free-text ``rationale``.
    """

    model_config = ConfigDict(frozen=True)

    answers: MiFIDAnswers  # per-pillar inputs snapshot
    appetite_from_tolerance: float = Field(ge=0.0, le=1.0)
    appetite_from_capacity: float = Field(ge=0.0, le=1.0)
    a_gamma: float = Field(gt=0.0)  # binding min(tolerance, capacity) aversion
    band: RiskToleranceBand  # recorded named MiFID category
    esg_exclusions: tuple[GicsSector, ...] = ()  # derived HARD block
    inconsistency_flags: tuple[str, ...] = ()  # surfaced at the HITL gate
    rationale: str = ""


# ---------------------------------------------------------------------------
# The default questionnaire — one entry per pillar sub-datum. Presentation only;
# the adviser records the typed ``MiFIDAnswers`` the deterministic mapping reads.
# ---------------------------------------------------------------------------
QUESTION_BANK: tuple[MiFIDQuestion, ...] = (
    MiFIDQuestion(
        id="ke_1",
        pillar="knowledge",
        text="How would you describe your knowledge and experience with investing?",
        kind="single_choice",
        options=tuple(level.value for level in KnowledgeLevel),
    ),
    MiFIDQuestion(
        id="cap_1",
        pillar="capacity",
        text=(
            "What is the largest one-year loss you could absorb without changing "
            "your lifestyle, as a percentage of the invested capital?"
        ),
        kind="numeric",
    ),
    MiFIDQuestion(
        id="cap_2",
        pillar="capacity",
        text=(
            "How many months of essential expenses does your cash buffer cover "
            "outside this portfolio?"
        ),
        kind="numeric",
    ),
    MiFIDQuestion(
        id="obj_1",
        pillar="objectives",
        text="What is the primary objective for this portfolio?",
        kind="single_choice",
        options=tuple(goal.value for goal in ObjectiveChoice),
    ),
    MiFIDQuestion(
        id="obj_2",
        pillar="objectives",
        text="Over what horizon do you plan to stay invested?",
        kind="single_choice",
        options=tuple(horizon.value for horizon in Horizon),
    ),
    MiFIDQuestion(
        id="obj_3",
        pillar="objectives",
        text=(
            "I am comfortable accepting short-term losses in exchange for higher "
            "long-term returns."
        ),
        kind="likert",
    ),
    MiFIDQuestion(
        id="obj_4",
        pillar="objectives",
        text="If this portfolio fell 20% in a month, what would you most likely do?",
        kind="single_choice",
        options=tuple(reaction.value for reaction in LossReaction),
    ),
    MiFIDQuestion(
        id="esg_1",
        pillar="esg",
        text="Which economic sectors, if any, do you wish to exclude on ESG grounds?",
        kind="multi_choice",
        options=tuple(sector.value for sector in GicsSector),
    ),
)
