"""Optimisation-inputs questionnaire I/O (Fase 5, runtime step 0).

Three pure, frozen, serialisable pydantic-v2 shapes the profiler consumes and
produces — building any of them imports **no** ``optimizer`` / ``deepagents``
code (the deterministic answers→knob mapping lives in ``agents/profiler.py``, and
the LLM only ever emits typed answers, never a knob):

* ``MiFIDQuestion`` (+ the ``QUESTION_BANK`` constant) — the questions the form
  asks, one per row, tagged by the input section it belongs to.
* ``MiFIDAnswers`` — the four input sections as nested frozen models: knowledge &
  experience (``KnowledgeAnswers``), financial / loss-capacity
  (``CapacityAnswers``), investment objectives incl. the risk-tolerance Likert +
  horizon + reaction-to-loss (``ObjectivesAnswers``), and ESG (``EsgAnswers``).
* ``SuitabilityAssessment`` — the structured record of the saved inputs: the
  per-section answer snapshot, the two appetite scores (tolerance / capacity, kept
  disjoint), the binding ``a_gamma`` + its named ``RiskToleranceBand``, the derived
  ESG exclusions, ``inconsistency_flags`` and a free-text ``rationale``.

Validation matches SPEC §5: an out-of-range Likert response, a non-ISO currency,
and an unknown GICS sector are all rejected at construction. Everything is
``frozen=True`` (hashable) and JSON round-trips as a tested invariant.

The answer models also *close* the object (``extra="forbid"``) and require every
pillar, ESG included, with no defaults. A small tool-calling model extracts these
from free text, and an omitted ``esg`` pillar or a mis-keyed one (``excluded_sectors``
next to a valid ``exclusions``) must raise and trigger the retry rather than read as
"no exclusions" and silently drop a client's hard ESG block.

Because every datum is required, a client text that never mentions one (a cash
buffer, say) would otherwise force the extractor to invent it. ``MiFIDAnswers.
unanswered`` is the escape hatch: the extractor lists the datum it could not ground in
the text, the profiler turns each entry into an inconsistency flag shown at the HITL
gate, and the user sees that the value was assumed rather than stated in the text.
The ESG section is the exception: excluding sectors is optional, so an empty
exclusion list is a complete answer and no ESG datum is ever listed beside it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
    RiskToleranceBand,
)

# ``QUESTION_BANK`` and the ``UnansweredDatum`` alias are public module-level names
# (accessed via ``fund.schemas.questionnaire.<name>``); they are deliberately not in
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

# Every datum of ``MiFIDAnswers`` as a dotted path. Closed on purpose: a mis-spelled
# path must fail validation (and trigger the extraction retry) instead of becoming a
# flag the user cannot map back to a field.
UnansweredDatum = Literal[
    "base_currency",
    "knowledge.level",
    "capacity.max_1yr_loss_pct",
    "capacity.buffer_months",
    "objectives.goal",
    "objectives.horizon",
    "objectives.likert_items",
    "objectives.loss_reaction",
    "esg.exclusions",
    "esg.look_through_tolerance",
]

# Owed only when sectors are excluded: with nothing excluded there is no fund screen
# for the figure to loosen.
LOOK_THROUGH_TOLERANCE_PATH: UnansweredDatum = "esg.look_through_tolerance"
# The ESG filter is optional, so with no sector excluded neither ESG datum can be
# missing. Both stay valid entries: stored records may list them, and a sector list
# the extraction model marks as unstated must still reach the review gate.
_ESG_DATA: frozenset[str] = frozenset({"esg.exclusions", LOOK_THROUGH_TOLERANCE_PATH})

_Pillar = Literal["knowledge", "capacity", "objectives", "esg"]
_QuestionKind = Literal["single_choice", "multi_choice", "likert", "numeric"]


# The LLM-extracted answer shapes: frozen (hashable) and closed to unknown keys.
_ANSWERS_CONFIG = ConfigDict(frozen=True, extra="forbid")


class MiFIDQuestion(BaseModel):
    """One questionnaire item, tagged by the input section it belongs to.

    Presentation-only metadata — the mapping never reads a question, only the
    typed ``MiFIDAnswers`` the user enters. ``options`` lists the allowed
    responses for a choice question (empty for ``likert`` / ``numeric``).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    pillar: _Pillar
    text: str
    kind: _QuestionKind
    options: tuple[str, ...] = ()


class KnowledgeAnswers(BaseModel):
    """Input section 1 — knowledge & experience. Low ``level`` (``none`` / ``basic``)
    drives the ``UniverseFilters`` restrictions the fund toolsets enforce."""

    model_config = _ANSWERS_CONFIG

    level: KnowledgeLevel = Field(
        description=(
            "Investing knowledge & experience — exactly one of: none, basic, "
            "informed, advanced. This is NOT the user's risk tolerance."
        ),
    )


class CapacityAnswers(BaseModel):
    """Input section 2 — financial situation / loss capacity (attitude-independent).

    ``max_1yr_loss_pct`` is the largest one-year loss the user can absorb as a
    fraction of the invested capital; ``buffer_months`` is the emergency cash
    buffer (months of essential expenses) held **outside** this portfolio. Both
    feed ``_appetite_from_capacity`` (SPEC §8.2) — never fused with tolerance.
    """

    model_config = _ANSWERS_CONFIG

    max_1yr_loss_pct: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Largest tolerable one-year loss as a FRACTION of capital in [0, 1] "
            "(e.g. 0.4 means 40%)."
        ),
    )
    buffer_months: float = Field(
        ge=0.0,
        description=(
            "Emergency cash buffer in months of essential expenses, held OUTSIDE "
            "this portfolio."
        ),
    )


class ObjectivesAnswers(BaseModel):
    """Input section 3 — investment objectives, horizon, attitudinal risk tolerance.

    ``likert_items`` are the attitudinal risk-tolerance responses (the appetite
    driver); ``loss_reaction`` is the reaction to an extreme drawdown scenario
    (the ``risk_measure`` / ``beta`` driver) — the two are deliberately separate.
    """

    model_config = _ANSWERS_CONFIG

    goal: ObjectiveChoice = Field(
        description=(
            "Primary objective — exactly one of: protection, income, growth, max."
        ),
    )
    horizon: Horizon = Field(
        description=(
            "Investment horizon bucket. Map any stated duration to exactly one of "
            "'short' (< ~3 years), 'medium' (~3-7 years), or 'long' (> ~7 years). "
            "Return ONLY this enum value — never a number of years or a "
            "'horizon_years' field."
        ),
    )
    likert_items: tuple[LikertScore, ...] = Field(
        min_length=1,
        description=(
            "Attitudinal risk-tolerance responses on a 1-7 agreement scale "
            "(1=strongly disagree, 7=strongly agree); at least one integer."
        ),
    )
    loss_reaction: LossReaction = Field(
        description=(
            "Reaction to a severe drawdown — exactly one of: sell_all, sell_some, "
            "hold, buy_more."
        ),
    )


class EsgAnswers(BaseModel):
    """Input section 4 — ESG preferences. ``exclusions`` are GICS sectors the user
    wants excluded; the profiler turns them into a HARD ``EsgPolicy`` block.
    ``exclusions`` has no default: "no preference" is an explicit empty tuple, so an
    omitted answer cannot pass as one. ``other_preferences`` keeps the stated
    preferences no sector expresses so the user sees them.

    ``look_through_tolerance`` is how much of a *fund* may sit in an excluded sector.
    A fund has no single sector, so without a stated share the only fund screen is
    the strictest one (0.0); it defaults to that, which keeps records stored before
    the field existed valid."""

    model_config = _ANSWERS_CONFIG

    exclusions: tuple[GicsSector, ...] = Field(
        description=(
            "GICS sectors the user wants excluded on ESG grounds (a subset of "
            "the 11 sectors). Optional: always present, as an empty list when the "
            "user names none, a complete answer. With no sector excluded list "
            "neither 'esg.exclusions' nor 'esg.look_through_tolerance' in "
            "unanswered; list 'esg.exclusions' only when it holds a sector you "
            "cannot tell the user named."
        ),
    )
    other_preferences: tuple[str, ...] = Field(
        default=(),
        description=(
            "ESG preferences the user stated that are NOT a GICS sector (e.g. "
            "'no tobacco', 'no weapons'), verbatim. Never map one onto an arbitrary "
            "sector; they are shown to the user, not enforced."
        ),
    )
    look_through_tolerance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Largest share of a FUND's weight allowed in an excluded sector, as a "
            "FRACTION in [0, 1] (0.05 means 5%). 0 means none at all. Set it only "
            "from a figure the user stated. When sectors are excluded and the user "
            "stated no figure, leave 0 and list 'esg.look_through_tolerance' in "
            "unanswered; with no sector excluded it needs no entry."
        ),
    )


class MiFIDAnswers(BaseModel):
    """The user's optimisation inputs: the four input sections the limits derive from.

    ``base_currency`` is the user's reporting currency (ISO-4217 alpha).
    The ``portfolio_id`` is **not** carried here — it is a persistence concern
    supplied to ``build_constraint_set`` / ``run_profiler`` at mapping time.

    ``unanswered`` names the data the user's text did not state; the values filled in
    for them are placeholders, not the user's answers, and the profiler flags each at
    the HITL gate. It defaults to empty so a fully-answered form, the TUI form path and
    records persisted before the field existed all validate unchanged.
    """

    model_config = _ANSWERS_CONFIG

    base_currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description=(
            "The user's reporting currency as an ISO-4217 alpha code, e.g. EUR, USD."
        ),
    )
    knowledge: KnowledgeAnswers
    capacity: CapacityAnswers
    objectives: ObjectivesAnswers
    esg: EsgAnswers
    unanswered: tuple[UnansweredDatum, ...] = Field(
        default=(),
        description=(
            "Dotted paths of every datum the user did NOT state (e.g. "
            "'capacity.buffer_months'). Empty when the user answered everything."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _flag_unstated_look_through_tolerance(cls, data: Any) -> Any:
        """Mark the tolerance unanswered when sectors are excluded but no figure given.

        The typed TUI form and the CLI ``--answers`` path hand the profiler a finished
        ``MiFIDAnswers`` and skip the extraction model, so nothing else would say the
        strict 0.0 was assumed. An explicit 0 counts as stated: only an absent key
        (or, for a typed ``EsgAnswers``, a field left out of the constructor) is
        unstated. Anything this cannot read is passed on for the field validators to
        reject.
        """
        esg = _raw_esg_answer(data)
        unanswered = _raw_unanswered(data)
        if esg is None or unanswered is None:
            return data
        excluded, stated = esg
        if not excluded or stated or LOOK_THROUGH_TOLERANCE_PATH in unanswered:
            return data
        return {**data, "unanswered": (*unanswered, LOOK_THROUGH_TOLERANCE_PATH)}

    @model_validator(mode="before")
    @classmethod
    def _no_exclusions_answer_the_esg_section(cls, data: Any) -> Any:
        """Drop the ESG data from ``unanswered`` when no sector is excluded.

        The ESG filter is optional and an empty list is the user's answer, while the
        fund tolerance only matters once a sector is excluded. Left listed, either
        path would reach the review gate as an assumed value the user was never
        required to give.
        """
        esg = _raw_esg_answer(data)
        unanswered = _raw_unanswered(data)
        if esg is None or unanswered is None:
            return data
        excluded, _ = esg
        kept = tuple(path for path in unanswered if path not in _ESG_DATA)
        if excluded or len(kept) == len(unanswered):
            return data
        return {**data, "unanswered": kept}


def _raw_esg_answer(data: Any) -> tuple[bool, bool] | None:
    """``(sectors excluded, tolerance stated)`` read from raw ``MiFIDAnswers`` input.

    ``None`` when ``data`` is no mapping or its ESG section is neither a mapping nor an
    ``EsgAnswers``, so the field validators report it. For a typed ``EsgAnswers`` the
    tolerance counts as stated only when its constructor was given one.
    """
    if not isinstance(data, dict):
        return None
    esg = data.get("esg")
    if isinstance(esg, EsgAnswers):
        return bool(esg.exclusions), "look_through_tolerance" in esg.model_fields_set
    if isinstance(esg, dict):
        return bool(esg.get("exclusions")), "look_through_tolerance" in esg
    return None


def _raw_unanswered(data: Any) -> list[Any] | tuple[Any, ...] | None:
    """The raw ``unanswered`` entries of a mapping input.

    ``None`` when ``data`` is no mapping or the entries are no sequence, so the field
    validators report it.
    """
    if not isinstance(data, dict):
        return None
    unanswered = data.get("unanswered", ())
    return unanswered if isinstance(unanswered, list | tuple) else None


class SuitabilityAssessment(BaseModel):
    """The structured record of a portfolio's saved optimisation inputs.

    Retains the per-section ``answers`` snapshot, the two disjoint appetite scores
    (``appetite_from_tolerance`` / ``appetite_from_capacity`` ∈ [0, 1]), the
    binding ``a_gamma`` and its named ``band``, the derived ESG ``esg_exclusions``
    (the HARD block), any ``inconsistency_flags`` surfaced at the HITL gate, and a
    free-text ``rationale``.
    """

    model_config = ConfigDict(frozen=True)

    answers: MiFIDAnswers
    appetite_from_tolerance: float = Field(ge=0.0, le=1.0)
    appetite_from_capacity: float = Field(ge=0.0, le=1.0)
    a_gamma: float = Field(gt=0.0)  # binding min(tolerance, capacity) aversion
    band: RiskToleranceBand
    esg_exclusions: tuple[GicsSector, ...] = ()  # derived HARD block
    inconsistency_flags: tuple[str, ...] = ()  # surfaced at the HITL gate
    rationale: str = ""


# ---------------------------------------------------------------------------
# The default questionnaire — one entry per pillar sub-datum. Presentation only;
# the user records the typed ``MiFIDAnswers`` the deterministic mapping reads.
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
    MiFIDQuestion(
        id="esg_2",
        pillar="esg",
        text=(
            "What share of a fund may sit in an excluded sector, as a percentage "
            "of the fund?"
        ),
        kind="numeric",
    ),
)
