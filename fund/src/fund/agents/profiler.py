"""MiFID II suitability profiler — the deterministic answers → knobs mapping.

Runtime *step 0*: turn a validated ``MiFIDAnswers`` (the four ESMA pillars) into a
``ConstraintSet`` (the risk profile every later agent reads) plus a structured
``SuitabilityAssessment`` for MiFID record-keeping. This module owns the **pure,
total, auditable** core — the deterministic mapping, the ESG hard-gate, the K&E
universe filters, and the inconsistency / anti-overconfidence check. The LLM
profiler agent + HITL persistence land in a later Task-7 slice.

Building a ``ConstraintSet`` imports **no** ``optimizer`` code — the mapping is
plain dict/arithmetic; the optimizer only appears when the caller feeds the result
through ``ConstraintSet.to_mean_risk_config()``.

Correctness-critical (SPEC §8, deep_agent.md ``01:142`` / ``30:41``):

* **appetite vs aversion direction** — ``A = min(tolerance, capacity)`` is on
  risk-**appetite** ``A ∈ [0, 1]`` (higher = *can take more risk*); the 5-band
  lookup maps it **monotonically decreasing** to ``a_gamma`` (low appetite → high
  aversion). A naive ``a_gamma = min(gamma_tol, gamma_cap)`` would be backwards.
* **tolerance and capacity are scored from disjoint answer fields** and never fused
  (regulatory double-binding): attitudinal Likert drives tolerance; financial
  loss-capacity / buffer drive capacity.

All maps are **total** plain-dict lookups (an unmapped enum member is a hard
``KeyError``, never a silent default), matching the ``fund/schemas`` ethos.
"""

from __future__ import annotations

from fund.schemas.constraint_set import ConstraintSet, EsgPolicy, UniverseFilters
from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
    RiskMeasureChoice,
    RiskToleranceBand,
)
from fund.schemas.questionnaire import (
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    ObjectivesAnswers,
    SuitabilityAssessment,
)

__all__ = [
    "FLAG_OBJECTIVE_CAPACITY_MISMATCH",
    "FLAG_OVERCONFIDENCE",
    "FLAG_REACTION_TOLERANCE_MISMATCH",
    "SuitabilityBreachError",
    "assess_suitability",
    "build_constraint_set",
    "run_mapping",
]

# Named inconsistency-flag vocabulary the suitability check can emit. These are
# surfaced (never auto-clamped) at the HITL gate in Task 7 (SPEC §8.3).
FLAG_OBJECTIVE_CAPACITY_MISMATCH = "objective_capacity_mismatch"
FLAG_OVERCONFIDENCE = "overconfidence"
FLAG_REACTION_TOLERANCE_MISMATCH = "reaction_tolerance_mismatch"


class SuitabilityBreachError(ValueError):
    """A HARD MiFID breach that cannot proceed to a portfolio (SPEC §8.3).

    Distinct from a soft ``inconsistency_flags`` entry (flagged and surfaced at the
    HITL gate): a breach hard-blocks outright. Raised when the ESG pillar excludes
    every GICS sector, leaving no investable universe. Subclasses ``ValueError`` so
    a caller may catch it broadly or specifically.
    """


# ---------------------------------------------------------------------------
# Resolved lookup tables (SPEC §8 — now contract, not re-litigated here).
# ---------------------------------------------------------------------------

# SPEC §8.1 — appetite buckets into 5 named MiFID categories, each a fixed
# ``a_gamma``. Ordered ascending by the band's exclusive upper bound; the final
# ``1.01`` sentinel captures the closed ``[0.8, 1.0]`` Aggressive band.
_AVERSION_BANDS: tuple[tuple[float, float], ...] = (
    (0.2, 12.0),  # [0.0, 0.2)  Defensive
    (0.4, 8.0),  # [0.2, 0.4)  Conservative
    (0.6, 5.0),  # [0.4, 0.6)  Balanced
    (0.8, 2.5),  # [0.6, 0.8)  Growth
    (1.01, 1.0),  # [0.8, 1.0]  Aggressive
)

# The recorded MiFID category for each band's ``a_gamma`` (total over the five
# values ``_appetite_to_aversion`` can emit).
_BAND_BY_AVERSION: dict[float, RiskToleranceBand] = {
    12.0: RiskToleranceBand.DEFENSIVE,
    8.0: RiskToleranceBand.CONSERVATIVE,
    5.0: RiskToleranceBand.BALANCED,
    2.5: RiskToleranceBand.GROWTH,
    1.0: RiskToleranceBand.AGGRESSIVE,
}

# Objective spectrum (SPEC §1) — the client-stated goal *is* the objective knob;
# an explicit total identity map keeps the "unmapped ⇒ KeyError" guarantee and a
# single place to diverge later.
_OBJECTIVE_BY_GOAL: dict[ObjectiveChoice, ObjectiveChoice] = {
    ObjectiveChoice.PROTECTION: ObjectiveChoice.PROTECTION,
    ObjectiveChoice.INCOME: ObjectiveChoice.INCOME,
    ObjectiveChoice.GROWTH: ObjectiveChoice.GROWTH,
    ObjectiveChoice.MAX: ObjectiveChoice.MAX,
}

# Horizon passthrough (SPEC §1) — same shape, kept explicit + total.
_HORIZON_BY_BUCKET: dict[Horizon, Horizon] = {
    Horizon.SHORT: Horizon.SHORT,
    Horizon.MEDIUM: Horizon.MEDIUM,
    Horizon.LONG: Horizon.LONG,
}

# Reaction to an extreme drawdown → downside risk-measure (SPEC §1,
# "protection → CVaR/CDaR/MaxDD"). Ordered most-protective (capitulates) to
# most-comfortable (buys the dip); each reaction picks a distinct measure so the
# panic-seller gets peak-to-trough (MaxDD) control and the contrarian rides the
# symmetric variance base (D34).
_MEASURE_BY_REACTION: dict[LossReaction, RiskMeasureChoice] = {
    LossReaction.SELL_ALL: RiskMeasureChoice.MAX_DRAWDOWN,
    LossReaction.SELL_SOME: RiskMeasureChoice.CDAR,
    LossReaction.HOLD: RiskMeasureChoice.CVAR,
    LossReaction.BUY_MORE: RiskMeasureChoice.VARIANCE,
}

# Tail confidence for the CVaR/CDaR measures (SPEC §1, "beta da tolleranza"): the
# more protective the reaction, the deeper the tail the profile controls. All in
# the open interval (0, 1) ``ConstraintSet.beta`` requires.
_BETA_BY_REACTION: dict[LossReaction, float] = {
    LossReaction.SELL_ALL: 0.99,
    LossReaction.SELL_SOME: 0.975,
    LossReaction.HOLD: 0.95,
    LossReaction.BUY_MORE: 0.90,
}


# ---------------------------------------------------------------------------
# Appetite scoring (SPEC §8.2 — normalized-average composite, all linear).
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    """Clamp to the unit interval ``[0, 1]``."""
    return min(max(x, 0.0), 1.0)


def _appetite_from_tolerance(objectives: ObjectivesAnswers) -> float:
    """Attitudinal risk appetite ∈ [0, 1]: mean Likert rescaled ``(x - 1) / 6``.

    Reads **only** the Likert items — the attitude pillar, kept disjoint from the
    financial capacity score.
    """
    mean_likert = sum(objectives.likert_items) / len(objectives.likert_items)
    return (mean_likert - 1.0) / 6.0


def _appetite_from_capacity(capacity: CapacityAnswers) -> float:
    """Financial risk appetite ∈ [0, 1]: ``mean(loss_sub, buffer_sub)``.

    ``loss_sub = clip(max_1yr_loss_pct / 0.50)`` and
    ``buffer_sub = clip(buffer_months / 12)`` — reads **only** the financial
    fields, never the attitudinal Likert.
    """
    loss_sub = _clip01(capacity.max_1yr_loss_pct / 0.50)
    buffer_sub = _clip01(capacity.buffer_months / 12.0)
    return (loss_sub + buffer_sub) / 2.0


def _appetite_to_aversion(appetite: float) -> float:
    """5-band MiFID lookup → ``a_gamma`` (monotone-DECREASING; SPEC §8.1)."""
    for upper, a_gamma in _AVERSION_BANDS:
        if appetite < upper:
            return a_gamma
    return 1.0  # appetite clamped ≤ 1.0 upstream; total fallthrough guard


def _category(a_gamma: float) -> RiskToleranceBand:
    """Recorded MiFID category for a band ``a_gamma`` (unmapped ⇒ ``KeyError``)."""
    return _BAND_BY_AVERSION[a_gamma]


def _nu_tiers(capacity: CapacityAnswers) -> tuple[float, float, float]:
    """Escalating drawdown-ceiling tiers as fractions of capital (deep_agent ``20:75``).

    Anchored on the stated one-year loss tolerance: a soft warning at half the
    tolerance, the hard ceiling at the tolerance itself, and an absolute stop at
    1.5x (clamped to 1.0). Emitted here, enforced by the Fase-7 ``risk_check``.
    """
    ceiling = capacity.max_1yr_loss_pct
    nu1 = ceiling * 0.5
    nu2 = ceiling
    nu3 = min(ceiling * 1.5, 1.0)
    return (nu1, nu2, nu3)


# ---------------------------------------------------------------------------
# ESG hard gate (D9/D16) + K&E universe filters (D32).
# ---------------------------------------------------------------------------


def _esg_policy(esg: EsgAnswers) -> EsgPolicy:
    """Client ESG exclusions → HARD ``EsgPolicy`` block (D9/D16).

    Copies the declared GICS-sector exclusions verbatim (deduplicated,
    order-preserving); nothing else feeds this, so an excluded sector can never be
    re-admitted by another answer. Excluding *every* sector leaves no investable
    universe — a legal breach that raises ``SuitabilityBreachError`` rather than
    silently emptying the mandate.
    """
    exclusions = tuple(dict.fromkeys(esg.exclusions))
    if set(exclusions) >= set(GicsSector):
        raise SuitabilityBreachError(
            "ESG exclusions remove every GICS sector — no investable universe."
        )
    return EsgPolicy(exclusions=exclusions)


# Low K&E (none/basic) → complex/leverage banned + a tightened per-position cap;
# informed/advanced stay unrestricted. Total over ``KnowledgeLevel`` (an unmapped
# member is a hard ``KeyError``, matching the rest of the mapping).
_FILTERS_BY_KNOWLEDGE: dict[KnowledgeLevel, UniverseFilters] = {
    KnowledgeLevel.NONE: UniverseFilters(
        no_complex=True, no_leverage=True, max_position_cap=0.05
    ),
    KnowledgeLevel.BASIC: UniverseFilters(
        no_complex=True, no_leverage=True, max_position_cap=0.10
    ),
    KnowledgeLevel.INFORMED: UniverseFilters(),
    KnowledgeLevel.ADVANCED: UniverseFilters(),
}


def _ke_filters(knowledge: KnowledgeAnswers) -> UniverseFilters:
    """K&E level → ``UniverseFilters`` restrictions (unmapped ⇒ ``KeyError``)."""
    return _FILTERS_BY_KNOWLEDGE[knowledge.level]


# ---------------------------------------------------------------------------
# Inconsistency / anti-overconfidence check (SPEC §8.3 — flag, never clamp).
# ---------------------------------------------------------------------------

# Appetite thresholds for the soft contradiction rules. All expressed on the
# appetite scale ∈ [0, 1] so they read against the same bands as ``a_gamma``.
_LOW_CAPACITY_APPETITE = 0.4  # below the Balanced floor ⇒ conservative capacity
_OVERCONFIDENCE_GAP = 0.4  # attitude far outstripping financial capacity
_HIGH_TOLERANCE_APPETITE = 0.6  # comfortable-with-loss attitudinal appetite


def _inconsistency_flags(
    answers: MiFIDAnswers, a_tol: float, a_cap: float
) -> tuple[str, ...]:
    """Flag contradictory answers (SPEC §8.3, deep_agent ``01:338``) — no clamp.

    Pure over the already-scored appetites plus the raw objective / reaction
    answers. Each rule is independent and additive; the flags are surfaced to the
    adviser at the HITL gate — the binding ``a_gamma`` is left untouched.
    """
    flags: list[str] = []
    objectives = answers.objectives
    if (
        objectives.goal in (ObjectiveChoice.GROWTH, ObjectiveChoice.MAX)
        and a_cap < _LOW_CAPACITY_APPETITE
    ):
        flags.append(FLAG_OBJECTIVE_CAPACITY_MISMATCH)
    if a_tol - a_cap >= _OVERCONFIDENCE_GAP:
        flags.append(FLAG_OVERCONFIDENCE)
    if (
        a_tol >= _HIGH_TOLERANCE_APPETITE
        and objectives.loss_reaction is LossReaction.SELL_ALL
    ):
        flags.append(FLAG_REACTION_TOLERANCE_MISMATCH)
    return tuple(flags)


# ---------------------------------------------------------------------------
# The deterministic mapping (pure, total, imports no optimizer code).
# ---------------------------------------------------------------------------


def build_constraint_set(
    answers: MiFIDAnswers,
    *,
    portfolio_id: str,
    base_currency: str | None = None,
) -> ConstraintSet:
    """Map validated MiFID answers → ``ConstraintSet``.

    ``portfolio_id`` is a persistence concern not carried on ``MiFIDAnswers``, so it
    is threaded in here. ``base_currency`` defaults to ``answers.base_currency``
    (the client's reporting currency validated at questionnaire time); an explicit
    override is accepted for callers that reconcile against a portfolio's currency.

    HARD rules (SPEC §8, tested):

    * ``A = min(appetite_from_tolerance, appetite_from_capacity)`` → ``a_gamma`` via
      the inverting 5-band lookup;
    * tolerance (attitude) and capacity (finance) scored from **disjoint** fields.

    The ESG pillar becomes a HARD ``EsgPolicy`` block (declared exclusions,
    unoverridable; every-sector exclusion raises ``SuitabilityBreachError``) and
    low K&E tightens ``UniverseFilters``. The ``SuitabilityAssessment`` record is
    built by ``assess_suitability`` / ``run_mapping``.
    """
    a_tol = _appetite_from_tolerance(answers.objectives)
    a_cap = _appetite_from_capacity(answers.capacity)
    appetite = min(a_tol, a_cap)  # regulatory double-binding on appetite

    reaction = answers.objectives.loss_reaction
    nu1, nu2, nu3 = _nu_tiers(answers.capacity)

    return ConstraintSet(
        portfolio_id=portfolio_id,
        base_currency=(
            base_currency if base_currency is not None else answers.base_currency
        ),
        a_gamma=_appetite_to_aversion(appetite),  # monotone-DECREASING map
        objective=_OBJECTIVE_BY_GOAL[answers.objectives.goal],
        risk_measure=_MEASURE_BY_REACTION[reaction],
        beta=_BETA_BY_REACTION[reaction],
        nu1=nu1,
        nu2=nu2,
        nu3=nu3,
        horizon=_HORIZON_BY_BUCKET[answers.objectives.horizon],
        esg=_esg_policy(answers.esg),  # HARD gate (D9/D16); may hard-block
        universe_filters=_ke_filters(answers.knowledge),  # low K&E ⇒ restricted
    )


def assess_suitability(
    answers: MiFIDAnswers, constraint_set: ConstraintSet
) -> SuitabilityAssessment:
    """Assemble the structured MiFID suitability record for a mapped profile.

    Pure and LLM-free: recomputes the two disjoint appetite scores, records the
    binding ``a_gamma`` and its named band, mirrors the ESG block, and runs the
    inconsistency check. ``rationale`` is a deterministic, human-readable trail of
    how the profile was derived (what MiFID record-keeping retains).
    """
    a_tol = _appetite_from_tolerance(answers.objectives)
    a_cap = _appetite_from_capacity(answers.capacity)
    band = _category(constraint_set.a_gamma)
    binding = "capacity" if a_cap <= a_tol else "tolerance"
    rationale = (
        f"Risk appetite {min(a_tol, a_cap):.2f} "
        f"(tolerance {a_tol:.2f}, capacity {a_cap:.2f}); {binding} binds; "
        f"band {band.value} → a_gamma {constraint_set.a_gamma:g}."
    )
    return SuitabilityAssessment(
        answers=answers,
        appetite_from_tolerance=a_tol,
        appetite_from_capacity=a_cap,
        a_gamma=constraint_set.a_gamma,
        band=band,
        esg_exclusions=constraint_set.esg.exclusions,
        inconsistency_flags=_inconsistency_flags(answers, a_tol, a_cap),
        rationale=rationale,
    )


def run_mapping(
    answers: MiFIDAnswers,
    *,
    portfolio_id: str,
    base_currency: str | None = None,
) -> tuple[ConstraintSet, SuitabilityAssessment]:
    """Pure Task-3 wrapper: ``answers -> (ConstraintSet, SuitabilityAssessment)``.

    The LLM profiler agent (Task 7) calls this after normalising free-text into a
    typed ``MiFIDAnswers``; it runs no LLM and touches no DB. An ESG/legal breach
    hard-blocks here via ``build_constraint_set``.
    """
    constraint_set = build_constraint_set(
        answers, portfolio_id=portfolio_id, base_currency=base_currency
    )
    return constraint_set, assess_suitability(answers, constraint_set)
