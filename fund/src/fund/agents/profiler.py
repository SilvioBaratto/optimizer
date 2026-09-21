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

import datetime as dt
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fund.agents.prompts import PROFILER_SYSTEM_PROMPT
from fund.config import FundConfig, settings
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

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from sqlalchemy.orm import Session

__all__ = [
    "FLAG_OBJECTIVE_CAPACITY_MISMATCH",
    "FLAG_OVERCONFIDENCE",
    "FLAG_REACTION_TOLERANCE_MISMATCH",
    "ProfilerRun",
    "SuitabilityBreachError",
    "assess_suitability",
    "build_constraint_set",
    "build_profiler_agent",
    "run_mapping",
    "run_profiler",
]

# The single HITL-gated persistence tool the profiler agent exposes (SPEC §8.5).
_SAVE_PROFILE_TOOL = "save_profile"

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


# ---------------------------------------------------------------------------
# Task 7 — the LLM profiler agent + always-on HITL persistence (SPEC §8.5).
#
# The LLM *interprets* free-text answers into a typed ``MiFIDAnswers`` (via the
# Fase-4 ``structured_call`` helper) and *decides* to persist; the deterministic
# mapping above computes every knob. Persistence is a single ``save_profile`` tool
# gated behind ``interrupt_on`` + a checkpointer, so the profiler *always* pauses
# for adviser sign-off before writing. Heavy deps (deepagents / langchain /
# langgraph / the audit repos) are imported lazily so the pure mapping above stays
# cheap to import (test_profiler_mapping imports it without the agent stack).
# ---------------------------------------------------------------------------


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string).

    ``ConstraintSet.portfolio_id`` is a free ``str`` but ``mifid_profiles`` /
    ``agent_runs`` key on ``UUID`` — so a run needs a real UUID (or its canonical
    string), and the ``str`` form is threaded to the mapping / Store.
    """
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _hash(payload: str) -> str:
    """sha256 hex of a payload — the audit trail may keep the hash, not the text."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_messages(questionnaire: Any) -> Any:
    """Coerce a bare questionnaire string into a one-message list; pass lists as-is."""
    if isinstance(questionnaire, str):
        return [{"role": "user", "content": questionnaire}]
    return questionnaire


def _prompt_text(questionnaire: Any) -> str:
    """A stable string rendering of the questionnaire input for the audit trail."""
    if isinstance(questionnaire, str):
        return questionnaire
    return json.dumps(questionnaire, default=str, sort_keys=True)


def _persist_instruction(portfolio_id: str) -> str:
    """The human turn that drives the agent to call ``save_profile`` once."""
    return (
        f"The suitability assessment for portfolio {portfolio_id} is complete and "
        f"validated. Call save_profile with this portfolio id to persist it."
    )


def _interrupt_description(
    portfolio_id: str, suitability: SuitabilityAssessment
) -> str:
    """The HITL pause message — surfaces the band + any inconsistency flags."""
    flags = ", ".join(suitability.inconsistency_flags) or "none"
    return (
        f"Persist MiFID suitability profile for portfolio {portfolio_id}. "
        f"Risk band {suitability.band.value}; a_gamma {suitability.a_gamma:g}; "
        f"inconsistency flags: {flags}. "
        f"Approve to write the profile, or reject to discard."
    )


def _extract_interrupt(result: Any) -> dict[str, Any] | None:
    """Pull the HITL interrupt payload from a compiled-graph invoke result."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    value = interrupts[0].value
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value)


@dataclass(frozen=True)
class ProfilerRun:
    """Handle for one profiling run paused at the adviser-confirmation gate.

    Carries the interpreted ``answers``, the deterministically-mapped
    ``constraint_set`` + ``suitability`` (with any ``inconsistency_flags``), the
    ``interrupt`` payload the adviser reviews, and enough state to resume:
    ``resume("approve")`` persists, ``resume("reject")`` persists nothing.
    """

    answers: MiFIDAnswers
    constraint_set: ConstraintSet
    suitability: SuitabilityAssessment
    run_id: uuid.UUID
    interrupt: dict[str, Any] | None
    agent: Any = field(repr=False, compare=False)
    thread_config: dict[str, Any] = field(repr=False, compare=False)
    session: Any = field(repr=False, compare=False)

    def resume(self, decision: str) -> dict[str, Any]:
        """Resume the paused agent with an adviser ``decision`` (approve / reject).

        ``approve`` runs the gated ``save_profile`` tool (the write happens inside
        it). Any other decision writes nothing but records the HITL choice in the
        audit trail and finalises the run as rejected.
        """
        from langgraph.types import Command

        from fund.audit import AgentRunRepository

        result: dict[str, Any] = self.agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=self.thread_config,
        )
        if decision != "approve":
            audit = AgentRunRepository(self.session)
            audit.append_decision(
                self.run_id,
                agent="profiler",
                step=_SAVE_PROFILE_TOOL,
                hitl_decision={"decision": decision},
            )
            audit.finalize_run(self.run_id, weights={}, status="rejected")
        return result


def _make_save_profile(
    *,
    session: Session,
    store: Any | None,
    config: FundConfig,
    answers: MiFIDAnswers,
    constraint_set: ConstraintSet,
    suitability: SuitabilityAssessment,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
) -> BaseTool:
    """Build the closure ``save_profile`` tool for one run.

    The write is driven entirely by the captured, deterministically-mapped values
    — the LLM-supplied ``portfolio_id`` argument is advisory only, so no knob can
    enter through a tool argument (the load-bearing rule). Idempotent per the HITL
    re-run contract (SPEC D3): a second call for an already-profiled portfolio
    never writes a new version (re-profiling is deferred to Phase 9, D11), but it
    still finalises the (otherwise orphaned) run and records the idempotent
    approve so the audit trail reflects what actually applied. A same-run replay
    (already finalised) is a true no-op — no double-log.
    """
    from langchain_core.tools import tool

    from fund.audit import (
        AgentRunRepository,
        MifidProfileRepository,
        put_constraint_set,
    )

    store_key = config.constraint_set_store_key
    portfolio_id_uuid = portfolio_id  # trusted UUID bound into the closure

    @tool
    def save_profile(portfolio_id: str) -> str:
        """Persist the approved MiFID profile: append the ``mifid_profiles`` row
        and cache the active ``ConstraintSet`` in the Store. Call once, after
        adviser approval, with the portfolio id."""
        repo = MifidProfileRepository(session)
        existing = repo.get_active(portfolio_id_uuid)
        if existing is not None:
            # Re-profiling to a new version is deferred (D11, Phase 9). A fresh
            # run that hits this branch (a manual re-profile) is otherwise
            # orphaned at "pending"; finalise it + record the idempotent approve
            # so the audit trail reflects what actually applied. A same-run
            # replay (already finalised) is left untouched — no double-log
            # (SPEC-D3 precedent).
            audit = AgentRunRepository(session)
            run = audit.get_run(run_id)
            if run is not None and run.finished_at is None:
                audit.append_decision(
                    run_id,
                    agent="profiler",
                    step=_SAVE_PROFILE_TOOL,
                    hitl_decision={
                        "decision": "approve",
                        "idempotent": True,
                        "existing_version": existing.version,
                        "flags": list(suitability.inconsistency_flags),
                    },
                )
                audit.finalize_run(run_id, weights={}, status="completed")
            return json.dumps(
                {"saved": True, "idempotent": True, "version": existing.version}
            )

        profile = repo.add_version(
            portfolio_id=portfolio_id_uuid,
            questionnaire=answers.model_dump(mode="json"),
            constraint_set=constraint_set.model_dump(mode="json"),
            suitability=suitability.model_dump(mode="json"),
            store_key=store_key,
        )
        if store is not None:
            put_constraint_set(store, constraint_set, store_key=store_key)

        audit = AgentRunRepository(session)
        audit.append_decision(
            run_id,
            agent="profiler",
            step=_SAVE_PROFILE_TOOL,
            constraint_set=constraint_set.model_dump(mode="json"),
            hitl_decision={
                "decision": "approve",
                "flags": list(suitability.inconsistency_flags),
            },
        )
        audit.finalize_run(run_id, weights={}, status="completed")
        return json.dumps(
            {
                "saved": True,
                "idempotent": False,
                "profile_id": str(profile.id),
                "version": profile.version,
                "store_key": store_key,
            }
        )

    return save_profile


def build_profiler_agent(
    model: Any,
    tools: list[BaseTool],
    *,
    checkpointer: Any,
    store: Any | None = None,
    interrupt_config: dict[str, Any] | None = None,
) -> Any:
    """Assemble the ``deepagents`` profiler agent (SPEC §8.5).

    System prompt = ``PROFILER_SYSTEM_PROMPT``; ``tools`` is the single
    ``save_profile`` tool, gated behind ``interrupt_on`` (always-confirm) — which
    requires the ``checkpointer``. ``interrupt_config`` (allowed decisions +
    flag-surfacing description) customises the pause; a bare ``True`` falls back to
    the default gate. ``temperature=0`` / the DeepSeek route are carried by
    ``model`` (built from ``FundConfig``), not passed here.
    """
    from deepagents import create_deep_agent

    # bool | InterruptOnConfig (a langchain TypedDict) — kept Any to avoid a
    # top-level import of the agent stack (the module stays import-light).
    gate: Any = interrupt_config if interrupt_config else True
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=PROFILER_SYSTEM_PROMPT,
        interrupt_on={_SAVE_PROFILE_TOOL: gate},
        checkpointer=checkpointer,
        store=store,
    )


def run_profiler(
    model: Any,
    questionnaire: Any,
    *,
    portfolio_id: uuid.UUID | str,
    session: Session,
    checkpointer: Any,
    store: Any | None = None,
    config: FundConfig = settings,
    fallback: Any | None = None,
    base_currency: str | None = None,
    asof: dt.date | None = None,
    thread_id: str | None = None,
) -> ProfilerRun:
    """Run the MiFID profiler to the adviser-confirmation gate (runtime step 0).

    Orchestrates: (1) normalise free-text ``questionnaire`` answers into a typed
    ``MiFIDAnswers`` via ``structured_call`` (retry once, then ``fallback``);
    (2) run the deterministic mapping (``run_mapping`` → ``ConstraintSet`` +
    ``SuitabilityAssessment``; an ESG/legal breach raises ``SuitabilityBreachError``
    and blocks before any write); (3) build the ``deepagents`` agent and invoke it
    so the ``save_profile`` tool pauses at the HITL gate. Returns a
    :class:`ProfilerRun`; call ``.resume("approve" | "reject")`` to persist or not.

    ``model`` must expose ``with_structured_output`` (normalisation) and be a
    tool-calling chat model (the agent). Every write goes through the injected
    ``session`` (caller owns the transaction) and, when given, the ``store``.
    """
    from fund.audit import AgentRunRepository
    from fund.schemas.structured import structured_call

    pid_uuid = _coerce_uuid(portfolio_id)
    pid_str = str(pid_uuid)
    run_asof = asof if asof is not None else dt.date.today()

    audit = AgentRunRepository(session)
    run = audit.create_run(
        portfolio_id=pid_uuid,
        asof=run_asof,
        seed=None,
        universe=[],
        optimizer_config={"step": "profiler"},
    )

    # (1) The LLM interprets the client's answers into typed inputs.
    answers = structured_call(
        model,
        MiFIDAnswers,
        _normalise_messages(questionnaire),
        retries=1,
        fallback=fallback,
    )
    answers_json = answers.model_dump_json()
    audit.append_decision(
        run.id,
        agent="profiler",
        step="normalize_answers",
        llm_prompt=_prompt_text(questionnaire),
        llm_response=answers_json,
        llm_response_hash=_hash(answers_json),
    )

    # (2) The deterministic mapping computes every knob (may hard-block).
    try:
        constraint_set, suitability = run_mapping(
            answers, portfolio_id=pid_str, base_currency=base_currency
        )
    except SuitabilityBreachError as exc:
        # HARD MiFID breach: a terminal business outcome, not an infra error.
        # Record it and finalise the run so no 'pending' orphan is left behind
        # (mirrors run_fund's early-exit finalisation; the caller still sees the
        # exception).
        audit.append_decision(
            run.id,
            agent="profiler",
            step="suitability_breach",
            hitl_decision={"decision": "blocked", "reason": str(exc)},
        )
        audit.finalize_run(run.id, weights={}, status="blocked")
        raise

    # (3) Persist behind the always-on HITL gate.
    save_profile = _make_save_profile(
        session=session,
        store=store,
        config=config,
        answers=answers,
        constraint_set=constraint_set,
        suitability=suitability,
        run_id=run.id,
        portfolio_id=pid_uuid,
    )
    interrupt_config = {
        "allowed_decisions": ["approve", "reject"],
        "description": _interrupt_description(pid_str, suitability),
    }
    agent = build_profiler_agent(
        model,
        [save_profile],
        checkpointer=checkpointer,
        store=store,
        interrupt_config=interrupt_config,
    )
    thread_config = {"configurable": {"thread_id": thread_id or pid_str}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _persist_instruction(pid_str)}]},
        config=thread_config,
    )
    return ProfilerRun(
        answers=answers,
        constraint_set=constraint_set,
        suitability=suitability,
        run_id=run.id,
        interrupt=_extract_interrupt(result),
        agent=agent,
        thread_config=thread_config,
        session=session,
    )
