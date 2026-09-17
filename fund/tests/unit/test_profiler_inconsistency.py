"""Task 3 — inconsistency / anti-overconfidence check + suitability assembly.

The deterministic layer flags contradictory answers (SPEC §5, deep_agent.md
``01:338`` anti-overconfidence) **without auto-clamping** — the flags are surfaced
to the adviser at the HITL gate, the ``a_gamma`` still comes from the
``min(tolerance, capacity)`` binding. Only a legal breach (an empty ESG universe)
hard-blocks outright.

``run_mapping`` is the pure Task-3 wrapper the LLM agent (Task 7) calls: it returns
the ``(ConstraintSet, SuitabilityAssessment)`` pair, imports no ``deepagents`` and
touches no DB.
"""

from __future__ import annotations

import pytest

from fund.agents.profiler import (
    FLAG_OBJECTIVE_CAPACITY_MISMATCH,
    FLAG_OVERCONFIDENCE,
    FLAG_REACTION_TOLERANCE_MISMATCH,
    SuitabilityBreachError,
    _category,
    assess_suitability,
    build_constraint_set,
    run_mapping,
)
from fund.schemas.constraint_set import ConstraintSet
from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
)
from fund.schemas.questionnaire import (
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    ObjectivesAnswers,
    SuitabilityAssessment,
)


def _answers(
    *,
    likert: tuple[int, ...] = (5, 6, 4),
    max_loss: float = 0.25,
    buffer: float = 6.0,
    goal: ObjectiveChoice = ObjectiveChoice.GROWTH,
    reaction: LossReaction = LossReaction.HOLD,
    exclusions: tuple[GicsSector, ...] = (),
) -> MiFIDAnswers:
    return MiFIDAnswers(
        base_currency="EUR",
        knowledge=KnowledgeAnswers(level=KnowledgeLevel.INFORMED),
        capacity=CapacityAnswers(max_1yr_loss_pct=max_loss, buffer_months=buffer),
        objectives=ObjectivesAnswers(
            goal=goal,
            horizon=Horizon.LONG,
            likert_items=likert,
            loss_reaction=reaction,
        ),
        esg=EsgAnswers(exclusions=exclusions),
    )


# --- consistent baseline ---------------------------------------------------


def test_consistent_answers_produce_no_flags() -> None:
    _, sa = run_mapping(_answers(), portfolio_id="pf-1")
    assert sa.inconsistency_flags == ()


# --- individual inconsistency rules (isolated) -----------------------------


def test_aggressive_objective_with_low_capacity_flags_mismatch() -> None:
    # "max growth" + "cannot lose anything" — the SPEC canonical contradiction.
    _, sa = run_mapping(
        _answers(goal=ObjectiveChoice.MAX, max_loss=0.02, buffer=0.0),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH in sa.inconsistency_flags


def test_high_tolerance_low_capacity_flags_overconfidence() -> None:
    # Bold attitude, thin wallet — anti-overconfidence. goal=INCOME keeps the
    # objective/capacity rule out of it so overconfidence is isolated.
    _, sa = run_mapping(
        _answers(
            likert=(7, 7, 7), max_loss=0.02, buffer=0.0, goal=ObjectiveChoice.INCOME
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OVERCONFIDENCE in sa.inconsistency_flags
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH not in sa.inconsistency_flags


def test_high_tolerance_but_panic_reaction_flags_mismatch() -> None:
    # High stated tolerance yet "sell everything" on a drop — a behavioural
    # contradiction. Capacity is high (a_cap = 1.0) so overconfidence stays out.
    _, sa = run_mapping(
        _answers(
            likert=(7, 7, 7),
            max_loss=0.50,
            buffer=12.0,
            goal=ObjectiveChoice.INCOME,
            reaction=LossReaction.SELL_ALL,
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_REACTION_TOLERANCE_MISMATCH in sa.inconsistency_flags
    assert FLAG_OVERCONFIDENCE not in sa.inconsistency_flags


# --- no auto-clamp: flags do not move a_gamma ------------------------------


def test_flags_do_not_auto_clamp_a_gamma() -> None:
    answers = _answers(
        likert=(7, 7, 7), max_loss=0.02, buffer=0.0, goal=ObjectiveChoice.INCOME
    )
    cs, sa = run_mapping(answers, portfolio_id="pf-1")
    # capacity binds at appetite ≈ 0.02 → Defensive; the flag does not clamp it
    # further, it merely surfaces at the HITL gate.
    assert cs.a_gamma == 12.0
    assert sa.inconsistency_flags != ()
    # a_gamma is exactly the min-binding output, unchanged by the flag.
    assert sa.a_gamma == build_constraint_set(answers, portfolio_id="pf-1").a_gamma


# --- ESG/legal breach hard-blocks (not a soft flag) ------------------------


def test_esg_breach_hard_blocks_run_mapping() -> None:
    with pytest.raises(SuitabilityBreachError):
        run_mapping(_answers(exclusions=tuple(GicsSector)), portfolio_id="pf-1")


# --- SuitabilityAssessment assembly ----------------------------------------


def test_run_mapping_returns_constraint_set_and_assessment() -> None:
    cs, sa = run_mapping(_answers(), portfolio_id="pf-1")
    assert isinstance(cs, ConstraintSet)
    assert isinstance(sa, SuitabilityAssessment)


def test_assessment_records_appetites_band_and_esg() -> None:
    answers = _answers(exclusions=(GicsSector.ENERGY,))
    cs, sa = run_mapping(answers, portfolio_id="pf-1")
    assert sa.answers == answers
    assert 0.0 <= sa.appetite_from_tolerance <= 1.0
    assert 0.0 <= sa.appetite_from_capacity <= 1.0
    assert sa.a_gamma == cs.a_gamma
    assert sa.band is _category(cs.a_gamma)
    assert sa.esg_exclusions == cs.esg.exclusions
    assert GicsSector.ENERGY in sa.esg_exclusions
    assert sa.rationale != ""


def test_assess_suitability_matches_constraint_set() -> None:
    answers = _answers()
    cs = build_constraint_set(answers, portfolio_id="pf-1")
    sa = assess_suitability(answers, cs)
    assert sa.a_gamma == cs.a_gamma
    assert sa.band is _category(cs.a_gamma)
    assert sa.esg_exclusions == cs.esg.exclusions


def test_assessment_json_round_trips() -> None:
    _, sa = run_mapping(_answers(exclusions=(GicsSector.ENERGY,)), portfolio_id="pf-1")
    restored = SuitabilityAssessment.model_validate_json(sa.model_dump_json())
    assert restored == sa
