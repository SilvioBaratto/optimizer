"""Inconsistency / anti-overconfidence check and suitability assembly.

The deterministic layer flags contradictory answers without auto-clamping — flags
are surfaced to the adviser at the HITL gate; ``a_gamma`` still comes from the
``min(tolerance, capacity)`` binding. Only a legal breach (an empty ESG universe)
hard-blocks outright.

``run_mapping`` is the pure wrapper the LLM agent calls: it returns the
``(ConstraintSet, SuitabilityAssessment)`` pair, imports no ``deepagents`` and
touches no DB.
"""

from __future__ import annotations

import pytest

from fund.agents.profiler import (
    FLAG_OBJECTIVE_CAPACITY_MISMATCH,
    FLAG_OVERCONFIDENCE,
    FLAG_REACTION_TOLERANCE_MISMATCH,
    FLAG_UNANSWERED_INPUT,
    FLAG_UNENFORCED_ESG_PREFERENCE,
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
    other_preferences: tuple[str, ...] = (),
    unanswered: tuple[str, ...] = (),
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
        esg=EsgAnswers(exclusions=exclusions, other_preferences=other_preferences),
        unanswered=unanswered,  # type: ignore[arg-type]
    )


# --- consistent baseline ---------------------------------------------------


def test_consistent_answers_produce_no_flags() -> None:
    _, sa = run_mapping(_answers(), portfolio_id="pf-1")
    assert sa.inconsistency_flags == ()


# --- individual inconsistency rules (isolated) -----------------------------


def test_aggressive_objective_with_low_capacity_flags_mismatch() -> None:
    # "max growth" + "cannot lose anything" — the canonical contradiction.
    _, sa = run_mapping(
        _answers(goal=ObjectiveChoice.MAX, max_loss=0.02, buffer=0.0),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH in sa.inconsistency_flags


def test_zero_loss_capacity_with_a_large_buffer_still_flags_mismatch() -> None:
    # "max growth" + "can lose 0%": a big emergency buffer lifts the blended
    # capacity score above the floor, but the stated loss capacity alone is the
    # contradiction the adviser must see.
    _, sa = run_mapping(
        _answers(goal=ObjectiveChoice.MAX, likert=(6, 6, 6), max_loss=0.0, buffer=12.0),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH in sa.inconsistency_flags


def test_averse_attitude_with_deep_pockets_flags_aggressive_objective() -> None:
    # Likert all 1 (strongly averse) but ample finances: the binding appetite is the
    # attitude, and a max-growth objective ignores a_gamma, so the flag is the only
    # guard against the contradiction.
    _, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.MAX,
            likert=(1, 1, 1),
            max_loss=0.50,
            buffer=12.0,
            reaction=LossReaction.SELL_ALL,
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH in sa.inconsistency_flags


def test_aggressive_objective_with_matching_answers_is_not_flagged() -> None:
    _, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.MAX, likert=(7, 7, 7), max_loss=0.50, buffer=12.0
        ),
        portfolio_id="pf-1",
    )
    assert sa.inconsistency_flags == ()


def test_protection_objective_with_cautious_answers_is_not_flagged() -> None:
    _, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.PROTECTION, likert=(1, 1, 1), max_loss=0.0, buffer=0.0
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH not in sa.inconsistency_flags


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


@pytest.mark.parametrize(
    ("likert", "max_loss", "buffer"),
    [
        ((5,), 0.10, 4.0),  # tolerance 2/3, capacity 4/15: gap exactly 0.4
        ((5, 5, 4, 4, 5), 0.10, 2.4),  # tolerance 0.6, capacity 0.2: gap exactly 0.4
    ],
)
def test_overconfidence_gap_at_exactly_the_threshold_is_flagged(
    likert: tuple[int, ...], max_loss: float, buffer: float
) -> None:
    # goal=INCOME and a HOLD reaction keep the other two rules out of it.
    _, sa = run_mapping(
        _answers(
            likert=likert,
            max_loss=max_loss,
            buffer=buffer,
            goal=ObjectiveChoice.INCOME,
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OVERCONFIDENCE in sa.inconsistency_flags


def test_overconfidence_gap_just_under_the_threshold_is_not_flagged() -> None:
    # tolerance 0.6, capacity 0.2 + 0.01: gap 0.39.
    _, sa = run_mapping(
        _answers(
            likert=(5, 5, 4, 4, 5),
            max_loss=0.10,
            buffer=2.64,
            goal=ObjectiveChoice.INCOME,
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OVERCONFIDENCE not in sa.inconsistency_flags


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


# --- unstated data is flagged, never passed off as client-provided ---------


def test_unanswered_datum_is_flagged_for_the_adviser() -> None:
    answers = _answers(unanswered=("capacity.buffer_months",))
    _, sa = run_mapping(answers, portfolio_id="pf-1")
    assert f"{FLAG_UNANSWERED_INPUT}:capacity.buffer_months" in sa.inconsistency_flags


def test_each_unanswered_datum_gets_its_own_flag_once() -> None:
    answers = _answers(
        unanswered=(
            "capacity.buffer_months",
            "objectives.likert_items",
            "capacity.buffer_months",
        )
    )
    _, sa = run_mapping(answers, portfolio_id="pf-1")
    unanswered = [
        f for f in sa.inconsistency_flags if f.startswith(f"{FLAG_UNANSWERED_INPUT}:")
    ]
    assert unanswered == [
        f"{FLAG_UNANSWERED_INPUT}:capacity.buffer_months",
        f"{FLAG_UNANSWERED_INPUT}:objectives.likert_items",
    ]


# --- placeholders for unanswered data and unenforceable ESG preferences ------


def test_unanswered_fields_are_flagged_and_given_conservative_placeholders() -> None:
    # The model filled bold-looking values for answers the client never gave; they
    # must not drive the band. Placeholders are the most conservative values.
    answers = _answers(
        likert=(7, 7, 7),
        max_loss=0.5,
        buffer=12.0,
        goal=ObjectiveChoice.PROTECTION,
        unanswered=("objectives.likert_items", "capacity.buffer_months"),
    )

    cs, sa = run_mapping(answers, portfolio_id="pf-1")

    assert any(
        f.startswith(f"{FLAG_UNANSWERED_INPUT}:") for f in sa.inconsistency_flags
    )
    assert sa.answers.objectives.likert_items == (1,)
    assert sa.answers.capacity.buffer_months == 0.0
    assert sa.appetite_from_tolerance == 0.0
    assert cs.a_gamma == 12.0
    assert (
        sa.answers.unanswered == answers.unanswered
    )  # the record keeps what was missing


def test_placeholders_leave_the_stated_fields_untouched() -> None:
    answers = _answers(
        likert=(6, 6, 6),
        max_loss=0.4,
        buffer=9.0,
        unanswered=("capacity.buffer_months",),
    )

    _, sa = run_mapping(answers, portfolio_id="pf-1")

    assert sa.answers.objectives.likert_items == (6, 6, 6)
    assert sa.answers.capacity.max_1yr_loss_pct == 0.4
    assert sa.answers.capacity.buffer_months == 0.0


def test_unanswered_loss_reaction_does_not_fabricate_a_reaction_mismatch() -> None:
    # The SELL_ALL placeholder is a conservative stand-in, not something the client
    # said, so a bold stated attitude cannot be "contradicted" by it.
    _, sa = run_mapping(
        _answers(
            likert=(7, 7, 7),
            max_loss=0.50,
            buffer=12.0,
            goal=ObjectiveChoice.INCOME,
            unanswered=("objectives.loss_reaction",),
        ),
        portfolio_id="pf-1",
    )
    assert any(
        f.startswith(f"{FLAG_UNANSWERED_INPUT}:") for f in sa.inconsistency_flags
    )
    assert FLAG_REACTION_TOLERANCE_MISMATCH not in sa.inconsistency_flags


def test_unanswered_likert_does_not_fabricate_an_objective_mismatch() -> None:
    # Growth goal with ample stated capacity; the (1,) placeholder attitude is the
    # only thing that looks cautious, and the client never said it.
    cs, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.GROWTH,
            max_loss=0.50,
            buffer=12.0,
            unanswered=("objectives.likert_items",),
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH not in sa.inconsistency_flags
    assert cs.a_gamma == 12.0  # the band itself still takes the conservative value


def test_unanswered_likert_still_flags_a_stated_capacity_contradiction() -> None:
    # What the client did state (a growth goal with no loss capacity) is still a
    # contradiction, whatever the unanswered attitude pillar holds.
    _, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.MAX,
            max_loss=0.02,
            buffer=0.0,
            unanswered=("objectives.likert_items",),
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH in sa.inconsistency_flags


def test_unanswered_capacity_does_not_fabricate_overconfidence() -> None:
    # Bold attitude against a capacity score the client only half-gave: the zero
    # buffer is a placeholder, so no attitude/capacity gap has been shown.
    _, sa = run_mapping(
        _answers(
            likert=(7, 7, 7),
            max_loss=0.50,
            buffer=12.0,
            goal=ObjectiveChoice.INCOME,
            unanswered=("capacity.buffer_months",),
        ),
        portfolio_id="pf-1",
    )
    assert any(
        f.startswith(f"{FLAG_UNANSWERED_INPUT}:") for f in sa.inconsistency_flags
    )
    assert FLAG_OVERCONFIDENCE not in sa.inconsistency_flags


def test_unanswered_loss_capacity_does_not_fabricate_an_objective_mismatch() -> None:
    _, sa = run_mapping(
        _answers(
            goal=ObjectiveChoice.MAX,
            likert=(7, 7, 7),
            max_loss=0.0,
            buffer=0.0,
            unanswered=("capacity.max_1yr_loss_pct", "capacity.buffer_months"),
        ),
        portfolio_id="pf-1",
    )
    assert FLAG_OBJECTIVE_CAPACITY_MISMATCH not in sa.inconsistency_flags
    assert FLAG_OVERCONFIDENCE not in sa.inconsistency_flags


def test_fully_stated_answers_carry_no_unanswered_flag() -> None:
    _, sa = run_mapping(_answers(), portfolio_id="pf-1")
    assert not any(
        f.startswith(f"{FLAG_UNANSWERED_INPUT}:") for f in sa.inconsistency_flags
    )


def test_non_sector_esg_preference_is_flagged_not_silently_dropped() -> None:
    answers = _answers(other_preferences=("no tobacco", "no weapons"))

    cs, sa = run_mapping(answers, portfolio_id="pf-1")

    assert FLAG_UNENFORCED_ESG_PREFERENCE in sa.inconsistency_flags
    assert cs.esg.exclusions == ()  # no arbitrary sector was substituted
    assert sa.answers.esg.other_preferences == ("no tobacco", "no weapons")


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
