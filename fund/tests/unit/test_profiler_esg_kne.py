"""Task 3 — ESG hard gate + K&E universe filters (deterministic, zero LLM).

Two HARD MiFID rules land here (deep_agent.md ``01:335`` / ``01:331``,
SPEC §5 required-coverage rows):

* **ESG gate = hard block (D9/D16).** The client's declared GICS-sector
  exclusions are copied verbatim into ``ConstraintSet.esg.exclusions`` and no
  other answer can re-admit an excluded sector — the gate is applied from the ESG
  pillar alone. Excluding *every* sector leaves no investable universe and is a
  legal breach that hard-blocks outright.
* **Low K&E ⇒ tightened universe (D32).** A ``none`` / ``basic`` knowledge level
  sets ``UniverseFilters.no_complex`` / ``no_leverage`` and a tightened
  ``max_position_cap``; ``informed`` / ``advanced`` leave the defaults.

Both maps are **total** over their driving enum (unmapped ⇒ hard ``KeyError``),
matching the ``fund/schemas`` ethos and Task 2's mapping tables.
"""

from __future__ import annotations

import pytest

from fund.agents.profiler import (
    SuitabilityBreachError,
    _esg_policy,
    _ke_filters,
    build_constraint_set,
)
from fund.schemas.constraint_set import EsgPolicy, UniverseFilters
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
)


def _answers(
    *,
    exclusions: tuple[GicsSector, ...] = (),
    knowledge: KnowledgeLevel = KnowledgeLevel.INFORMED,
    goal: ObjectiveChoice = ObjectiveChoice.GROWTH,
) -> MiFIDAnswers:
    return MiFIDAnswers(
        base_currency="EUR",
        knowledge=KnowledgeAnswers(level=knowledge),
        capacity=CapacityAnswers(max_1yr_loss_pct=0.25, buffer_months=6.0),
        objectives=ObjectivesAnswers(
            goal=goal,
            horizon=Horizon.LONG,
            likert_items=(5, 6, 4),
            loss_reaction=LossReaction.HOLD,
        ),
        esg=EsgAnswers(exclusions=exclusions),
    )


# --- ESG hard gate ---------------------------------------------------------


def test_declared_exclusions_appear_in_constraint_set() -> None:
    answers = _answers(exclusions=(GicsSector.ENERGY, GicsSector.UTILITIES))
    cs = build_constraint_set(answers, portfolio_id="pf-1")
    assert GicsSector.ENERGY in cs.esg.exclusions
    assert GicsSector.UTILITIES in cs.esg.exclusions


def test_no_esg_preference_leaves_exclusions_empty() -> None:
    cs = build_constraint_set(_answers(), portfolio_id="pf-1")
    assert cs.esg.exclusions == ()


def test_esg_exclusion_cannot_be_overridden_by_other_answers() -> None:
    # A maximally aggressive profile (goal=MAX) must NOT re-admit an excluded
    # sector — the gate reads the ESG pillar alone and is applied hard.
    answers = _answers(exclusions=(GicsSector.ENERGY,), goal=ObjectiveChoice.MAX)
    cs = build_constraint_set(answers, portfolio_id="pf-1")
    assert GicsSector.ENERGY in cs.esg.exclusions


def test_esg_policy_dedupes_preserving_order() -> None:
    policy = _esg_policy(
        EsgAnswers(
            exclusions=(GicsSector.ENERGY, GicsSector.ENERGY, GicsSector.FINANCIALS)
        )
    )
    assert isinstance(policy, EsgPolicy)
    assert policy.exclusions == (GicsSector.ENERGY, GicsSector.FINANCIALS)


def test_excluding_every_sector_hard_blocks() -> None:
    all_sectors = tuple(GicsSector)
    with pytest.raises(SuitabilityBreachError):
        _esg_policy(EsgAnswers(exclusions=all_sectors))
    # and the breach propagates through the full mapping
    with pytest.raises(SuitabilityBreachError):
        build_constraint_set(_answers(exclusions=all_sectors), portfolio_id="pf-1")


# --- K&E universe filters --------------------------------------------------


@pytest.mark.parametrize("level", [KnowledgeLevel.NONE, KnowledgeLevel.BASIC])
def test_low_knowledge_tightens_universe(level: KnowledgeLevel) -> None:
    cs = build_constraint_set(_answers(knowledge=level), portfolio_id="pf-1")
    filters = cs.universe_filters
    assert filters.no_complex is True
    assert filters.no_leverage is True
    assert filters.max_position_cap is not None
    assert 0.0 < filters.max_position_cap < 1.0


@pytest.mark.parametrize("level", [KnowledgeLevel.INFORMED, KnowledgeLevel.ADVANCED])
def test_high_knowledge_leaves_filters_default(level: KnowledgeLevel) -> None:
    cs = build_constraint_set(_answers(knowledge=level), portfolio_id="pf-1")
    assert cs.universe_filters == UniverseFilters()


def test_none_knowledge_is_at_least_as_strict_as_basic() -> None:
    none_cap = _ke_filters(KnowledgeAnswers(level=KnowledgeLevel.NONE)).max_position_cap
    basic_cap = _ke_filters(
        KnowledgeAnswers(level=KnowledgeLevel.BASIC)
    ).max_position_cap
    assert none_cap is not None and basic_cap is not None
    assert none_cap <= basic_cap


def test_ke_filters_total_over_every_knowledge_level() -> None:
    # Every KnowledgeLevel is a key — no member falls through to a KeyError.
    for level in KnowledgeLevel:
        assert isinstance(_ke_filters(KnowledgeAnswers(level=level)), UniverseFilters)
