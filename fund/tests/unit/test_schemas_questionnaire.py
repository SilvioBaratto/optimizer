"""Task 1 — ``fund.schemas.questionnaire`` pins the MiFID questionnaire I/O.

``MiFIDQuestion`` (+ the ``QUESTION_BANK`` constant), ``MiFIDAnswers`` (the four
ESMA pillars: knowledge & experience, financial/loss-capacity, objectives incl.
the risk-tolerance Likert + horizon, and ESG) and ``SuitabilityAssessment`` are
pure, frozen, serialisable pydantic-v2 data — building one imports **no**
``optimizer`` / ``deepagents`` code. These tests assert validation (accept good,
reject the SPEC §5 bad cases: bad currency, out-of-range Likert, unknown GICS),
the JSON round-trip invariant, and the question-bank shape.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    ObjectiveChoice,
    RiskToleranceBand,
)
from fund.schemas.questionnaire import (
    QUESTION_BANK,
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    MiFIDQuestion,
    ObjectivesAnswers,
    SuitabilityAssessment,
)


def _valid_answers(**overrides: object) -> MiFIDAnswers:
    kwargs: dict[str, object] = {
        "base_currency": "EUR",
        "knowledge": KnowledgeAnswers(level=KnowledgeLevel.INFORMED),
        "capacity": CapacityAnswers(max_1yr_loss_pct=0.25, buffer_months=6.0),
        "objectives": ObjectivesAnswers(
            goal=ObjectiveChoice.GROWTH,
            horizon=Horizon.LONG,
            likert_items=(5, 6, 4),
            loss_reaction=LossReaction.HOLD,
        ),
        "esg": EsgAnswers(exclusions=(GicsSector.ENERGY,)),
    }
    kwargs.update(overrides)
    return MiFIDAnswers(**kwargs)  # type: ignore[arg-type]


def _valid_suitability(**overrides: object) -> SuitabilityAssessment:
    kwargs: dict[str, object] = {
        "answers": _valid_answers(),
        "appetite_from_tolerance": 0.7,
        "appetite_from_capacity": 0.5,
        "a_gamma": 5.0,
        "band": RiskToleranceBand.BALANCED,
        "esg_exclusions": (GicsSector.ENERGY,),
        "inconsistency_flags": (),
        "rationale": "Capacity binds; balanced band.",
    }
    kwargs.update(overrides)
    return SuitabilityAssessment(**kwargs)  # type: ignore[arg-type]


# --- happy path -----------------------------------------------------------


def test_valid_answers_accepted():
    a = _valid_answers()
    assert a.base_currency == "EUR"
    assert a.knowledge.level is KnowledgeLevel.INFORMED
    assert a.capacity.max_1yr_loss_pct == 0.25
    assert a.capacity.buffer_months == 6.0
    assert a.objectives.goal is ObjectiveChoice.GROWTH
    assert a.objectives.horizon is Horizon.LONG
    assert a.objectives.likert_items == (5, 6, 4)
    assert a.objectives.loss_reaction is LossReaction.HOLD
    assert a.esg.exclusions == (GicsSector.ENERGY,)


def test_esg_defaults_to_no_exclusions():
    a = _valid_answers(esg=EsgAnswers())
    assert a.esg.exclusions == ()


def test_all_pillar_models_are_frozen():
    a = _valid_answers()
    with pytest.raises(pydantic.ValidationError):
        a.base_currency = "USD"  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        a.capacity.buffer_months = 12.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        a.objectives.likert_items = (1,)  # type: ignore[misc]


def test_models_are_hashable():
    # frozen=True keeps the schema hashable (matches the optimizer ethos).
    assert hash(_valid_answers()) == hash(_valid_answers())
    assert hash(_valid_suitability()) == hash(_valid_suitability())


# --- SPEC §5 bad cases: bad currency / out-of-range Likert / unknown GICS --


@pytest.mark.parametrize("bad_currency", ["eur", "EURO", "EU", "E1R", "US$"])
def test_rejects_non_iso_currency(bad_currency: str):
    with pytest.raises(pydantic.ValidationError):
        _valid_answers(base_currency=bad_currency)


@pytest.mark.parametrize("bad", [0, 8, -1, 100])
def test_rejects_out_of_range_likert(bad: int):
    # SPEC §8.2 rescales the Likert mean by (x-1)/6, so responses live in [1, 7].
    with pytest.raises(pydantic.ValidationError):
        ObjectivesAnswers(
            goal=ObjectiveChoice.GROWTH,
            horizon=Horizon.LONG,
            likert_items=(4, bad),
            loss_reaction=LossReaction.HOLD,
        )


def test_rejects_empty_likert_items():
    with pytest.raises(pydantic.ValidationError):
        ObjectivesAnswers(
            goal=ObjectiveChoice.GROWTH,
            horizon=Horizon.LONG,
            likert_items=(),
            loss_reaction=LossReaction.HOLD,
        )


def test_rejects_unknown_gics_sector_in_esg():
    with pytest.raises(pydantic.ValidationError):
        EsgAnswers(exclusions=("not_a_sector",))  # type: ignore[arg-type]


def test_accepts_known_gics_sectors_in_esg():
    esg = EsgAnswers(exclusions=(GicsSector.ENERGY, GicsSector.UTILITIES))
    assert GicsSector.ENERGY in esg.exclusions


@pytest.mark.parametrize("bad_loss", [-0.01, 1.01])
def test_rejects_loss_pct_outside_unit_interval(bad_loss: float):
    with pytest.raises(pydantic.ValidationError):
        CapacityAnswers(max_1yr_loss_pct=bad_loss, buffer_months=6.0)


def test_rejects_negative_buffer_months():
    with pytest.raises(pydantic.ValidationError):
        CapacityAnswers(max_1yr_loss_pct=0.25, buffer_months=-1.0)


@pytest.mark.parametrize("bad_appetite", [-0.01, 1.01])
def test_suitability_rejects_appetite_outside_unit_interval(bad_appetite: float):
    with pytest.raises(pydantic.ValidationError):
        _valid_suitability(appetite_from_tolerance=bad_appetite)


@pytest.mark.parametrize("bad_gamma", [0.0, -1.0])
def test_suitability_rejects_non_positive_a_gamma(bad_gamma: float):
    with pytest.raises(pydantic.ValidationError):
        _valid_suitability(a_gamma=bad_gamma)


# --- JSON round-trip ------------------------------------------------------


def test_answers_json_round_trip():
    a = _valid_answers(
        esg=EsgAnswers(exclusions=(GicsSector.ENERGY, GicsSector.UTILITIES)),
    )
    assert MiFIDAnswers.model_validate(a.model_dump(mode="json")) == a


def test_suitability_json_round_trip():
    sa = _valid_suitability(
        inconsistency_flags=("max_growth_but_zero_loss_capacity",),
        esg_exclusions=(GicsSector.ENERGY,),
    )
    assert SuitabilityAssessment.model_validate(sa.model_dump(mode="json")) == sa


def test_suitability_embeds_per_pillar_inputs():
    sa = _valid_suitability()
    assert sa.answers == _valid_answers()  # per-pillar snapshot retained
    assert sa.band is RiskToleranceBand.BALANCED  # recorded named category
    assert sa.appetite_from_tolerance == 0.7
    assert sa.appetite_from_capacity == 0.5


# --- question bank --------------------------------------------------------


def test_question_bank_covers_the_four_pillars():
    pillars = {q.pillar for q in QUESTION_BANK}
    assert pillars == {"knowledge", "capacity", "objectives", "esg"}


def test_question_bank_ids_are_unique():
    ids = [q.id for q in QUESTION_BANK]
    assert len(ids) == len(set(ids))


def test_question_bank_entries_are_frozen_questions():
    assert QUESTION_BANK  # non-empty
    for q in QUESTION_BANK:
        assert isinstance(q, MiFIDQuestion)
        with pytest.raises(pydantic.ValidationError):
            q.text = "mutated"  # type: ignore[misc]


def test_mifid_question_rejects_unknown_pillar():
    with pytest.raises(pydantic.ValidationError):
        MiFIDQuestion(
            id="x",
            pillar="nonsense",  # type: ignore[arg-type]
            text="?",
            kind="single_choice",
        )


def test_mifid_question_rejects_unknown_kind():
    with pytest.raises(pydantic.ValidationError):
        MiFIDQuestion(
            id="x",
            pillar="knowledge",
            text="?",
            kind="freeform",  # type: ignore[arg-type]
        )


# --- hygiene: schemas stay optimizer / agent-stack free -------------------


def test_module_imports_nothing_forbidden():
    import fund.schemas.questionnaire as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "optimizer",
        "deepagents",
        "langgraph",
        "langchain",
        "app",
        "skfolio",
    ):
        assert f"import {forbidden}" not in src
        assert f"from {forbidden}" not in src
