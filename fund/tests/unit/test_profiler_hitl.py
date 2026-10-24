"""The always-on HITL gate: the profiler never persists silently.

Every profiling run pauses at the ``save_profile`` tool (behind ``interrupt_on`` +
a ``MemorySaver`` checkpointer) before any write. ``Command(resume=approve)``
persists; ``Command(resume=reject)`` persists nothing. Inconsistency /
anti-overconfidence flags surface in the interrupt payload so the adviser sees
them at the confirmation gate. Zero live LLM, zero network.
"""

from __future__ import annotations

import dataclasses
import uuid
from types import SimpleNamespace

import pytest
from _profiler_fakes import (
    ScriptedProfilerModel,
    final_reply,
    make_answers,
    make_model,
    save_tool_call,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision

from fund.agents.profiler import run_profiler
from fund.audit import AgentRunRepository, MifidProfileRepository
from fund.schemas.enums import GicsSector, KnowledgeLevel, LossReaction, ObjectiveChoice

_MESSAGES = [{"role": "user", "content": "questionnaire ..."}]


def _run(model, db_session, *, portfolio_id, store=None):
    return run_profiler(
        model,
        _MESSAGES,
        portfolio_id=portfolio_id,
        session=db_session,
        checkpointer=MemorySaver(),
        store=store,
    )


def _overconfident_answers():
    # High attitude (all-7 Likert) but a thin buffer / low loss capacity, asking
    # for MAX growth: capacity binds conservative AND contradictions are flagged.
    return make_answers(
        likert=(7, 7, 7),
        max_loss=0.10,
        buffer=1.0,
        goal=ObjectiveChoice.MAX,
        reaction=LossReaction.SELL_ALL,
    )


# --- always pause before persist --------------------------------------------


def test_always_interrupts_before_persisting(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    assert run.interrupt is not None
    # The gate paused: nothing is written until the adviser resumes.
    assert MifidProfileRepository(db_session).get_active(pid) is None


def test_resume_approve_persists_the_profile(db_session):
    pid = uuid.uuid4()
    store = InMemoryStore()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=store)
    run.resume("approve")

    active = MifidProfileRepository(db_session).get_active(pid)
    assert active is not None
    assert active.version == 1


def test_resume_reject_persists_nothing(db_session):
    pid = uuid.uuid4()
    store = InMemoryStore()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=store)
    run.resume("reject")

    assert MifidProfileRepository(db_session).get_active(pid) is None
    assert store.get((str(pid),), "constraint_set") is None


# --- flags surface at the gate ----------------------------------------------


def test_inconsistency_flags_surface_in_interrupt_payload(db_session):
    pid = uuid.uuid4()
    model = make_model(_overconfident_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    assert run.suitability.inconsistency_flags
    description = run.interrupt["action_requests"][0]["description"]
    for flag in run.suitability.inconsistency_flags:
        assert flag in description


def test_unanswered_datum_surfaces_in_interrupt_payload(db_session):
    pid = uuid.uuid4()
    answers = make_answers(unanswered=("capacity.buffer_months",))
    model = make_model(answers, portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    description = run.interrupt["action_requests"][0]["description"]
    assert "unanswered_input:capacity.buffer_months" in description


def test_interrupt_description_shows_the_extracted_answers(db_session):
    pid = uuid.uuid4()
    answers = make_answers(
        likert=(3, 4, 5),
        max_loss=0.30,
        buffer=4.0,
        knowledge=KnowledgeLevel.BASIC,
        exclusions=(GicsSector.ENERGY,),
        other_preferences=("no tobacco",),
    )

    run = _run(make_model(answers, portfolio_id=str(pid)), db_session, portfolio_id=pid)

    description = run.interrupt["action_requests"][0]["description"]
    assert "knowledge basic" in description
    assert "max 1y loss 30%" in description
    assert "buffer 4 months" in description
    assert "likert 3, 4, 5" in description
    assert "other ESG preferences (not enforced): no tobacco" in description
    assert "unenforced_esg_preference" in description


# --- a malformed save_profile call must not burn the adviser's approval ------


def _model_calling_save_profile_with(args: dict, *, answers=None):
    call = AIMessage(
        content="",
        tool_calls=[{"name": "save_profile", "args": args, "id": "call_save_profile"}],
    )
    return ScriptedProfilerModel(
        structured_outcomes=[answers or make_answers()],
        chat_responses=[call, final_reply()],
    )


@pytest.mark.parametrize(
    "args",
    [{}, {"portfolio_id": 5}, {"portfolio_id": "not-the-id", "extra": 1}],
    ids=["no-args", "mistyped-portfolio-id", "foreign-args"],
)
def test_approve_persists_whatever_arguments_the_model_passes(db_session, args):
    pid = uuid.uuid4()
    run = _run(_model_calling_save_profile_with(args), db_session, portfolio_id=pid)

    run.resume("approve")

    active = MifidProfileRepository(db_session).get_active(pid)
    assert active is not None
    assert active.version == 1
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


def test_approve_that_leaves_the_run_unfinished_is_reported_not_silent(db_session):
    pid = uuid.uuid4()
    run = _run(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        portfolio_id=pid,
    )
    # An agent whose resume never reaches the write (e.g. the tool call errored).
    stalled = dataclasses.replace(run, agent=SimpleNamespace(invoke=lambda *a, **k: {}))

    with pytest.raises(RuntimeError, match="save_profile"):
        stalled.resume("approve")

    assert MifidProfileRepository(db_session).get_active(pid) is None


# --- audit trail records the HITL decision ----------------------------------


def test_approve_records_hitl_decision_in_audit(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)
    run.resume("approve")

    decisions = (
        db_session.query(AgentDecision).filter(AgentDecision.run_id == run.run_id).all()
    )
    hitl = [d for d in decisions if d.hitl_decision]
    assert any(d.hitl_decision.get("decision") == "approve" for d in hitl)


def test_reject_records_hitl_decision_in_audit(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)
    run.resume("reject")

    decisions = (
        db_session.query(AgentDecision).filter(AgentDecision.run_id == run.run_id).all()
    )
    assert any(
        d.hitl_decision and d.hitl_decision.get("decision") == "reject"
        for d in decisions
    )


# --- a decided run cannot be decided again ----------------------------------


def test_resume_after_approve_cannot_reject_the_live_profile(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=InMemoryStore())
    run.resume("approve")

    with pytest.raises(ValueError, match="completed"):
        run.resume("reject")

    # The audit trail still agrees with the live profile.
    assert MifidProfileRepository(db_session).get_active(pid) is not None
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


def test_resume_after_reject_cannot_approve(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=InMemoryStore())
    run.resume("reject")

    with pytest.raises(ValueError, match="rejected"):
        run.resume("approve")

    assert MifidProfileRepository(db_session).get_active(pid) is None
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


# --- the closing model turn cannot undo an approved write -------------------


def test_resume_approve_survives_a_failing_closing_model_turn(committing_session):
    pid = uuid.uuid4()
    store = InMemoryStore()
    # No closing reply is scripted: the turn after ``save_profile`` raises.
    model = ScriptedProfilerModel(
        structured_outcomes=[make_answers()],
        chat_responses=[save_tool_call(str(pid))],
    )

    run = _run(model, committing_session, portfolio_id=pid, store=store)
    run.resume("approve")
    committing_session.commit()

    assert MifidProfileRepository(committing_session).get_active(pid) is not None
    row = AgentRunRepository(committing_session).get_run(run.run_id)
    assert row.status == "completed"
    assert store.get((str(pid),), "constraint_set") is not None


# --- a run that never reaches the gate is not left pending -------------------


def test_run_that_never_reaches_the_gate_is_finalised_incomplete(db_session):
    pid = uuid.uuid4()
    model = ScriptedProfilerModel(
        structured_outcomes=[make_answers()], chat_responses=[final_reply()]
    )

    run = _run(model, db_session, portfolio_id=pid)

    assert run.interrupt is None
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert row.finished_at is not None
    assert MifidProfileRepository(db_session).get_active(pid) is None


# --- re-profiling appends the next version ----------------------------------


def test_reprofile_appends_a_new_version_and_refreshes_the_store(committing_session):
    pid = uuid.uuid4()
    store = InMemoryStore()

    run1 = _run(
        make_model(make_answers(), portfolio_id=str(pid)),
        committing_session,
        portfolio_id=pid,
        store=store,
    )
    run1.resume("approve")
    committing_session.commit()
    assert MifidProfileRepository(committing_session).get_active(pid).version == 1

    # Second profile for the SAME portfolio with materially different answers.
    diff = make_answers(
        likert=(7, 7, 7),
        max_loss=0.5,
        buffer=12.0,
        goal=ObjectiveChoice.MAX,
        reaction=LossReaction.BUY_MORE,
    )
    run2 = _run(
        make_model(diff, portfolio_id=str(pid)),
        committing_session,
        portfolio_id=pid,
        store=store,
    )
    run2.resume("approve")
    # The Store cache is released only once the session commits.
    committing_session.commit()

    # The amended assessment is the next append-only version and is the active one.
    active = MifidProfileRepository(committing_session).get_active(pid)
    assert active.version == 2
    assert active.constraint_set == run2.constraint_set.model_dump(mode="json")
    assert run2.constraint_set != run1.constraint_set
    cached = store.get((str(pid),), "constraint_set")
    assert cached.value == run2.constraint_set.model_dump(mode="json")

    run2_row = AgentRunRepository(committing_session).get_run(run2.run_id)
    assert run2_row.status == "completed"
    assert run2_row.finished_at is not None
    saves = [
        d
        for d in committing_session.query(AgentDecision)
        .filter(AgentDecision.run_id == run2.run_id)
        .all()
        if d.step == "save_profile"
    ]
    assert len(saves) == 1
    assert saves[0].hitl_decision.get("decision") == "approve"
    assert not saves[0].hitl_decision.get("idempotent")


def test_save_profile_replay_of_a_completed_run_writes_nothing(db_session):
    """A same-run replay (SPEC D3) appends no version and re-logs no decision."""
    import json

    from fund.agents.profiler import _make_save_profile
    from fund.config import settings

    pid = uuid.uuid4()
    run = _run(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        portfolio_id=pid,
        store=InMemoryStore(),
    )
    run.resume("approve")

    replay = _make_save_profile(
        session=db_session,
        store=None,
        config=settings,
        answers=run.answers,
        constraint_set=run.constraint_set,
        suitability=run.suitability,
        run_id=run.run_id,
        portfolio_id=pid,
    )
    result = json.loads(replay.invoke({"portfolio_id": str(pid)}))

    assert result == {"saved": True, "idempotent": True, "version": 1}
    assert MifidProfileRepository(db_session).get_active(pid).version == 1
    saves = [
        d
        for d in db_session.query(AgentDecision)
        .filter(AgentDecision.run_id == run.run_id)
        .all()
        if d.step == "save_profile"
    ]
    assert len(saves) == 1
