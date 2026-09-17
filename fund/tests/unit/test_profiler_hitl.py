"""Task 7 — the always-on HITL gate: the profiler never persists silently.

Every profiling run pauses at the ``save_profile`` tool (behind ``interrupt_on`` +
a ``MemorySaver`` checkpointer) before any write. ``Command(resume=approve)``
persists; ``Command(resume=reject)`` persists nothing. Inconsistency /
anti-overconfidence flags surface in the interrupt payload so the adviser sees
them at the confirmation gate. Zero live LLM, zero network.
"""

from __future__ import annotations

import uuid

from _profiler_fakes import make_answers, make_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision

from fund.agents.profiler import run_profiler
from fund.audit import MifidProfileRepository
from fund.schemas.enums import LossReaction, ObjectiveChoice

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

    assert run.suitability.inconsistency_flags  # contradiction detected
    description = run.interrupt["action_requests"][0]["description"]
    for flag in run.suitability.inconsistency_flags:
        assert flag in description


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
