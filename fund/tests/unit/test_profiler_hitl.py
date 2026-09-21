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
from fund.audit import AgentRunRepository, MifidProfileRepository
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


# --- idempotent re-profile still finalises its (otherwise orphaned) run ------


def test_reprofile_idempotent_hit_finalizes_run_and_logs_decision(db_session):
    pid = uuid.uuid4()
    store = InMemoryStore()

    # First profile -> writes v1, finalises run1.
    run1 = _run(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        portfolio_id=pid,
        store=store,
    )
    run1.resume("approve")
    assert MifidProfileRepository(db_session).get_active(pid).version == 1

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
        db_session,
        portfolio_id=pid,
        store=store,
    )
    run2.resume("approve")

    # Deferred-by-design (D11, Phase 9): no v2; v1 (run1's mapping) stays active.
    active = MifidProfileRepository(db_session).get_active(pid)
    assert active.version == 1
    assert active.constraint_set == run1.constraint_set.model_dump(mode="json")

    # Audit-integrity: run2 is finalised, not orphaned at "pending".
    run2_row = AgentRunRepository(db_session).get_run(run2.run_id)
    assert run2_row.status == "completed"
    assert run2_row.finished_at is not None

    # ...and run2's save step is recorded as an idempotent approve.
    saves = [
        d
        for d in db_session.query(AgentDecision)
        .filter(AgentDecision.run_id == run2.run_id)
        .all()
        if d.step == "save_profile"
    ]
    assert saves and saves[0].hitl_decision.get("idempotent") is True
    assert saves[0].hitl_decision.get("existing_version") == 1
