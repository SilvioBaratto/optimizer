"""``resume_profiler`` + ``resume_run`` dispatch — cross-process approve/reject of
a paused MiFID profiler run (the profiler counterpart of ``test_resume_fund``).

``run_profiler`` drives the profiler deep agent to the ``save_profile`` HITL gate,
marking the run ``paused`` and persisting ``thread_id`` + the typed answers (the
``normalize_answers`` decision). A human observer then resumes it **without the
live** :class:`~fund.agents.profiler.ProfilerRun` **handle**: ``resume_profiler``
rebuilds the profiler agent against the *same* checkpointer + thread in a fresh
instance, re-derives the ``ConstraintSet`` from the audited answers (a pure
function — nothing is persisted before the gate), and approves (writes v1) or
rejects (writes nothing) — parity with the in-process ``ProfilerRun.resume``.
``resume_run`` dispatches a paused run to this path (vs ``resume_fund``) by its
recorded ``optimizer_config["step"]``. Zero live LLM, zero network:
``ScriptedProfilerModel`` + ``MemorySaver`` + ``InMemoryStore`` + the in-memory
``db_session`` (SPEC §5 / §4d).
"""

from __future__ import annotations

import uuid

import pytest
from _profiler_fakes import (
    ScriptedProfilerModel,
    final_reply,
    make_answers,
    make_model,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision

from fund.agents.graph import resume_run
from fund.agents.profiler import resume_profiler, run_profiler
from fund.audit import AgentRunRepository, MifidProfileRepository
from fund.config import settings

_MESSAGES = [{"role": "user", "content": "questionnaire ..."}]


# --- helpers ----------------------------------------------------------------


def _resume_model() -> ScriptedProfilerModel:
    """A model for the resume leg only.

    On resume the answers are recovered from the audit trail — no ``structured``
    normalisation pass runs — so only ``chat_responses`` are consumed: after the
    resumed tool (approve) or rejection, the agent calls the model once for a
    closing message. Two replies give a harmless buffer.
    """
    return ScriptedProfilerModel(chat_responses=[final_reply(), final_reply()])


def _run_to_gate(
    model: ScriptedProfilerModel,
    db_session,
    saver: MemorySaver,
    store: InMemoryStore,
    *,
    portfolio_id: uuid.UUID,
):
    return run_profiler(
        model,
        _MESSAGES,
        portfolio_id=portfolio_id,
        session=db_session,
        checkpointer=saver,
        store=store,
    )


def _decisions(db_session, run_id) -> list[AgentDecision]:
    return db_session.query(AgentDecision).filter(AgentDecision.run_id == run_id).all()


# --- approve: fresh agent persists v1 + caches the Store + completes ---------


def test_resume_profiler_approve_persists_and_completes(db_session) -> None:
    pid = uuid.uuid4()
    saver, store = MemorySaver(), InMemoryStore()

    run = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        saver,
        store,
        portfolio_id=pid,
    )
    # Paused at the gate: nothing is written until the adviser resumes.
    assert run.interrupt is not None
    assert MifidProfileRepository(db_session).get_active(pid) is None

    # A FRESH model/agent resumes — no reliance on the live ProfilerRun handle.
    resume_profiler(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_resume_model(),
    )

    active = MifidProfileRepository(db_session).get_active(pid)
    assert active is not None
    assert active.version == 1
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.finished_at is not None
    # The reconstructed ConstraintSet is cached in the Store for the rebalance path.
    assert store.get((str(pid),), settings.constraint_set_store_key) is not None
    # The save step is recorded as a HITL approve (via the resumed tool).
    saves = [d for d in _decisions(db_session, run.run_id) if d.step == "save_profile"]
    assert len(saves) == 1
    assert saves[0].hitl_decision.get("decision") == "approve"


# --- reject: no profile, no Store cache, rejected ---------------------------


def test_resume_profiler_reject_persists_nothing(db_session) -> None:
    pid = uuid.uuid4()
    saver, store = MemorySaver(), InMemoryStore()

    run = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        saver,
        store,
        portfolio_id=pid,
    )
    resume_profiler(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_resume_model(),
    )

    assert MifidProfileRepository(db_session).get_active(pid) is None
    assert store.get((str(pid),), settings.constraint_set_store_key) is None
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "rejected"
    assert any(
        d.hitl_decision and d.hitl_decision.get("decision") == "reject"
        for d in _decisions(db_session, run.run_id)
    )


# --- the ConstraintSet is re-derived from the audited answers, not persisted -


def test_resume_profiler_reconstructs_constraint_set_from_audit(db_session) -> None:
    pid = uuid.uuid4()
    saver, store = MemorySaver(), InMemoryStore()

    run = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        saver,
        store,
        portfolio_id=pid,
    )
    # The mapping is a pure function of the answers; resume must reproduce exactly
    # what run_profiler computed in-process (nothing was persisted before the gate).
    original_cs = run.constraint_set.model_dump(mode="json")

    resume_profiler(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_resume_model(),
    )

    active = MifidProfileRepository(db_session).get_active(pid)
    assert active.constraint_set == original_cs


# --- parity: ProfilerRun.resume and resume_profiler leave identical state ----


def test_resume_profiler_matches_profilerrun_resume(db_session) -> None:
    # Portfolio A — resumed via the live in-process handle (ProfilerRun.resume).
    pid_a = uuid.uuid4()
    run_a = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid_a)),
        db_session,
        MemorySaver(),
        InMemoryStore(),
        portfolio_id=pid_a,
    )
    run_a.resume("approve")

    # Portfolio B — resumed via the rebuild-to-resume path (fresh agent).
    pid_b = uuid.uuid4()
    saver_b, store_b = MemorySaver(), InMemoryStore()
    run_b = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid_b)),
        db_session,
        saver_b,
        store_b,
        portfolio_id=pid_b,
    )
    resume_profiler(
        run_b.run_id,
        "approve",
        session=db_session,
        checkpointer=saver_b,
        store=store_b,
        model=_resume_model(),
    )

    repo = AgentRunRepository(db_session)
    row_a, row_b = repo.get_run(run_a.run_id), repo.get_run(run_b.run_id)
    assert row_a.status == row_b.status == "completed"
    prof = MifidProfileRepository(db_session)
    assert prof.get_active(pid_a).version == prof.get_active(pid_b).version == 1


# --- guards -----------------------------------------------------------------


def test_resume_profiler_unknown_run_id_raises(db_session) -> None:
    with pytest.raises(LookupError):
        resume_profiler(
            uuid.uuid4(),
            "approve",
            session=db_session,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
            model=_resume_model(),
        )


def test_resume_profiler_rejects_unknown_decision(db_session) -> None:
    with pytest.raises(ValueError, match="approve"):
        resume_profiler(
            uuid.uuid4(),
            "edit",
            session=db_session,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
            model=_resume_model(),
        )


# --- resume_run dispatches by the run's step --------------------------------


def test_resume_run_dispatches_profiler_run_to_resume_profiler(db_session) -> None:
    pid = uuid.uuid4()
    saver, store = MemorySaver(), InMemoryStore()

    run = _run_to_gate(
        make_model(make_answers(), portfolio_id=str(pid)),
        db_session,
        saver,
        store,
        portfolio_id=pid,
    )
    # A profiler run (optimizer_config step == "profiler") must route to
    # resume_profiler — NOT resume_fund (which would hunt a non-existent
    # place_orders gate / active ConstraintSet and fail).
    resume_run(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_resume_model(),
    )

    assert MifidProfileRepository(db_session).get_active(pid).version == 1
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


def test_resume_run_unknown_run_id_raises(db_session) -> None:
    with pytest.raises(LookupError):
        resume_run(
            uuid.uuid4(),
            "approve",
            session=db_session,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
            model=_resume_model(),
        )
