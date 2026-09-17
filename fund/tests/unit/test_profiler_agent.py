"""Task 7 — the ``deepagents`` profiler agent (mock model), normalisation + persist.

Drives ``run_profiler`` with a fully-scripted chat model (``_profiler_fakes``): the
agent normalises free-text answers into a typed ``MiFIDAnswers`` via
``structured_call`` (retry once on malformed, then a fallback model), runs the
deterministic mapping (``build_constraint_set`` — never an LLM-emitted knob), and,
on adviser approval, persists the ``mifid_profiles`` row + the active
``ConstraintSet`` in the Store + the ``agent_runs`` audit trail. Zero live LLM,
zero network — a ``MemorySaver`` + ``InMemoryStore`` + the in-memory ``db_session``.

The always-on HITL gate itself is covered in ``test_profiler_hitl.py``.
"""

from __future__ import annotations

import uuid

import pytest
from _profiler_fakes import (
    ScriptedProfilerModel,
    final_reply,
    make_answers,
    make_model,
    save_tool_call,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import MifidProfile

from fund.agents.profiler import SuitabilityBreachError, run_profiler
from fund.audit import MifidProfileRepository, resolve_constraint_set
from fund.schemas import ConstraintSet, ConstraintSetRef
from fund.schemas.enums import GicsSector, RiskToleranceBand

_MESSAGES = [{"role": "user", "content": "Here are my questionnaire answers ..."}]


def _run(model, db_session, *, portfolio_id, store=None, fallback=None):
    return run_profiler(
        model,
        _MESSAGES,
        portfolio_id=portfolio_id,
        session=db_session,
        checkpointer=MemorySaver(),
        store=store,
        fallback=fallback,
    )


# --- normalisation via structured_call --------------------------------------


def test_normalises_free_text_into_typed_answers(db_session):
    pid = uuid.uuid4()
    answers = make_answers()
    model = make_model(answers, portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    assert run.answers == answers
    assert model.with_structured_output_calls >= 1
    assert model.structured_invoke_calls == 1  # one clean normalisation


def test_maps_answers_to_a_valid_constraint_set(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    assert isinstance(run.constraint_set, ConstraintSet)
    assert run.constraint_set.portfolio_id == str(pid)
    # Balanced band from the default answers (appetite 0.5 → a_gamma 5.0).
    assert run.constraint_set.a_gamma == 5.0
    assert run.suitability.band is RiskToleranceBand.BALANCED
    # The mapping is deterministic — the LLM emits no knob.
    run.constraint_set.to_mean_risk_config()


def test_retries_normalisation_once_then_succeeds(db_session):
    pid = uuid.uuid4()
    answers = make_answers()
    # First structured attempt is malformed (missing fields) → retried once.
    model = make_model(
        answers, portfolio_id=str(pid), structured_prefix=({"nope": True},)
    )

    run = _run(model, db_session, portfolio_id=pid)

    assert run.answers == answers
    assert model.structured_invoke_calls == 2  # malformed, then valid


def test_falls_back_when_primary_normalisation_is_exhausted(db_session):
    pid = uuid.uuid4()
    answers = make_answers()
    # Primary normalisation always malformed → fallback supplies the answers; the
    # primary still drives the agent afterwards (fallback is structured-call only).
    primary = ScriptedProfilerModel(
        structured_outcomes=[{"bad": 1}, {"bad": 2}],  # both malformed
        chat_responses=[save_tool_call(str(pid)), final_reply()],
    )
    fallback = ScriptedProfilerModel(structured_outcomes=[answers], chat_responses=[])

    run = _run(primary, db_session, portfolio_id=pid, fallback=fallback)

    assert run.answers == answers
    assert primary.structured_invoke_calls == 2  # retries=1 → two primary tries
    assert fallback.structured_invoke_calls == 1  # fallback tried exactly once


# --- persistence on approval ------------------------------------------------


def test_approve_persists_profile_store_and_audit(db_session):
    pid = uuid.uuid4()
    store = InMemoryStore()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=store)
    run.resume("approve")

    # (1) the mifid_profiles row is the durable system of record.
    profile = MifidProfileRepository(db_session).get_active(pid)
    assert profile is not None
    assert profile.version == 1
    assert profile.status == "active"
    # (2) the persisted ConstraintSet is the deterministically-mapped one.
    assert profile.constraint_set == run.constraint_set.model_dump(mode="json")
    # (3) the active ConstraintSet resolves through a Phase-4 ConstraintSetRef.
    ref = ConstraintSetRef(portfolio_id=str(pid), store_key=profile.store_key)
    assert resolve_constraint_set(store, ref) == run.constraint_set


def test_approve_without_store_still_persists_the_row(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=None)
    run.resume("approve")

    assert MifidProfileRepository(db_session).get_active(pid) is not None


# --- ESG / legal hard block -------------------------------------------------


def test_esg_all_sectors_excluded_hard_blocks_before_persist(db_session):
    pid = uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))  # empties the universe
    model = make_model(answers, portfolio_id=str(pid))

    with pytest.raises(SuitabilityBreachError):
        _run(model, db_session, portfolio_id=pid)

    # Nothing was persisted — the breach hard-blocks outright.
    assert db_session.query(MifidProfile).count() == 0
