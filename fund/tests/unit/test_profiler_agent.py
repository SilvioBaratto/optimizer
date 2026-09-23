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
from portopt_db.models import AgentRun, MifidProfile

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


def test_esg_breach_finalizes_run_no_orphan_pending(db_session):
    pid = uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))  # empties the universe
    model = make_model(answers, portfolio_id=str(pid))

    with pytest.raises(SuitabilityBreachError):
        _run(model, db_session, portfolio_id=pid)

    runs = db_session.query(AgentRun).filter(AgentRun.portfolio_id == pid).all()
    assert len(runs) == 1  # the pending run was created
    run = runs[0]
    assert run.status == "blocked"  # finalised to a terminal status
    assert run.status != "pending"
    assert run.finished_at is not None  # no dangling open run
    steps = [d.step for d in run.decisions]
    assert "normalize_answers" in steps  # LLM audit trail preserved
    assert "suitability_breach" in steps  # breach recorded


# --- build_profiler_agent: create_deep_agent kwargs (spy, no LLM) ------------


def test_build_profiler_agent_wires_the_theory_backend_and_skill(tmp_path, monkeypatch):
    """The profiler (step 0) must run under the theory-staged, virtual-mode backend
    and load its ``mifid-profiling`` skill through it (Risk R1) — the same wiring the
    PM gets — so the consultation protocol can reach ``optimizer-theory/``."""
    import deepagents
    from deepagents.backends import FilesystemBackend

    from fund.agents import profiler as profiler_mod
    from fund.agents.prompts import PROFILER_SYSTEM_PROMPT
    from fund.agents.skills import skill_sources

    captured: dict[str, object] = {}

    def _spy(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "PROFILER_AGENT"

    monkeypatch.setattr(deepagents, "create_deep_agent", _spy)
    # Avoid staging the real theory tree: a real virtual-mode backend at a temp dir
    # keeps the virtual_mode assertion honest without touching fund/.runtime.
    fake_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(profiler_mod, "build_backend", lambda config: fake_backend)

    agent = profiler_mod.build_profiler_agent(
        object(), [], checkpointer="CKPT", store="STORE"
    )

    assert agent == "PROFILER_AGENT"
    assert captured["system_prompt"] == PROFILER_SYSTEM_PROMPT
    # Root-relative profiler skill + the virtual-mode theory backend.
    assert captured["skills"] == skill_sources("profiler")
    assert captured["backend"] is fake_backend
    assert captured["backend"].virtual_mode is True
    # The always-on HITL save gate + persistence handles are still threaded through.
    assert captured["interrupt_on"] == {"save_profile": True}
    assert captured["checkpointer"] == "CKPT"
    assert captured["store"] == "STORE"
