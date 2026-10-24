"""The ``deepagents`` profiler agent (mock model), normalisation + persist.

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

import dataclasses
import datetime as dt
import uuid
from collections.abc import Generator
from contextlib import contextmanager

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
from portopt_db.models import AgentRun, Base, MifidProfile
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fund.agents.profiler import (
    SuitabilityBreachError,
    resume_profiler,
    run_mapping,
    run_profiler,
)
from fund.audit import (
    AgentRunRepository,
    MifidProfileRepository,
    resolve_constraint_set,
)
from fund.config import settings
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
    assert model.structured_invoke_calls == 1


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
    assert fallback.structured_invoke_calls == 1


# --- typed questionnaire input is never re-interpreted by the LLM ------------


def _declared_answers():
    # What the adviser typed into the form: an ESG hard-block, a tight loss
    # capacity and a cautious Likert profile.
    return make_answers(likert=(2, 3), max_loss=0.10, exclusions=(GicsSector.ENERGY,))


def _drifted_answers():
    # A schema-valid but wrong re-extraction: the exclusion is dropped, the loss
    # capacity quadruples and the attitude inflates.
    return make_answers(likert=(5, 6), max_loss=0.40)


def test_typed_answers_json_skips_the_llm_and_is_used_verbatim(db_session):
    pid = uuid.uuid4()
    declared = _declared_answers()
    model = make_model(_drifted_answers(), portfolio_id=str(pid))

    run = run_profiler(
        model,
        declared.model_dump_json(),
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
    )

    assert run.answers == declared
    assert model.structured_invoke_calls == 0
    assert run.constraint_set.esg.exclusions == (GicsSector.ENERGY,)


def test_typed_answers_are_what_the_approved_profile_persists(db_session):
    pid = uuid.uuid4()
    declared = _declared_answers()
    model = make_model(_drifted_answers(), portfolio_id=str(pid))

    run = run_profiler(
        model,
        declared.model_dump_json(),
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
        store=InMemoryStore(),
    )
    run.resume("approve")

    profile = MifidProfileRepository(db_session).get_active(pid)
    assert profile.questionnaire == declared.model_dump(mode="json")


def test_non_answer_text_still_goes_through_the_llm(db_session):
    pid = uuid.uuid4()
    answers = make_answers()
    model = make_model(answers, portfolio_id=str(pid))

    run = run_profiler(
        model,
        "I am fairly relaxed about risk, my horizon is ten years.",
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
    )

    assert run.answers == answers
    assert model.structured_invoke_calls == 1


def test_gate_description_shows_all_four_pillars_of_the_answers(db_session):
    pid = uuid.uuid4()
    answers = _declared_answers()
    model = make_model(answers, portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    description = run.interrupt["action_requests"][0]["description"]
    assert "ESG exclusions: energy" in description
    assert "max 1y loss 10%" in description
    assert "knowledge informed" in description
    assert "goal growth" in description
    assert "horizon long" in description
    assert "likert 2, 3" in description


def test_gate_description_shows_the_base_currency_that_will_be_written(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(currency="EUR"), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid)

    description = run.interrupt["action_requests"][0]["description"]
    assert "base currency EUR" in description


def test_gate_description_flags_an_overridden_base_currency(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(currency="EUR"), portfolio_id=str(pid))

    run = run_profiler(
        model,
        _MESSAGES,
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
        base_currency="USD",
    )

    description = run.interrupt["action_requests"][0]["description"]
    assert "base currency USD" in description
    assert "answers say EUR" in description
    assert run.constraint_set.base_currency == "USD"


# --- an agent that never reaches the approval gate --------------------------


def _prose_only_model(answers):
    # The model answers in prose and never calls ``save_profile``.
    return ScriptedProfilerModel(
        structured_outcomes=[answers],
        chat_responses=[AIMessage(content="Understood, I will wait for approval.")],
    )


def test_agent_that_never_calls_save_profile_is_finalized_incomplete(db_session):
    pid = uuid.uuid4()

    run = _run(_prose_only_model(make_answers()), db_session, portfolio_id=pid)

    row = db_session.get(AgentRun, run.run_id)
    assert row.status == "incomplete"  # not left 'pending' forever
    assert row.finished_at is not None
    assert run.interrupt is None
    assert run.status == "incomplete"
    assert AgentRunRepository(db_session).list_paused_runs(pid) == []
    assert MifidProfileRepository(db_session).get_active(pid) is None


def test_gate_reaching_run_reports_paused_status(db_session):
    pid = uuid.uuid4()

    run = _run(
        make_model(make_answers(), portfolio_id=str(pid)), db_session, portfolio_id=pid
    )

    assert run.status == "paused"


def test_resuming_a_run_that_never_reached_the_gate_is_refused(db_session):
    pid = uuid.uuid4()
    saver, store = MemorySaver(), InMemoryStore()
    run = run_profiler(
        _prose_only_model(make_answers()),
        _MESSAGES,
        portfolio_id=pid,
        session=db_session,
        checkpointer=saver,
        store=store,
    )

    with pytest.raises(ValueError, match="incomplete"):
        resume_profiler(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedProfilerModel(),
        )
    assert MifidProfileRepository(db_session).get_active(pid) is None


def test_save_profile_tool_description_says_the_call_raises_the_gate(db_session):
    from fund.agents.profiler import _make_save_profile

    answers = make_answers()
    constraint_set, suitability = run_mapping(answers, portfolio_id=str(uuid.uuid4()))
    tool = _make_save_profile(
        session=db_session,
        store=None,
        config=settings,
        answers=answers,
        constraint_set=constraint_set,
        suitability=suitability,
        run_id=uuid.uuid4(),
        portfolio_id=uuid.uuid4(),
    )

    # The system prompt tells the agent to call the tool now; a description that
    # says "call after adviser approval" contradicts it and invites withholding.
    assert "after adviser approval" not in tool.description
    assert "pauses" in tool.description


# --- per-provider structured-output method ----------------------------------


def test_profiler_defaults_to_function_calling_structured_output(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    _run(model, db_session, portfolio_id=pid)

    # Default config → normalisation uses ``function_calling``.
    assert model.structured_output_methods == ["function_calling"]


def test_profiler_forwards_configured_structured_output_method(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))
    config = dataclasses.replace(settings, structured_output_method="json_mode")

    run_profiler(
        model,
        _MESSAGES,
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
        config=config,
    )

    assert model.structured_output_methods == ["json_mode"]


# --- persistence on approval ------------------------------------------------


def test_approve_persists_profile_store_and_audit(committing_session):
    pid = uuid.uuid4()
    store = InMemoryStore()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, committing_session, portfolio_id=pid, store=store)
    run.resume("approve")
    committing_session.commit()  # the Store cache follows the row's commit

    # (1) the mifid_profiles row is the durable system of record.
    profile = MifidProfileRepository(committing_session).get_active(pid)
    assert profile is not None
    assert profile.version == 1
    assert profile.status == "active"
    # (2) the persisted ConstraintSet is the deterministically-mapped one.
    assert profile.constraint_set == run.constraint_set.model_dump(mode="json")
    # (3) the active ConstraintSet resolves through a ConstraintSetRef.
    ref = ConstraintSetRef(portfolio_id=str(pid), store_key=profile.store_key)
    assert resolve_constraint_set(store, ref) == run.constraint_set


def test_approve_without_store_still_persists_the_row(db_session):
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = _run(model, db_session, portfolio_id=pid, store=None)
    run.resume("approve")

    assert MifidProfileRepository(db_session).get_active(pid) is not None


# --- ESG / legal hard block -------------------------------------------------
#
# A breach rolls the session back to commit only the blocked trail, so these tests
# run on a real-commit engine: the shared ``db_session`` SAVEPOINT harness would
# have its outer transaction torn down by that rollback.


@pytest.fixture
def private_engine() -> Generator[Engine, None, None]:
    """A pristine in-memory DB whose commits are real (no outer SAVEPOINT)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@contextmanager
def _rollback_on_error_session(engine: Engine) -> Generator[Session, None, None]:
    """Mirror ``DatabaseManager.get_session``: roll back on any exception."""
    session = Session(bind=engine)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_esg_all_sectors_excluded_hard_blocks_before_persist(private_engine):
    pid = uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))  # empties the universe
    model = make_model(answers, portfolio_id=str(pid))

    with Session(bind=private_engine) as session:
        with pytest.raises(SuitabilityBreachError):
            _run(model, session, portfolio_id=pid)

        # Nothing was persisted — the breach hard-blocks outright.
        assert session.query(MifidProfile).count() == 0


def test_esg_breach_finalizes_run_no_orphan_pending(private_engine):
    pid = uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))  # empties the universe
    model = make_model(answers, portfolio_id=str(pid))

    with Session(bind=private_engine) as session:
        with pytest.raises(SuitabilityBreachError):
            _run(model, session, portfolio_id=pid)

        runs = session.query(AgentRun).filter(AgentRun.portfolio_id == pid).all()
        assert len(runs) == 1  # the pending run was created
        run = runs[0]
        assert run.status == "blocked"  # finalised to a terminal status
        assert run.finished_at is not None  # no dangling open run
        steps = [d.step for d in run.decisions]
        assert "normalize_answers" in steps  # LLM audit trail preserved
        assert "suitability_breach" in steps  # breach recorded


def test_blocked_breach_record_survives_the_callers_rollback(private_engine):
    pid = uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))  # empties the universe
    model = make_model(answers, portfolio_id=str(pid))

    with (
        pytest.raises(SuitabilityBreachError),
        _rollback_on_error_session(private_engine) as session,
    ):
        run_profiler(
            model,
            _MESSAGES,
            portfolio_id=pid,
            session=session,
            checkpointer=MemorySaver(),
        )

    with Session(bind=private_engine) as check:
        runs = check.query(AgentRun).filter(AgentRun.portfolio_id == pid).all()
        assert [r.status for r in runs] == ["blocked"]
        steps = [d.step for d in runs[0].decisions]
        assert "normalize_answers" in steps
        assert "suitability_breach" in steps


def test_breach_commits_the_blocked_trail_but_not_the_callers_staged_work(
    private_engine,
):
    pid, other = uuid.uuid4(), uuid.uuid4()
    answers = make_answers(exclusions=tuple(GicsSector))
    model = make_model(answers, portfolio_id=str(pid))

    with (
        pytest.raises(SuitabilityBreachError),
        _rollback_on_error_session(private_engine) as session,
    ):
        # Work the caller staged before the profiler ran (flushed, not committed).
        AgentRunRepository(session).create_run(
            portfolio_id=other,
            asof=dt.date(2026, 1, 2),
            seed=None,
            universe=[],
            optimizer_config={"step": "staged-by-caller"},
        )
        run_profiler(
            model,
            _MESSAGES,
            portfolio_id=pid,
            session=session,
            checkpointer=MemorySaver(),
        )

    with Session(bind=private_engine) as check:
        persisted = check.query(AgentRun).all()
        assert [(r.portfolio_id, r.status) for r in persisted] == [(pid, "blocked")]
        decisions = persisted[0].decisions
        assert [d.step for d in decisions] == [
            "normalize_answers",
            "suitability_breach",
        ]
        assert decisions[0].llm_response == answers.model_dump_json()


# --- build_profiler_agent: create_deep_agent kwargs (spy, no LLM) ------------


def test_build_profiler_agent_wires_the_theory_backend_and_skill(tmp_path, monkeypatch):
    """The profiler must run under the theory-staged, virtual-mode backend and load
    its ``mifid-profiling`` skill through it — the same wiring the PM gets — so the
    consultation protocol can reach ``optimizer-theory/``."""
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
    assert captured["skills"] == skill_sources("profiler")
    assert captured["backend"] is fake_backend
    assert captured["backend"].virtual_mode is True
    # The always-on HITL save gate + persistence handles are still threaded through.
    assert captured["interrupt_on"] == {"save_profile": True}
    assert captured["checkpointer"] == "CKPT"
    assert captured["store"] == "STORE"


def test_build_profiler_agent_applies_the_summary_override(tmp_path, monkeypatch):
    """``FUND_SUMMARY_*`` must bound the profiler's transcript, not just the PM's."""
    import deepagents
    from deepagents.backends import FilesystemBackend

    from fund.agents import profiler as profiler_mod

    captured: dict[str, object] = {}

    def _spy(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "PROFILER_AGENT"

    monkeypatch.setattr(deepagents, "create_deep_agent", _spy)
    fake_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(profiler_mod, "build_backend", lambda config: fake_backend)
    config = dataclasses.replace(
        settings, summary_token_threshold=24000, summary_messages_to_keep=4
    )

    profiler_mod.build_profiler_agent(
        ScriptedProfilerModel(), [], checkpointer="CKPT", store=None, config=config
    )

    summarizers = [
        m
        for m in captured.get("middleware", [])  # type: ignore[attr-defined]
        if m.name == "SummarizationMiddleware"
    ]
    assert len(summarizers) == 1
    assert summarizers[0]._lc_helper.trigger == ("tokens", 24000)
    assert summarizers[0]._lc_helper.keep == ("messages", 4)
    assert summarizers[0]._lc_helper.trim_tokens_to_summarize is None


def test_build_profiler_agent_keeps_deepagents_summarization_without_override(
    tmp_path, monkeypatch
):
    import deepagents
    from deepagents.backends import FilesystemBackend

    from fund.agents import profiler as profiler_mod

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda *a, **k: captured.update(k)
    )
    fake_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(profiler_mod, "build_backend", lambda config: fake_backend)

    profiler_mod.build_profiler_agent(
        ScriptedProfilerModel(), [], checkpointer="CKPT", store=None
    )

    assert not captured.get("middleware")


def test_extraction_prompt_tells_the_model_to_mark_unstated_data_not_invent_it():
    from fund.agents.profiler import _EXTRACTION_SYSTEM_PROMPT

    assert "unanswered" in _EXTRACTION_SYSTEM_PROMPT
    # The old blanket instruction forced a fabricated value for any omitted datum.
    assert "Fill every required field from the client's statements" not in (
        _EXTRACTION_SYSTEM_PROMPT
    )
