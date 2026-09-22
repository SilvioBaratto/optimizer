"""Task 4 — cost caps: the PM round-cap middleware + the profiler recursion guard.

Two deterministic, network-free caps land here:

* ``PMRoundCapMiddleware`` bounds the PM's delegation rounds. A runaway PM that
  keeps re-delegating without committing is jumped straight to ``end`` once
  ``max_pm_rounds`` model rounds elapse, and ``run_fund`` finalises the run
  ``incomplete`` with reason ``"pm_round_cap"`` — a deterministic cap kept
  **distinct** from both the langgraph ``recursion_limit`` round-cap
  (``"round_cap"``) and the blocking-risk early exit (``"no_order"``).
* ``run_profiler`` now carries ``config.recursion_limit`` on its ``thread_config``
  (``run_fund`` already did; the profiler matches).

The round-cap run drives the scripted ``overrun`` PM (never commits) under a LOW
``max_pm_rounds`` and the default (high) ``recursion_limit`` so the *middleware*
cap trips first — the very ``can_jump_to`` edge the cap depends on. Zero live LLM,
zero network (a ``MemorySaver`` + ``InMemoryStore`` + the in-memory ``db_session``).
"""

from __future__ import annotations

import dataclasses
import uuid

from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    make_constraint_set,
    make_mandate,
    seed_panel,
)
from _profiler_fakes import make_answers, make_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, PaperOrder

from fund.agents.graph import run_fund
from fund.agents.profiler import run_profiler
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.tools.optimize import optimize_portfolio


def _store_with_cs() -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    return store


def _run_fund(model, db_session, *, config):
    return run_fund(
        model,
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=MemorySaver(),
        store=_store_with_cs(),
        config=config,
    )


def _reasons(db_session, run_id) -> list[str]:
    return [
        d.hitl_decision.get("reason")
        for d in db_session.query(AgentDecision)
        .filter(AgentDecision.run_id == run_id)
        .all()
        if d.hitl_decision
    ]


# --- PM round cap → incomplete / pm_round_cap --------------------------------


def test_pm_round_cap_ends_run_incomplete(db_session) -> None:
    seed_panel(db_session)
    # Runaway PM (never commits) + a LOW pm-round cap under the default (high)
    # recursion_limit: the middleware cap must trip before the langgraph guard, so
    # the run *returns* incomplete instead of raising GraphRecursionError.
    model = ScriptedFundModel(universe=UNIVERSE, weights={"AAA": 1.0}, overrun=True)
    capped = dataclasses.replace(settings, max_pm_rounds=3)

    run = _run_fund(model, db_session, config=capped)

    assert run.status == "incomplete"
    assert run.interrupt is None
    # The cap left nothing to commit — no paper ticket.
    assert db_session.query(PaperOrder).all() == []


def test_pm_round_cap_surfaces_distinct_reason(db_session) -> None:
    seed_panel(db_session)
    model = ScriptedFundModel(universe=UNIVERSE, weights={"AAA": 1.0}, overrun=True)
    capped = dataclasses.replace(settings, max_pm_rounds=3)

    run = _run_fund(model, db_session, config=capped)

    finalized = AgentRunRepository(db_session).get_run(run.run_id)
    assert finalized.status == "incomplete"
    reasons = _reasons(db_session, run.run_id)
    # Surfaced as "pm_round_cap" — distinct from the recursion-limit "round_cap"
    # (the middleware ended it, not the langgraph guard) and from "no_order".
    assert "pm_round_cap" in reasons
    assert "round_cap" not in reasons
    assert "no_order" not in reasons


def test_below_cap_still_reaches_gate(db_session) -> None:
    # A happy-path run makes ~5 PM rounds; the default cap (10) never trips, so the
    # middleware must not stop a legitimate run before the place_orders gate.
    seed_panel(db_session)
    weights = optimize_portfolio(db_session, ASOF, UNIVERSE)["data"]["weights"]
    model = ScriptedFundModel(universe=UNIVERSE, weights=weights)

    run = _run_fund(model, db_session, config=settings)

    assert run.status == "paused"
    assert run.interrupt is not None


# --- profiler recursion_limit ------------------------------------------------


def test_profiler_thread_config_carries_recursion_limit(db_session) -> None:
    pid = uuid.uuid4()
    model = make_model(make_answers(), portfolio_id=str(pid))

    run = run_profiler(
        model,
        [{"role": "user", "content": "questionnaire ..."}],
        portfolio_id=pid,
        session=db_session,
        checkpointer=MemorySaver(),
    )

    assert run.thread_config["recursion_limit"] == settings.recursion_limit
