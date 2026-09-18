"""T7.6 — ``run_fund`` + ``FundRun``: the one complete paper-run path.

Drives ``run_fund`` with a fully-scripted, network-free chat model
(:mod:`_fund_fakes`): the PM deep agent delegates economist → allocator → risk →
executor, the allocator runs the **real** bound ``optimize_portfolio`` (skfolio
computes the weights over a seeded SQLite panel — never the LLM), and the run
pauses at the ``place_orders`` HITL gate. ``FundRun.resume`` either commits the
paper ticket (approve → completed) or discards it (reject → rejected). Zero live
LLM, zero network — a ``MemorySaver`` + ``InMemoryStore`` + the in-memory
``db_session``.

Pins the SPEC-§4d contract: pause at the gate, approve/reject outcomes, the
missing-ConstraintSet guard, the blocking risk gate (executor un-invoked, no
ticket), the delegation round cap (incomplete), and the reproducibility record
(seed + temperature=0 + 3y lookback).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import uuid

import pytest
from _fund_fakes import ScriptedFundModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, AgentRun, PaperOrder
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.agents.graph import FundRun, run_fund
from fund.agents.toolsets import RunContext
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice
from fund.schemas.mandate import PortfolioMandate, RunTriggers
from fund.tools.optimize import optimize_portfolio

_START = dt.date(2024, 1, 1)
_N_DAYS = 40
_ASOF = dt.date(2024, 1, 30)  # decision bar (index 29); bars 30..39 are future
_UNIVERSE = ["AAA", "BBB", "CCC"]
# A UUID with hex letters: an all-digit UUID gets coerced to a float by SQLite's
# numeric affinity when a UUID column round-trips through ``refresh``.
_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_SERIES = {"AAA": (100.0, 0.0), "BBB": (50.0, 1.3), "CCC": (25.0, 2.6)}


def _close_on(start: float, phase: float, i: int) -> float:
    """Deterministic close for day ``i``: a phased oscillation off ``start``."""
    return round(start * (1.0 + 0.02 * math.sin(0.5 * i + phase)), 6)


def _seed_panel(db_session) -> None:
    """Seed an ``_N_DAYS`` x 3 close panel extending past ``_ASOF``."""
    for ticker, (start, phase) in _SERIES.items():
        ex = Exchange(name=f"EX-{ticker}")
        db_session.add(ex)
        db_session.flush()
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=ex.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(inst)
        db_session.flush()
        for i in range(_N_DAYS):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(start, phase, i),
                    volume=1000,
                )
            )
    db_session.flush()


def _constraint_set() -> ConstraintSet:
    return ConstraintSet(
        portfolio_id=str(_PORTFOLIO_ID),
        base_currency="EUR",
        a_gamma=2.5,
        objective=ObjectiveChoice.GROWTH,
        risk_measure=RiskMeasureChoice.VARIANCE,
        beta=0.95,
        nu1=0.05,
        nu2=0.10,
        nu3=0.20,
        horizon=Horizon.LONG,
    )


def _mandate() -> PortfolioMandate:
    from decimal import Decimal

    return PortfolioMandate(
        portfolio_id=str(_PORTFOLIO_ID),
        capital=Decimal("100000"),
        base_currency="EUR",
        drift_l1_threshold=0.1,
        triggers=RunTriggers(cron=True, drift=True),
    )


def _expected_weights(db_session) -> dict[str, float]:
    """The optimizer's weights the allocator will produce — computed the same way
    (default bounds), so the ticket / audit / finalised run must all match these."""
    result = optimize_portfolio(db_session, _ASOF, _UNIVERSE)
    assert result["ok"] is True
    return result["data"]["weights"]


def _store_with_cs() -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(
        store, _constraint_set(), store_key=settings.constraint_set_store_key
    )
    return store


def _run(model, db_session, *, store, config=settings) -> FundRun:
    return run_fund(
        model,
        _mandate(),
        portfolio_id=_PORTFOLIO_ID,
        asof=_ASOF,
        session=db_session,
        checkpointer=MemorySaver(),
        store=store,
        config=config,
    )


def _paper_orders(db_session) -> list[PaperOrder]:
    return db_session.query(PaperOrder).all()


def _decisions(db_session, run_id) -> list[AgentDecision]:
    return db_session.query(AgentDecision).filter(AgentDecision.run_id == run_id).all()


# --- pause at the HITL gate --------------------------------------------------


def test_pauses_at_place_orders_gate(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected)

    run = _run(model, db_session, store=_store_with_cs())

    assert isinstance(run, FundRun)
    assert run.status == "paused"
    assert run.interrupt is not None
    # The gate paused: no paper ticket is written until the adviser resumes.
    assert _paper_orders(db_session) == []
    # The load-bearing weights came from the optimizer, captured before the pause.
    assert run.weights == expected
    assert run.constraint_set == _constraint_set()


def test_interrupt_surfaces_the_place_orders_action(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected)

    run = _run(model, db_session, store=_store_with_cs())

    action = run.interrupt["action_requests"][0]
    assert action["name"] == "place_orders"


# --- approve → paper ticket + completed --------------------------------------


def test_approve_writes_ticket_and_completes(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected)

    run = _run(model, db_session, store=_store_with_cs())
    run.resume("approve")

    # (1) exactly one paper ticket, carrying the optimizer's weights.
    orders = _paper_orders(db_session)
    assert len(orders) == 1
    assert orders[0].weights == expected
    # (2) the run is finalised completed with those same optimizer weights.
    finalized = AgentRunRepository(db_session).get_run(run.run_id)
    assert finalized.status == "completed"
    assert finalized.weights == expected
    assert math.isclose(sum(finalized.weights.values()), 1.0, abs_tol=1e-6)
    # (3) the executor's execution decision + HITL approve is on the audit trail.
    executor = [
        d
        for d in _decisions(db_session, run.run_id)
        if d.agent == "executor" and d.step == "place_orders"
    ]
    assert len(executor) == 1
    assert executor[0].hitl_decision == {"decision": "approve"}


# --- reject → no order + rejected + HITL recorded ----------------------------


def test_reject_places_no_order_and_records_hitl(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected)

    run = _run(model, db_session, store=_store_with_cs())
    run.resume("reject")

    # No ticket was written — the adviser declined the commit.
    assert _paper_orders(db_session) == []
    finalized = AgentRunRepository(db_session).get_run(run.run_id)
    assert finalized.status == "rejected"
    assert finalized.weights == {}
    # The HITL rejection is recorded in the audit trail.
    assert any(
        d.hitl_decision and d.hitl_decision.get("decision") == "reject"
        for d in _decisions(db_session, run.run_id)
    )


# --- missing ConstraintSet → RuntimeError routing to the profiler ------------


def test_missing_constraint_set_raises(db_session) -> None:
    _seed_panel(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights={"AAA": 1.0})

    # An empty store has no active ConstraintSet for the portfolio.
    with pytest.raises(RuntimeError, match="profiler"):
        _run(model, db_session, store=InMemoryStore())

    # Fail-fast before any run row is opened.
    assert db_session.query(AgentRun).count() == 0


def test_missing_store_raises(db_session) -> None:
    _seed_panel(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights={"AAA": 1.0})

    with pytest.raises(RuntimeError):
        _run(model, db_session, store=None)


# --- blocking risk gate → executor un-invoked, no ticket ---------------------


def test_risk_violation_skips_executor_and_places_no_order(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected, risk_passes=False)

    run = _run(model, db_session, store=_store_with_cs())

    # The blocking gate stopped the pipeline before the HITL commit.
    assert run.status == "incomplete"
    assert run.interrupt is None
    assert _paper_orders(db_session) == []
    # The executor never proposed a rebalance — no executor decision was logged.
    assert not [d for d in _decisions(db_session, run.run_id) if d.agent == "executor"]
    finalized = AgentRunRepository(db_session).get_run(run.run_id)
    assert finalized.status == "incomplete"


# --- delegation round cap → incomplete + surfaced ----------------------------


def test_round_cap_finalizes_incomplete(db_session) -> None:
    _seed_panel(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights={"AAA": 1.0}, overrun=True)
    # A low recursion limit makes the PM's runaway re-delegation trip the cap fast.
    capped = dataclasses.replace(settings, recursion_limit=8)

    run = _run(model, db_session, store=_store_with_cs(), config=capped)

    assert run.status == "incomplete"
    assert run.interrupt is None
    assert _paper_orders(db_session) == []
    finalized = AgentRunRepository(db_session).get_run(run.run_id)
    assert finalized.status == "incomplete"
    # The round-cap outcome is surfaced for the adviser (D22).
    assert any(
        d.hitl_decision and d.hitl_decision.get("reason") == "round_cap"
        for d in _decisions(db_session, run.run_id)
    )


# --- reproducibility: seed + temperature=0 + 3y lookback (D31/D18) -----------


def test_records_seed_temperature_and_lookback(db_session) -> None:
    _seed_panel(db_session)
    expected = _expected_weights(db_session)
    model = ScriptedFundModel(universe=_UNIVERSE, weights=expected)

    run = _run(model, db_session, store=_store_with_cs())

    created = AgentRunRepository(db_session).get_run(run.run_id)
    assert created.seed is not None
    assert created.optimizer_config["temperature"] == 0.0
    assert created.optimizer_config["lookback_days"] == 756
    # The run-config lookback matches the RunContext default it is threaded through.
    ctx = RunContext(
        session=db_session,
        asof=_ASOF,
        store=None,
        config=settings,
        run_id=run.run_id,
        portfolio_id=_PORTFOLIO_ID,
    )
    assert ctx.lookback_days == created.optimizer_config["lookback_days"]
