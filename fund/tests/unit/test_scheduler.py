"""Autonomy enforcement in the fund scheduler (T7).

``_autonomy_blocks_round`` reads the ``fund_autonomy_state`` posture (lazy-imported
from the submodule, so ``import fund.scheduler`` stays agent-stack-free) and reports
whether a FROZEN fund **or** FROZEN portfolio should block a round. ``_drive_rebalance``
calls it after the paused-run guard and skips a frozen portfolio without opening a run.
Driven headless over the SQLite ``db_session`` harness — no model, no network.
"""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

import pytest
from _fund_fakes import ASOF, PORTFOLIO_ID, make_constraint_set, make_mandate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

import fund.agents.graph as graph_mod
from fund import scheduler
from fund.audit import (
    AgentRunRepository,
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
    MandateRepository,
    put_constraint_set,
)
from fund.config import settings


def _persistence_with_cs():
    """A persistence stand-in whose store carries the fake portfolio's ConstraintSet."""
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    return SimpleNamespace(store=store, saver=MemorySaver())


def test_autonomy_blocks_round_true_when_portfolio_frozen(db_session):
    """A FROZEN portfolio blocks its own round."""
    AutonomyRepository(db_session).set_state(
        AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN, reason="halt"
    )
    assert scheduler._autonomy_blocks_round(db_session, PORTFOLIO_ID) is True


def test_autonomy_blocks_round_true_when_fund_frozen(db_session):
    """A FROZEN fund blocks every portfolio's round."""
    AutonomyRepository(db_session).set_state(
        AutonomyScope.FUND, None, AutonomyState.FROZEN
    )
    assert scheduler._autonomy_blocks_round(db_session, PORTFOLIO_ID) is True


def test_autonomy_blocks_round_false_when_armed(db_session):
    """An armed fund and armed portfolio do not block a round."""
    assert scheduler._autonomy_blocks_round(db_session, PORTFOLIO_ID) is False


def test_drive_rebalance_skips_a_frozen_portfolio(db_session, monkeypatch):
    """A frozen portfolio is skipped — ``run_fund`` is never driven — even with a full
    mandate + ConstraintSet that would otherwise reach it."""
    MandateRepository(db_session).upsert(make_mandate())
    AutonomyRepository(db_session).set_state(
        AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN, reason="halt"
    )
    driven: list[dict] = []
    monkeypatch.setattr(graph_mod, "run_fund", lambda *a, **k: driven.append(k))

    scheduler._drive_rebalance(
        PORTFOLIO_ID,
        ASOF,
        session=db_session,
        persistence=_persistence_with_cs(),
        model=object(),
        fallback=None,
    )

    assert driven == []  # frozen → no run driven
    assert AgentRunRepository(db_session).list_runs_for_portfolio(PORTFOLIO_ID) == []


def test_drive_rebalance_runs_an_armed_portfolio(db_session, monkeypatch):
    """Control: with the same setup but armed, ``run_fund`` is reached and driven."""
    MandateRepository(db_session).upsert(make_mandate())
    driven: list[dict] = []

    def _fake_run_fund(*a, **k):
        driven.append(k)
        return SimpleNamespace(run_id=uuid.uuid4(), status="paused")

    monkeypatch.setattr(graph_mod, "run_fund", _fake_run_fund)

    scheduler._drive_rebalance(
        PORTFOLIO_ID,
        ASOF,
        session=db_session,
        persistence=_persistence_with_cs(),
        model=object(),
        fallback=None,
    )

    assert len(driven) == 1  # armed → the rebalance is driven


@pytest.mark.parametrize(
    ("status", "note"),
    [
        ("paused", "at HITL gate"),
        ("rejected", "no approval gate"),
        ("incomplete", "no approval gate"),
    ],
)
def test_drive_rebalance_logs_the_outcome_the_round_reached(
    db_session, monkeypatch, caplog, status, note
):
    """Only a paused round sits at a gate; a round a freeze halted (or one that ended
    with no ticket) must not be logged as sitting 'at HITL gate'."""
    MandateRepository(db_session).upsert(make_mandate())
    monkeypatch.setattr(
        graph_mod,
        "run_fund",
        lambda *a, **k: SimpleNamespace(run_id=uuid.uuid4(), status=status),
    )

    with caplog.at_level(logging.INFO, logger="fund.scheduler"):
        scheduler._drive_rebalance(
            PORTFOLIO_ID,
            ASOF,
            session=db_session,
            persistence=_persistence_with_cs(),
            model=object(),
            fallback=None,
        )

    message = next(
        r.getMessage() for r in caplog.records if f"status={status}" in r.getMessage()
    )
    assert note in message
    assert ("HITL gate" in message) == (status == "paused")
