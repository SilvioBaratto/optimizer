"""A database error inside a frozen tool must not wipe the run's own transaction.

The bound agent tools run inside a SAVEPOINT (``_serialised``) on the run's shared
session, while the frozen ``fund.tools`` functions they call carry their own
``tool_envelope``. That inner envelope used to roll the whole session back, root
transaction included, taking the uncommitted ``agent_runs`` row and every decision
with it before the SAVEPOINT owner could recover.
"""

from __future__ import annotations

import datetime as dt
import threading
import uuid
from types import SimpleNamespace

import pytest
from portopt_db.models import AgentDecision, AgentRun
from sqlalchemy import func, select, text

from fund.agents.toolsets import _serialised
from fund.audit import AgentRunRepository
from fund.audit.orders_repository import OrderRepository, weights_hash
from fund.tools import _base
from fund.tools import orders as orders_mod
from fund.tools.orders import place_orders

_ASOF = dt.date(2026, 10, 1)
_WEIGHTS = {"AAA": 1.0}


@pytest.fixture
def raced_ticket(committing_session, monkeypatch):
    """A ticket another writer already committed, plus a lookup that misses it.

    Stands in for the ``paper_orders`` unique-key race: the find step sees nothing,
    the insert then collides.
    """
    pid = uuid.uuid4()
    OrderRepository(committing_session).create(
        portfolio_id=pid,
        asof=_ASOF,
        weights_hash=weights_hash(_WEIGHTS),
        weights=_WEIGHTS,
        fill_date=_ASOF,
        lines=[],
        notional=1.0,
        total_commission=0.0,
        total_slippage_cost=0.0,
    )
    committing_session.commit()
    monkeypatch.setattr(OrderRepository, "get_by_key", lambda self, **kw: None)
    monkeypatch.setattr(
        orders_mod,
        "_next_close",
        lambda repo, ticker, asof: orders_mod._Fill(
            date=_ASOF + dt.timedelta(days=1), close=10.0, unit="EUR"
        ),
    )
    return pid


def _open_run(session, pid):
    repo = AgentRunRepository(session)
    run = repo.create_run(
        portfolio_id=pid, asof=_ASOF, seed=1, universe=[], optimizer_config={}
    )
    repo.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response='{"weights":{"AAA":1.0}}',
    )
    return run.id


def _bound_place(ctx, pid):
    @_base.tool_envelope
    @_serialised
    def impl(context):
        """Mirror a bound tool: a frozen tool called inside the SAVEPOINT body."""
        return place_orders(context.session, _ASOF, _WEIGHTS, pid, base_currency="EUR")

    return impl(ctx)


class TestFrozenToolFailureInsideSerialisedBody:
    """The SAVEPOINT owner, not the inner envelope, recovers the session."""

    def test_run_row_and_decisions_survive_a_unique_key_collision(
        self, committing_session, raced_ticket
    ) -> None:
        """The uncommitted run row and its decision outlive the failed tool."""
        run_id = _open_run(committing_session, raced_ticket)
        ctx = SimpleNamespace(session=committing_session, lock=threading.RLock())

        result = _bound_place(ctx, raced_ticket)

        assert result["ok"] is False
        assert committing_session.get(AgentRun, run_id) is not None
        decisions = committing_session.execute(
            select(func.count()).select_from(AgentDecision)
        ).scalar_one()
        assert decisions == 1

    def test_session_is_usable_for_the_next_tool_after_the_collision(
        self, committing_session, raced_ticket
    ) -> None:
        """The session stays in its transaction and accepts the next statement."""
        _open_run(committing_session, raced_ticket)
        ctx = SimpleNamespace(session=committing_session, lock=threading.RLock())

        _bound_place(ctx, raced_ticket)

        assert committing_session.execute(text("SELECT 1")).scalar_one() == 1
        assert committing_session.in_transaction() is True

    def test_the_run_survives_a_commit_after_the_failed_tool(
        self, committing_session, raced_ticket
    ) -> None:
        """What the run wrote before the failure is what the later commit persists."""
        run_id = _open_run(committing_session, raced_ticket)
        ctx = SimpleNamespace(session=committing_session, lock=threading.RLock())

        _bound_place(ctx, raced_ticket)
        committing_session.commit()

        assert committing_session.get(AgentRun, run_id) is not None


class TestUnownedSessionStillRollsBack:
    """Outside a SAVEPOINT owner the envelope keeps rolling the session back."""

    def test_a_frozen_tool_called_without_a_savepoint_owner_recovers_the_session(
        self, committing_session, raced_ticket
    ) -> None:
        """A bare frozen-tool call has no one else to recover its failed flush."""
        result = place_orders(
            committing_session, _ASOF, _WEIGHTS, raced_ticket, base_currency="EUR"
        )

        assert result["ok"] is False
        assert committing_session.in_transaction() is False
        assert committing_session.execute(text("SELECT 1")).scalar_one() == 1
