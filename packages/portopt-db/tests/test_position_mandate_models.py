"""T1 — ``positions`` / ``portfolio_mandates`` models + ``agent_runs.thread_id``.

Phase 8 of the ``fund`` bridge adds two pure-SQLAlchemy ``BaseModel`` subclasses
and one nullable column on the shared ``Base.metadata`` so SQLite builds them in
tests (mirroring ``agent_run.py`` / ``paper_order.py``: UUID PK, ``_JSON``
variant, named indexes/constraints). These tests prove:

* both new tables register on ``Base.metadata``;
* ``Position`` round-trips its snapshot columns, ``shares``/``notional`` default
  ``None``, and ``(portfolio_id, ticker)`` rejects a duplicate holding while the
  same ticker under a *different* portfolio is allowed;
* ``PortfolioMandate`` round-trips the ``mandate`` JSONB→JSON payload + scalar
  mirror columns, ``status`` carries its ``"active"`` server default, and the
  ``portfolio_id`` UNIQUE collapses a second mandate;
* the new nullable ``agent_runs.thread_id`` accepts ``None`` and a set value.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import AgentRun, PortfolioMandate, Position


def test_both_new_tables_are_registered_on_metadata(test_engine):
    tables = set(inspect(test_engine).get_table_names())
    assert {"positions", "portfolio_mandates"} <= tables


def test_position_round_trips_snapshot_columns(db_session):
    pid = uuid.uuid4()
    pos = Position(
        portfolio_id=pid,
        ticker="AAPL",
        weight=0.25,
        shares=10.0,
        notional=2500.0,
        asof=date(2026, 1, 2),
    )
    db_session.add(pos)
    db_session.flush()

    fetched = db_session.execute(
        select(Position).where(Position.id == pos.id)
    ).scalar_one()
    assert fetched.portfolio_id == pid
    assert fetched.ticker == "AAPL"
    assert fetched.weight == 0.25
    assert fetched.shares == 10.0
    assert fetched.notional == 2500.0
    assert fetched.asof == date(2026, 1, 2)
    assert fetched.paper_order_id is None
    assert fetched.created_at is not None


def test_position_shares_and_notional_default_null(db_session):
    pos = Position(
        portfolio_id=uuid.uuid4(),
        ticker="MSFT",
        weight=1.0,
        asof=date(2026, 1, 2),
    )
    db_session.add(pos)
    db_session.flush()
    assert pos.shares is None
    assert pos.notional is None


def test_duplicate_ticker_for_same_portfolio_is_rejected(db_session):
    pid = uuid.uuid4()
    db_session.add(
        Position(portfolio_id=pid, ticker="AAPL", weight=0.5, asof=date(2026, 1, 2))
    )
    db_session.flush()
    db_session.add(
        Position(portfolio_id=pid, ticker="AAPL", weight=0.5, asof=date(2026, 1, 2))
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_same_ticker_under_a_different_portfolio_is_allowed(db_session):
    db_session.add(
        Position(
            portfolio_id=uuid.uuid4(),
            ticker="AAPL",
            weight=1.0,
            asof=date(2026, 1, 2),
        )
    )
    db_session.add(
        Position(
            portfolio_id=uuid.uuid4(),
            ticker="AAPL",
            weight=1.0,
            asof=date(2026, 1, 2),
        )
    )
    db_session.flush()  # must NOT raise — distinct portfolio_id


def test_portfolio_mandate_round_trips_json_and_scalars(db_session):
    pid = uuid.uuid4()
    mandate = PortfolioMandate(
        portfolio_id=pid,
        base_currency="EUR",
        capital=100_000.0,
        drift_l1_threshold=0.1,
        benchmark="^GSPC",
        mandate={"capital": 100_000.0, "hitl_gates": ["place_orders"]},
    )
    db_session.add(mandate)
    db_session.flush()

    fetched = db_session.execute(
        select(PortfolioMandate).where(PortfolioMandate.id == mandate.id)
    ).scalar_one()
    assert fetched.portfolio_id == pid
    assert fetched.base_currency == "EUR"
    assert fetched.capital == 100_000.0
    assert fetched.drift_l1_threshold == 0.1
    assert fetched.benchmark == "^GSPC"
    assert fetched.mandate == {"capital": 100_000.0, "hitl_gates": ["place_orders"]}
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


def test_mandate_status_carries_active_default(db_session):
    mandate = PortfolioMandate(
        portfolio_id=uuid.uuid4(),
        base_currency="USD",
        capital=1.0,
        drift_l1_threshold=0.05,
        mandate={},
    )
    db_session.add(mandate)
    db_session.flush()
    db_session.refresh(mandate)
    assert mandate.status == "active"
    assert mandate.benchmark is None


def test_duplicate_portfolio_mandate_is_rejected(db_session):
    pid = uuid.uuid4()
    db_session.add(
        PortfolioMandate(
            portfolio_id=pid,
            base_currency="USD",
            capital=1.0,
            drift_l1_threshold=0.05,
            mandate={},
        )
    )
    db_session.flush()
    db_session.add(
        PortfolioMandate(
            portfolio_id=pid,
            base_currency="USD",
            capital=2.0,
            drift_l1_threshold=0.05,
            mandate={},
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_agent_run_thread_id_defaults_null_and_accepts_a_value(db_session):
    run = AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})
    db_session.add(run)
    db_session.flush()
    assert run.thread_id is None

    run.thread_id = str(run.id)
    db_session.flush()
    fetched = db_session.execute(
        select(AgentRun).where(AgentRun.id == run.id)
    ).scalar_one()
    assert fetched.thread_id == str(run.id)
