"""Audit timestamps must reflect when a row was written, not when the
surrounding transaction began.

PostgreSQL's ``now()`` is the transaction start time, so every row a long
agent run flushes inside one transaction used to share one stale
``created_at``. The ORM/Core write paths therefore stamp with the statement
wall clock (``clock_timestamp()`` on PostgreSQL).
"""

from __future__ import annotations

import time
from datetime import date

from sqlalchemy import insert, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from portopt_db.models import AgentDecision, AgentRun
from portopt_db.models.macro.macro_regime import EconomicIndicator
from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

# Long enough to clear the sub-millisecond resolution of the SQLite stand-in
# clock, short enough not to slow the suite.
_TICK_SECONDS = 0.02


def _new_run() -> AgentRun:
    return AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})


def test_postgres_insert_stamps_created_at_with_statement_clock() -> None:
    sql = str(
        insert(AgentDecision.__table__).compile(
            dialect=postgresql.dialect(),
            column_keys=["run_id", "decision_index", "agent", "step"],
        )
    )

    assert "clock_timestamp()" in sql
    assert "now()" not in sql


def test_postgres_update_refreshes_updated_at_with_statement_clock() -> None:
    sql = str(
        update(AgentRun.__table__)
        .values(status="running")
        .compile(dialect=postgresql.dialect())
    )

    assert "updated_at=clock_timestamp()" in sql
    assert "now()" not in sql


def test_rows_flushed_in_one_transaction_get_increasing_created_at(
    db_session: Session,
) -> None:
    run = _new_run()
    db_session.add(run)
    db_session.flush()
    first = AgentDecision(run_id=run.id, decision_index=0, agent="pm", step="a")
    db_session.add(first)
    db_session.flush()
    time.sleep(_TICK_SECONDS)
    second = AgentDecision(run_id=run.id, decision_index=1, agent="pm", step="b")
    db_session.add(second)
    db_session.flush()

    db_session.refresh(first)
    db_session.refresh(second)
    assert first.created_at < second.created_at


def test_updated_at_advances_past_created_at_inside_one_transaction(
    db_session: Session,
) -> None:
    run = _new_run()
    db_session.add(run)
    db_session.flush()
    time.sleep(_TICK_SECONDS)

    run.status = "running"
    db_session.flush()
    db_session.refresh(run)

    assert run.updated_at > run.created_at


def test_started_at_tracks_insert_time_not_transaction_start(
    db_session: Session,
) -> None:
    first, second = _new_run(), _new_run()
    db_session.add(first)
    db_session.flush()
    time.sleep(_TICK_SECONDS)
    db_session.add(second)
    db_session.flush()

    db_session.refresh(first)
    db_session.refresh(second)
    assert first.started_at < second.started_at


def test_upsert_conflict_stamp_is_later_than_the_original_insert(
    db_session: Session,
) -> None:
    repo = MacroRegimeRepository(db_session)
    payload = {
        "last_inflation": 2.1,
        "inflation_6m": None,
        "inflation_10y_avg": None,
        "gdp_growth_6m": None,
        "earnings_12m": None,
        "eps_expected_12m": None,
        "peg_ratio": None,
        "lt_rate_forecast": None,
        "reference_date": None,
    }
    repo.upsert_economic_indicator("CLOCK_TEST", payload)
    db_session.flush()
    row = db_session.query(EconomicIndicator).filter_by(country="CLOCK_TEST").one()
    inserted_at = row.updated_at
    time.sleep(_TICK_SECONDS)

    repo.upsert_economic_indicator("CLOCK_TEST", {**payload, "last_inflation": 3.5})
    db_session.flush()
    db_session.expire(row)

    assert row.updated_at > inserted_at
