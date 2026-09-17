"""T2.2 — ``agent_runs`` / ``agent_decisions`` ORM models on SQLite.

The two audit models live in ``portopt_db`` (mirroring ``background_job.py``:
UUID PK, ``_JSON`` variant, indexes) so the shared ``Base.metadata`` builds them
under SQLite in tests. These tests prove the JSONB→JSON variant creates the
tables on SQLite, that JSON columns round-trip Python dicts/lists, that the
``run → decisions`` relationship cascades, and that ``(run_id, decision_index)``
is unique.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import AgentDecision, AgentRun


def test_both_audit_tables_are_registered_on_metadata(test_engine):
    tables = set(inspect(test_engine).get_table_names())
    assert {"agent_runs", "agent_decisions"} <= tables


def test_agent_run_round_trips_json_columns(db_session):
    run = AgentRun(
        portfolio_id=None,
        asof=date(2026, 1, 2),
        seed=42,
        universe=["AAPL", "MSFT", "GOOG"],
        optimizer_config={"objective": "min_variance", "l2_coef": 0.1},
        status="pending",
    )
    db_session.add(run)
    db_session.flush()

    fetched = db_session.execute(
        select(AgentRun).where(AgentRun.id == run.id)
    ).scalar_one()
    assert fetched.universe == ["AAPL", "MSFT", "GOOG"]
    assert fetched.optimizer_config == {"objective": "min_variance", "l2_coef": 0.1}
    assert fetched.status == "pending"
    assert fetched.seed == 42
    assert fetched.created_at is not None
    assert fetched.started_at is not None


def test_weights_default_null_until_set(db_session):
    run = AgentRun(
        asof=date(2026, 1, 2),
        universe=["AAPL"],
        optimizer_config={},
    )
    db_session.add(run)
    db_session.flush()
    assert run.weights is None

    run.weights = {"AAPL": 1.0}
    db_session.flush()
    assert run.weights == {"AAPL": 1.0}


def test_run_decisions_relationship_orders_by_index(db_session):
    run = AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})
    db_session.add(run)
    db_session.flush()

    db_session.add_all(
        [
            AgentDecision(
                run_id=run.id, decision_index=1, agent="risk", step="risk_check"
            ),
            AgentDecision(run_id=run.id, decision_index=0, agent="pm", step="allocate"),
        ]
    )
    db_session.flush()
    db_session.refresh(run)

    assert [d.decision_index for d in run.decisions] == [0, 1]
    assert run.decisions[0].agent == "pm"


def test_decision_json_columns_round_trip(db_session):
    run = AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})
    db_session.add(run)
    db_session.flush()

    dec = AgentDecision(
        run_id=run.id,
        decision_index=0,
        agent="pm",
        step="allocate",
        constraint_set={"max_weight": 0.1},
        views={"AAPL": {"type": "absolute", "value": 0.05}},
        llm_prompt="choose weights",
        llm_response="done",
        llm_response_hash="abc123",
        hitl_decision={"approved": True},
    )
    db_session.add(dec)
    db_session.flush()

    fetched = db_session.execute(
        select(AgentDecision).where(AgentDecision.id == dec.id)
    ).scalar_one()
    assert fetched.constraint_set == {"max_weight": 0.1}
    assert fetched.views == {"AAPL": {"type": "absolute", "value": 0.05}}
    assert fetched.hitl_decision == {"approved": True}
    assert fetched.llm_response_hash == "abc123"


def test_duplicate_decision_index_for_same_run_is_rejected(db_session):
    run = AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})
    db_session.add(run)
    db_session.flush()

    db_session.add(AgentDecision(run_id=run.id, decision_index=0, agent="pm", step="a"))
    db_session.flush()
    db_session.add(
        AgentDecision(run_id=run.id, decision_index=0, agent="risk", step="b")
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_run_cascades_to_its_decisions(db_session):
    run = AgentRun(asof=date(2026, 1, 2), universe=["AAPL"], optimizer_config={})
    db_session.add(run)
    db_session.flush()
    db_session.add(AgentDecision(run_id=run.id, decision_index=0, agent="pm", step="a"))
    db_session.flush()

    db_session.delete(run)
    db_session.flush()

    remaining = db_session.execute(select(AgentDecision)).scalars().all()
    assert remaining == []
