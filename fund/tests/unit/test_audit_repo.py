"""``AgentRunRepository`` CRUD round-trips on SQLite.

The repo is the fund-side behavior layer over the ``agent_runs`` /
``agent_decisions`` models that live in ``portopt_db`` (the split mirrors
``background_jobs``: model in the shared DB package, behavior in the consumer).
It sits on ``portopt_db.repository.RepositoryBase`` and opens no session of its
own — the caller injects one (sync).
"""

from __future__ import annotations

from datetime import date

from portopt_db.models import AgentDecision, AgentRun

from fund.audit import AgentRunRepository


def test_create_run_persists_and_returns_row(db_session):
    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=None,
        asof=date(2026, 1, 2),
        seed=7,
        universe=["AAPL", "MSFT"],
        optimizer_config={"objective": "min_variance"},
    )

    assert run.id is not None
    assert run.status == "pending"
    assert run.universe == ["AAPL", "MSFT"]
    fetched = db_session.get(AgentRun, run.id)
    assert fetched is not None
    assert fetched.optimizer_config == {"objective": "min_variance"}


def test_get_run_returns_none_for_unknown_id(db_session):
    import uuid

    repo = AgentRunRepository(db_session)
    assert repo.get_run(uuid.uuid4()) is None


def test_append_decision_auto_increments_index(db_session):
    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=None,
        asof=date(2026, 1, 2),
        seed=1,
        universe=["AAPL"],
        optimizer_config={},
    )

    d0 = repo.append_decision(run.id, agent="pm", step="allocate")
    d1 = repo.append_decision(
        run.id, agent="risk", step="risk_check", constraint_set={"max_weight": 0.1}
    )

    assert d0.decision_index == 0
    assert d1.decision_index == 1
    assert d1.constraint_set == {"max_weight": 0.1}
    rows = db_session.query(AgentDecision).filter_by(run_id=run.id).all()
    assert len(rows) == 2


def test_finalize_run_writes_weights_and_status(db_session):
    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=None,
        asof=date(2026, 1, 2),
        seed=1,
        universe=["AAPL", "MSFT"],
        optimizer_config={},
    )

    finalized = repo.finalize_run(
        run.id, weights={"AAPL": 0.6, "MSFT": 0.4}, status="completed"
    )

    assert finalized is not None
    assert finalized.status == "completed"
    assert finalized.weights == {"AAPL": 0.6, "MSFT": 0.4}
    assert finalized.finished_at is not None


def test_finalize_run_returns_none_for_unknown_id(db_session):
    import uuid

    repo = AgentRunRepository(db_session)
    assert repo.finalize_run(uuid.uuid4(), weights={}) is None


def test_list_runs_for_portfolio_filters_by_portfolio(db_session):
    import uuid

    repo = AgentRunRepository(db_session)
    pid_a = uuid.uuid4()
    pid_b = uuid.uuid4()
    repo.create_run(
        portfolio_id=pid_a,
        asof=date(2026, 1, 2),
        seed=1,
        universe=["AAPL"],
        optimizer_config={},
    )
    repo.create_run(
        portfolio_id=pid_a,
        asof=date(2026, 1, 3),
        seed=2,
        universe=["MSFT"],
        optimizer_config={},
    )
    repo.create_run(
        portfolio_id=pid_b,
        asof=date(2026, 1, 2),
        seed=3,
        universe=["GOOG"],
        optimizer_config={},
    )

    runs_a = repo.list_runs_for_portfolio(pid_a)
    assert len(runs_a) == 2
    assert {r.seed for r in runs_a} == {1, 2}


def test_list_portfolio_ids_distinct_excludes_null(db_session):
    import uuid

    repo = AgentRunRepository(db_session)
    pid = uuid.uuid4()
    # Two runs for the same portfolio → one id; a fund-wide run (NULL) → none.
    repo.create_run(
        portfolio_id=pid,
        asof=date(2026, 1, 2),
        seed=1,
        universe=["AAPL"],
        optimizer_config={},
    )
    repo.create_run(
        portfolio_id=pid,
        asof=date(2026, 1, 3),
        seed=2,
        universe=["MSFT"],
        optimizer_config={},
    )
    repo.create_run(
        portfolio_id=None,
        asof=date(2026, 1, 4),
        seed=3,
        universe=["GOOG"],
        optimizer_config={},
    )

    ids = repo.list_portfolio_ids()

    assert set(ids) == {pid}
    assert len(ids) == 1


def test_list_decisions_joins_portfolio_and_filters(db_session):
    import uuid

    repo = AgentRunRepository(db_session)
    pid_a = uuid.UUID("aaaa0001-0000-4000-8000-000000000001")
    pid_b = uuid.UUID("bbbb0002-0000-4000-8000-000000000002")
    run_a = repo.create_run(
        portfolio_id=pid_a,
        asof=date(2026, 1, 2),
        seed=1,
        universe=[],
        optimizer_config={},
    )
    run_b = repo.create_run(
        portfolio_id=pid_b,
        asof=date(2026, 1, 2),
        seed=2,
        universe=[],
        optimizer_config={},
    )
    repo.append_decision(run_a.id, agent="allocator", step="optimize_portfolio")
    repo.append_decision(run_a.id, agent="risk", step="risk_check")
    repo.append_decision(run_b.id, agent="allocator", step="optimize_portfolio")

    # Cross-portfolio: all three decisions, each carrying its run's portfolio_id.
    everyone = repo.list_decisions()
    assert len(everyone) == 3
    assert {pid for _, pid in everyone} == {pid_a, pid_b}

    # Scoped to one portfolio.
    only_a = repo.list_decisions(portfolio_id=pid_a)
    assert len(only_a) == 2
    assert all(pid == pid_a for _, pid in only_a)

    # Scoped by step (the decision "type").
    optimize = repo.list_decisions(step="optimize_portfolio")
    assert len(optimize) == 2
    assert all(dec.step == "optimize_portfolio" for dec, _ in optimize)
