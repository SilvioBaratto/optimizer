"""T3 — ``MandateRepository`` upsert/get + ``AgentRunRepository`` thread/pause ext.

``MandateRepository`` is the single writer of the ``portfolio_mandates`` row
(one per portfolio, UNIQUE ``portfolio_id``): the pydantic
``fund.schemas.PortfolioMandate`` is the input, the DB model
(``PortfolioMandateModel``) is stored — full mandate in the ``mandate`` JSON
source-of-truth column, scalars mirrored out. The ``AgentRunRepository`` gains
the two mutators (``set_thread_id`` / ``mark_paused``) and the paused-run query
the resume path and observers need. Both sit on ``portopt_db.repository``'s
``RepositoryBase`` and open no session of their own (sync, D1).
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from portopt_db.models import AgentRun
from portopt_db.models import PortfolioMandate as PortfolioMandateModel

from fund.audit import AgentRunRepository, MandateRepository
from fund.schemas.mandate import PortfolioMandate, RunTriggers


def _mandate(
    portfolio_id: str,
    *,
    capital: str = "100000",
    base_currency: str = "USD",
    drift: float = 0.1,
    benchmark: str | None = None,
) -> PortfolioMandate:
    return PortfolioMandate(
        portfolio_id=portfolio_id,
        capital=Decimal(capital),
        base_currency=base_currency,
        drift_l1_threshold=drift,
        triggers=RunTriggers(cron=True, drift=False),
        benchmark=benchmark,
    )


def _run(repo: AgentRunRepository, portfolio_id: uuid.UUID | None) -> AgentRun:
    return repo.create_run(
        portfolio_id=portfolio_id,
        asof=date(2026, 1, 2),
        seed=1,
        universe=["AAPL"],
        optimizer_config={},
    )


# --- MandateRepository ------------------------------------------------------


def test_mandate_repository_exported_from_audit():
    from fund.audit import MandateRepository as Exported

    assert Exported is MandateRepository


def test_upsert_inserts_then_updates_same_row(db_session):
    pid = str(uuid.uuid4())
    repo = MandateRepository(db_session)

    first = repo.upsert(_mandate(pid, capital="100000", drift=0.1))
    assert first.id is not None

    second = repo.upsert(_mandate(pid, capital="250000", drift=0.2, benchmark="SPY"))

    rows = (
        db_session.query(PortfolioMandateModel)
        .filter_by(portfolio_id=uuid.UUID(pid))
        .all()
    )
    assert len(rows) == 1  # one row per portfolio_id — the second upsert updated
    assert second.id == first.id
    assert float(second.capital) == 250000.0
    assert second.drift_l1_threshold == 0.2
    assert second.benchmark == "SPY"


def test_upsert_derives_scalar_columns_from_mandate(db_session):
    pid = str(uuid.uuid4())
    repo = MandateRepository(db_session)

    row = repo.upsert(
        _mandate(
            pid,
            capital="500000",
            base_currency="EUR",
            drift=0.15,
            benchmark="^STOXX50E",
        )
    )

    assert row.base_currency == "EUR"
    assert float(row.capital) == 500000.0
    assert row.drift_l1_threshold == 0.15
    assert row.benchmark == "^STOXX50E"
    assert row.status == "active"


def test_upsert_stores_full_mandate_json_that_round_trips(db_session):
    pid = str(uuid.uuid4())
    mandate = _mandate(
        pid, capital="100000", base_currency="GBP", drift=0.05, benchmark="^FTSE"
    )
    repo = MandateRepository(db_session)
    repo.upsert(mandate)

    fetched = repo.get(uuid.UUID(pid))
    assert fetched is not None
    assert fetched.mandate == mandate.model_dump(mode="json")
    # The JSON source of truth reconstructs the exact pydantic mandate.
    assert PortfolioMandate.model_validate(fetched.mandate) == mandate


def test_get_returns_none_for_unknown_portfolio(db_session):
    repo = MandateRepository(db_session)
    assert repo.get(uuid.uuid4()) is None


# --- AgentRunRepository thread/pause extensions -----------------------------


def test_set_thread_id_mutates_run(db_session):
    repo = AgentRunRepository(db_session)
    run = _run(repo, None)

    updated = repo.set_thread_id(run.id, "thread-xyz")

    assert updated is not None
    assert updated.thread_id == "thread-xyz"
    assert db_session.get(AgentRun, run.id).thread_id == "thread-xyz"


def test_set_thread_id_returns_none_for_unknown_run(db_session):
    repo = AgentRunRepository(db_session)
    assert repo.set_thread_id(uuid.uuid4(), "t") is None


def test_mark_paused_sets_status(db_session):
    repo = AgentRunRepository(db_session)
    run = _run(repo, None)

    paused = repo.mark_paused(run.id)

    assert paused is not None
    assert paused.status == "paused"
    assert db_session.get(AgentRun, run.id).status == "paused"


def test_mark_paused_returns_none_for_unknown_run(db_session):
    repo = AgentRunRepository(db_session)
    assert repo.mark_paused(uuid.uuid4()) is None


def test_list_paused_runs_returns_only_paused_optionally_scoped(db_session):
    repo = AgentRunRepository(db_session)
    pid = uuid.uuid4()
    other_pid = uuid.uuid4()

    _run(repo, pid)  # stays pending
    paused_same = _run(repo, pid)
    repo.mark_paused(paused_same.id)
    paused_other = _run(repo, other_pid)
    repo.mark_paused(paused_other.id)

    all_paused = repo.list_paused_runs()
    assert {r.id for r in all_paused} == {paused_same.id, paused_other.id}
    assert all(r.status == "paused" for r in all_paused)

    scoped = repo.list_paused_runs(portfolio_id=pid)
    assert {r.id for r in scoped} == {paused_same.id}
