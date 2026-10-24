from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest
from portopt_db.models import AgentRun, PaperOrder
from portopt_db.models import PortfolioMandate as PortfolioMandateModel

from fund.audit import AgentRunRepository, MandateRepository
from fund.audit.mandate_repository import BaseCurrencyLockedError
from fund.audit.nav_history_repository import NavHistoryRepository
from fund.audit.positions_repository import PositionRepository
from fund.schemas.mandate import PortfolioMandate, RunTriggers


def _mandate(
    portfolio_id: str,
    *,
    capital: str = "100000",
    base_currency: str = "USD",
    drift: float = 0.1,
    benchmark: str | None = None,
    cron: bool = True,
    drift_trigger: bool = False,
) -> PortfolioMandate:
    return PortfolioMandate(
        portfolio_id=portfolio_id,
        capital=Decimal(capital),
        base_currency=base_currency,
        drift_l1_threshold=drift,
        triggers=RunTriggers(cron=cron, drift=drift_trigger),
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
    assert len(rows) == 1
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
    assert PortfolioMandate.model_validate(fetched.mandate) == mandate


def test_upsert_round_trips_remembered_directives(db_session):
    pid = str(uuid.uuid4())
    mandate = PortfolioMandate(
        portfolio_id=pid,
        capital=Decimal("100000"),
        base_currency="EUR",
        drift_l1_threshold=0.05,
        triggers=RunTriggers(cron=True, drift=False),
        remembered_directives=("hold cash heavy", "avoid energy"),
    )
    repo = MandateRepository(db_session)
    repo.upsert(mandate)

    fetched = repo.get(uuid.UUID(pid))
    assert fetched is not None
    rehydrated = PortfolioMandate.model_validate(fetched.mandate)
    assert rehydrated.remembered_directives == ("hold cash heavy", "avoid energy")


def test_get_returns_none_for_unknown_portfolio(db_session):
    repo = MandateRepository(db_session)
    assert repo.get(uuid.uuid4()) is None


# --- MandateRepository base-currency lock ------------------------------------
#
# ``nav_history`` / ``positions`` / ``paper_orders`` store no currency, so re-basing
# a portfolio that already has any of them would mix currencies in one series.


def _add_nav(session, pid: uuid.UUID) -> None:
    NavHistoryRepository(session).upsert_navs(
        [
            {
                "portfolio_id": pid,
                "date": date(2026, 1, 2),
                "nav": Decimal("100000"),
                "benchmark": None,
            }
        ]
    )


def _add_position(session, pid: uuid.UUID) -> None:
    PositionRepository(session).set_holdings(
        pid, [{"ticker": "AAA", "weight": 1.0}], asof=date(2026, 1, 2)
    )


def _add_paper_order(session, pid: uuid.UUID) -> None:
    session.add(
        PaperOrder(
            portfolio_id=pid,
            asof=date(2026, 1, 2),
            weights_hash="h1",
            weights={"AAA": 1.0},
            fill_date=date(2026, 1, 3),
            lines=[],
            notional=1000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    session.flush()


@pytest.mark.parametrize("seed", [_add_nav, _add_position, _add_paper_order])
def test_upsert_rejects_a_base_currency_change_once_the_ledger_has_history(
    db_session, seed
):
    pid = uuid.uuid4()
    repo = MandateRepository(db_session)
    repo.upsert(_mandate(str(pid), base_currency="EUR"))
    seed(db_session, pid)

    with pytest.raises(BaseCurrencyLockedError, match="EUR"):
        repo.upsert(_mandate(str(pid), base_currency="USD"))

    row = repo.get(pid)
    assert row is not None
    assert row.base_currency == "EUR"
    assert row.mandate["base_currency"] == "EUR"


def test_base_currency_lock_error_is_a_value_error():
    # Every caller already turns ``ValueError`` into a clean message.
    assert issubclass(BaseCurrencyLockedError, ValueError)


@pytest.mark.parametrize("seed", [_add_nav, _add_position, _add_paper_order])
def test_upsert_allows_other_edits_on_a_portfolio_with_history(db_session, seed):
    pid = uuid.uuid4()
    repo = MandateRepository(db_session)
    repo.upsert(_mandate(str(pid), base_currency="EUR", drift=0.1))
    seed(db_session, pid)

    row = repo.upsert(_mandate(str(pid), base_currency="EUR", drift=0.3))

    assert row.drift_l1_threshold == 0.3


def test_upsert_allows_a_base_currency_change_before_any_history(db_session):
    pid = uuid.uuid4()
    repo = MandateRepository(db_session)
    repo.upsert(_mandate(str(pid), base_currency="EUR"))

    row = repo.upsert(_mandate(str(pid), base_currency="USD"))

    assert row.base_currency == "USD"


def test_upsert_history_of_another_portfolio_does_not_lock_this_one(db_session):
    pid, other = uuid.uuid4(), uuid.uuid4()
    repo = MandateRepository(db_session)
    repo.upsert(_mandate(str(pid), base_currency="EUR"))
    _add_nav(db_session, other)

    assert repo.upsert(_mandate(str(pid), base_currency="USD")).base_currency == "USD"


@pytest.mark.parametrize("seed", [_add_nav, _add_position, _add_paper_order])
def test_has_ledger_history_reflects_each_ledger_table(db_session, seed):
    pid = uuid.uuid4()
    repo = MandateRepository(db_session)
    assert repo.has_ledger_history(pid) is False

    seed(db_session, pid)

    assert repo.has_ledger_history(pid) is True


# --- MandateRepository.list_active ------------------------------------------


def test_list_active_returns_only_active_rows(db_session):
    repo = MandateRepository(db_session)
    active_pids = {str(uuid.uuid4()), str(uuid.uuid4())}
    for pid in active_pids:
        repo.upsert(_mandate(pid))
    # The sweep must ignore a retired mandate — upsert stamps "active", so flip
    # one row's status directly to exercise the filter.
    archived = repo.upsert(_mandate(str(uuid.uuid4())))
    archived.status = "archived"
    db_session.flush()

    rows = repo.list_active()

    assert {str(r.portfolio_id) for r in rows} == active_pids
    assert all(r.status == "active" for r in rows)


def test_list_active_orders_deterministically_by_portfolio_id(db_session):
    repo = MandateRepository(db_session)
    pids = [str(uuid.uuid4()) for _ in range(4)]
    for pid in pids:
        repo.upsert(_mandate(pid))

    rows = repo.list_active()

    returned = [str(r.portfolio_id) for r in rows]
    assert set(returned) == set(pids)
    assert returned == sorted(returned)


def test_list_active_rows_rehydrate_to_pydantic_mandate_with_triggers(db_session):
    pid = str(uuid.uuid4())
    repo = MandateRepository(db_session)
    repo.upsert(_mandate(pid, cron=True, drift_trigger=True))

    (row,) = repo.list_active()

    # Trigger filtering is not in SQL: the caller rehydrates the JSON payload and
    # reads `.triggers.cron` / `.triggers.drift`.
    rehydrated = PortfolioMandate.model_validate(row.mandate)
    assert rehydrated.triggers.cron is True
    assert rehydrated.triggers.drift is True


def test_list_active_returns_empty_list_when_no_active_mandates(db_session):
    repo = MandateRepository(db_session)
    assert repo.list_active() == []


# --- MandateRepository.list_portfolio_ids -----------------------------------


def test_list_portfolio_ids_returns_distinct_mandate_portfolios(db_session):
    repo = MandateRepository(db_session)
    pids = {uuid.uuid4(), uuid.uuid4()}
    for pid in pids:
        repo.upsert(_mandate(str(pid)))

    assert set(repo.list_portfolio_ids()) == pids


def test_list_portfolio_ids_empty_when_no_mandates(db_session):
    assert MandateRepository(db_session).list_portfolio_ids() == []


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


def test_set_universe_replaces_the_run_universe(db_session):
    repo = AgentRunRepository(db_session)
    run = _run(repo, None)
    assert run.universe == ["AAPL"]

    updated = repo.set_universe(run.id, ["AAA", "BBB"])

    assert updated is not None
    assert db_session.get(AgentRun, run.id).universe == ["AAA", "BBB"]


def test_set_universe_returns_none_for_unknown_run(db_session):
    repo = AgentRunRepository(db_session)
    assert repo.set_universe(uuid.uuid4(), ["AAA"]) is None


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
