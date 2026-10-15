"""Daily NAV-mark scheduler step (SPEC Phase 05 T10).

``run_nav_mark`` claims the ``fund_nav_mark`` slot, heartbeats, builds each active
mandate's FX-normalized NAV series, and upserts it into ``nav_history`` — backfilling
on the first run and converging idempotently on a re-run. Pure compute (no runtime /
agent stack).

Because ``FundJobRepository`` commits (the slot must land for a reaper in another
process) and the heartbeat pulses from a separate thread, these tests use a
private, function-scoped **file-based** SQLite engine, not the shared SAVEPOINT
``db_session``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal

import pytest
from portopt_db.models import BackgroundJob, Base, NavHistory
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from fund.audit import FUND_JOB_TYPES
from fund.audit.fund_job_repository import FundJobRepository
from fund.audit.mandate_repository import MandateRepository
from fund.scheduler import JOB_NAV_MARK, run_nav_mark
from fund.schemas.mandate import PortfolioMandate, RunTriggers

SessionFactory = Callable[[], "contextmanager[Session]"]

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_FILL = date(2026, 10, 2)
_NEXT = date(2026, 10, 3)
_ASOF = date(2026, 10, 3)


@pytest.fixture
def session_factory(tmp_path) -> Generator[SessionFactory, None, None]:
    """A function-scoped file-based SQLite engine + session-context factory.

    File-based (not StaticPool ``:memory:``) so the step's session and the
    heartbeat thread's session each get their own connection while committing to
    one shared DB. Discarded on teardown.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'navmark.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _factory() -> Generator[Session, None, None]:
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    try:
        yield _factory
    finally:
        engine.dispose()


def _seed(factory: SessionFactory) -> None:
    """Seed an active EUR mandate + a USD/GBp ledger with two days of prices/FX."""
    with factory() as session:
        ex = Exchange(name="X")
        session.add(ex)
        session.flush()

        def _inst(ticker):
            inst = Instrument(
                ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
            )
            session.add(inst)
            session.flush()
            return inst

        us1, lon = _inst("US1"), _inst("LON.L")
        session.add_all(
            [
                PriceHistory(
                    instrument_id=us1.id,
                    date=_FILL,
                    close=Decimal("100"),
                    price_unit="USD",
                ),
                PriceHistory(
                    instrument_id=us1.id,
                    date=_NEXT,
                    close=Decimal("110"),
                    price_unit="USD",
                ),
                PriceHistory(
                    instrument_id=lon.id,
                    date=_FILL,
                    close=Decimal("500"),
                    price_unit="GBp",
                ),
                PriceHistory(
                    instrument_id=lon.id,
                    date=_NEXT,
                    close=Decimal("520"),
                    price_unit="GBp",
                ),
            ]
        )
        from portopt_db.repositories.market_data.fx_rates_repository import (
            FxRatesRepository,
        )

        FxRatesRepository(session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": _FILL,
                    "rate": 0.90,
                },
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": _NEXT,
                    "rate": 0.92,
                },
                {
                    "base_currency": "EUR",
                    "quote_currency": "GBP",
                    "date": _FILL,
                    "rate": 1.17,
                },
                {
                    "base_currency": "EUR",
                    "quote_currency": "GBP",
                    "date": _NEXT,
                    "rate": 1.18,
                },
            ]
        )
        session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=date(2026, 10, 1),
                weights_hash="h1",
                weights={"US1": 0.6, "LON.L": 0.4},
                fill_date=_FILL,
                lines=[
                    {
                        "ticker": "US1",
                        "notional": 6000.0,
                        "fill_price": 100.0,
                        "shares": 60.0,
                    },
                    {
                        "ticker": "LON.L",
                        "notional": 4000.0,
                        "fill_price": 500.0,
                        "shares": 8.0,
                    },
                ],
                notional=10000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
        MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("10000"),
                base_currency="EUR",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        session.commit()


def _navs(factory: SessionFactory) -> dict[date, float]:
    with factory() as session:
        rows = session.query(NavHistory).filter_by(portfolio_id=_PID).all()
        return {r.date: float(r.nav) for r in rows}


def _slots(factory: SessionFactory) -> list[BackgroundJob]:
    with factory() as session:
        return list(session.query(BackgroundJob).filter_by(job_type=JOB_NAV_MARK).all())


def test_nav_mark_job_type_registered_for_reaper():
    """The reaper auto-scopes the new slot via FUND_JOB_TYPES."""
    assert "fund_nav_mark" in FUND_JOB_TYPES


def test_nav_mark_upserts_and_backfills(session_factory):
    """Every bar from the first fill is persisted, with FX-correct NAV values."""
    _seed(session_factory)

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True

    navs = _navs(session_factory)
    assert set(navs) == {_FILL, _NEXT}  # backfilled from the earliest fill
    assert navs[_FILL] == pytest.approx(10000.0, rel=1e-6)
    expected_next = (6000.0 / 90.0) * 101.2 + (4000.0 / 5.85) * 6.136
    assert navs[_NEXT] == pytest.approx(expected_next, rel=1e-6)

    slots = _slots(session_factory)
    assert len(slots) == 1
    assert slots[0].status == "completed"


def test_nav_mark_is_idempotent_on_rerun(session_factory):
    """A second run leaves the same rows and stable values (upsert, no duplicates)."""
    _seed(session_factory)
    run_nav_mark(session_factory=session_factory, asof=_ASOF)
    first = _navs(session_factory)

    run_nav_mark(session_factory=session_factory, asof=_ASOF)
    second = _navs(session_factory)

    assert first == second
    with session_factory() as session:
        assert session.query(NavHistory).count() == 2


def test_nav_mark_slot_busy_skips(session_factory):
    """A pre-existing active slot makes the step skip without a second claim."""
    _seed(session_factory)
    with session_factory() as session:
        FundJobRepository(session).claim_or_create(JOB_NAV_MARK)

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is False
    assert len(_slots(session_factory)) == 1
    assert _navs(session_factory) == {}


def test_nav_mark_no_mandates_completes_without_rows(session_factory):
    """With no active mandates the step completes and writes nothing."""
    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True
    assert _navs(session_factory) == {}
    slots = _slots(session_factory)
    assert len(slots) == 1 and slots[0].status == "completed"
