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
from datetime import date, timedelta
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


def _seed(
    factory: SessionFactory, *, base_currency: str = "EUR", capital: str = "10000"
) -> None:
    """Seed an active mandate + a USD/GBp ledger with two days of prices/FX.

    FX rows are always EUR-base (the feed's only base), so a non-EUR
    ``base_currency`` leaves the USD holding uncovered — the fail-loud path.
    """
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
                capital=Decimal(capital),
                base_currency=base_currency,
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


def test_nav_mark_skips_an_unreadable_mandate_and_marks_the_rest(session_factory):
    """One mandate that no longer validates must not stop every other portfolio."""
    _seed(session_factory)
    # Sorts before ``_PID`` in ``list_active`` order, so it is hit first.
    bad = uuid.UUID("0000000a-0000-4000-8000-00000000000a")
    with session_factory() as session:
        row = MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(bad),
                capital=Decimal("10000"),
                base_currency="EUR",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        row.mandate = {**row.mandate, "capital": "-1"}
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True

    assert set(_navs(session_factory)) == {_FILL, _NEXT}
    assert _slots(session_factory)[0].status == "completed"


def test_nav_mark_skips_mandate_without_fx_coverage(session_factory):
    """A non-EUR mandate with foreign holdings is skipped loudly, not mis-marked.

    The FX feed writes EUR-base rows only, so a GBP mandate holding a USD name has
    no rate for USD under GBP — the step skips it (no rows) and still completes,
    rather than silently marking in listing currency.
    """
    _seed(session_factory, base_currency="GBP")

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True
    assert _navs(session_factory) == {}  # skipped, not a wrong NAV
    slots = _slots(session_factory)
    assert len(slots) == 1 and slots[0].status == "completed"


def test_nav_mark_survives_a_stored_mandate_that_no_longer_validates(session_factory):
    """A mandate stored before the ``(0, 2)`` drift bound must not stop other marks."""
    _seed(session_factory)
    stale_pid = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")  # sorts before _PID
    with session_factory() as session:
        repo = MandateRepository(session)
        row = repo.upsert(
            PortfolioMandate(
                portfolio_id=str(stale_pid),
                capital=Decimal("10000"),
                base_currency="EUR",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        row.mandate = {**row.mandate, "drift_l1_threshold": 5.0}
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True

    assert set(_navs(session_factory)) == {_FILL, _NEXT}  # _PID still marked
    slots = _slots(session_factory)
    assert len(slots) == 1 and slots[0].status == "completed"


def test_nav_mark_incremental_extends_without_rewriting(session_factory):
    """A later run picks up a new bar and leaves the prior bars' values stable."""
    _seed(session_factory)
    assert run_nav_mark(session_factory=session_factory, asof=_NEXT) is True
    first = _navs(session_factory)
    assert set(first) == {_FILL, _NEXT}

    d3 = date(2026, 10, 6)
    with session_factory() as session:
        us1 = session.query(Instrument).filter_by(yfinance_ticker="US1").one()
        lon = session.query(Instrument).filter_by(yfinance_ticker="LON.L").one()
        session.add_all(
            [
                PriceHistory(
                    instrument_id=us1.id,
                    date=d3,
                    close=Decimal("120"),
                    price_unit="USD",
                ),
                PriceHistory(
                    instrument_id=lon.id,
                    date=d3,
                    close=Decimal("540"),
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
                    "date": d3,
                    "rate": 0.93,
                },
                {
                    "base_currency": "EUR",
                    "quote_currency": "GBP",
                    "date": d3,
                    "rate": 1.19,
                },
            ]
        )
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=d3) is True
    second = _navs(session_factory)
    assert set(second) == {_FILL, _NEXT, d3}
    assert second[_FILL] == pytest.approx(first[_FILL], rel=1e-6)
    assert second[_NEXT] == pytest.approx(first[_NEXT], rel=1e-6)


_C1, _C2, _C3, _C4 = (date(2026, 10, d) for d in (1, 2, 3, 4))
_C20, _C21, _C22 = (date(2026, 10, d) for d in (20, 21, 22))
_CHAIN_CLOSES = {
    _C1: "100",
    _C2: "150",
    _C3: "150",
    _C4: "165",
    _C20: "165",
    _C21: "180",
    _C22: "180",
}
# One name, two flat-notional tickets, the second filled on _C3 with the same weights.
# Chain-linked, the +50% earned by _C3 survives the rebalance.
_CHAIN_LINKED = {
    _C1: 10_000.0,
    _C2: 15_000.0,
    _C3: 15_000.0,
    _C4: 16_500.0,
    _C20: 16_500.0,
    _C21: 18_000.0,
    _C22: 18_000.0,
}
# What the pre-chain-link marker persisted: ticket 2 snaps back to its flat notional.
_FLAT_NOTIONAL = {
    _C1: 10_000.0,
    _C2: 15_000.0,
    _C3: 10_000.0,
    _C4: 11_000.0,
    _C20: 11_000.0,
    _C21: 12_000.0,
    _C22: 12_000.0,
}


def _seed_chained(factory: SessionFactory, persisted: dict[date, float]) -> None:
    """Seed a USD book with two tickets plus already-persisted ``nav_history`` rows."""
    with factory() as session:
        ex = Exchange(name="CX")
        session.add(ex)
        session.flush()
        inst = Instrument(
            ticker="AAA", short_name="AAA", exchange=ex, yfinance_ticker="AAA"
        )
        session.add(inst)
        session.flush()
        session.add_all(
            PriceHistory(
                instrument_id=inst.id, date=day, close=Decimal(close), price_unit="USD"
            )
            for day, close in _CHAIN_CLOSES.items()
        )
        for fill, key in ((_C1, "c1"), (_C3, "c2")):
            session.add(
                PaperOrder(
                    portfolio_id=_PID,
                    asof=fill - timedelta(days=1),
                    weights_hash=key,
                    weights={"AAA": 1.0},
                    fill_date=fill,
                    lines=[
                        {
                            "ticker": "AAA",
                            "notional": 10_000.0,
                            "fill_date": fill.isoformat(),
                        }
                    ],
                    notional=10_000.0,
                    total_commission=0.0,
                    total_slippage_cost=0.0,
                )
            )
        session.add_all(
            NavHistory(portfolio_id=_PID, date=day, nav=Decimal(str(nav)))
            for day, nav in persisted.items()
        )
        MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("10000"),
                base_currency="USD",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        session.commit()


def test_nav_mark_repairs_flat_notional_history_older_than_the_lookback(
    session_factory,
):
    """History persisted before chain-linking is re-marked in full, not just a week.

    The daily mark only re-emits the last week, so a flat-notional cliff at an older
    rebalance would survive, and the rewritten recent window (now chain-linked) would
    add a second, spurious jump at the window boundary.
    """
    _seed_chained(session_factory, _FLAT_NOTIONAL)

    assert run_nav_mark(session_factory=session_factory, asof=_C22) is True

    navs = _navs(session_factory)
    assert set(navs) == set(_CHAIN_LINKED)
    for day, expected in _CHAIN_LINKED.items():
        assert navs[day] == pytest.approx(expected, rel=1e-6), day


def test_nav_mark_leaves_consistent_old_history_alone(session_factory):
    """A healthy series is still extended incrementally, not rewritten each run."""
    _seed_chained(session_factory, _CHAIN_LINKED)
    sentinel = 1.0  # a value no real mark would produce, on a bar outside the window
    with session_factory() as session:
        session.query(NavHistory).filter_by(
            portfolio_id=_PID, date=_C2
        ).one().nav = Decimal(str(sentinel))
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=_C22) is True

    navs = _navs(session_factory)
    assert navs[_C2] == pytest.approx(sentinel)
    assert navs[_C22] == pytest.approx(_CHAIN_LINKED[_C22], rel=1e-6)


def test_nav_mark_full_rewrites_the_whole_series_on_request(session_factory):
    """``full=True`` is the explicit operator hook for a one-time full re-mark."""
    _seed_chained(session_factory, _CHAIN_LINKED)
    with session_factory() as session:
        session.query(NavHistory).filter_by(
            portfolio_id=_PID, date=_C2
        ).one().nav = Decimal("1.0")
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=_C22, full=True) is True

    navs = _navs(session_factory)
    for day, expected in _CHAIN_LINKED.items():
        assert navs[day] == pytest.approx(expected, rel=1e-6), day


_HIST_START = date(2026, 9, 1)
_HIST_DAYS = 30
_HIST_END = date(2026, 9, 30)


def _day(offset: int) -> date:
    return date.fromordinal(_HIST_START.toordinal() + offset)


def _seed_long_history(factory: SessionFactory) -> None:
    """A month of daily bars for two USD names with a drifting EUR/USD rate.

    Two names with different trajectories and a moving FX rate make any two books
    (and any two base currencies) mark to visibly different NAVs, so a stale row is
    detectable. Ticket A holds ``US1`` only and fills on the first bar.
    """
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    with factory() as session:
        ex = Exchange(name="X")
        session.add(ex)
        session.flush()
        insts = {}
        for ticker in ("US1", "ALT"):
            inst = Instrument(
                ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
            )
            session.add(inst)
            session.flush()
            insts[ticker] = inst
        for i in range(_HIST_DAYS):
            session.add_all(
                [
                    PriceHistory(
                        instrument_id=insts["US1"].id,
                        date=_day(i),
                        close=Decimal(100 + i),
                        price_unit="USD",
                    ),
                    PriceHistory(
                        instrument_id=insts["ALT"].id,
                        date=_day(i),
                        close=Decimal(50 + 3 * i),
                        price_unit="USD",
                    ),
                ]
            )
        FxRatesRepository(session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": _day(i),
                    "rate": 0.90 + 0.002 * i,
                }
                for i in range(_HIST_DAYS)
            ]
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
    _add_ticket(factory, ticker="US1", fill=_day(0))


def _add_ticket(factory: SessionFactory, *, ticker: str, fill: date) -> None:
    with factory() as session:
        session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=date.fromordinal(fill.toordinal() - 1),
                weights_hash=f"h-{ticker}-{fill.isoformat()}",
                weights={ticker: 1.0},
                fill_date=fill,
                lines=[
                    {
                        "ticker": ticker,
                        "notional": 10000.0,
                        "fill_price": 100.0,
                        "shares": 100.0,
                    }
                ],
                notional=10000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
        session.commit()


def _recomputed_navs(factory: SessionFactory, base_currency: str) -> dict[date, float]:
    """The NAV series rebuilt from every ticket: what ``nav_history`` must equal."""
    from fund.performance import build_nav_series

    with factory() as session:
        points = build_nav_series(
            _PID, session=session, base_currency=base_currency, asof=_HIST_END
        )
    return {p.date: p.nav for p in points}


def test_nav_mark_remarks_history_older_than_the_window_for_a_back_dated_ticket(
    session_factory,
):
    """A ticket filled long before the last mark rewrites the rows it changes.

    An approval delayed past the incremental window files a ticket whose fill date
    precedes ``last_nav_date - window``. Re-marking only the recent bars would leave
    the older rows on the previous book and put a seam in the series at the window
    edge.
    """
    _seed_long_history(session_factory)
    assert run_nav_mark(session_factory=session_factory, asof=_HIST_END) is True

    _add_ticket(session_factory, ticker="ALT", fill=_day(10))
    assert run_nav_mark(session_factory=session_factory, asof=_HIST_END) is True

    assert _navs(session_factory) == pytest.approx(
        _recomputed_navs(session_factory, "EUR"), rel=1e-6
    )


def test_nav_mark_history_stays_in_one_currency_because_the_base_cannot_change(
    session_factory,
):
    """Re-basing a marked portfolio is refused, so rows never mix two currencies.

    Paper tickets store their notionals without a currency, so a re-based mandate would
    read old EUR amounts as USD; the mandate repository therefore locks the base
    currency once the ledger has history, and the persisted series stays whole.
    """
    from fund.audit.mandate_repository import BaseCurrencyLockedError

    _seed_long_history(session_factory)
    assert run_nav_mark(session_factory=session_factory, asof=_HIST_END) is True
    before = _navs(session_factory)

    with session_factory() as session, pytest.raises(BaseCurrencyLockedError):
        MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("10000"),
                base_currency="USD",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )

    assert run_nav_mark(session_factory=session_factory, asof=_HIST_END) is True
    assert _navs(session_factory) == pytest.approx(before)
    assert _navs(session_factory) == pytest.approx(
        _recomputed_navs(session_factory, "EUR"), rel=1e-6
    )


def test_nav_mark_skips_an_invalid_mandate_row_and_marks_the_rest(session_factory):
    """One mandate whose stored JSON no longer validates must not starve the others.

    The invalid row sorts before the valid portfolio, so a validation error escaping
    the per-portfolio isolation would leave every later portfolio unmarked on every
    daily run.
    """
    _seed(session_factory)
    bad_pid = uuid.UUID("0a0a0a0a-0a0a-4a0a-8a0a-0a0a0a0a0a0a")
    with session_factory() as session:
        repo = MandateRepository(session)
        repo.upsert(
            PortfolioMandate(
                portfolio_id=str(bad_pid),
                capital=Decimal("10000"),
                base_currency="EUR",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        row = repo.get(bad_pid)
        assert row is not None
        row.mandate = {k: v for k, v in row.mandate.items() if k != "triggers"}
        session.commit()

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True

    assert set(_navs(session_factory)) == {_FILL, _NEXT}
    slots = _slots(session_factory)
    assert len(slots) == 1 and slots[0].status == "completed"


def test_nav_mark_denominates_nav_in_the_mandate_capital(session_factory):
    """The persisted NAV starts at the mandate's capital, not the ticket's notional."""
    _seed(session_factory, capital="20000")  # ticket was sized against 10,000

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF) is True

    navs = _navs(session_factory)
    assert navs[_FILL] == pytest.approx(20000.0, rel=1e-6)
    expected_next = 2 * ((6000.0 / 90.0) * 101.2 + (4000.0 / 5.85) * 6.136)
    assert navs[_NEXT] == pytest.approx(expected_next, rel=1e-6)


def test_nav_mark_rewrites_history_when_a_ticket_is_approved_after_the_fact(
    session_factory,
):
    """A late-approved ticket re-bases the bars it fills over, not just the last week.

    Ticket 2 fills on bar 2 but is approved after the daily mark has already run to
    bar 20, so its fill predates the incremental window. The persisted bars between
    its fill and the window still hold the old book; they must be rewritten so the
    stored series is the one a full rebuild gives.
    """
    days = [date(2026, 10, 1) + timedelta(days=i) for i in range(25)]
    with session_factory() as session:
        ex = Exchange(name="X")
        session.add(ex)
        session.flush()
        for ticker, start, step in (("A", 100.0, 1.01), ("B", 50.0, 1.02)):
            inst = Instrument(
                ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
            )
            session.add(inst)
            session.flush()
            session.add_all(
                PriceHistory(
                    instrument_id=inst.id,
                    date=day,
                    close=Decimal(str(start * step**i)),
                    price_unit="USD",
                )
                for i, day in enumerate(days)
            )
        MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("100000"),
                base_currency="USD",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        session.commit()

    def _ticket(ticker: str, fill_index: int, key: str) -> None:
        with session_factory() as session:
            session.add(
                PaperOrder(
                    portfolio_id=_PID,
                    asof=days[fill_index] - timedelta(days=1),
                    weights_hash=key,
                    weights={ticker: 1.0},
                    fill_date=days[fill_index],
                    lines=[
                        {
                            "ticker": ticker,
                            "notional": 100_000.0,
                            "fill_date": days[fill_index].isoformat(),
                        }
                    ],
                    notional=100_000.0,
                    total_commission=0.0,
                    total_slippage_cost=0.0,
                )
            )
            session.commit()

    _ticket("A", 0, "h1")
    assert run_nav_mark(session_factory=session_factory, asof=days[20]) is True
    _ticket("B", 2, "h2")
    assert run_nav_mark(session_factory=session_factory, asof=days[24]) is True

    navs = _navs(session_factory)
    handover = 100_000.0 * 1.01**2
    for i, day in enumerate(days):
        expected = 100_000.0 * 1.01**i if i < 2 else handover * 1.02 ** (i - 2)
        assert navs[day] == pytest.approx(expected, rel=1e-6), day


def _seed_benchmark_book(factory: SessionFactory, *, benchmark: str) -> None:
    """Twelve USD bars for one holding plus two candidate benchmarks (EUR base).

    SPY trades near 400 and ACWI near 100, so their base levels are unmistakable;
    twelve bars outrun the 7-day incremental re-mark window.
    """
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    days = [date(2026, 10, 1) + timedelta(days=i) for i in range(12)]
    with factory() as session:
        ex = Exchange(name="X")
        session.add(ex)
        session.flush()
        for ticker, start in (("US1", 100), ("SPY", 400), ("ACWI", 100)):
            inst = Instrument(
                ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
            )
            session.add(inst)
            session.flush()
            session.add_all(
                PriceHistory(
                    instrument_id=inst.id,
                    date=d,
                    close=Decimal(str(start + i)),
                    price_unit="USD",
                )
                for i, d in enumerate(days)
            )
        FxRatesRepository(session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": d,
                    "rate": 0.9,
                }
                for d in days
            ]
        )
        session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=date(2026, 9, 30),
                weights_hash="hb",
                weights={"US1": 1.0},
                fill_date=days[0],
                lines=[
                    {
                        "ticker": "US1",
                        "notional": 9000.0,
                        "fill_price": 100.0,
                        "shares": 90.0,
                    }
                ],
                notional=9000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
        session.commit()
    _set_benchmark(factory, benchmark)


def _set_benchmark(factory: SessionFactory, benchmark: str) -> None:
    with factory() as session:
        MandateRepository(session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("10000"),
                base_currency="EUR",
                benchmark=benchmark,
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=False),
            )
        )
        session.commit()


def _benchmark_rows(factory: SessionFactory) -> dict[date, tuple[float, str | None]]:
    with factory() as session:
        rows = session.query(NavHistory).filter_by(portfolio_id=_PID).all()
        return {r.date: (float(r.benchmark), r.benchmark_ticker) for r in rows}


def test_nav_mark_remarks_full_history_when_the_mandate_benchmark_changes(
    session_factory,
):
    """A benchmark switch rewrites every persisted level, not just the last 7 days.

    Rows older than the incremental window would otherwise keep the previous
    benchmark's level beside the new one, and the stitched series would show a
    phantom benchmark return at the join.
    """
    _seed_benchmark_book(session_factory, benchmark="SPY")
    assert run_nav_mark(session_factory=session_factory, asof=date(2026, 10, 12))
    first = _benchmark_rows(session_factory)
    assert first[date(2026, 10, 1)] == (pytest.approx(0.9 * 400), "SPY")

    _set_benchmark(session_factory, "ACWI")
    assert run_nav_mark(session_factory=session_factory, asof=date(2026, 10, 12))

    rows = _benchmark_rows(session_factory)
    assert {ticker for _, ticker in rows.values()} == {"ACWI"}
    assert rows[date(2026, 10, 1)][0] == pytest.approx(0.9 * 100)
    assert rows[date(2026, 10, 12)][0] == pytest.approx(0.9 * 111)


def test_nav_mark_stays_incremental_while_the_benchmark_is_unchanged(session_factory):
    """With the same benchmark the older persisted rows are left untouched."""
    _seed_benchmark_book(session_factory, benchmark="SPY")
    run_nav_mark(session_factory=session_factory, asof=date(2026, 10, 12))
    with session_factory() as session:
        oldest = (
            session.query(NavHistory)
            .filter_by(portfolio_id=_PID, date=date(2026, 10, 1))
            .one()
        )
        oldest.benchmark = Decimal("1")  # a sentinel an incremental run won't rewrite
        session.commit()

    run_nav_mark(session_factory=session_factory, asof=date(2026, 10, 12))

    assert _benchmark_rows(session_factory)[date(2026, 10, 1)][0] == pytest.approx(1.0)
