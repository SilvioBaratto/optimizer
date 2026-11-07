"""NAV chain-linking edge cases: undeployed cash, unvaluable switch bars, contributions.

Runs against in-memory SQLite like ``test_performance_nav``; the helpers here seed
USD books so base currency needs no FX unless a test adds rates itself.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument

from fund import performance
from fund.performance import build_nav_series, holding_marks

_PID = uuid.UUID("bbbbbbbb-cccc-dddd-eeee-ffffffffffff")
_DAYS = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(5)]


def _seed_series(
    db_session,
    series: dict[str, dict[dt.date, float]],
    units: dict[str, str] | None = None,
) -> dict[str, Instrument]:
    """Seed ``{ticker: {date: close}}``; the raw ``price_unit`` defaults to USD."""
    ex = Exchange(name="RX")
    db_session.add(ex)
    db_session.flush()
    instruments: dict[str, Instrument] = {}
    for ticker, closes in series.items():
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=day,
                close=Decimal(str(px)),
                price_unit=(units or {}).get(ticker, "USD"),
            )
            for day, px in closes.items()
        )
        instruments[ticker] = inst
    db_session.flush()
    return instruments


def _seed_closes(db_session, closes: dict[str, list[float]]) -> None:
    _seed_series(
        db_session,
        {
            ticker: dict(zip(_DAYS, series, strict=True))
            for ticker, series in closes.items()
        },
    )


def _seed_eur_fx(db_session, quote: str, rates: dict[dt.date, float]) -> None:
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": quote,
                "date": day,
                "rate": rate,
            }
            for day, rate in rates.items()
        ]
    )
    db_session.flush()


def _weekdays(start: dt.date, count: int) -> list[dt.date]:
    days: list[dt.date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def _ticket(
    db_session,
    *,
    key: str,
    fill: dt.date,
    notionals: dict[str, float],
    gross: float,
    pid: uuid.UUID = _PID,
) -> None:
    db_session.add(
        PaperOrder(
            portfolio_id=pid,
            asof=fill - dt.timedelta(days=1),
            weights_hash=key,
            weights={t: n / gross for t, n in notionals.items()},
            fill_date=fill,
            lines=[
                {"ticker": t, "notional": n, "fill_date": fill.isoformat()}
                for t, n in notionals.items()
            ],
            notional=gross,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()


def test_nav_starts_at_capital_when_the_first_ticket_leaves_cash_undeployed(
    db_session,
):
    """A later ticket that invests the reserve must not lever the NAV.

    Ticket 1 puts 50k of 100k into A, ticket 2 puts all of it in. The 50k reserve is
    cash inside NAV from the first bar, so when A moves +10% the NAV moves +10% (it
    held 100k of A) and when A then halves the NAV halves.
    """
    _seed_closes(db_session, {"A": [100.0, 100.0, 100.0, 110.0, 55.0]})
    _ticket(db_session, key="t1", fill=_DAYS[0], notionals={"A": 50_000.0}, gross=1e5)
    _ticket(db_session, key="t2", fill=_DAYS[2], notionals={"A": 100_000.0}, gross=1e5)

    points = build_nav_series(
        _PID, session=db_session, base_currency="USD", capital=100_000.0
    )

    navs = {p.date: p.nav for p in points}
    assert navs[_DAYS[0]] == pytest.approx(100_000.0)
    assert navs[_DAYS[2]] == pytest.approx(100_000.0)
    assert navs[_DAYS[3]] == pytest.approx(110_000.0)
    assert navs[_DAYS[4]] == pytest.approx(55_000.0)
    marks = holding_marks(
        _PID, session=db_session, base_currency="USD", capital=100_000.0
    )
    assert marks[0].value == pytest.approx(navs[_DAYS[4]])
    summary = performance.portfolio_metrics(_PID, points)
    assert summary.max_drawdown == pytest.approx(0.5)


def test_holding_contributions_sum_to_the_nav_one_day_return_with_cash(db_session):
    """Contributions are weighted on the whole NAV, cash included."""
    _seed_closes(db_session, {"A": [100.0, 120.0, 130.0, 135.0, 140.0]})
    _ticket(db_session, key="h1", fill=_DAYS[0], notionals={"A": 100_000.0}, gross=1e5)
    _ticket(db_session, key="h2", fill=_DAYS[2], notionals={"A": 50_000.0}, gross=1e5)

    points = build_nav_series(_PID, session=db_session, base_currency="USD")
    marks = holding_marks(_PID, session=db_session, base_currency="USD")

    one_day = points[-1].nav / points[-2].nav - 1.0
    assert sum(m.contribution for m in marks) == pytest.approx(one_day)


def test_a_switch_bar_with_a_stale_fx_line_still_carries_the_accumulated_pnl(
    db_session,
):
    """A rebalance on a bar where the old book cannot be valued keeps its profit.

    EU1 doubles while the USD leg's FX feed has a gap. The second ticket fills inside
    the gap, where the outgoing book (EU1 + USD line) has no complete mark, so the
    hand-over waits for the first bar both books price. Falling back to the previous
    scale reset the NAV to the ticket's flat 100,000 and dropped the 50,000 gained.
    """
    days = _weekdays(dt.date(2026, 10, 5), 16)
    eu = {day: 100.0 + 12.5 * min(i, 8) for i, day in enumerate(days)}
    us = dict.fromkeys(days, 100.0)
    _seed_series(db_session, {"EU1": eu, "US1": us}, units={"EU1": "EUR"})
    live = days[:4] + days[12:]
    _seed_eur_fx(db_session, "USD", dict.fromkeys(live, 0.90))
    _ticket(
        db_session,
        key="h1",
        fill=days[0],
        notionals={"EU1": 50_000.0, "US1": 50_000.0},
        gross=100_000.0,
    )
    _ticket(
        db_session, key="h2", fill=days[10], notionals={"EU1": 100_000.0}, gross=1e5
    )

    points = build_nav_series(_PID, session=db_session, base_currency="EUR")

    navs = {p.date: p.nav for p in points}
    assert navs[days[8]] == pytest.approx(150_000.0)
    assert days[9] not in navs
    assert navs[days[10]] == pytest.approx(150_000.0)
    assert navs[days[12]] == pytest.approx(150_000.0)


def test_a_held_currency_with_no_fx_rate_fails_loud_even_when_others_are_covered(
    db_session,
):
    """A CHF line is not valued 1:1 in EUR because a USD rate happens to exist."""
    days = _weekdays(dt.date(2026, 10, 5), 5)
    _seed_series(
        db_session,
        {"US1": dict.fromkeys(days, 100.0), "CH1": dict.fromkeys(days, 100.0)},
        units={"CH1": "CHF"},
    )
    _seed_eur_fx(db_session, "USD", dict.fromkeys(days, 0.90))
    _ticket(
        db_session,
        key="h1",
        fill=days[0],
        notionals={"US1": 50_000.0, "CH1": 50_000.0},
        gross=100_000.0,
    )

    with pytest.raises(performance.FxCoverageError, match="CHF"):
        build_nav_series(_PID, session=db_session, base_currency="EUR")
    with pytest.raises(performance.FxCoverageError, match="CHF"):
        holding_marks(_PID, session=db_session, base_currency="EUR")
    with pytest.raises(performance.FxCoverageError, match="CHF"):
        performance.marked_weights(_PID, session=db_session, base_currency="EUR")


@pytest.mark.parametrize("newest", [None, 10])
def test_price_frame_unit_is_the_newest_rows_whichever_way_it_is_read(
    db_session, newest
):
    """A unit flip in the stored bars reads the same on every path.

    The NAV read goes oldest first and the holdings read newest first, so taking the
    first row's unit put the two 100x apart. Both now take the newest unit and express
    the older pence bars in it.
    """
    inst_days = _DAYS[:3]
    ex = Exchange(name="LSE")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker="VOD", short_name="VOD", exchange=ex, yfinance_ticker="VOD.L"
    )
    db_session.add(inst)
    db_session.flush()
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=inst.id,
                date=inst_days[0],
                close=Decimal("500"),
                price_unit="GBp",
            ),
            PriceHistory(
                instrument_id=inst.id,
                date=inst_days[1],
                close=Decimal("510"),
                price_unit="GBp",
            ),
            PriceHistory(
                instrument_id=inst.id,
                date=inst_days[2],
                close=Decimal("5.2"),
                price_unit="GBP",
            ),
        ]
    )
    db_session.flush()

    prices, currency_map = performance._load_price_frame(
        db_session, ["VOD.L"], newest=newest
    )

    assert currency_map["VOD.L"] == "GBP"
    assert prices["VOD.L"].tolist() == pytest.approx([5.0, 5.1, 5.2])
