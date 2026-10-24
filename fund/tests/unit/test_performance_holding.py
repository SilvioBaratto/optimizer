"""Performance tracker — per-holding mark-to-market (SPEC Phase 05 T9).

Each holding's value / returns / contribution / unrealized P&L is marked against a
hand-computed ledger including a non-base (GBp) holding, proving shares are
re-derived with FX. A holding with no price degrades to None fields, not a crash.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.performance import (
    _HOLDING_LOOKBACK_BARS,
    FxCoverageError,
    build_nav_series,
    holding_marks,
    marked_weights,
)

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_D1 = dt.date(2026, 10, 2)  # fill bar
_D2 = dt.date(2026, 10, 3)  # current bar


def _seed(db_session, *, missing_price_holding: bool = False):
    """Seed one ticket (USD + GBp holdings) + two days of prices/FX."""
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()

    def _inst(ticker):
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        return inst

    us1, lon = _inst("US1"), _inst("LON.L")
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=us1.id, date=_D1, close=Decimal("100"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id, date=_D2, close=Decimal("110"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=lon.id, date=_D1, close=Decimal("500"), price_unit="GBp"
            ),
            PriceHistory(
                instrument_id=lon.id, date=_D2, close=Decimal("520"), price_unit="GBp"
            ),
        ]
    )
    lines = [
        {"ticker": "US1", "notional": 6000.0, "fill_price": 100.0, "shares": 60.0},
        {"ticker": "LON.L", "notional": 4000.0, "fill_price": 500.0, "shares": 8.0},
    ]
    if missing_price_holding:
        _inst("NOPX")  # instrument exists but no price history
        lines.append({"ticker": "NOPX", "notional": 1000.0, "fill_price": 10.0})

    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _D1,
                "rate": 0.90,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _D2,
                "rate": 0.92,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "date": _D1,
                "rate": 1.17,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "date": _D2,
                "rate": 1.18,
            },
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=dt.date(2026, 10, 1),
            weights_hash="h1",
            weights={"US1": 0.6, "LON.L": 0.4},
            fill_date=_D1,
            lines=lines,
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()


def test_holding_marks_fx_correct_value_and_pnl(db_session):
    """Value, weight, 1D return, contribution, P&L match the hand computation."""
    _seed(db_session)
    marks = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }

    # US1: shares 6000/90; current base 110*0.92 = 101.2
    us_value = (6000.0 / 90.0) * 101.2
    # LON.L: shares 4000/5.85; current base (520/100)*1.18 = 6.136
    lon_value = (4000.0 / 5.85) * 6.136
    total = us_value + lon_value

    us, lon = marks["US1"], marks["LON.L"]
    assert us.value == pytest.approx(us_value, rel=1e-6)
    assert lon.value == pytest.approx(lon_value, rel=1e-6)
    assert us.weight == pytest.approx(us_value / total, rel=1e-6)
    assert lon.weight == pytest.approx(lon_value / total, rel=1e-6)

    assert us.return_1d == pytest.approx(101.2 / 90.0 - 1.0, rel=1e-6)
    assert lon.return_1d == pytest.approx(6.136 / 5.85 - 1.0, rel=1e-6)
    assert us.contribution == pytest.approx(us.weight * us.return_1d, rel=1e-6)

    # Cost basis is the deployed base notional (shares × price_base@fill).
    assert us.unrealized_pnl == pytest.approx(us_value - 6000.0, rel=1e-6)
    assert lon.unrealized_pnl == pytest.approx(lon_value - 4000.0, rel=1e-6)


def test_holding_marks_shares_not_read_from_stale_lines(db_session):
    """The GBp holding's shares are re-derived, not the currency-mixed lines value."""
    _seed(db_session)
    lon = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }["LON.L"]
    assert lon.shares == pytest.approx(4000.0 / 5.85, rel=1e-6)  # not the stale 8.0


def test_holding_marks_short_window_returns_none_for_month_year(db_session):
    """A two-day window has no bar a month or year back."""
    _seed(db_session)
    us = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }["US1"]
    assert us.return_1m is None
    assert us.return_1y is None


def test_holding_marks_missing_price_degrades_to_none(db_session):
    """A holding with no price history yields None fields, not a crash."""
    _seed(db_session, missing_price_holding=True)
    marks = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }
    nopx = marks["NOPX"]
    assert nopx.value is None
    assert nopx.shares is None
    assert nopx.weight is None
    assert nopx.unrealized_pnl is None
    # The priced holdings are still marked.
    assert marks["US1"].value is not None


def test_holding_marks_no_orders_returns_empty(db_session):
    """A portfolio with no paper orders yields an empty list."""
    assert holding_marks(_PID, session=db_session, base_currency="EUR") == []


def test_holding_marks_non_base_mandate_without_fx_coverage_raises(db_session):
    """holding_marks fails loud for a non-EUR mandate with no FX coverage (C1).

    The ledger seeds EUR-base rates only; marking in GBP leaves the USD holding
    with no rate, so it raises rather than marking in listing currency.
    """
    _seed(db_session)
    with pytest.raises(FxCoverageError):
        holding_marks(_PID, session=db_session, base_currency="GBP")


def test_holding_marks_returns_use_own_calendar_not_carry_forward(db_session):
    """A holding idle on the mark day gets its real 1D move, not a 0 carry-forward.

    US2 does not trade on the last union bar (US1 does), so a ffilled series would
    show a 0 % 1D move; returns must instead use US2's own last two real closes.
    """
    d1, d2, d3 = dt.date(2026, 10, 2), dt.date(2026, 10, 3), dt.date(2026, 10, 6)
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()

    def _inst(ticker):
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        return inst

    us1, us2 = _inst("US1"), _inst("US2")
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=us1.id, date=d1, close=Decimal("100"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id, date=d2, close=Decimal("110"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id, date=d3, close=Decimal("120"), price_unit="USD"
            ),  # US1 trades on d3
            PriceHistory(
                instrument_id=us2.id, date=d1, close=Decimal("200"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us2.id, date=d2, close=Decimal("210"), price_unit="USD"
            ),  # US2 is idle after d2
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=dt.date(2026, 10, 1),
            weights_hash="h1",
            weights={"US1": 0.6, "US2": 0.4},
            fill_date=d1,
            lines=[
                {"ticker": "US1", "notional": 6000.0, "fill_price": 100.0},
                {"ticker": "US2", "notional": 4000.0, "fill_price": 200.0},
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()

    marks = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="USD", asof=d3)
    }
    us2 = marks["US2"]
    # US2's 1D move is its own 210/200 - 1, not a 0 carry-forward over idle d3.
    assert us2.return_1d == pytest.approx(210.0 / 200.0 - 1.0, rel=1e-6)
    assert us2.return_1d != pytest.approx(0.0, abs=1e-9)
    # Value still marks on the carried-forward (asof) close: 20 shares × 210.
    assert us2.value == pytest.approx(20.0 * 210.0, rel=1e-6)


def _seed_long_history(db_session, *, days: int) -> None:
    """Add ``days`` of daily pre-fill bars + FX rows to the two-day ``_seed`` ledger."""
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )
    from sqlalchemy import select

    instruments = {
        inst.yfinance_ticker: inst
        for inst in db_session.execute(select(Instrument)).scalars()
    }
    units = {"US1": "USD", "LON.L": "GBp"}
    closes = {"US1": "95", "LON.L": "480"}
    bars = []
    rates = []
    for offset in range(1, days + 1):
        day = _D1 - dt.timedelta(days=offset)
        for ticker, unit in units.items():
            bars.append(
                PriceHistory(
                    instrument_id=instruments[ticker].id,
                    date=day,
                    close=Decimal(closes[ticker]),
                    price_unit=unit,
                )
            )
        for quote, rate in (("USD", 0.88), ("GBP", 1.15)):
            rates.append(
                {
                    "base_currency": "EUR",
                    "quote_currency": quote,
                    "date": day,
                    "rate": rate,
                }
            )
    db_session.add_all(bars)
    FxRatesRepository(db_session).upsert_rates(rates)
    db_session.flush()


def _count_history_rows(monkeypatch) -> list[int]:
    """Record how many price + FX rows each repository read pulls into memory."""
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )
    from portopt_db.repositories.market_data.yfinance_repository import (
        YFinanceRepository,
    )

    rows_read: list[int] = []
    original_prices = YFinanceRepository.get_price_history
    original_rates = FxRatesRepository.get_rates

    def prices_spy(self, *args, **kwargs):
        rows = original_prices(self, *args, **kwargs)
        rows_read.append(len(rows))
        return rows

    def rates_spy(self, *args, **kwargs):
        rows = original_rates(self, *args, **kwargs)
        rows_read.append(len(rows))
        return rows

    monkeypatch.setattr(YFinanceRepository, "get_price_history", prices_spy)
    monkeypatch.setattr(FxRatesRepository, "get_rates", rates_spy)
    return rows_read


def test_marked_weights_match_the_holding_marks_weights(db_session):
    """The cheap weight read agrees with the full mark on the hand-built ledger."""
    _seed(db_session)
    expected = {
        m.ticker: m.weight
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }

    weights = marked_weights(_PID, session=db_session, base_currency="EUR")

    assert weights == pytest.approx(expected, rel=1e-9)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_marked_weights_read_only_the_bars_near_the_fill(db_session, monkeypatch):
    """Years of pre-fill history must not be loaded just to weight the current book.

    The drift poll calls this every couple of seconds on the TUI event loop, so the
    read has to be bounded by the book's age, not by the instrument's listing age.
    """
    _seed(db_session)
    _seed_long_history(db_session, days=300)
    expected = {
        m.ticker: m.weight
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }
    rows_read = _count_history_rows(monkeypatch)

    weights = marked_weights(_PID, session=db_session, base_currency="EUR")

    assert weights == pytest.approx(expected, rel=1e-9)
    # 2 tickers x 300 pre-fill bars + 2 FX quotes x 300 rates is 1200 rows unbounded.
    assert sum(rows_read) < 200


def test_marked_weights_none_without_a_paper_ticket(db_session):
    assert marked_weights(_PID, session=db_session, base_currency="EUR") is None


def test_marked_weights_none_when_a_holding_cannot_be_priced(db_session):
    """Weights over only the priced holdings would misstate the book, so none."""
    _seed(db_session, missing_price_holding=True)

    assert marked_weights(_PID, session=db_session, base_currency="EUR") is None


def test_marked_weights_fail_loud_without_fx_coverage(db_session):
    _seed(db_session)

    with pytest.raises(FxCoverageError):
        marked_weights(_PID, session=db_session, base_currency="GBP")


def test_holding_marks_scale_to_mandate_capital(db_session):
    """Shares, value and P&L are on the mandate's capital, not the ticket's notional.

    The ticket was sized against 10,000; marking against a 20,000 mandate doubles
    every absolute figure while weights and returns stay put.
    """
    _seed(db_session)
    plain = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }
    scaled = {
        m.ticker: m
        for m in holding_marks(
            _PID, session=db_session, base_currency="EUR", capital=20_000.0
        )
    }

    for ticker in ("US1", "LON.L"):
        assert scaled[ticker].shares == pytest.approx(2 * plain[ticker].shares)
        assert scaled[ticker].value == pytest.approx(2 * plain[ticker].value)
        assert scaled[ticker].weight == pytest.approx(plain[ticker].weight)
        assert scaled[ticker].return_1d == pytest.approx(plain[ticker].return_1d)
    cost_us = 2 * 6000.0
    assert scaled["US1"].unrealized_pnl == pytest.approx(
        scaled["US1"].value - cost_us, rel=1e-6
    )


def test_holding_marks_value_matches_nav_after_a_rebalance(db_session):
    """After a rebalance the marked book is worth the NAV, not a fresh notional.

    Asset A doubles, then a ticket switches the whole book into B. The holdings table
    must total the chain-linked NAV (200,000), not the second ticket's flat 100,000.
    """
    d = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(4)]
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    for ticker, closes in (("A", [100.0, 150.0, 200.0, 200.0]), ("B", [50.0] * 4)):
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
                price_unit="USD",
            )
            for day, px in zip(d, closes, strict=True)
        )
    for key, ticker, fill in (("h1", "A", d[0]), ("h2", "B", d[2])):
        db_session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=fill - dt.timedelta(days=1),
                weights_hash=key,
                weights={ticker: 1.0},
                fill_date=fill,
                lines=[
                    {
                        "ticker": ticker,
                        "notional": 100_000.0,
                        "fill_date": fill.isoformat(),
                    }
                ],
                notional=100_000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
    db_session.flush()

    marks = holding_marks(_PID, session=db_session, base_currency="USD")

    assert [m.ticker for m in marks] == ["B"]
    assert marks[0].shares == pytest.approx(4_000.0)  # 200,000 carried / 50
    assert marks[0].value == pytest.approx(200_000.0)
    assert marks[0].unrealized_pnl == pytest.approx(0.0, abs=1e-6)


def test_holding_marks_value_only_the_invested_part_of_a_half_cash_book(db_session):
    """A holding is worth its share of the book, not the whole NAV, beside cash.

    A rises 100 -> 130 and a second ticket then holds it at 50% of a 100,000 gross.
    The book is worth 130,000 on the hand-over bar: 65,000 in A, 65,000 in cash. The
    holdings table shows the 65,000 invested, bought at cost, with no P&L yet.
    """
    d = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(3)]
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(ticker="A", short_name="A", exchange=ex, yfinance_ticker="A")
    db_session.add(inst)
    db_session.flush()
    db_session.add_all(
        PriceHistory(
            instrument_id=inst.id, date=day, close=Decimal(str(px)), price_unit="USD"
        )
        for day, px in zip(d, [100.0, 115.0, 130.0], strict=True)
    )
    for key, notional, fill in (("h1", 100_000.0, d[0]), ("h2", 50_000.0, d[2])):
        db_session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=fill - dt.timedelta(days=1),
                weights_hash=key,
                weights={"A": notional / 100_000.0},
                fill_date=fill,
                lines=[
                    {"ticker": "A", "notional": notional, "fill_date": fill.isoformat()}
                ],
                notional=100_000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
    db_session.flush()

    marks = holding_marks(_PID, session=db_session, base_currency="USD")

    assert [m.ticker for m in marks] == ["A"]
    assert marks[0].value == pytest.approx(65_000.0)
    assert marks[0].shares == pytest.approx(500.0)  # 65,000 / 130
    assert marks[0].unrealized_pnl == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("bars", "priced"),
    [
        (9, True),  # latest bar is the fifth past the last live rate: still usable
        (10, False),  # the sixth: the rate has expired
    ],
)
def test_holding_marks_degrades_a_holding_whose_fx_rate_is_stale(
    db_session, bars, priced
):
    """A holding is valued on a carried rate for five bars, then degrades to ``None``.

    The USD rate stops after bar 4 while the stock keeps climbing. The stale-rate
    bars used to be filled with the last converted price without limit, so the mark
    showed a value from many bars ago. The EUR holding is unaffected either way.
    """
    # Weekdays only: the FX staleness horizon is counted in business days.
    days = [
        day
        for day in (
            dt.date(2026, 10, 5) + dt.timedelta(days=i) for i in range(2 * bars)
        )
        if day.weekday() < 5
    ][:bars]
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    for ticker, unit, prices in (
        ("EU1", "EUR", [100.0] * bars),
        ("US1", "USD", [100.0 * 1.01**i for i in range(bars)]),
    ):
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
                price_unit=unit,
            )
            for day, px in zip(days, prices, strict=True)
        )
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": day,
                "rate": 0.90,
            }
            for day in days[:4]
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=days[0] - dt.timedelta(days=1),
            weights_hash="h1",
            weights={"EU1": 0.5, "US1": 0.5},
            fill_date=days[0],
            lines=[
                {"ticker": "EU1", "notional": 5000.0, "fill_date": days[0].isoformat()},
                {"ticker": "US1", "notional": 5000.0, "fill_date": days[0].isoformat()},
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()

    marks = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }

    assert marks["EU1"].value == pytest.approx(5000.0)
    if priced:
        # The rate is constant, so the mark tracks the stock exactly.
        assert marks["US1"].value == pytest.approx(5000.0 * 1.01 ** (bars - 1))
    else:
        assert marks["US1"].value is None
        assert marks["US1"].shares is None


def test_holding_marks_price_the_latest_bar_of_a_history_past_the_row_cap(db_session):
    """A long backfill is marked at today's close, not at its 5,000th-oldest bar.

    The price read caps rows at 5,000 and, ordered oldest-first, kept the *earliest*
    ones — so a ``--period max`` instrument was valued at a years-old price and its
    fill bar fell outside the loaded frame, pinning unrealized P&L to exactly zero.
    """
    n = 5_010
    last = dt.date(2026, 10, 3)
    days = [last - dt.timedelta(days=n - 1 - i) for i in range(n)]
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(ticker="A", short_name="A", exchange=ex, yfinance_ticker="A")
    db_session.add(inst)
    db_session.flush()
    db_session.add_all(
        PriceHistory(
            instrument_id=inst.id,
            date=day,
            close=Decimal(str(round(100.0 + 0.01 * i, 2))),
            price_unit="USD",
        )
        for i, day in enumerate(days)
    )
    fill = days[-3]
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=fill - dt.timedelta(days=1),
            weights_hash="h1",
            weights={"A": 1.0},
            fill_date=fill,
            lines=[
                {"ticker": "A", "notional": 10_000.0, "fill_date": fill.isoformat()}
            ],
            notional=10_000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()

    (mark,) = holding_marks(_PID, session=db_session, base_currency="USD")

    fill_price, latest_price = 100.0 + 0.01 * (n - 3), 100.0 + 0.01 * (n - 1)
    assert mark.value == pytest.approx(10_000.0 * latest_price / fill_price)
    assert mark.unrealized_pnl == pytest.approx(
        10_000.0 * (latest_price / fill_price - 1.0)
    )


def test_holding_marks_raise_for_a_holding_with_no_resolvable_currency(db_session):
    """A holding priced in an unknown currency is not marked as if it were base."""
    _seed(db_session)
    db_session.query(PriceHistory).filter(PriceHistory.price_unit == "USD").update(
        {"price_unit": None}
    )
    db_session.flush()

    with pytest.raises(FxCoverageError, match="US1"):
        holding_marks(_PID, session=db_session, base_currency="EUR")


@pytest.mark.parametrize("latest", ["AAA", "BBB"])
def test_same_bar_tickets_resolve_to_the_latest_created_for_nav_and_holdings(
    db_session, latest
):
    """Tickets sharing asof and fill_date supersede each other by creation time.

    NAV and the holdings table must follow the *same* ticket, whichever physical
    row order the database returns for the tied sort keys.
    """
    d1, d2, d3 = dt.date(2026, 10, 2), dt.date(2026, 10, 3), dt.date(2026, 10, 6)
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    paths = {"AAA": ("100", "105", "110"), "BBB": ("100", "100", "100")}
    for ticker, closes in paths.items():
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        for day, close in zip((d1, d2, d3), closes, strict=True):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=day,
                    close=Decimal(close),
                    price_unit="USD",
                )
            )
    early = dt.datetime(2026, 10, 1, 9, 0, tzinfo=dt.UTC)
    late = early + dt.timedelta(hours=1)
    for ticker in ("AAA", "BBB"):
        db_session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=dt.date(2026, 10, 1),
                weights_hash=f"h-{ticker}",
                weights={ticker: 1.0},
                fill_date=d1,
                lines=[{"ticker": ticker, "notional": 10000.0, "fill_price": 100.0}],
                notional=10000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
                created_at=late if ticker == latest else early,
            )
        )
    db_session.flush()

    marks = holding_marks(_PID, session=db_session, base_currency="USD")
    nav = build_nav_series(_PID, session=db_session, base_currency="USD")

    assert [m.ticker for m in marks] == [latest]
    expected_nav = 11000.0 if latest == "AAA" else 10000.0
    assert nav[-1].nav == pytest.approx(expected_nav, rel=1e-9)


def _add_instrument(db_session, ticker):
    ex = Exchange(name=f"X-{ticker}")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _seed_daily_history(db_session, inst, last_day, n_bars):
    """Seed ``n_bars`` consecutive daily USD closes ending on ``last_day``.

    Closes climb by 1 per bar, so every bar is distinguishable and a return over any
    lookback has a closed form.
    """
    closes: dict[dt.date, float] = {}
    for i in range(n_bars):
        day = last_day - dt.timedelta(days=n_bars - 1 - i)
        closes[day] = 100.0 + i
        db_session.add(
            PriceHistory(
                instrument_id=inst.id,
                date=day,
                close=Decimal(str(100.0 + i)),
                price_unit="USD",
            )
        )
    db_session.flush()
    return closes


def _ticket(fill_day, ticker="AAA", notional=10000.0):
    return PaperOrder(
        portfolio_id=_PID,
        asof=fill_day,
        weights_hash="h1",
        weights={ticker: 1.0},
        fill_date=fill_day,
        lines=[{"ticker": ticker, "notional": notional, "fill_price": 100.0}],
        notional=notional,
        total_commission=0.0,
        total_slippage_cost=0.0,
    )


def _spy_loaded_rows(monkeypatch):
    """Record every price and FX row ``holding_marks`` reads from the repositories."""
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )
    from portopt_db.repositories.market_data.yfinance_repository import (
        YFinanceRepository,
    )

    prices: list[dt.date] = []
    fx: list[dt.date] = []
    original_prices = YFinanceRepository.get_price_history
    original_fx = FxRatesRepository.get_rates

    def price_spy(self, *args, **kwargs):
        rows = original_prices(self, *args, **kwargs)
        prices.extend(row.date for row in rows)
        return rows

    def fx_spy(self, *args, **kwargs):
        rows = original_fx(self, *args, **kwargs)
        fx.extend(row.date for row in rows)
        return rows

    monkeypatch.setattr(YFinanceRepository, "get_price_history", price_spy)
    monkeypatch.setattr(FxRatesRepository, "get_rates", fx_spy)
    return prices, fx


def test_holding_marks_loads_a_bounded_trailing_window_not_the_whole_history(
    db_session, monkeypatch
):
    """Only the newest bars the returns need are read, however long the history is."""
    inst = _add_instrument(db_session, "AAA")
    closes = _seed_daily_history(db_session, inst, _D2, 700)
    db_session.add(_ticket(_D1))
    db_session.flush()
    prices, _ = _spy_loaded_rows(monkeypatch)

    mark = holding_marks(_PID, session=db_session, base_currency="USD")[0]

    assert len(prices) <= _HOLDING_LOOKBACK_BARS
    assert min(closes) not in prices
    year_ago = closes[_D2 - dt.timedelta(days=365)]
    assert mark.return_1y == pytest.approx(closes[_D2] / year_ago - 1.0, rel=1e-9)


def test_holding_marks_for_a_book_older_than_the_window_stays_bounded(
    db_session, monkeypatch
):
    """A years-old book prices its shares at the fill but loads no history in between.

    The fill bar lies beyond the newest-bars window, so it is read through a narrow
    lookup around the fill date; the price and FX loads must not grow with book age.
    """
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    n_bars = 900
    inst = _add_instrument(db_session, "AAA")
    closes = _seed_daily_history(db_session, inst, _D2, n_bars)
    rates = {
        day: 0.90 + 0.0001 * i for i, day in enumerate(sorted(closes))
    }  # base-per-USD
    FxRatesRepository(db_session).upsert_rates(
        [
            {"base_currency": "EUR", "quote_currency": "USD", "date": d, "rate": r}
            for d, r in rates.items()
        ]
    )
    fill_day = _D2 - dt.timedelta(days=600)
    db_session.add(_ticket(fill_day))
    db_session.flush()
    prices, fx = _spy_loaded_rows(monkeypatch)

    mark = holding_marks(_PID, session=db_session, base_currency="EUR")[0]

    slack = 60  # the fill-date lookup and the FX margin on both windows
    assert len(prices) <= _HOLDING_LOOKBACK_BARS + slack
    assert len(fx) <= _HOLDING_LOOKBACK_BARS + slack
    shares = 10000.0 / (closes[fill_day] * rates[fill_day])
    assert mark.shares == pytest.approx(shares, rel=1e-9)
    assert mark.value == pytest.approx(shares * closes[_D2] * rates[_D2], rel=1e-9)
    year_ago_day = _D2 - dt.timedelta(days=365)
    expected_1y = (closes[_D2] * rates[_D2]) / (
        closes[year_ago_day] * rates[year_ago_day]
    )
    assert mark.return_1y == pytest.approx(expected_1y - 1.0, rel=1e-9)


def test_holding_marks_chain_link_an_old_rebalance_without_loading_the_gap(
    db_session, monkeypatch
):
    """Two tickets filled long before the newest-bars window still chain-link.

    The switch bar of the second ticket lies in a narrow lookup around its fill, so the
    holdings total the full-history NAV although the years between are never read.
    """
    n_bars = 900
    aaa = _add_instrument(db_session, "AAA")
    bbb = _add_instrument(db_session, "BBB")
    _seed_daily_history(db_session, aaa, _D2, n_bars)
    for i in range(n_bars):
        db_session.add(
            PriceHistory(
                instrument_id=bbb.id,
                date=_D2 - dt.timedelta(days=n_bars - 1 - i),
                close=Decimal(str(50.0 + 0.5 * i)),
                price_unit="USD",
            )
        )
    first_fill = _D2 - dt.timedelta(days=600)
    second_fill = _D2 - dt.timedelta(days=560)
    db_session.add(_ticket(first_fill, "AAA"))
    second = _ticket(second_fill, "BBB")
    second.weights_hash = "h2"
    db_session.add(second)
    db_session.flush()
    prices, _ = _spy_loaded_rows(monkeypatch)

    marks = holding_marks(_PID, session=db_session, base_currency="USD")
    marks_rows = len(prices)
    nav = build_nav_series(_PID, session=db_session, base_currency="USD")

    assert [m.ticker for m in marks] == ["BBB"]
    assert marks[0].value == pytest.approx(nav[-1].nav, rel=1e-9)
    # Two tickers, each read for its newest bars plus the narrow fill lookups.
    assert marks_rows <= 2 * (_HOLDING_LOOKBACK_BARS + 60)
