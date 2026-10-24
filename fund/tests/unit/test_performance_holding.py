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

from fund.performance import FxCoverageError, holding_marks, marked_weights

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
