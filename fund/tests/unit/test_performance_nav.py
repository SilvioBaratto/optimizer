"""Performance tracker — FX-aware price reader (SPEC Phase 05 T6) + NAV series (T7).

The FX-conversion cases (``test_nav_fx_*``) drive the pure base-currency price
reader against in-memory fixtures (incl. a GBp minor-unit ticker); the DB reader
cases run against real SQLite. The module must import no agent stack
(optimizer/skfolio are allowed — they are not the deep-agent stack).
"""

from __future__ import annotations

import ast
import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument

from fund import performance
from fund.performance import (
    _load_fx_rates_frame,
    _load_price_frame,
    _to_base_prices,
    build_nav_series,
)

_D1 = dt.date(2026, 10, 1)
_D2 = dt.date(2026, 10, 2)
_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _price_frame():
    return pd.DataFrame(
        {"US1": [100.0, 101.0], "LON": [500.0, 510.0]},
        index=pd.to_datetime([_D1, _D2]),
    )


def _fx_frame():
    return pd.DataFrame(
        {"USD": [0.90, 0.91], "GBP": [1.17, 1.18]},
        index=pd.to_datetime([_D1, _D2]),
    )


def test_nav_fx_converts_minor_unit_and_multiplies_by_rate():
    """GBp prices are /100 then × GBP rate; USD prices × USD rate; base per date."""
    out = _to_base_prices(
        _price_frame(),
        base_currency="EUR",
        currency_map={"US1": "USD", "LON": "GBp"},
        fx_rates=_fx_frame(),
    )
    assert out["US1"].tolist() == pytest.approx([90.0, 91.91])
    assert out["LON"].tolist() == pytest.approx([5.85, 6.018])


def test_nav_fx_missing_rate_degrades_without_raise():
    """A ticker whose currency has no FX column is left unconverted, not raised."""
    prices = pd.DataFrame({"JPN": [1000.0, 1010.0]}, index=pd.to_datetime([_D1, _D2]))
    out = _to_base_prices(
        prices,
        base_currency="EUR",
        currency_map={"JPN": "JPY"},
        fx_rates=_fx_frame(),
    )
    assert out["JPN"].tolist() == pytest.approx([1000.0, 1010.0])


def test_nav_fx_empty_prices_returns_empty():
    """An empty price frame round-trips to empty without touching optimizer.fx."""
    out = _to_base_prices(
        pd.DataFrame(),
        base_currency="EUR",
        currency_map={},
        fx_rates=_fx_frame(),
    )
    assert out.empty


def test_nav_fx_load_price_frame_carries_raw_price_unit(db_session):
    """The DB reader keys currency_map on the RAW price_unit (e.g. 'GBp')."""
    ex = Exchange(name="LSE")
    db_session.add(ex)
    db_session.flush()
    lon = Instrument(
        ticker="LON", short_name="Lon", exchange=ex, yfinance_ticker="LON.L"
    )
    db_session.add(lon)
    db_session.flush()
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=lon.id, date=_D1, close=Decimal("500"), price_unit="GBp"
            ),
            PriceHistory(
                instrument_id=lon.id, date=_D2, close=Decimal("510"), price_unit="GBp"
            ),
        ]
    )
    db_session.flush()

    prices, currency_map = _load_price_frame(db_session, ["LON.L"])
    assert currency_map["LON.L"] == "GBp"  # raw, not pre-normalized to GBP
    assert prices["LON.L"].tolist() == pytest.approx([500.0, 510.0])


def test_nav_fx_load_fx_rates_frame_pivots_to_currency_columns(db_session):
    """The DB reader pivots fx_rate rows into a dates × quote-currency frame."""
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
                "rate": 0.91,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "date": _D1,
                "rate": 1.17,
            },
        ]
    )
    db_session.flush()

    frame = _load_fx_rates_frame(db_session, "EUR", ["USD", "GBP"])
    assert set(frame.columns) == {"USD", "GBP"}
    assert float(frame.loc[pd.Timestamp(_D1), "USD"]) == pytest.approx(0.90)
    assert float(frame.loc[pd.Timestamp(_D2), "USD"]) == pytest.approx(0.91)


def _seed_ledger_two_day(db_session, *, with_benchmark: bool = False):
    """Seed one paper ticket (USD + GBp holdings) + two days of prices/FX.

    A EUR-base portfolio with 10,000 capital split 6,000 US1 / 4,000 LON.L, filled
    on day 2 (``_D2``). Prices/FX are chosen so NAV at the fill equals the capital
    and moves to a hand-computed value the next day — proving shares are re-derived
    with FX, not read from the currency-mixed ``lines.shares``.
    """
    d2 = dt.date(2026, 10, 3)  # day after the fill, for a NAV move
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
                instrument_id=us1.id, date=_D2, close=Decimal("100"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id, date=d2, close=Decimal("110"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=lon.id, date=_D2, close=Decimal("500"), price_unit="GBp"
            ),
            PriceHistory(
                instrument_id=lon.id, date=d2, close=Decimal("520"), price_unit="GBp"
            ),
        ]
    )
    if with_benchmark:
        spx = _inst("SPX")
        db_session.add_all(
            [
                PriceHistory(
                    instrument_id=spx.id,
                    date=_D2,
                    close=Decimal("5000"),
                    price_unit="USD",
                ),
                PriceHistory(
                    instrument_id=spx.id,
                    date=d2,
                    close=Decimal("5100"),
                    price_unit="USD",
                ),
            ]
        )

    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _D2,
                "rate": 0.90,
            },
            {"base_currency": "EUR", "quote_currency": "USD", "date": d2, "rate": 0.92},
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "date": _D2,
                "rate": 1.17,
            },
            {"base_currency": "EUR", "quote_currency": "GBP", "date": d2, "rate": 1.18},
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=_D1,
            weights_hash="h1",
            weights={"US1": 0.6, "LON.L": 0.4},
            fill_date=_D2,
            lines=[
                # shares here are the stale currency-mixed values — must be ignored.
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
    db_session.flush()
    return d2


def test_nav_series_rederives_fx_correct_shares(db_session):
    """NAV equals capital at the fill and the hand-computed value next day.

    Using the stale ``lines.shares`` would wildly mis-mark the GBp holding, so the
    match proves shares are re-derived as notional_base / price_base@fill.
    """
    d3 = _seed_ledger_two_day(db_session)

    points = build_nav_series(_PID, session=db_session, base_currency="EUR")

    navs = {p.date: p.nav for p in points}
    assert navs[_D2] == pytest.approx(10000.0, rel=1e-6)
    # US1: (6000/90)*101.2 ; LON.L: (4000/5.85)*6.136
    expected_d3 = (6000.0 / 90.0) * 101.2 + (4000.0 / 5.85) * 6.136
    assert navs[d3] == pytest.approx(expected_d3, rel=1e-6)


def test_nav_series_benchmark_fx_converted_and_aligned(db_session):
    """The benchmark level is FX-converted to base and date-aligned to the NAV."""
    d3 = _seed_ledger_two_day(db_session, with_benchmark=True)

    points = build_nav_series(
        _PID, session=db_session, base_currency="EUR", benchmark="SPX"
    )
    bench = {p.date: p.benchmark for p in points}
    assert bench[_D2] == pytest.approx(5000.0 * 0.90)
    assert bench[d3] == pytest.approx(5100.0 * 0.92)


def test_nav_series_no_benchmark_leaves_field_none(db_session):
    """Without a benchmark ticker every point's benchmark is None."""
    _seed_ledger_two_day(db_session)
    points = build_nav_series(_PID, session=db_session, base_currency="EUR")
    assert points
    assert all(p.benchmark is None for p in points)


def test_nav_series_empty_history_returns_empty(db_session):
    """A portfolio with no paper orders yields an empty series (no raise)."""
    assert build_nav_series(_PID, session=db_session, base_currency="EUR") == []


def test_nav_series_non_base_mandate_without_fx_coverage_raises(db_session):
    """A non-EUR mandate with a foreign holding but no FX rows fails loud (C1).

    The ledger seeds EUR-base rates only; marking in GBP leaves the USD holding
    with no rate, which would silently mix currencies — so it raises instead.
    """
    _seed_ledger_two_day(db_session)
    with pytest.raises(performance.FxCoverageError):
        build_nav_series(_PID, session=db_session, base_currency="GBP")


def test_nav_series_foreign_benchmark_without_fx_does_not_block(db_session):
    """A foreign benchmark with no FX rate degrades; it must not block an all-base NAV.

    The coverage guard considers holdings only — an all-EUR book with a USD
    benchmark and an empty FX table still marks NAV (regression guard for the
    benchmark-currency false positive).
    """
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

    eu1, spx = _inst("EU1"), _inst("SPX")
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=eu1.id, date=_D1, close=Decimal("100"), price_unit="EUR"
            ),
            PriceHistory(
                instrument_id=eu1.id, date=_D2, close=Decimal("110"), price_unit="EUR"
            ),
            PriceHistory(
                instrument_id=spx.id, date=_D1, close=Decimal("5000"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=spx.id, date=_D2, close=Decimal("5100"), price_unit="USD"
            ),
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=dt.date(2026, 9, 30),
            weights_hash="h1",
            weights={"EU1": 1.0},
            fill_date=_D1,
            lines=[{"ticker": "EU1", "notional": 10000.0, "fill_price": 100.0}],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()

    points = build_nav_series(
        _PID, session=db_session, base_currency="EUR", benchmark="SPX"
    )
    navs = {p.date: p.nav for p in points}
    assert navs[_D1] == pytest.approx(10000.0, rel=1e-6)  # pure-EUR NAV, not blocked
    assert navs[_D2] == pytest.approx(11000.0, rel=1e-6)  # 100 shares × 110


def test_nav_series_since_bounds_emitted_points(db_session):
    """``since`` trims emitted bars while shares still derive from full history."""
    d3 = _seed_ledger_two_day(db_session)  # fills on _D2; next bar d3
    points = build_nav_series(_PID, session=db_session, base_currency="EUR", since=d3)
    assert [p.date for p in points] == [d3]  # the _D2 fill bar is trimmed
    # The value still reflects shares re-derived from the full fill history.
    expected_d3 = (6000.0 / 90.0) * 101.2 + (4000.0 / 5.85) * 6.136
    assert points[0].nav == pytest.approx(expected_d3, rel=1e-6)


def test_nav_series_uses_each_lines_own_fill_date(db_session):
    """Staggered fills: each line's shares use the price at *its* fill date.

    The ticket's ``fill_date`` is ``min`` across lines; a line that fills later must
    not be priced at that earlier date. Base USD / USD tickers isolate the share
    derivation (no FX).
    """
    d_a = dt.date(2026, 10, 2)  # ticket min fill (US1 fills here)
    d_b = dt.date(2026, 10, 3)  # US2 fills here
    d_last = dt.date(2026, 10, 4)
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
                instrument_id=us1.id, date=d_a, close=Decimal("100"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id, date=d_b, close=Decimal("110"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us1.id,
                date=d_last,
                close=Decimal("120"),
                price_unit="USD",
            ),
            PriceHistory(
                instrument_id=us2.id, date=d_a, close=Decimal("200"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us2.id, date=d_b, close=Decimal("210"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=us2.id,
                date=d_last,
                close=Decimal("220"),
                price_unit="USD",
            ),
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=dt.date(2026, 10, 1),
            weights_hash="h1",
            weights={"US1": 0.6, "US2": 0.4},
            fill_date=d_a,  # min across lines
            lines=[
                {
                    "ticker": "US1",
                    "notional": 6000.0,
                    "fill_price": 100.0,
                    "fill_date": d_a.isoformat(),
                },
                {
                    "ticker": "US2",
                    "notional": 4000.0,
                    "fill_price": 200.0,
                    "fill_date": d_b.isoformat(),
                },
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()

    points = build_nav_series(_PID, session=db_session, base_currency="USD")
    navs = {p.date: p.nav for p in points}
    # US2 shares = 4000 / 210 (its OWN fill), not 4000 / 200 (ticket min).
    expected_last = (6000.0 / 100.0) * 120.0 + (4000.0 / 210.0) * 220.0
    assert navs[d_last] == pytest.approx(expected_last, rel=1e-6)


def test_performance_module_declares_no_agent_stack_imports():
    """performance.py imports no agent stack (optimizer/skfolio are allowed)."""
    forbidden = ("deepagents", "langgraph", "langchain", "fund.agents", "fund.audit")
    source = Path(performance.__file__).read_text(encoding="utf-8")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    offenders = [n for n in names if any(n.startswith(f) for f in forbidden)]
    assert offenders == [], (
        f"agent-stack imports leaked into performance.py: {offenders}"
    )
