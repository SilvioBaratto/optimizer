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
    holding_marks,
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


def _seed_eur_book_with_usd_benchmark(db_session, *, gbp_rates: bool = False):
    """Seed an all-EUR single-stock book plus a USD-priced ``SPX`` benchmark.

    ``gbp_rates`` adds an FX table that covers GBP only, so USD has no rate column.
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
    if gbp_rates:
        from portopt_db.repositories.market_data.fx_rates_repository import (
            FxRatesRepository,
        )

        FxRatesRepository(db_session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "GBP",
                    "date": day,
                    "rate": 1.17,
                }
                for day in (_D1, _D2)
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


def test_nav_series_foreign_benchmark_without_fx_does_not_block(db_session):
    """A foreign benchmark with no FX rate degrades; it must not block an all-base NAV.

    The coverage guard considers holdings only — an all-EUR book with a USD
    benchmark and an empty FX table still marks NAV (regression guard for the
    benchmark-currency false positive).
    """
    _seed_eur_book_with_usd_benchmark(db_session)

    points = build_nav_series(
        _PID, session=db_session, base_currency="EUR", benchmark="SPX"
    )
    navs = {p.date: p.nav for p in points}
    assert navs[_D1] == pytest.approx(10000.0, rel=1e-6)  # pure-EUR NAV, not blocked
    assert navs[_D2] == pytest.approx(11000.0, rel=1e-6)  # 100 shares × 110


@pytest.mark.parametrize("gbp_rates", [False, True], ids=["empty-fx", "gbp-only-fx"])
def test_nav_series_unconverted_foreign_benchmark_is_not_emitted(db_session, gbp_rates):
    """A benchmark whose currency has no FX column must not leak in as raw USD points.

    Emitting 5000/5100 index points beside a EUR NAV would make excess return,
    tracking error and beta compare two currencies with no FX return between them.
    """
    _seed_eur_book_with_usd_benchmark(db_session, gbp_rates=gbp_rates)

    points = build_nav_series(
        _PID, session=db_session, base_currency="EUR", benchmark="SPX"
    )

    assert points
    assert all(p.benchmark is None for p in points)


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
    # No look-ahead across the stagger: on d_a US2 has not filled, so it is carried at
    # base-currency book value (its 4000 notional), never marked at the future d_b fill
    # price — NAV stays whole (6000 US1 + 4000 committed) instead of dipping then
    # recovering as a fake gain.
    assert navs[d_a] == pytest.approx(6000.0 + 4000.0, rel=1e-6)
    # On d_b US2 fills: its shares (4000 / 210) mark at the d_b close (210) == notional.
    assert navs[d_b] == pytest.approx((6000.0 / 100.0) * 110.0 + 4000.0, rel=1e-6)


_R1 = dt.date(2026, 11, 1)
_R2 = dt.date(2026, 11, 2)
_R3 = dt.date(2026, 11, 3)
_R4 = dt.date(2026, 11, 4)


def _seed_usd_closes(db_session, closes: dict[str, dict[dt.date, str]]) -> None:
    """Seed USD price rows for ``{ticker: {date: close}}`` (base USD ⇒ no FX)."""
    ex = Exchange(name="RX")
    db_session.add(ex)
    db_session.flush()
    for ticker, by_date in closes.items():
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=day,
                close=Decimal(close),
                price_unit="USD",
            )
            for day, close in by_date.items()
        )
    db_session.flush()


def _add_ticket(
    db_session,
    *,
    asof: dt.date,
    key: str,
    lines: dict[str, tuple[float, dt.date]],
) -> None:
    """Persist a paper ticket from ``{ticker: (notional, line_fill_date)}``."""
    total = sum(n for n, _ in lines.values())
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=asof,
            weights_hash=key,
            weights={t: n / total for t, (n, _) in lines.items()},
            fill_date=min(f for _, f in lines.values()),
            lines=[
                {
                    "ticker": ticker,
                    "notional": notional,
                    "fill_date": fill.isoformat(),
                }
                for ticker, (notional, fill) in lines.items()
            ],
            notional=total,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    db_session.flush()


def test_nav_series_rebalance_into_same_weights_conserves_nav(db_session):
    """A rebalance neither creates nor destroys value: NAV is continuous at the fill.

    Every ticket is sized off the flat ticket notional, so a naive reconstruction
    re-marks the book back to that notional on the new ticket's first bar — a
    spurious one-day loss that also discards all accumulated P&L.
    """
    _seed_usd_closes(
        db_session,
        {"AAA": {_R1: "100", _R2: "110", _R3: "120", _R4: "130"}},
    )
    _add_ticket(
        db_session,
        asof=_R1 - dt.timedelta(days=1),
        key="h1",
        lines={"AAA": (10_000.0, _R1)},
    )
    _add_ticket(db_session, asof=_R2, key="h2", lines={"AAA": (10_000.0, _R3)})

    points = build_nav_series(_PID, session=db_session, base_currency="USD")

    navs = {p.date: p.nav for p in points}
    assert navs[_R2] == pytest.approx(11_000.0)
    assert navs[_R3] == pytest.approx(12_000.0)
    assert navs[_R4] == pytest.approx(13_000.0)


def test_nav_series_rebalance_carries_nav_into_new_weights(db_session):
    """The new book is sized to the NAV in force, and ``since`` does not change it.

    100% AAA doubles to 15,000 (+50%), then the portfolio rebalances to 50/50
    AAA/BBB: the series must carry on from 15,000, and re-emitting only a recent
    window must reproduce the same levels as the full backfill.
    """
    _seed_usd_closes(
        db_session,
        {
            "AAA": {_R1: "100", _R2: "150", _R3: "150", _R4: "165"},
            "BBB": {_R3: "100", _R4: "110"},
        },
    )
    _add_ticket(
        db_session,
        asof=_R1 - dt.timedelta(days=1),
        key="h1",
        lines={"AAA": (10_000.0, _R1)},
    )
    _add_ticket(
        db_session,
        asof=_R2,
        key="h2",
        lines={"AAA": (5_000.0, _R3), "BBB": (5_000.0, _R3)},
    )

    full = {
        p.date: p.nav
        for p in build_nav_series(_PID, session=db_session, base_currency="USD")
    }
    assert full[_R2] == pytest.approx(15_000.0)
    assert full[_R3] == pytest.approx(15_000.0)
    # 1.5x scale on the 50/50 book: (5000/150*165 + 5000/100*110) * 1.5.
    assert full[_R4] == pytest.approx(16_500.0)

    recent = build_nav_series(_PID, session=db_session, base_currency="USD", since=_R4)
    assert [p.date for p in recent] == [_R4]
    assert recent[0].nav == pytest.approx(16_500.0)


def test_nav_series_rebalance_with_staggered_new_line_stays_continuous(db_session):
    """A not-yet-filled line of the new ticket must not distort the switch bar."""
    _seed_usd_closes(
        db_session,
        {
            "AAA": {_R1: "100", _R2: "150", _R3: "150", _R4: "165"},
            "BBB": {_R3: "100", _R4: "110"},
        },
    )
    _add_ticket(
        db_session,
        asof=_R1 - dt.timedelta(days=1),
        key="h1",
        lines={"AAA": (10_000.0, _R1)},
    )
    _add_ticket(
        db_session,
        asof=_R2,
        key="h2",
        lines={"AAA": (5_000.0, _R3), "BBB": (5_000.0, _R4)},
    )

    navs = {
        p.date: p.nav
        for p in build_nav_series(_PID, session=db_session, base_currency="USD")
    }
    assert navs[_R3] == pytest.approx(15_000.0)
    # BBB fills at the R4 close (110): its 5000 notional is carried until then.
    assert navs[_R4] == pytest.approx((5_000.0 / 150.0 * 165.0 + 5_000.0) * 1.5)


def _seed_rebalanced_eur_book(
    db_session,
    closes: dict[str, list[float]],
    tickets: list[tuple[int, dict[str, float]]],
) -> list[dt.date]:
    """Seed EUR instruments plus fixed-notional tickets; returns the bar dates.

    ``tickets`` holds ``(bar index of the fill, weights)``; every ticket is sized at
    the flat 100k notional ``place_orders`` applies, whatever the book is worth.
    """
    bars = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(5)]
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    for ticker, series in closes.items():
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=bar,
                close=Decimal(str(close)),
                price_unit="EUR",
            )
            for bar, close in zip(bars, series, strict=True)
        )
    for n, (bar_idx, weights) in enumerate(tickets):
        db_session.add(
            PaperOrder(
                portfolio_id=_PID,
                asof=bars[bar_idx] - dt.timedelta(days=1),
                weights_hash=f"h{n}",
                weights=weights,
                fill_date=bars[bar_idx],
                lines=[
                    {
                        "ticker": ticker,
                        "notional": weight * 100_000.0,
                        "fill_date": bars[bar_idx].isoformat(),
                    }
                    for ticker, weight in weights.items()
                ],
                notional=100_000.0,
                total_commission=0.0,
                total_slippage_cost=0.0,
            )
        )
    db_session.flush()
    return bars


def test_nav_series_stays_continuous_across_a_same_weights_rebalance(db_session):
    """A rebalance fill must not re-base NAV to the ticket's flat notional.

    The book is worth 132k when the second 100k-notional ticket fills; the series has
    to carry that value across the switch instead of dropping back to 100k, which
    would print a fake loss and a fake drawdown for a monotonically rising asset.
    """
    _seed_rebalanced_eur_book(
        db_session,
        {"AAA": [100.0, 120.0, 120.0, 132.0, 132.0]},
        [(0, {"AAA": 1.0}), (3, {"AAA": 1.0})],
    )

    points = build_nav_series(_PID, session=db_session, base_currency="EUR")

    assert [p.nav for p in points] == pytest.approx(
        [100_000.0, 120_000.0, 120_000.0, 132_000.0, 132_000.0]
    )
    summary = performance.portfolio_metrics(_PID, points)
    assert summary.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert summary.returns["max"] == pytest.approx(0.32)


def test_nav_series_carries_accumulated_value_into_a_reweighted_book(db_session):
    """Switching assets keeps NAV at the old book's value and compounds from there."""
    _seed_rebalanced_eur_book(
        db_session,
        {
            "AAA": [100.0, 110.0, 110.0, 110.0, 110.0],
            "BBB": [50.0, 50.0, 50.0, 55.0, 55.0],
        },
        [(0, {"AAA": 1.0}), (2, {"BBB": 1.0})],
    )

    points = build_nav_series(_PID, session=db_session, base_currency="EUR")

    assert [p.nav for p in points] == pytest.approx(
        [100_000.0, 110_000.0, 110_000.0, 121_000.0, 121_000.0]
    )


def test_nav_series_since_keeps_the_rebalance_chain(db_session):
    """Trimming with ``since`` must not drop the value carried over earlier switches."""
    bars = _seed_rebalanced_eur_book(
        db_session,
        {"AAA": [100.0, 120.0, 120.0, 132.0, 132.0]},
        [(0, {"AAA": 1.0}), (3, {"AAA": 1.0})],
    )

    points = build_nav_series(
        _PID, session=db_session, base_currency="EUR", since=bars[4]
    )

    assert [(p.date, p.nav) for p in points] == [(bars[4], pytest.approx(132_000.0))]


def test_holding_marks_value_matches_the_chain_linked_nav(db_session):
    """Holdings are sized on the same chain as NAV, so they sum to the last NAV.

    The second flat-100k ticket fills when the book is worth 130k; NAV carries that
    value, so the holding must hold the shares that 130k buys (1000), not the 769 a
    flat 100k would. Unrealized P&L is measured from the value carried in at the
    rebalance (130k), on the same scale as NAV.
    """
    _seed_rebalanced_eur_book(
        db_session,
        {"AAA": [100.0, 120.0, 130.0, 150.0, 170.0]},
        [(0, {"AAA": 1.0}), (2, {"AAA": 1.0})],
    )

    nav = build_nav_series(_PID, session=db_session, base_currency="EUR")[-1].nav
    (mark,) = holding_marks(_PID, session=db_session, base_currency="EUR")

    assert nav == pytest.approx(170_000.0)
    assert mark.value == pytest.approx(nav)
    assert mark.shares == pytest.approx(1000.0)
    assert mark.unrealized_pnl == pytest.approx(170_000.0 - 130_000.0)


def test_holding_marks_scale_every_line_of_a_reweighted_book(db_session):
    """A reweighted ticket scales all its lines by the one carried-over factor."""
    _seed_rebalanced_eur_book(
        db_session,
        {
            "AAA": [100.0, 110.0, 120.0, 130.0, 140.0],
            "BBB": [50.0, 50.0, 60.0, 60.0, 66.0],
        },
        [(0, {"AAA": 0.5, "BBB": 0.5}), (2, {"AAA": 0.25, "BBB": 0.75})],
    )

    nav = build_nav_series(_PID, session=db_session, base_currency="EUR")[-1].nav
    marks = {
        m.ticker: m
        for m in holding_marks(_PID, session=db_session, base_currency="EUR")
    }

    # The old book is worth 120k at the switch, so the new one is sized 1.2x flat.
    assert marks["AAA"].shares == pytest.approx(250.0)
    assert marks["BBB"].shares == pytest.approx(1500.0)
    assert sum(m.value for m in marks.values()) == pytest.approx(nav)
    assert marks["AAA"].unrealized_pnl == pytest.approx(250.0 * 140.0 - 30_000.0)
    assert marks["BBB"].unrealized_pnl == pytest.approx(1500.0 * 66.0 - 90_000.0)
    assert marks["AAA"].weight == pytest.approx(35_000.0 / nav)


def test_holding_marks_single_ticket_book_keeps_the_flat_notional(db_session):
    """With no rebalance the chain is trivial: shares and P&L are the flat ticket's."""
    _seed_rebalanced_eur_book(
        db_session,
        {"AAA": [100.0, 120.0, 130.0, 150.0, 170.0]},
        [(0, {"AAA": 1.0})],
    )

    (mark,) = holding_marks(_PID, session=db_session, base_currency="EUR")

    assert mark.shares == pytest.approx(1000.0)
    assert mark.unrealized_pnl == pytest.approx(70_000.0)


_DAYS = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(5)]


def _seed_usd_floats(db_session, series: dict[str, dict[dt.date, float]]) -> None:
    """Seed USD closes given as ``{ticker: {date: float}}``."""
    _seed_usd_closes(
        db_session,
        {
            ticker: {day: str(px) for day, px in closes.items()}
            for ticker, closes in series.items()
        },
    )


def _add_flat_ticket(
    db_session, *, fill: dt.date, notionals: dict[str, float], key: str
) -> None:
    """Persist one ticket whose lines all fill on ``fill`` (asof the day before)."""
    _add_ticket(
        db_session,
        asof=fill - dt.timedelta(days=1),
        key=key,
        lines={ticker: (notional, fill) for ticker, notional in notionals.items()},
    )


def _set_ticket_gross(db_session, key: str, gross: float) -> None:
    """Widen a ticket's flat paper notional beyond the sum of its lines."""
    db_session.query(PaperOrder).filter(PaperOrder.weights_hash == key).update(
        {"notional": gross}
    )
    db_session.flush()


def test_nav_series_keeps_a_later_tickets_cash_weight(db_session):
    """A rebalance into a half-invested book parks the other half in cash.

    A rises 100 -> 140. A second ticket holds A at 50% of its 100k gross and fills at
    130, when the book is worth 130,000: 65,000 goes into A and 65,000 stays in cash.
    The level carries through the hand-over, and the last bar is 65,000 * 140/130 +
    65,000 -- not the 140,000 a book scaled back up to fully invested would show.
    """
    prices = dict(zip(_DAYS, [100.0, 120.0, 130.0, 135.0, 140.0], strict=True))
    _seed_usd_floats(db_session, {"A": prices})
    _add_flat_ticket(db_session, fill=_DAYS[0], notionals={"A": 100_000.0}, key="h1")
    _add_flat_ticket(db_session, fill=_DAYS[2], notionals={"A": 50_000.0}, key="h2")
    _set_ticket_gross(db_session, "h2", 100_000.0)

    points = build_nav_series(_PID, session=db_session, base_currency="USD")

    navs = {p.date: p.nav for p in points}
    assert navs[_DAYS[1]] == pytest.approx(120_000.0)
    assert navs[_DAYS[2]] == pytest.approx(130_000.0)
    assert navs[_DAYS[3]] == pytest.approx(65_000.0 * 135.0 / 130.0 + 65_000.0)
    assert navs[_DAYS[4]] == pytest.approx(135_000.0)
    assert performance.nav_return_series(points).iloc[1:].min() >= 0.0


def test_nav_series_keeps_a_partly_invested_book_as_exposed_after_a_rebalance(
    db_session,
):
    """Repeating a 90% ticket keeps 90% of total wealth in the market.

    The first ticket's 10,000 of cash stays outside NAV (NAV starts at the 90,000
    deployed) but is still part of the wealth the second ticket is sized on. Rebalancing
    at 120, where wealth is 118,000, puts 106,200 in A and 11,800 in cash, so NAV is
    continuous at 108,000 and then moves with 106,200 of exposure, not 108,000.
    """
    prices = dict(zip(_DAYS, [100.0, 105.0, 110.0, 120.0, 126.0], strict=True))
    _seed_usd_floats(db_session, {"A": prices})
    _add_flat_ticket(db_session, fill=_DAYS[0], notionals={"A": 90_000.0}, key="h1")
    _add_flat_ticket(db_session, fill=_DAYS[3], notionals={"A": 90_000.0}, key="h2")
    _set_ticket_gross(db_session, "h1", 100_000.0)
    _set_ticket_gross(db_session, "h2", 100_000.0)

    points = build_nav_series(_PID, session=db_session, base_currency="USD")

    navs = {p.date: p.nav for p in points}
    assert navs[_DAYS[0]] == pytest.approx(90_000.0)
    assert navs[_DAYS[3]] == pytest.approx(108_000.0)
    assert navs[_DAYS[4]] == pytest.approx(
        106_200.0 * 126.0 / 120.0 + 11_800.0 - 10_000.0
    )


def test_nav_series_denominates_the_first_ticket_in_mandate_capital(db_session):
    """NAV is on the mandate's capital, not the ticket's flat paper notional.

    The ticket sizes weights against 100,000; with a 250,000 mandate the book starts
    at 250,000 and then moves with the asset, rebalances included.
    """
    prices = dict(zip(_DAYS, [100.0, 105.0, 110.0, 120.0, 126.0], strict=True))
    _seed_usd_floats(db_session, {"A": prices})
    _add_flat_ticket(db_session, fill=_DAYS[0], notionals={"A": 100_000.0}, key="h1")
    _add_flat_ticket(db_session, fill=_DAYS[3], notionals={"A": 100_000.0}, key="h2")

    points = build_nav_series(
        _PID, session=db_session, base_currency="USD", capital=250_000.0
    )

    navs = {p.date: p.nav for p in points}
    assert navs[_DAYS[0]] == pytest.approx(250_000.0)
    assert navs[_DAYS[1]] == pytest.approx(262_500.0)
    assert navs[_DAYS[4]] == pytest.approx(315_000.0)


def test_nav_series_capital_keeps_the_undeployed_cash_weight(db_session):
    """A ticket weighted 90% in the market deploys 90% of the mandate capital."""
    _seed_usd_floats(db_session, {"A": dict.fromkeys(_DAYS, 100.0)})
    _add_flat_ticket(db_session, fill=_DAYS[0], notionals={"A": 90_000.0}, key="h1")
    # The ticket's gross is its line sum here, so widen it to the 100k book it was
    # sized against.
    db_session.query(PaperOrder).update({"notional": 100_000.0})
    db_session.flush()

    points = build_nav_series(
        _PID, session=db_session, base_currency="USD", capital=200_000.0
    )

    assert points[0].nav == pytest.approx(180_000.0)


def _agent_stack_offenders(module) -> list[str]:
    """AST-scan a module's source for any top-level agent-stack import."""
    forbidden = ("deepagents", "langgraph", "langchain", "fund.agents", "fund.audit")
    source = Path(module.__file__).read_text(encoding="utf-8")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return [n for n in names if any(n.startswith(f) for f in forbidden)]


def test_performance_module_declares_no_agent_stack_imports():
    """performance.py imports no agent stack (optimizer/skfolio are allowed)."""
    offenders = _agent_stack_offenders(performance)
    assert offenders == [], (
        f"agent-stack imports leaked into performance.py: {offenders}"
    )


def test_attribution_module_declares_no_agent_stack_imports():
    """attribution.py imports no agent stack (optimizer/skfolio are allowed)."""
    from fund import attribution

    offenders = _agent_stack_offenders(attribution)
    assert offenders == [], (
        f"agent-stack imports leaked into attribution.py: {offenders}"
    )
