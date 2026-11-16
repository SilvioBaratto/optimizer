"""``fund.performance``: NAV inputs are loaded once; the NAV maths need no session.

``build_nav_series`` is ``load_nav_inputs`` (the reads, the FX-coverage guard and the
dropped-benchmark warning) followed by the pure ``compute_nav``. The golden series
were printed by ``build_nav_series`` at commit ``afe2dba5``, before the split, over
the panel, FX rates and tickets ``_seed_nav_book`` builds; the composed path must
reproduce them to the last bit.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

import numpy as np
import pandas as pd
import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository

from fund.audit.orders_repository import FilledTicket, OrderRepository, weights_hash
from fund.performance import (
    FxCoverageError,
    NavInputs,
    NavPoint,
    build_nav_series,
    compute_nav,
    load_nav_inputs,
)

_D0 = dt.date(2024, 3, 4)


def _day(offset: int) -> dt.date:
    return _D0 + dt.timedelta(days=offset)


def _hand_built(
    *, since: dt.date | None = None, benchmark: str | None = None
) -> NavInputs:
    """One ticket on AAA and BBB; AAA has no price on the third bar."""
    ticket = FilledTicket(
        id=uuid.uuid4(),
        fill_date=_D0,
        created_at=dt.datetime(2024, 3, 1, 12, 0),
        notional=1_000.0,
        lines=(
            {"ticker": "AAA", "notional": 600.0, "fill_date": _D0.isoformat()},
            {"ticker": "BBB", "notional": 400.0, "fill_date": _D0.isoformat()},
        ),
    )
    base_prices = pd.DataFrame(
        {"AAA": [10.0, 11.0, np.nan, 12.0], "BBB": [20.0, 20.0, 21.0, 22.0]},
        index=pd.DatetimeIndex([pd.Timestamp(_day(i)) for i in range(4)]),
    )
    return NavInputs(
        tickets=(ticket,),
        base_prices=base_prices,
        benchmark=benchmark,
        capital=None,
        since=since,
    )


class TestComputeNav:
    """``compute_nav`` marks a hand-built book without a session."""

    def test_a_bar_missing_a_held_price_is_not_marked(self) -> None:
        """60 AAA and 20 BBB from the fill; the bar without an AAA price is skipped."""
        points = compute_nav(_hand_built())

        assert points == [
            NavPoint(date=_day(0), nav=1_000.0, benchmark=None),
            NavPoint(date=_day(1), nav=1_060.0, benchmark=None),
            NavPoint(date=_day(3), nav=1_160.0, benchmark=None),
        ]

    def test_since_bounds_the_emitted_bars(self) -> None:
        """Bars before ``since`` are marked from the same shares but not emitted."""
        points = compute_nav(_hand_built(since=_day(1)))

        assert [point.date for point in points] == [_day(1), _day(3)]

    def test_the_benchmark_column_is_emitted_beside_nav(self) -> None:
        """The benchmark level is the base-currency price on each emitted bar."""
        points = compute_nav(_hand_built(benchmark="BBB"))

        assert [point.benchmark for point in points] == [20.0, 20.0, 22.0]

    def test_a_book_with_no_priced_line_has_no_points(self) -> None:
        """Tickets whose lines have no price column produce an empty series."""
        inputs = _hand_built()
        unpriced = NavInputs(
            tickets=inputs.tickets,
            base_prices=inputs.base_prices.rename(columns={"AAA": "X", "BBB": "Y"}),
            benchmark=None,
            capital=None,
            since=None,
        )

        assert compute_nav(unpriced) == []


# --- golden series --------------------------------------------------------------

_START = dt.date(2024, 1, 1)
_N_DAYS = 33
_LISTINGS = {"LON": (2500.0, "GBp"), "NYS": (200.0, "USD"), "EUX": (80.0, "EUR")}
_NAV_PID = uuid.UUID("66666666-6666-6666-6666-666666666666")
_RUNS = (
    {"benchmark": "EUX", "capital": 120_000.0},
    {"benchmark": "NYS", "since": dt.date(2024, 1, 25)},
    {"benchmark": "QQQ", "asof": dt.date(2024, 1, 28)},
)

_GOLDEN_WITH_CAPITAL = [
    ("2024-01-10", 120000.0, 81.74536),
    ("2024-01-11", 117923.33573577799, 79.4192),
    ("2024-01-12", 120292.80037975978, 81.144),
    ("2024-01-13", 121752.57419434818, 82.87296),
    ("2024-01-14", 119379.12204489793, 80.53848),
    ("2024-01-15", 121941.68045070296, 82.27056),
    ("2024-01-16", 118581.16906972471, 79.9288),
    ("2024-01-17", 120963.58787345557, 81.664),
    ("2024-01-18", 123601.05980499898, 83.40336),
    ("2024-01-19", 120226.20059036127, 81.05328),
    ("2024-01-20", 121698.54725020542, 82.79576),
    ("2024-01-21", 118905.3622616796, 80.4384),
    ("2024-01-22", 121022.47367856346, 82.184),
    ("2024-01-23", 123080.92341983304, 83.93376),
    ("2024-01-24", 120254.92866405431, 81.56808),
    ("2024-01-25", 122294.18113635533, 83.32096),
    ("2024-01-26", 119568.4206197387, 80.948),
    ("2024-01-27", 121612.4293246943, 82.704),
    ("2024-01-28", 123660.40477752077, 84.46416),
    ("2024-01-29", 120838.6593528162, 82.08288),
    ("2024-01-30", 122977.86917391635, 83.84616),
    ("2024-01-31", 120146.46978160368, 81.4576),
    ("2024-02-01", 122224.02990428926, 83.224),
    ("2024-02-02", 124261.71252118595, 84.99456),
]

_GOLDEN_SINCE = [
    ("2024-01-25", 101911.81761362944, 190.32590288),
    ("2024-01-26", 99640.3505164489, 185.330446),
    ("2024-01-27", 101343.69110391191, 185.18878700000002),
    ("2024-01-28", 103050.33731460063, 193.08506976),
    ("2024-01-29", 100698.8827940135, 187.49781864),
    ("2024-01-30", 102481.55764493028, 191.96578331999999),
    ("2024-01-31", 100122.05815133639, 186.3546244),
    ("2024-02-01", 101853.35825357436, 190.39570600000002),
    ("2024-02-02", 103551.42710098828, 194.14882368),
]

_GOLDEN_THROUGH_ASOF = [
    ("2024-01-10", 100000.0, None),
    ("2024-01-11", 98269.44644648167, None),
    ("2024-01-12", 100244.00031646647, None),
    ("2024-01-13", 101460.47849529015, None),
    ("2024-01-14", 99482.60170408161, None),
    ("2024-01-15", 101618.06704225246, None),
    ("2024-01-16", 98817.64089143724, None),
    ("2024-01-17", 100802.9898945463, None),
    ("2024-01-18", 103000.88317083247, None),
    ("2024-01-19", 100188.50049196774, None),
    ("2024-01-20", 101415.45604183784, None),
    ("2024-01-21", 99087.80188473298, None),
    ("2024-01-22", 100852.06139880288, None),
    ("2024-01-23", 102567.4361831942, None),
    ("2024-01-24", 100212.44055337859, None),
    ("2024-01-25", 101911.81761362944, None),
    ("2024-01-26", 99640.3505164489, None),
    ("2024-01-27", 101343.69110391191, None),
    ("2024-01-28", 103050.33731460063, None),
]


def _close_on(start: float, i: int) -> float:
    """A drifting, jittered close for day ``i``."""
    return round(start * (1.0 + 0.0013 * i), 6) * (1.0 + 0.01 * ((i * 7) % 5 - 2))


def _seed_listing(db_session, ticker: str, start: float, unit: str | None) -> None:
    """One instrument; NYS skips every seventh bar from the sixth on."""
    exchange = Exchange(name=f"EX-{ticker}")
    db_session.add(exchange)
    db_session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=exchange.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    db_session.add(instrument)
    db_session.flush()
    for i in range(_N_DAYS):
        if ticker == "NYS" and i % 7 == 5:
            continue
        db_session.add(
            PriceHistory(
                instrument_id=instrument.id,
                date=_START + dt.timedelta(days=i),
                close=_close_on(start, i),
                price_unit=unit,
                volume=1000,
            )
        )


def _seed_ticket(db_session, asof, fill, lines) -> None:
    """A ticket whose lines are ``(ticker, base notional, fill date)``."""
    notional = 100_000.0
    weights = {ticker: line_notional / notional for ticker, line_notional, _ in lines}
    OrderRepository(db_session).create(
        portfolio_id=_NAV_PID,
        asof=asof,
        weights_hash=weights_hash(weights),
        weights=weights,
        fill_date=fill,
        lines=[
            {"ticker": ticker, "notional": line_notional, "fill_date": fill_day}
            for ticker, line_notional, fill_day in lines
        ],
        notional=notional,
        total_commission=0.0,
        total_slippage_cost=0.0,
    )


def _seed_nav_book(db_session) -> None:
    """Three listings, EUR rates with gaps, and two tickets (staggered fills)."""
    for ticker, (start, unit) in _LISTINGS.items():
        _seed_listing(db_session, ticker, start, unit)
    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": quote,
                "date": _START + dt.timedelta(days=k),
                "rate": round(rate + 0.0007 * ((k * 3) % 4), 6),
            }
            for quote, rate in (("GBP", 1.1713), ("USD", 0.9137))
            for k in range(_N_DAYS)
            if k % 9 != 4
        ]
    )
    _seed_ticket(
        db_session,
        dt.date(2024, 1, 9),
        dt.date(2024, 1, 10),
        [
            ("LON", 40_000.0, "2024-01-10"),
            ("NYS", 35_000.0, "2024-01-11"),
            ("EUX", 20_000.0, "2024-01-10"),
        ],
    )
    _seed_ticket(
        db_session,
        dt.date(2024, 1, 19),
        dt.date(2024, 1, 20),
        [("LON", 30_000.0, "2024-01-20"), ("EUX", 50_000.0, "2024-01-20")],
    )
    db_session.flush()


def _as_points(rows) -> list[NavPoint]:
    return [
        NavPoint(date=dt.date.fromisoformat(day), nav=nav, benchmark=bench)
        for day, nav, bench in rows
    ]


_GOLDEN = (_GOLDEN_WITH_CAPITAL, _GOLDEN_SINCE, _GOLDEN_THROUGH_ASOF)


class TestGoldenNavSeries:
    """The composed ``build_nav_series`` reproduces the series from before the split."""

    @pytest.mark.parametrize(
        ("kwargs", "golden"), list(zip(_RUNS, _GOLDEN, strict=True))
    )
    def test_build_nav_series_equals_the_golden_series(
        self, db_session, kwargs, golden
    ) -> None:
        """Each run's points equal the printed ones, float for float."""
        _seed_nav_book(db_session)

        points = build_nav_series(
            _NAV_PID, session=db_session, base_currency="EUR", **kwargs
        )

        assert points == _as_points(golden)

    @pytest.mark.parametrize("kwargs", _RUNS)
    def test_build_nav_series_is_compute_nav_of_the_loaded_inputs(
        self, db_session, kwargs
    ) -> None:
        """``compute_nav(load_nav_inputs(...))`` gives the same points."""
        _seed_nav_book(db_session)

        inputs = load_nav_inputs(
            _NAV_PID, session=db_session, base_currency="EUR", **kwargs
        )

        assert inputs is not None
        assert compute_nav(inputs) == build_nav_series(
            _NAV_PID, session=db_session, base_currency="EUR", **kwargs
        )


class TestLoadNavInputs:
    """``load_nav_inputs`` keeps the reads, the guards and the warning."""

    def test_no_tickets_means_nothing_to_mark(self, db_session) -> None:
        """A portfolio with no filled ticket loads no inputs."""
        assert (
            load_nav_inputs(uuid.uuid4(), session=db_session, base_currency="EUR")
            is None
        )

    def test_a_holding_without_a_currency_fails_loud(self, db_session) -> None:
        """A held listing with no price currency raises ``FxCoverageError``."""
        _seed_listing(db_session, "ZZZ", 50.0, None)
        _seed_ticket(
            db_session,
            dt.date(2024, 1, 9),
            dt.date(2024, 1, 10),
            [("ZZZ", 100_000.0, "2024-01-10")],
        )
        db_session.flush()

        with pytest.raises(FxCoverageError):
            load_nav_inputs(_NAV_PID, session=db_session, base_currency="EUR")

    def test_a_benchmark_left_out_is_logged_and_not_emitted(
        self, db_session, caplog
    ) -> None:
        """An unknown benchmark is warned about once, by the loader."""
        _seed_nav_book(db_session)

        with caplog.at_level(logging.WARNING, logger="fund.performance"):
            inputs = load_nav_inputs(
                _NAV_PID, session=db_session, base_currency="EUR", benchmark="QQQ"
            )
            assert inputs is not None
            warned = [r for r in caplog.records if "left out of the NAV" in r.message]
            compute_nav(inputs)

        assert inputs.benchmark is None
        assert len(warned) == 1
        assert [
            r for r in caplog.records if "left out of the NAV" in r.message
        ] == warned
