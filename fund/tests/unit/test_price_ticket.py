"""``price_ticket``: the paper cost model as a pure function, and its golden numbers.

The unit tests build fill quotes by hand and need no database. The golden tests
pin the exact lines and totals of three tickets: the literals were printed by
``place_orders`` at commit ``afe2dba5`` (before pricing moved out of it) over the
price panel and FX rates ``_seed_golden_panel`` builds. ``price_ticket`` fed the
same quotes, and ``place_orders`` over the same panel, must reproduce them to the
last bit, which shows the move changed no number in them.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository

from fund.pricing import (
    BPS_PER_UNIT,
    CostModel,
    FillQuote,
    held_weights,
    price_ticket,
)
from fund.tools.orders import place_orders

_DAY = dt.date(2024, 1, 31)
_COSTS = CostModel(slippage_bps=5.0, commission_bps=1.0)
_NOTIONAL = 100_000.0


def _quote(price: float, *, base_price: float | None = None) -> FillQuote:
    """A fill on ``_DAY`` at ``price``, worth ``base_price`` (default ``price``)."""
    return FillQuote(
        date=_DAY,
        price=price,
        base_price=price if base_price is None else base_price,
        unit="EUR",
    )


def _lines_by_ticker(priced) -> dict[str, dict]:
    return {line["ticker"]: line for line in priced.lines}


class TestCostModel:
    """``fund.pricing.price_ticket``: the per-line cost model and the two totals."""

    def test_bps_per_unit_is_ten_thousand(self) -> None:
        """The constant equals the ``1e4`` literal it replaced, bit for bit."""
        assert BPS_PER_UNIT == 10_000.0
        assert BPS_PER_UNIT == 1e4

    def test_buy_pays_slippage_up(self) -> None:
        """A buy fills above the close and buys its shares at that price."""
        priced = price_ticket(
            {"AAA": 0.5}, {}, {"AAA": _quote(100.0)}, _NOTIONAL, _COSTS
        )

        line = priced.lines[0]
        assert line["traded_notional"] == 50_000.0
        assert line["effective_price"] == 100.0 * (1.0 + 5.0 / 1e4)
        assert line["shares"] == 50_000.0 / (100.0 * (1.0 + 5.0 / 1e4))

    def test_sell_receives_less(self) -> None:
        """A trim of a long is a sell and fills below the close."""
        priced = price_ticket(
            {"AAA": 0.4}, {"AAA": 0.6}, {"AAA": _quote(100.0)}, _NOTIONAL, _COSTS
        )

        line = priced.lines[0]
        assert line["traded_notional"] < 0
        assert line["effective_price"] < line["fill_price"]
        assert line["effective_price"] == 100.0 * (1.0 - 5.0 / 1e4)

    def test_unchanged_name_fills_at_the_close(self) -> None:
        """A name held at its prior weight pays no slippage and fills at the close."""
        priced = price_ticket(
            {"AAA": 0.5}, {"AAA": 0.5}, {"AAA": _quote(100.0)}, _NOTIONAL, _COSTS
        )

        line = priced.lines[0]
        assert line["effective_price"] == line["fill_price"]
        assert line["shares"] == 50_000.0 / 100.0
        assert line["slippage_cost"] == 0.0

    def test_commission_applies_only_to_traded_notional(self) -> None:
        """Commission is charged on the move from the prior weight, not the line."""
        priced = price_ticket(
            {"AAA": 0.6, "BBB": 0.4},
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": _quote(100.0), "BBB": _quote(50.0)},
            _NOTIONAL,
            _COSTS,
        )

        lines = _lines_by_ticker(priced)
        for line in lines.values():
            assert math.isclose(line["commission"], 10_000.0 * 1.0 / 1e4)
            assert line["notional"] != line["traded_notional"]
        assert math.isclose(priced.total_commission, 20_000.0 * 1.0 / 1e4)

    def test_zero_traded_notional_costs_nothing(self) -> None:
        """A ticket that equals the prior one has zero cost totals."""
        priced = price_ticket(
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": _quote(100.0), "BBB": _quote(50.0)},
            _NOTIONAL,
            _COSTS,
        )

        assert priced.total_commission == 0.0
        assert priced.total_slippage_cost == 0.0

    def test_dropped_name_adds_exit_cost_but_no_line(self) -> None:
        """A prior name missing from the target is charged its sale, with no line."""
        priced = price_ticket(
            {"AAA": 0.5},
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": _quote(100.0)},
            _NOTIONAL,
            _COSTS,
        )

        assert [line["ticker"] for line in priced.lines] == ["AAA"]
        assert priced.total_commission == 50_000.0 * 1.0 / 1e4
        assert priced.total_slippage_cost == 50_000.0 * 5.0 / 1e4

    def test_shares_divide_by_the_base_currency_price(self) -> None:
        """Shares use the base-currency price; the quoted price stays on the line."""
        priced = price_ticket(
            {"LON": 0.1},
            {},
            {"LON": _quote(2500.0, base_price=29.25)},
            _NOTIONAL,
            _COSTS,
        )

        line = priced.lines[0]
        assert line["fill_price"] == 2500.0
        assert line["fill_price_base"] == 29.25
        assert line["effective_price"] == 2500.0 * (1.0 + 5.0 / 1e4)
        assert line["shares"] == 10_000.0 / (29.25 * (1.0 + 5.0 / 1e4))

    def test_a_zero_base_price_gives_no_shares(self) -> None:
        """A zero base price yields zero shares instead of a division error."""
        priced = price_ticket({"AAA": 0.5}, {}, {"AAA": _quote(0.0)}, _NOTIONAL, _COSTS)

        assert priced.lines[0]["shares"] == 0.0

    def test_line_carries_the_stored_ticket_keys(self) -> None:
        """A line has the keys of a stored ticket line, in their written order."""
        priced = price_ticket(
            {"AAA": 0.5}, {}, {"AAA": _quote(100.0)}, _NOTIONAL, _COSTS
        )

        assert list(priced.lines[0]) == [
            "ticker",
            "weight",
            "fill_date",
            "currency",
            "fill_price",
            "fill_price_base",
            "effective_price",
            "slippage_bps",
            "notional",
            "traded_notional",
            "shares",
            "commission",
            "slippage_cost",
        ]

    def test_a_held_name_without_a_quote_raises(self) -> None:
        """Every held name needs a quote; a missing one is a ``KeyError``."""
        with pytest.raises(KeyError):
            price_ticket({"AAA": 0.5}, {}, {}, _NOTIONAL, _COSTS)


class TestHeldWeights:
    """``fund.pricing.held_weights``: which weights carry an order."""

    def test_drops_zero_and_dust_weights(self) -> None:
        """Weights at zero or below the dust threshold carry no order."""
        assert held_weights({"AAA": 0.5, "BBB": 0.0, "CCC": 1e-9}) == {"AAA": 0.5}

    def test_keeps_a_nan_weight_so_it_fails_downstream(self) -> None:
        """A NaN weight is kept, so it fails loudly instead of passing as dust."""
        held = held_weights({"AAA": float("nan")})

        assert list(held) == ["AAA"]


# --- golden numbers -------------------------------------------------------------

_START = dt.date(2024, 1, 1)
_ASOF = dt.date(2024, 1, 30)
_NEXT_ASOF = dt.date(2024, 1, 31)
_N_DAYS = 33
_LISTINGS = {
    "AAA": (100.0, None),
    "LON": (2500.0, "GBp"),
    "NYS": (200.0, "USD"),
    "EUX": (80.0, "EUR"),
}
_PID = uuid.UUID("44444444-4444-4444-4444-444444444444")
_QUOTED_PID = uuid.UUID("55555555-5555-5555-5555-555555555555")

_FIRST_WEIGHTS = {"AAA": 0.51, "LON": 0.29, "NYS": 0.19}
_SECOND_WEIGHTS = {"AAA": 0.437, "LON": 0.3, "EUX": 0.173, "ZZZ": 1e-9}
_QUOTED_WEIGHTS = {"AAA": 0.61, "NYS": 0.39}
_QUOTED_COSTS = {"notional": 250_000.0, "slippage_bps": 7.3, "commission_bps": 1.7}

_FIRST_LINES = [
    {
        "ticker": "AAA",
        "weight": 0.51,
        "fill_date": "2024-01-31",
        "currency": None,
        "fill_price": 103.9,
        "fill_price_base": 103.9,
        "effective_price": 103.95195,
        "slippage_bps": 5.0,
        "notional": 51000.0,
        "traded_notional": 51000.0,
        "shares": 490.61128723415004,
        "commission": 5.1,
        "slippage_cost": 25.5,
    },
    {
        "ticker": "LON",
        "weight": 0.29,
        "fill_date": "2024-01-31",
        "currency": "GBp",
        "fill_price": 2597.5,
        "fill_price_base": 30.4427,
        "effective_price": 2598.79875,
        "slippage_bps": 5.0,
        "notional": 28999.999999999996,
        "traded_notional": 28999.999999999996,
        "shares": 952.1332617138694,
        "commission": 2.8999999999999995,
        "slippage_cost": 14.499999999999996,
    },
    {
        "ticker": "NYS",
        "weight": 0.19,
        "fill_date": "2024-01-31",
        "currency": "USD",
        "fill_price": 207.8,
        "fill_price_base": 190.01232000000002,
        "effective_price": 207.9039,
        "slippage_bps": 5.0,
        "notional": 19000.0,
        "traded_notional": 19000.0,
        "shares": 99.94354443767745,
        "commission": 1.9,
        "slippage_cost": 9.5,
    },
]
_FIRST_TOTALS = (9.899999999999999, 49.5)

_SECOND_LINES = [
    {
        "ticker": "AAA",
        "weight": 0.437,
        "fill_date": "2024-02-01",
        "currency": None,
        "fill_price": 104.03,
        "fill_price_base": 104.03,
        "effective_price": 103.977985,
        "slippage_bps": 5.0,
        "notional": 43700.0,
        "traded_notional": -7300.000000000001,
        "shares": 420.0360297422575,
        "commission": 0.7300000000000001,
        "slippage_cost": 3.650000000000001,
    },
    {
        "ticker": "LON",
        "weight": 0.3,
        "fill_date": "2024-02-01",
        "currency": "GBp",
        "fill_price": 2600.75,
        "fill_price_base": 30.498995250000004,
        "effective_price": 2602.050375,
        "slippage_bps": 5.0,
        "notional": 30000.0,
        "traded_notional": 1000.0000000000009,
        "shares": 983.622575234673,
        "commission": 0.10000000000000009,
        "slippage_cost": 0.5000000000000004,
    },
    {
        "ticker": "EUX",
        "weight": 0.173,
        "fill_date": "2024-02-01",
        "currency": "EUR",
        "fill_price": 83.224,
        "fill_price_base": 83.224,
        "effective_price": 83.265612,
        "slippage_bps": 5.0,
        "notional": 17300.0,
        "traded_notional": 17300.0,
        "shares": 207.76884459817578,
        "commission": 1.73,
        "slippage_cost": 8.65,
    },
]
_SECOND_TOTALS = (4.46, 22.3)

_QUOTED_LINES = [
    {
        "ticker": "AAA",
        "weight": 0.61,
        "fill_date": "2024-01-31",
        "currency": None,
        "fill_price": 103.9,
        "fill_price_base": 103.9,
        "effective_price": 103.975847,
        "slippage_bps": 7.3,
        "notional": 152500.0,
        "traded_notional": 152500.0,
        "shares": 1466.6867777475281,
        "commission": 25.925,
        "slippage_cost": 111.325,
    },
    {
        "ticker": "NYS",
        "weight": 0.39,
        "fill_date": "2024-01-31",
        "currency": "USD",
        "fill_price": 207.8,
        "fill_price_base": 207.8,
        "effective_price": 207.951694,
        "slippage_bps": 7.3,
        "notional": 97500.0,
        "traded_notional": 97500.0,
        "shares": 468.8588879684721,
        "commission": 16.575,
        "slippage_cost": 71.175,
    },
]
_QUOTED_TOTALS = (42.5, 182.5)


def _quotes_from(lines: list[dict]) -> dict[str, FillQuote]:
    """The fill quotes the stored ``lines`` were priced from."""
    return {
        line["ticker"]: FillQuote(
            date=dt.date.fromisoformat(line["fill_date"]),
            price=line["fill_price"],
            base_price=line["fill_price_base"],
            unit=line["currency"],
        )
        for line in lines
    }


class TestGoldenPriceTicket:
    """``price_ticket`` reproduces the tickets the inline pricing produced."""

    def test_first_ticket(self) -> None:
        """The first ticket: three buys, quoted without a unit, in pence and in USD."""
        priced = price_ticket(
            held_weights(_FIRST_WEIGHTS),
            {},
            _quotes_from(_FIRST_LINES),
            _NOTIONAL,
            _COSTS,
        )

        assert list(priced.lines) == _FIRST_LINES
        assert (priced.total_commission, priced.total_slippage_cost) == _FIRST_TOTALS

    def test_rebalance_with_a_sell_an_exit_and_a_dust_weight(self) -> None:
        """The rebalance: a sell, a top-up, a new name, an exit and a dust weight."""
        priced = price_ticket(
            held_weights(_SECOND_WEIGHTS),
            _FIRST_WEIGHTS,
            _quotes_from(_SECOND_LINES),
            _NOTIONAL,
            _COSTS,
        )

        assert list(priced.lines) == _SECOND_LINES
        assert (
            priced.total_commission,
            priced.total_slippage_cost,
        ) == _SECOND_TOTALS

    def test_ticket_priced_as_quoted_with_cost_overrides(self) -> None:
        """No base currency, a larger notional and other slippage and commission."""
        priced = price_ticket(
            held_weights(_QUOTED_WEIGHTS),
            {},
            _quotes_from(_QUOTED_LINES),
            _QUOTED_COSTS["notional"],
            CostModel(
                slippage_bps=_QUOTED_COSTS["slippage_bps"],
                commission_bps=_QUOTED_COSTS["commission_bps"],
            ),
        )

        assert list(priced.lines) == _QUOTED_LINES
        assert (
            priced.total_commission,
            priced.total_slippage_cost,
        ) == _QUOTED_TOTALS


def _close_on(start: float, i: int) -> float:
    """Deterministic close for day ``i``: linear drift off ``start``."""
    return round(start * (1.0 + 0.0013 * i), 6)


def _seed_golden_panel(db_session) -> None:
    """Four listings (no unit, pence, dollars, euros) and EUR-per-GBP/USD rates."""
    for ticker, (start, unit) in _LISTINGS.items():
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
            db_session.add(
                PriceHistory(
                    instrument_id=instrument.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(start, i),
                    price_unit=unit,
                    volume=1000,
                )
            )
    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": quote,
                "date": _ASOF + dt.timedelta(days=k),
                "rate": round(rate + 0.0007 * k, 6),
            }
            for quote, rate in (("GBP", 1.1713), ("USD", 0.9137))
            for k in range(4)
        ]
    )
    db_session.flush()


class TestGoldenPlaceOrders:
    """``place_orders`` over the golden panel stores the same tickets."""

    def test_first_and_rebalance_tickets(self, db_session) -> None:
        """The first ticket and the rebalance after it, in EUR."""
        _seed_golden_panel(db_session)

        first = place_orders(
            db_session, _ASOF, _FIRST_WEIGHTS, _PID, base_currency="EUR"
        )["data"]
        second = place_orders(
            db_session, _NEXT_ASOF, _SECOND_WEIGHTS, _PID, base_currency="EUR"
        )["data"]

        assert first["fill_date"] == "2024-01-31"
        assert first["lines"] == _FIRST_LINES
        assert (
            first["total_commission"],
            first["total_slippage_cost"],
        ) == _FIRST_TOTALS
        assert second["fill_date"] == "2024-02-01"
        assert second["lines"] == _SECOND_LINES
        assert (
            second["total_commission"],
            second["total_slippage_cost"],
        ) == _SECOND_TOTALS

    def test_ticket_priced_as_quoted_with_cost_overrides(self, db_session) -> None:
        """No base currency, a larger notional and other slippage and commission."""
        _seed_golden_panel(db_session)

        quoted = place_orders(
            db_session, _ASOF, _QUOTED_WEIGHTS, _QUOTED_PID, **_QUOTED_COSTS
        )["data"]

        assert quoted["fill_date"] == "2024-01-31"
        assert quoted["lines"] == _QUOTED_LINES
        assert (
            quoted["total_commission"],
            quoted["total_slippage_cost"],
        ) == _QUOTED_TOTALS
