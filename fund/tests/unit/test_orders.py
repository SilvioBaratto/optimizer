"""``place_orders``: the idempotent paper-execution tool.

``place_orders`` turns a target-weight vector into a **simulated** order ticket
filled at the **next close strictly after** the decision bar (no look-ahead),
with a simple slippage + commission model, and persists it to the
``paper_orders`` table. It must be **idempotent**: on a HITL ``Command(resume=…)``
the interrupting node re-runs from the top, so two calls with the same
``(portfolio_id, asof, weights)`` must yield exactly **one** ticket.

These tests pin, over a seeded SQLite price panel:

* a filled ticket is persisted, priced at the next close after ``asof``;
* every fill bar is strictly after the decision bar (no look-ahead);
* slippage + commission are applied to the fill;
* a double call with the same key returns the same ticket — one DB row;
* the ``{ok, error}`` contract holds (empty weights / no future bar → error).
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import func, select

from fund.tools.orders import fill_shortfalls, place_orders

_START = dt.date(2024, 1, 1)
_ASOF = dt.date(2024, 1, 30)  # decision bar (index 29)
_N_DAYS = 33  # 30 in-sample + 3 future bars (indices 30, 31, 32)
_FILL_DATE = dt.date(2024, 1, 31)  # first close strictly after _ASOF (index 30)
_UNIVERSE = ["AAA", "BBB", "CCC"]
_WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
_PORTFOLIO_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
# Deterministic per-ticker starting price + daily drift — no RNG, so the whole
# fill path is reproducible and the next-close price is trivially assertable.
_SERIES = {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}


def _close_on(start: float, i: int) -> float:
    """Deterministic close for day ``i``: linear drift off ``start``."""
    return round(start * (1.0 + 0.001 * i), 6)


def _seed_panel(db_session) -> None:
    """Seed an ``_N_DAYS`` x 3 close panel that extends past ``_ASOF``."""
    for ticker, start in _SERIES.items():
        ex = Exchange(name=f"EX-{ticker}")
        db_session.add(ex)
        db_session.flush()
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=ex.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(inst)
        db_session.flush()
        for i in range(_N_DAYS):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(start, i),
                    volume=1000,
                )
            )
    db_session.flush()


def _fill_index() -> int:
    """0-based day index of the fill bar (first strictly after ``_ASOF``)."""
    return (_FILL_DATE - _START).days


class TestPlaceOrders:
    def test_ticket_filled_at_next_close_and_persisted(self, db_session) -> None:
        _seed_panel(db_session)

        result = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)

        assert result["ok"] is True
        ticket = result["data"]
        assert ticket["fill_date"] == _FILL_DATE.isoformat()
        assert ticket["status"] == "filled"
        assert {line["ticker"] for line in ticket["lines"]} == set(_UNIVERSE)

        idx = _fill_index()
        by_ticker = {line["ticker"]: line for line in ticket["lines"]}
        for ticker, start in _SERIES.items():
            assert by_ticker[ticker]["fill_price"] == _close_on(start, idx)

        stored = db_session.execute(select(PaperOrder)).scalars().all()
        assert len(stored) == 1
        assert str(stored[0].id) == ticket["order_id"]

    def test_fill_bar_strictly_after_decision_bar(self, db_session) -> None:
        _seed_panel(db_session)

        ticket = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)["data"]

        assert dt.date.fromisoformat(ticket["fill_date"]) > _ASOF
        for line in ticket["lines"]:
            assert dt.date.fromisoformat(line["fill_date"]) > _ASOF

    def test_slippage_and_commission_applied(self, db_session) -> None:
        _seed_panel(db_session)

        ticket = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)["data"]

        assert ticket["total_commission"] > 0.0
        assert ticket["total_slippage_cost"] > 0.0
        for line in ticket["lines"]:
            # Buys pay up: the effective price exceeds the raw fill.
            assert line["effective_price"] > line["fill_price"]
            assert line["commission"] >= 0.0

    def test_sell_side_receives_less_than_fill(self, db_session) -> None:
        _seed_panel(db_session)

        # AAA long (pays up), BBB short (receives less) — budget still 1.0.
        ticket = place_orders(
            db_session, _ASOF, {"AAA": 1.2, "BBB": -0.2}, _PORTFOLIO_ID
        )["data"]

        by_ticker = {line["ticker"]: line for line in ticket["lines"]}
        assert by_ticker["AAA"]["effective_price"] > by_ticker["AAA"]["fill_price"]
        # Sells cross the spread the other way: effective below the raw fill.
        assert by_ticker["BBB"]["effective_price"] < by_ticker["BBB"]["fill_price"]

    def test_accepts_iso_string_asof(self, db_session) -> None:
        _seed_panel(db_session)

        result = place_orders(db_session, _ASOF.isoformat(), _WEIGHTS, _PORTFOLIO_ID)

        assert result["ok"] is True
        assert result["data"]["fill_date"] == _FILL_DATE.isoformat()

    def test_accepts_datetime_asof(self, db_session) -> None:
        _seed_panel(db_session)

        ticket = place_orders(
            db_session,
            dt.datetime(2024, 1, 30, 16, 0),
            _WEIGHTS,
            _PORTFOLIO_ID,
        )["data"]

        assert ticket["fill_date"] == _FILL_DATE.isoformat()

    def test_accepts_string_portfolio_id(self, db_session) -> None:
        _seed_panel(db_session)

        ticket = place_orders(db_session, _ASOF, _WEIGHTS, str(_PORTFOLIO_ID))["data"]

        assert ticket["portfolio_id"] == str(_PORTFOLIO_ID)

    def test_cost_overrides_scale_the_totals(self, db_session) -> None:
        _seed_panel(db_session)
        other_portfolio = uuid.UUID("22222222-2222-2222-2222-222222222222")

        base = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)["data"]
        # A distinct portfolio_id → a fresh ticket (cost params aren't in the key).
        pricier = place_orders(
            db_session, _ASOF, _WEIGHTS, other_portfolio, slippage_bps=50.0
        )["data"]

        assert pricier["total_slippage_cost"] > base["total_slippage_cost"]

    def test_cost_override_on_existing_key_is_not_silently_stale(
        self, db_session
    ) -> None:
        _seed_panel(db_session)
        first = place_orders(
            db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID, notional=100_000.0
        )
        assert first["ok"] is True
        assert first["data"]["notional"] == 100_000.0
        # Same (portfolio, asof, weights) key, different notional (a what-if).
        second = place_orders(
            db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID, notional=250_000.0
        )
        # The cost mismatch is surfaced instead of a stale ticket being returned.
        assert second["ok"] is False
        assert "error" in second
        # No double-place either way: exactly one row for the key.
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 1

    def test_same_key_rerun_with_same_cost_params_is_idempotent(
        self, db_session
    ) -> None:
        _seed_panel(db_session)
        first = place_orders(
            db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID, slippage_bps=5.0
        )
        # A same-key re-run flowing the identical cost params still hits the
        # idempotent path (the real HITL Command(resume=...) re-run).
        second = place_orders(
            db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID, slippage_bps=5.0
        )
        assert first["ok"] is True
        assert second["ok"] is True
        assert second["data"]["order_id"] == first["data"]["order_id"]
        assert second["data"]["idempotent"] is True

    def test_fill_is_earliest_bar_when_future_bars_exceed_repo_limit(
        self, db_session
    ) -> None:
        ex = Exchange(name="EX-AAA")
        db_session.add(ex)
        db_session.flush()
        inst = Instrument(
            ticker="AAA",
            short_name="AAA",
            exchange_id=ex.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker="AAA",
        )
        db_session.add(inst)
        db_session.flush()
        # 6001 consecutive daily bars starting at _START — more than the repo's
        # 5000-row cap, so a DESC+limit fetch would truncate away the true
        # next close.
        for i in range(6001):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(100.0, i),
                    volume=1000,
                )
            )
        db_session.flush()

        ticket = place_orders(db_session, _START, {"AAA": 1.0}, _PORTFOLIO_ID)["data"]

        # The fill is the first bar strictly after asof, not one ~1000 days late.
        assert ticket["fill_date"] == (_START + dt.timedelta(days=1)).isoformat()

    def test_idempotent_double_call_one_ticket(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        second = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)

        assert first["ok"] is True
        assert second["ok"] is True
        assert second["data"]["order_id"] == first["data"]["order_id"]
        assert second["data"]["idempotent"] is True
        assert first["data"]["idempotent"] is False

        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 1

    def test_distinct_weights_make_distinct_tickets(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        other = place_orders(db_session, _ASOF, {"AAA": 0.6, "BBB": 0.4}, _PORTFOLIO_ID)

        assert first["data"]["order_id"] != other["data"]["order_id"]
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 2

    def test_deterministic(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        second = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)

        assert first["data"]["weights_hash"] == second["data"]["weights_hash"]
        assert first["data"]["lines"] == second["data"]["lines"]

    def test_empty_weights_is_error(self, db_session) -> None:
        result = place_orders(db_session, _ASOF, {}, _PORTFOLIO_ID)

        assert result["ok"] is False
        assert "error" in result

    def test_no_future_bar_is_error(self, db_session) -> None:
        _seed_panel(db_session)

        # asof at the last seeded bar ⇒ no close strictly after it to fill on.
        last = _START + dt.timedelta(days=_N_DAYS - 1)
        result = place_orders(db_session, last, _WEIGHTS, _PORTFOLIO_ID)

        assert result["ok"] is False
        assert "error" in result

    def test_unknown_ticker_is_error(self, db_session) -> None:
        _seed_panel(db_session)

        result = place_orders(db_session, _ASOF, {"ZZZ": 1.0}, _PORTFOLIO_ID)

        assert result["ok"] is False
        assert "error" in result


_NEXT_ASOF = _ASOF + dt.timedelta(days=1)  # second decision bar; fills a bar later


class TestTurnoverCosts:
    """Costs are charged on traded notional (|target - held|), not the full book."""

    def test_first_ticket_trades_the_whole_book(self, db_session) -> None:
        _seed_panel(db_session)

        ticket = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)["data"]

        assert ticket["total_commission"] == pytest.approx(100_000.0 * 1.0 / 1e4)
        assert ticket["total_slippage_cost"] == pytest.approx(100_000.0 * 5.0 / 1e4)

    def test_small_rebalance_costs_only_the_turnover(self, db_session) -> None:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, {"AAA": 0.5, "BBB": 0.5}, _PORTFOLIO_ID)

        ticket = place_orders(
            db_session, _NEXT_ASOF, {"AAA": 0.51, "BBB": 0.49}, _PORTFOLIO_ID
        )["data"]

        # 2% of the 100k book changes hands (1% bought + 1% sold).
        traded = 0.02 * 100_000.0
        assert ticket["total_commission"] == pytest.approx(traded * 1.0 / 1e4)
        assert ticket["total_slippage_cost"] == pytest.approx(traded * 5.0 / 1e4)
        by_ticker = {line["ticker"]: line for line in ticket["lines"]}
        assert by_ticker["AAA"]["commission"] == pytest.approx(1_000.0 * 1.0 / 1e4)
        assert by_ticker["BBB"]["commission"] == pytest.approx(1_000.0 * 1.0 / 1e4)

    def test_exited_name_is_charged_for_the_sell(self, db_session) -> None:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, {"AAA": 0.6, "BBB": 0.4}, _PORTFOLIO_ID)

        ticket = place_orders(db_session, _NEXT_ASOF, {"AAA": 1.0}, _PORTFOLIO_ID)[
            "data"
        ]

        # Buy 40% AAA, sell out 40% BBB: 80k traded, though BBB has no ticket line.
        traded = 0.8 * 100_000.0
        assert ticket["total_commission"] == pytest.approx(traded * 1.0 / 1e4)
        assert ticket["total_slippage_cost"] == pytest.approx(traded * 5.0 / 1e4)
        assert {line["ticker"] for line in ticket["lines"]} == {"AAA"}

    def test_name_zeroed_by_the_optimiser_is_still_charged_for_the_sell(
        self, db_session
    ) -> None:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, {"AAA": 0.6, "BBB": 0.4}, _PORTFOLIO_ID)

        ticket = place_orders(
            db_session, _NEXT_ASOF, {"AAA": 1.0, "BBB": 0.0}, _PORTFOLIO_ID
        )["data"]

        traded = 0.8 * 100_000.0
        assert ticket["total_commission"] == pytest.approx(traded * 1.0 / 1e4)
        assert {line["ticker"] for line in ticket["lines"]} == {"AAA"}

    def test_unchanged_target_costs_nothing(self, db_session) -> None:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, {"AAA": 0.5, "BBB": 0.5}, _PORTFOLIO_ID)

        # Same weights at a later bar is a distinct key (asof is part of it).
        ticket = place_orders(
            db_session, _NEXT_ASOF, {"AAA": 0.5, "BBB": 0.5}, _PORTFOLIO_ID
        )["data"]

        assert ticket["total_commission"] == pytest.approx(0.0)
        assert ticket["total_slippage_cost"] == pytest.approx(0.0)

    def test_other_portfolios_holdings_are_not_netted(self, db_session) -> None:
        _seed_panel(db_session)
        other = uuid.UUID("33333333-3333-3333-3333-333333333333")
        place_orders(db_session, _ASOF, {"AAA": 0.5, "BBB": 0.5}, other)

        ticket = place_orders(
            db_session, _NEXT_ASOF, {"AAA": 0.5, "BBB": 0.5}, _PORTFOLIO_ID
        )["data"]

        assert ticket["total_commission"] == pytest.approx(100_000.0 * 1.0 / 1e4)

    def test_rerun_of_a_turnover_priced_ticket_is_idempotent(self, db_session) -> None:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, {"AAA": 0.5, "BBB": 0.5}, _PORTFOLIO_ID)
        weights = {"AAA": 0.51, "BBB": 0.49}
        first = place_orders(db_session, _NEXT_ASOF, weights, _PORTFOLIO_ID)

        # The HITL resume re-runs the tool with identical params: the stored
        # (turnover-priced) ticket must still be recognised, not reported as a
        # cost-param mismatch.
        second = place_orders(db_session, _NEXT_ASOF, weights, _PORTFOLIO_ID)

        assert second["ok"] is True
        assert second["data"]["idempotent"] is True
        assert second["data"]["order_id"] == first["data"]["order_id"]


class TestRebalanceLineExecution:
    """Per-line fill fields follow the trade direction, not the target weight's sign."""

    _SLIP = 5.0 / 1e4

    def _second_ticket(self, db_session, first, second) -> dict[str, dict]:
        _seed_panel(db_session)
        place_orders(db_session, _ASOF, first, _PORTFOLIO_ID)
        ticket = place_orders(db_session, _NEXT_ASOF, second, _PORTFOLIO_ID)["data"]
        return {line["ticker"]: line for line in ticket["lines"]}

    def test_unchanged_name_fills_at_the_raw_close(self, db_session) -> None:
        lines = self._second_ticket(
            db_session,
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.5, "BBB": 0.5},
        )

        for line in lines.values():
            assert line["slippage_cost"] == pytest.approx(0.0)
            assert line["effective_price"] == pytest.approx(line["fill_price"])
            assert line["shares"] == pytest.approx(
                line["notional"] / line["fill_price"]
            )

    def test_trimmed_name_is_a_sell_that_receives_less(self, db_session) -> None:
        lines = self._second_ticket(
            db_session,
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.51, "BBB": 0.49},
        )

        bbb = lines["BBB"]
        assert bbb["traded_notional"] < 0
        assert bbb["effective_price"] == pytest.approx(
            bbb["fill_price"] * (1.0 - self._SLIP)
        )

    def test_added_name_is_a_buy_that_pays_up(self, db_session) -> None:
        lines = self._second_ticket(
            db_session,
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.51, "BBB": 0.49},
        )

        aaa = lines["AAA"]
        assert aaa["traded_notional"] > 0
        assert aaa["effective_price"] == pytest.approx(
            aaa["fill_price"] * (1.0 + self._SLIP)
        )

    def test_slippage_applies_only_to_the_traded_part_of_the_shares(
        self, db_session
    ) -> None:
        lines = self._second_ticket(
            db_session,
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.51, "BBB": 0.49},
        )

        for line in lines.values():
            carried = line["notional"] - line["traded_notional"]
            expected = (
                carried / line["fill_price"]
                + line["traded_notional"] / line["effective_price"]
            )
            assert line["shares"] == pytest.approx(expected)
        # A trim of 2% of the book must not shave slippage off the other 98%.
        aaa = lines["AAA"]
        assert aaa["shares"] > aaa["notional"] / (
            aaa["fill_price"] * (1.0 + self._SLIP)
        )

    def test_first_ticket_lines_are_all_buys_at_the_slipped_price(
        self, db_session
    ) -> None:
        _seed_panel(db_session)

        ticket = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)["data"]

        for line in ticket["lines"]:
            assert line["traded_notional"] == pytest.approx(line["notional"])
            assert line["shares"] == pytest.approx(
                line["notional"] / line["effective_price"]
            )


class TestFillShortfalls:
    """The read-only dry run an approve uses before the gated ``place_orders`` runs."""

    def test_no_shortfall_when_every_line_can_fill(self, db_session) -> None:
        _seed_panel(db_session)

        assert fill_shortfalls(db_session, _ASOF, _WEIGHTS) == []

    def test_reports_each_ticker_without_a_bar_after_asof(self, db_session) -> None:
        _seed_panel(db_session)
        last = _START + dt.timedelta(days=_N_DAYS - 1)

        shortfalls = fill_shortfalls(db_session, last, ["AAA", "BBB"])

        assert len(shortfalls) == 2
        assert "'AAA'" in shortfalls[0]
        assert "'BBB'" in shortfalls[1]

    def test_reports_an_unknown_ticker(self, db_session) -> None:
        _seed_panel(db_session)

        shortfalls = fill_shortfalls(db_session, _ASOF, ["AAA", "ZZZ"])

        assert len(shortfalls) == 1
        assert "ZZZ" in shortfalls[0]

    def test_writes_no_ticket(self, db_session) -> None:
        _seed_panel(db_session)

        fill_shortfalls(db_session, _ASOF, _WEIGHTS)

        assert db_session.execute(select(PaperOrder)).scalars().all() == []


_EUR_PER_GBP = 1.17
_EUR_PER_USD = 0.9


def _seed_listing(
    db_session, ticker: str, *, unit: str, close: float, days: int = _N_DAYS
) -> None:
    """Seed one instrument with ``days`` daily bars quoted in ``unit``."""
    ex = Exchange(name=f"EX-{ticker}")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=ex.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    db_session.add(inst)
    db_session.flush()
    for i in range(days):
        db_session.add(
            PriceHistory(
                instrument_id=inst.id,
                date=_START + dt.timedelta(days=i),
                close=close,
                price_unit=unit,
                volume=1000,
            )
        )
    db_session.flush()


def _seed_eur_rates(db_session) -> None:
    """EUR-per-foreign rates on the decision bar and the fill bar."""
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
            for quote, rate in (("GBP", _EUR_PER_GBP), ("USD", _EUR_PER_USD))
            for day in (_ASOF, _FILL_DATE)
        ]
    )
    db_session.flush()


class TestShareCurrency:
    def test_pence_listing_shares_use_the_base_currency_price(self, db_session) -> None:
        _seed_listing(db_session, "LON", unit="GBp", close=2500.0)
        _seed_eur_rates(db_session)

        ticket = place_orders(
            db_session, _ASOF, {"LON": 0.10}, _PORTFOLIO_ID, base_currency="EUR"
        )["data"]

        line = ticket["lines"][0]
        price_eur = 2500.0 / 100 * _EUR_PER_GBP
        assert line["shares"] == pytest.approx(
            10_000.0 / (price_eur * (1 + 5e-4)), rel=1e-9
        )

    def test_foreign_major_currency_shares_apply_the_fx_rate(self, db_session) -> None:
        _seed_listing(db_session, "NYS", unit="USD", close=200.0)
        _seed_eur_rates(db_session)

        ticket = place_orders(
            db_session, _ASOF, {"NYS": 0.5}, _PORTFOLIO_ID, base_currency="EUR"
        )["data"]

        line = ticket["lines"][0]
        assert line["shares"] == pytest.approx(
            50_000.0 / (200.0 * _EUR_PER_USD * (1 + 5e-4)), rel=1e-9
        )
        assert line["currency"] == "USD"

    def test_foreign_listing_without_an_fx_rate_rejects_the_ticket(
        self, db_session
    ) -> None:
        _seed_listing(db_session, "NYS", unit="USD", close=200.0)

        result = place_orders(
            db_session, _ASOF, {"NYS": 1.0}, _PORTFOLIO_ID, base_currency="EUR"
        )

        assert result["ok"] is False
        assert "FX" in result["error"]
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 0

    def test_base_currency_listing_needs_no_fx_rate(self, db_session) -> None:
        _seed_listing(db_session, "EUX", unit="EUR", close=80.0)

        ticket = place_orders(
            db_session, _ASOF, {"EUX": 1.0}, _PORTFOLIO_ID, base_currency="EUR"
        )["data"]

        assert ticket["lines"][0]["shares"] == pytest.approx(
            100_000.0 / (80.0 * (1 + 5e-4)), rel=1e-9
        )


class TestZeroWeightLines:
    """A name the book does not hold needs no fill bar, so it cannot block a ticket."""

    @staticmethod
    def _seed_with_stale_name(db_session) -> None:
        _seed_panel(db_session)
        # Last bar is well before the decision bar: no fill bar exists after it.
        _seed_listing(db_session, "STALE", unit="EUR", close=10.0, days=20)

    @pytest.mark.parametrize("dust", [0.0, 1e-12, -1e-12])
    def test_zero_weight_name_without_a_fill_bar_does_not_block_the_ticket(
        self, db_session, dust
    ) -> None:
        self._seed_with_stale_name(db_session)
        weights = {"AAA": 0.6, "BBB": 0.4, "STALE": dust}

        result = place_orders(
            db_session, _ASOF, weights, _PORTFOLIO_ID, base_currency="EUR"
        )

        assert result["ok"] is True, result
        ticket = result["data"]
        assert {line["ticker"] for line in ticket["lines"]} == {"AAA", "BBB"}
        assert ticket["fill_date"] == _FILL_DATE.isoformat()
        assert sum(line["notional"] for line in ticket["lines"]) == pytest.approx(
            100_000.0
        )
        # The target vector is the audit record: the skipped name stays visible.
        assert ticket["weights"]["STALE"] == dust

    def test_zero_weight_foreign_listing_without_an_fx_rate_is_not_priced(
        self, db_session
    ) -> None:
        _seed_panel(db_session)
        _seed_listing(db_session, "NYS", unit="USD", close=200.0)

        result = place_orders(
            db_session,
            _ASOF,
            {"AAA": 1.0, "NYS": 0.0},
            _PORTFOLIO_ID,
            base_currency="EUR",
        )

        assert result["ok"] is True, result
        assert [line["ticker"] for line in result["data"]["lines"]] == ["AAA"]

    def test_unknown_zero_weight_ticker_does_not_block_the_ticket(
        self, db_session
    ) -> None:
        _seed_panel(db_session)

        result = place_orders(
            db_session,
            _ASOF,
            {"AAA": 1.0, "ZZZ": 0.0},
            _PORTFOLIO_ID,
            base_currency="EUR",
        )

        assert result["ok"] is True, result

    def test_held_name_without_a_fill_bar_still_rejects_the_ticket(
        self, db_session
    ) -> None:
        self._seed_with_stale_name(db_session)

        result = place_orders(
            db_session,
            _ASOF,
            {"AAA": 0.9, "STALE": 0.1},
            _PORTFOLIO_ID,
            base_currency="EUR",
        )

        assert result["ok"] is False
        assert "STALE" in result["error"]
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 0

    def test_all_zero_weights_have_nothing_to_place(self, db_session) -> None:
        _seed_panel(db_session)

        result = place_orders(
            db_session,
            _ASOF,
            {"AAA": 0.0, "BBB": 0.0},
            _PORTFOLIO_ID,
            base_currency="EUR",
        )

        assert result["ok"] is False
        assert "non-zero" in result["error"]
