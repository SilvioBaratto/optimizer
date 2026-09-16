"""T3.6 — ``place_orders``: the idempotent paper-execution tool.

``place_orders`` turns a target-weight vector into a **simulated** order ticket
(D5) filled at the **next close strictly after** the decision bar (D30 — no
look-ahead), with a simple slippage + commission model, and persists it to the
``paper_orders`` table. It must be **idempotent**: on a HITL ``Command(resume=…)``
the interrupting node re-runs from the top (SPEC D3 gotcha), so two calls with
the same ``(portfolio_id, asof, weights)`` must yield exactly **one** ticket.

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

from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import func, select

from fund.tools.orders import place_orders

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

        # The raw fill price is the next close after the decision bar.
        idx = _fill_index()
        by_ticker = {line["ticker"]: line for line in ticket["lines"]}
        for ticker, start in _SERIES.items():
            assert by_ticker[ticker]["fill_price"] == _close_on(start, idx)

        # Persisted exactly once, retrievable by the idempotency key.
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

    def test_idempotent_double_call_one_ticket(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        second = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)

        assert first["ok"] is True
        assert second["ok"] is True
        # Same key ⇒ same ticket id, and the second call is flagged idempotent.
        assert second["data"]["order_id"] == first["data"]["order_id"]
        assert second["data"]["idempotent"] is True
        assert first["data"]["idempotent"] is False

        # Exactly one row survives the double call.
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 1

    def test_distinct_weights_make_distinct_tickets(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        other = place_orders(
            db_session, _ASOF, {"AAA": 0.6, "BBB": 0.4}, _PORTFOLIO_ID
        )

        assert first["data"]["order_id"] != other["data"]["order_id"]
        count = db_session.execute(
            select(func.count()).select_from(PaperOrder)
        ).scalar_one()
        assert count == 2

    def test_deterministic(self, db_session) -> None:
        _seed_panel(db_session)

        first = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)
        second = place_orders(db_session, _ASOF, _WEIGHTS, _PORTFOLIO_ID)

        # Idempotent ⇒ the second envelope equals the first bar the flag.
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
