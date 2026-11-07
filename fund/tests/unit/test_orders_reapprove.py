"""Re-approving weights that equal an older same-bar ticket makes it the book again."""

from __future__ import annotations

import datetime as dt
import time
import uuid
from decimal import Decimal

import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.performance import _active_ticket, build_nav_series
from fund.tools.orders import place_orders

_PID = uuid.UUID("cccccccc-dddd-eeee-ffff-000000000000")
_ASOF = dt.date(2026, 10, 1)
_DAYS = [_ASOF + dt.timedelta(days=i) for i in range(1, 5)]


def _seed(db_session) -> None:
    ex = Exchange(name="RX")
    db_session.add(ex)
    db_session.flush()
    for ticker, closes in (("A", [100, 110, 121, 133.1]), ("B", [100, 100, 100, 100])):
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange=ex,
            yfinance_ticker=ticker,
            currency_code="EUR",
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=day,
                close=Decimal(str(px)),
                price_unit="EUR",
            )
            for day, px in zip(_DAYS, closes, strict=True)
        )
    db_session.flush()


def test_a_reapproved_ticket_becomes_the_active_book_again(db_session):
    _seed(db_session)
    only_a, only_b = {"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}
    first = place_orders(db_session, _ASOF, only_a, _PID, base_currency="EUR")
    time.sleep(0.01)
    place_orders(db_session, _ASOF, only_b, _PID, base_currency="EUR")
    time.sleep(0.01)

    again = place_orders(db_session, _ASOF, only_a, _PID, base_currency="EUR")

    assert again["data"]["idempotent"] is True
    assert again["data"]["order_id"] == first["data"]["order_id"]
    active = _active_ticket(db_session, _PID, None)
    assert str(active.id) == first["data"]["order_id"]
    navs = [
        p.nav for p in build_nav_series(_PID, session=db_session, base_currency="EUR")
    ]
    assert navs[-1] == pytest.approx(133_100.0)


def test_an_idempotent_replay_of_the_newest_ticket_leaves_the_order_alone(db_session):
    _seed(db_session)
    weights = {"A": 1.0}
    first = place_orders(db_session, _ASOF, weights, _PID, base_currency="EUR")
    stamp = _active_ticket(db_session, _PID, None).created_at

    place_orders(db_session, _ASOF, weights, _PID, base_currency="EUR")

    active = _active_ticket(db_session, _PID, None)
    assert str(active.id) == first["data"]["order_id"]
    assert active.created_at == stamp
