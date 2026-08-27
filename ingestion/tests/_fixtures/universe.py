"""DB seed builder for the universe domain: ``Exchange`` + ``Instrument``.

``Exchange.name`` is unique and ``Instrument`` is unique on
``(ticker, exchange_id)``, so creation is get-or-create — this lets
``seed_universe`` and ``seed_market_data`` compose in one transaction
without raising an IntegrityError on shared default keys.
"""

from __future__ import annotations

from typing import NamedTuple

from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy.orm import Session

from tests._fixtures._helpers import add_and_flush


class UniverseSeed(NamedTuple):
    exchange: Exchange
    instrument: Instrument


def get_or_create_exchange(session: Session, name: str) -> Exchange:
    existing = session.query(Exchange).filter_by(name=name).one_or_none()
    if existing is not None:
        return existing
    return add_and_flush(session, Exchange(name=name))


def get_or_create_instrument(
    session: Session, *, ticker: str, exchange: Exchange, short_name: str
) -> Instrument:
    existing = (
        session.query(Instrument)
        .filter_by(ticker=ticker, exchange_id=exchange.id)
        .one_or_none()
    )
    if existing is not None:
        return existing
    instrument = Instrument(ticker=ticker, short_name=short_name, exchange=exchange)
    return add_and_flush(session, instrument)


def seed_universe(
    session: Session,
    *,
    exchange_name: str = "Test Exchange",
    ticker: str = "AAPL",
    short_name: str = "Apple Inc.",
) -> UniverseSeed:
    exchange = get_or_create_exchange(session, exchange_name)
    instrument = get_or_create_instrument(
        session, ticker=ticker, exchange=exchange, short_name=short_name
    )
    return UniverseSeed(exchange=exchange, instrument=instrument)
