"""DB seed builders for the portopt-db test suite.

Copied from the ingestion daemon's ``tests/_fixtures`` (the subset the relocated
repository tests use). Builders insert via ``session.add`` + ``flush()`` and
never ``commit()`` — the SAVEPOINT ``db_session`` fixture owns the transaction.
"""

from __future__ import annotations

from typing import NamedTuple, TypeVar

from sqlalchemy.orm import Session

from portopt_db.models.universe.universe import Exchange, Instrument

T = TypeVar("T")


def add_and_flush(session: Session, obj: T) -> T:
    """Add a single ORM row and flush so its PK / FKs are assigned."""
    session.add(obj)
    session.flush()
    return obj


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
