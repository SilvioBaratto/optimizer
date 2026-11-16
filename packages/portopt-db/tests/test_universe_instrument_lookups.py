"""Instrument lookups by Yahoo symbol, on ``UniverseRepository`` and the facade.

The three lookups moved from ``YFinanceRepository`` to ``UniverseRepository``.
``YFinanceRepository`` keeps the three methods, with their signatures, and forwards
each call, so both must answer alike on one session.
"""

from __future__ import annotations

import datetime as dt

from _fixtures import add_and_flush, get_or_create_exchange
from sqlalchemy.orm import Session

from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from portopt_db.repositories.universe.universe_repository import UniverseRepository

_TODAY = dt.date(2026, 11, 16)


def _instrument(
    session: Session, ticker: str, symbol: str, delisted_at: dt.date | None = None
) -> Instrument:
    return add_and_flush(
        session,
        Instrument(
            ticker=ticker,
            short_name=ticker,
            yfinance_ticker=symbol,
            delisted_at=delisted_at,
            exchange=get_or_create_exchange(session, f"{ticker} Exchange"),
        ),
    )


def _bar(session: Session, instrument: Instrument, day: dt.date) -> None:
    add_and_flush(
        session, PriceHistory(instrument_id=instrument.id, date=day, close=10.0)
    )


def test_symbol_lookup_prefers_the_active_listing(db_session: Session) -> None:
    """Test ``UniverseRepository.get_instrument_by_yfinance_ticker``.

    Of a delisted and an active row sharing a symbol, the active one wins.
    """
    _instrument(db_session, "OLD", "VUSA.L", delisted_at=dt.date(2025, 1, 2))
    active = _instrument(db_session, "VUSA", "VUSA.L")

    picked = UniverseRepository(db_session).get_instrument_by_yfinance_ticker("VUSA.L")

    assert picked is not None
    assert picked.id == active.id


def test_ingest_list_leaves_out_a_superseded_listing(db_session: Session) -> None:
    """Test ``UniverseRepository.get_instruments_with_yfinance_ticker``.

    A delisted row whose symbol an active row now carries is left out.
    """
    _instrument(db_session, "OLD", "VUSA.L", delisted_at=dt.date(2025, 1, 2))
    active = _instrument(db_session, "VUSA", "VUSA.L")

    rows = UniverseRepository(db_session).get_instruments_with_yfinance_ticker(
        today=_TODAY
    )

    assert [row.id for row in rows] == [active.id]


def test_benchmark_coverage_counts_bars_and_reports_the_latest(
    db_session: Session,
) -> None:
    """Test ``UniverseRepository.get_benchmark_coverage``.

    A priced symbol reports its bar count and newest date; an unknown symbol
    reports ``(0, None)``.
    """
    spy = _instrument(db_session, "SPY", "SPY")
    _bar(db_session, spy, dt.date(2026, 11, 12))
    _bar(db_session, spy, dt.date(2026, 11, 13))

    coverage = UniverseRepository(db_session).get_benchmark_coverage(["SPY", "NOPE"])

    assert coverage == {"SPY": (2, dt.date(2026, 11, 13)), "NOPE": (0, None)}


def test_facade_lookups_answer_like_universe_repository(db_session: Session) -> None:
    """Test the three lookup forwarders of ``YFinanceRepository``.

    Each returns what the ``UniverseRepository`` method of the same name returns.
    """
    _instrument(db_session, "OLD", "VUSA.L", delisted_at=dt.date(2025, 1, 2))
    _instrument(db_session, "VUSA", "VUSA.L")
    spy = _instrument(db_session, "SPY", "SPY")
    _bar(db_session, spy, dt.date(2026, 11, 13))
    facade = YFinanceRepository(db_session)
    universe = UniverseRepository(db_session)

    assert facade.get_instrument_by_yfinance_ticker(
        "VUSA.L"
    ) is universe.get_instrument_by_yfinance_ticker("VUSA.L")
    assert list(facade.get_instruments_with_yfinance_ticker(today=_TODAY)) == list(
        universe.get_instruments_with_yfinance_ticker(today=_TODAY)
    )
    assert facade.get_benchmark_coverage(
        ["SPY", "NOPE"]
    ) == universe.get_benchmark_coverage(["SPY", "NOPE"])
