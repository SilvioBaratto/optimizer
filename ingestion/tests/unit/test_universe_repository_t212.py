"""UniverseRepository T212-annotation surface (SPEC D13/D14, task T11a).

Adds the nullable ``t212_ticker`` column and the repo helpers the annotation
step uses: ``set_t212_ticker`` (update by (ticker, exchange)) and
``get_active_instruments`` (non-delisted rows to resolve ISINs for).
"""

from __future__ import annotations

from datetime import date

from app.models.universe.universe import Exchange, Instrument
from app.repositories.universe.universe_repository import UniverseRepository
from tests._fixtures._helpers import add_and_flush


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


def test_instrument_t212_ticker_defaults_to_none(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert inst.t212_ticker is None


def test_set_t212_ticker_updates_matching_row(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert (
        _repo(db_session).set_t212_ticker(
            ticker="AAPL", exchange_id=ex.id, t212_ticker="AAPL_US_EQ"
        )
        is True
    )
    db_session.refresh(inst)
    assert inst.t212_ticker == "AAPL_US_EQ"


def test_set_t212_ticker_missing_row_returns_false(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    assert (
        _repo(db_session).set_t212_ticker(
            ticker="NOPE", exchange_id=ex.id, t212_ticker="X"
        )
        is False
    )


def test_get_active_instruments_excludes_delisted(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=ex,
            delisted_at=date(2020, 1, 1),
        ),
    )
    tickers = {i.ticker for i in _repo(db_session).get_active_instruments()}
    assert tickers == {"AAPL"}
