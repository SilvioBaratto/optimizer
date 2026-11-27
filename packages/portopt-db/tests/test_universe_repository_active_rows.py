"""Tests for the UniverseRepository active-row lookups.

Covers ``get_active_instruments`` (non-delisted rows),
``get_active_listings`` ((ticker, exchange name) of non-delisted rows) and
``has_active_instruments``.
"""

from __future__ import annotations

from datetime import date

from _fixtures import add_and_flush

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


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


def test_get_active_listings_pairs_ticker_with_exchange_name(db_session) -> None:
    nasdaq = add_and_flush(db_session, Exchange(name="NASDAQ"))
    xetra = add_and_flush(db_session, Exchange(name="Deutsche Börse Xetra"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=nasdaq)
    )
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=xetra)
    )
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=nasdaq,
            delisted_at=date(2020, 1, 1),
        ),
    )
    assert _repo(db_session).get_active_listings() == {
        ("AAPL", "NASDAQ"),
        ("AAPL", "Deutsche Börse Xetra"),
    }


def test_has_active_instruments_is_false_on_an_empty_table(db_session) -> None:
    assert _repo(db_session).has_active_instruments() is False


def test_has_active_instruments_is_true_with_a_live_row(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert _repo(db_session).has_active_instruments() is True


def test_has_active_instruments_ignores_delisted_rows(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=ex,
            delisted_at=date(2020, 1, 1),
        ),
    )
    assert _repo(db_session).has_active_instruments() is False
