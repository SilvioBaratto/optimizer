"""Tests for the UniverseRepository active-row lookups.

Covers ``get_active_instruments`` (non-delisted rows),
``get_active_listings`` ((ticker, exchange name) of non-delisted rows) and
``has_active_instruments``.
"""

from __future__ import annotations

from datetime import date

import pytest
from _fixtures import add_and_flush

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


@pytest.mark.criterion("REQ-DB-009")
def test_get_active_instruments_excludes_delisted(db_session) -> None:
    """Validates ``UniverseRepository.get_active_instruments``: delisted rows drop."""
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


@pytest.mark.criterion("REQ-DB-009")
def test_get_active_listings_pairs_ticker_with_exchange_name(db_session) -> None:
    """Validates ``UniverseRepository.get_active_listings``: only live rows pair.

    Each non-delisted row yields its ticker with the name of its exchange.
    """
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


@pytest.mark.criterion("REQ-DB-009")
def test_has_active_instruments_is_false_on_an_empty_table(db_session) -> None:
    """Validates ``UniverseRepository.has_active_instruments``: no rows, no match."""
    assert _repo(db_session).has_active_instruments() is False


@pytest.mark.criterion("REQ-DB-009")
def test_has_active_instruments_is_true_with_a_live_row(db_session) -> None:
    """Validates ``UniverseRepository.has_active_instruments``: a live row counts."""
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert _repo(db_session).has_active_instruments() is True


@pytest.mark.criterion("REQ-DB-009")
def test_has_active_instruments_ignores_delisted_rows(db_session) -> None:
    """Validates ``UniverseRepository.has_active_instruments``: delisted rows drop."""
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
