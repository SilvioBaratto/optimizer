"""A full universe build against the real repository keeps pinned instruments.

The unit tests in ``test_t212_builder.py`` use fakes; this one wires the real
``UniverseRepository`` and ``PinnedInstrumentRepository`` over the SQLite test
engine, so the pin lookup, the builder's subtraction and the ``mark_delisted``
guard are exercised together. A hand-added ETF the Screener never returns must
survive the weekly build that delists its unpinned neighbours.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.universe.pinned_instrument_repository import (
    PinnedInstrumentRepository,
)
from portopt_db.repositories.universe.universe_repository import UniverseRepository

from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig

_NASDAQ = {"name": "NASDAQ", "workingSchedules": [{"id": 1}]}


def _stock(short: str) -> dict[str, Any]:
    return {
        "ticker": f"{short}_US",
        "type": "STOCK",
        "shortName": short,
        "name": f"{short} Inc",
        "currencyCode": "USD",
        "workingScheduleId": 1,
    }


def _bond_etf(short: str) -> dict[str, Any]:
    return {
        "ticker": f"{short}_US",
        "type": "ETF",
        "shortName": short,
        "name": "iShares Core Aggregate Bond",
        "isin": f"US{short}",
        "currencyCode": "USD",
        "workingScheduleId": 1,
    }


def _build(db_session, instruments: list[dict[str, Any]]) -> Any:
    api = MagicMock()
    api.get_exchanges.return_value = [_NASDAQ]
    api.get_instruments.return_value = instruments
    mapper = MagicMock()
    mapper.discover.side_effect = lambda short, exchange: short
    return UniverseBuilder(
        config=UniverseBuilderConfig(),
        api_client=api,
        ticker_mapper=mapper,
        repository=UniverseRepository(db_session),
        max_workers=1,
    ).build()


def _row(db_session, ticker: str) -> Instrument:
    return (
        db_session.query(Instrument)
        .filter_by(ticker=ticker)
        .execution_options(populate_existing=True)
        .one()
    )


def test_pinned_etf_absent_from_the_snapshot_stays_active(db_session) -> None:
    """The weekly build delists the unpinned absentee and spares the pinned ETF."""
    _build(
        db_session,
        [
            _stock("KEEP"),
            _stock("GONE"),
            _bond_etf("AGG"),
            _bond_etf("BND"),
            _bond_etf("TLT"),
        ],
    )
    PinnedInstrumentRepository(db_session).pin("AGG", "manual")

    result = _build(db_session, [_stock("KEEP"), _bond_etf("TLT")])

    assert result.errors == []
    assert _row(db_session, "KEEP_US").delisted_at is None
    assert _row(db_session, "GONE_US").delisted_at is not None
    assert _row(db_session, "BND_US").delisted_at is not None
    assert _row(db_session, "AGG_US").delisted_at is None


def test_unpinning_makes_the_etf_delistable_again(db_session) -> None:
    """Releasing the last pin returns the instrument to the normal rule."""
    still_listed = [_stock("KEEP"), _bond_etf("TLT")]
    _build(db_session, [*still_listed, _bond_etf("AGG")])
    pins = PinnedInstrumentRepository(db_session)
    pins.pin("AGG", "manual")
    _build(db_session, still_listed)
    assert _row(db_session, "AGG_US").delisted_at is None

    pins.unpin("AGG", "manual")
    _build(db_session, still_listed)

    assert _row(db_session, "AGG_US").delisted_at is not None
