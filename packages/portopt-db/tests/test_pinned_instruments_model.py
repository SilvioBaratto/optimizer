"""``pinned_instruments`` ORM model on SQLite.

The table records tickers the weekly universe build must never delist. It is
keyed by ticker text rather than an ``instruments`` FK, so a pin can exist
before its instrument row does. These tests prove the table builds under the
shared metadata, the column widths match what the callers write, and the
``(ticker, source)`` uniqueness lets several owners pin the same ticker.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import String, inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import Instrument, PinnedInstrument
from portopt_db.models.universe import PinnedInstrument as UniversePinnedInstrument


def test_table_is_registered_on_metadata(test_engine):
    """The model builds the table under SQLite ``create_all``."""
    assert "pinned_instruments" in set(inspect(test_engine).get_table_names())


def test_model_is_reexported_from_universe_package():
    """The universe subpackage exposes the same class as the models package."""
    assert UniversePinnedInstrument is PinnedInstrument


def test_column_widths_match_callers():
    """``ticker`` matches ``instruments.yfinance_ticker``; ``source`` fits a mandate key."""
    ticker = PinnedInstrument.__table__.c.ticker.type
    source = PinnedInstrument.__table__.c.source.type
    yf_ticker = Instrument.__table__.c.yfinance_ticker.type
    assert isinstance(ticker, String)
    assert ticker.length == yf_ticker.length == 100
    assert isinstance(source, String)
    assert source.length == 64
    # "mandate:" + a 36-character UUID
    assert len(f"mandate:{uuid.uuid4()}") == 44 <= source.length


def test_columns_are_not_nullable():
    """A pin without a ticker or an owner is meaningless."""
    cols = PinnedInstrument.__table__.c
    assert not cols.ticker.nullable
    assert not cols.source.nullable


def test_row_round_trips(db_session):
    """A pin persists with its id and audit timestamps."""
    row = PinnedInstrument(ticker="VWCE.DE", source="manual")
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(PinnedInstrument).where(PinnedInstrument.id == row.id)
    ).scalar_one()
    assert isinstance(fetched.id, uuid.UUID)
    assert fetched.ticker == "VWCE.DE"
    assert fetched.source == "manual"
    assert fetched.created_at is not None


def test_unique_blocks_duplicate_ticker_source(db_session):
    """The same ticker pinned twice by the same source violates the UNIQUE."""
    db_session.add(PinnedInstrument(ticker="EGLN.L", source="manual"))
    db_session.flush()
    db_session.add(PinnedInstrument(ticker="EGLN.L", source="manual"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_same_ticker_under_two_sources_coexists(db_session):
    """Two owners may pin one ticker; each releases only its own row."""
    db_session.add_all(
        [
            PinnedInstrument(ticker="SXR8.DE", source="manual"),
            PinnedInstrument(ticker="SXR8.DE", source=f"mandate:{uuid.uuid4()}"),
        ]
    )
    db_session.flush()

    rows = db_session.execute(select(PinnedInstrument)).scalars().all()
    assert len(rows) == 2


def test_unique_constraint_and_source_index_are_named(test_engine):
    """Constraint and index carry the names the migration creates."""
    insp = inspect(test_engine)
    uniques = {u["name"] for u in insp.get_unique_constraints("pinned_instruments")}
    indexes = {i["name"] for i in insp.get_indexes("pinned_instruments")}
    assert "uq_pinned_instruments_ticker_source" in uniques
    assert indexes == {"ix_pinned_instruments_source"}
