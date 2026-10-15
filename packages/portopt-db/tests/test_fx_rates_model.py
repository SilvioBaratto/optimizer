"""``fx_rates`` ORM model on SQLite.

The daily FX-rate table lives in ``portopt_db`` so the shared ``Base.metadata``
builds it under SQLite in tests. These tests prove the table creates on SQLite,
a rate row round-trips, the composite ``(base_currency, quote_currency, date)``
uniqueness blocks a duplicate pair-date, and distinct pairs/dates coexist.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import FxRate


def test_table_is_registered_on_metadata(test_engine):
    """The model builds the table under SQLite ``create_all``."""
    assert "fx_rates" in set(inspect(test_engine).get_table_names())


def test_rate_round_trips(db_session):
    """A rate row persists and reads back, rate kept as ``Decimal``."""
    row = FxRate(
        base_currency="EUR",
        quote_currency="USD",
        date=datetime.date(2026, 10, 1),
        rate=Decimal("0.9123456789"),
    )
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(select(FxRate).where(FxRate.id == row.id)).scalar_one()
    assert fetched.base_currency == "EUR"
    assert fetched.quote_currency == "USD"
    assert fetched.date == datetime.date(2026, 10, 1)
    assert Decimal(str(fetched.rate)) == Decimal("0.9123456789")
    assert fetched.created_at is not None


def test_composite_unique_blocks_duplicate_pair_date(db_session):
    """Two rows sharing (base, quote, date) violate the composite UNIQUE."""
    d = datetime.date(2026, 10, 1)
    db_session.add(
        FxRate(base_currency="EUR", quote_currency="USD", date=d, rate=Decimal("0.91"))
    )
    db_session.flush()
    db_session.add(
        FxRate(base_currency="EUR", quote_currency="USD", date=d, rate=Decimal("0.92"))
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_distinct_pairs_and_dates_coexist(db_session):
    """Different quote currencies and dates for the same base coexist."""
    db_session.add_all(
        [
            FxRate(
                base_currency="EUR",
                quote_currency="USD",
                date=datetime.date(2026, 10, 1),
                rate=Decimal("0.91"),
            ),
            FxRate(
                base_currency="EUR",
                quote_currency="GBP",
                date=datetime.date(2026, 10, 1),
                rate=Decimal("1.17"),
            ),
            FxRate(
                base_currency="EUR",
                quote_currency="USD",
                date=datetime.date(2026, 10, 2),
                rate=Decimal("0.92"),
            ),
        ]
    )
    db_session.flush()

    rows = db_session.execute(select(FxRate)).scalars().all()
    assert len(rows) == 3
    assert all(isinstance(r.id, uuid.UUID) for r in rows)
