"""``nav_history`` ORM model on SQLite.

The persisted marked-to-market NAV series lives in ``portopt_db`` so the shared
``Base.metadata`` builds it under SQLite in tests. These tests prove the table
creates on SQLite, a NAV row round-trips (with an optional benchmark level), the
composite ``(portfolio_id, date)`` uniqueness blocks a duplicate, and distinct
portfolios/dates coexist. ``portfolio_id`` is seeded with a letter-bearing UUID
so SQLite does not REAL-coerce it.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import NavHistory

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_PID2 = uuid.UUID("ffffffff-1111-2222-3333-444444444abc")


def test_table_is_registered_on_metadata(test_engine):
    """The model builds the table under SQLite ``create_all``."""
    assert "nav_history" in set(inspect(test_engine).get_table_names())


def test_nav_row_round_trips(db_session):
    """A NAV row with a benchmark level persists and reads back."""
    row = NavHistory(
        portfolio_id=_PID,
        date=datetime.date(2026, 10, 1),
        nav=Decimal("101234.56"),
        benchmark=Decimal("5600.50"),
    )
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(NavHistory).where(NavHistory.id == row.id)
    ).scalar_one()
    assert fetched.portfolio_id == _PID
    assert fetched.date == datetime.date(2026, 10, 1)
    assert Decimal(str(fetched.nav)) == Decimal("101234.56")
    assert Decimal(str(fetched.benchmark)) == Decimal("5600.50")
    assert fetched.created_at is not None


def test_benchmark_is_optional(db_session):
    """Omitting the benchmark leaves it NULL."""
    row = NavHistory(
        portfolio_id=_PID,
        date=datetime.date(2026, 10, 2),
        nav=Decimal("100000"),
    )
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(NavHistory).where(NavHistory.id == row.id)
    ).scalar_one()
    assert fetched.benchmark is None


def test_composite_unique_blocks_duplicate_portfolio_date(db_session):
    """Two rows sharing (portfolio_id, date) violate the composite UNIQUE."""
    d = datetime.date(2026, 10, 1)
    db_session.add(NavHistory(portfolio_id=_PID, date=d, nav=Decimal("1")))
    db_session.flush()
    db_session.add(NavHistory(portfolio_id=_PID, date=d, nav=Decimal("2")))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_distinct_portfolios_and_dates_coexist(db_session):
    """Different portfolios and dates coexist."""
    db_session.add_all(
        [
            NavHistory(
                portfolio_id=_PID, date=datetime.date(2026, 10, 1), nav=Decimal("1")
            ),
            NavHistory(
                portfolio_id=_PID, date=datetime.date(2026, 10, 2), nav=Decimal("2")
            ),
            NavHistory(
                portfolio_id=_PID2, date=datetime.date(2026, 10, 1), nav=Decimal("3")
            ),
        ]
    )
    db_session.flush()

    rows = db_session.execute(select(NavHistory)).scalars().all()
    assert len(rows) == 3


def test_benchmark_ticker_round_trips_and_defaults_to_null(db_session):
    """The benchmark's identity is stored beside its level; legacy rows are NULL."""
    tagged = NavHistory(
        portfolio_id=_PID,
        date=datetime.date(2026, 10, 1),
        nav=Decimal("100"),
        benchmark=Decimal("450"),
        benchmark_ticker="SPY",
    )
    legacy = NavHistory(
        portfolio_id=_PID,
        date=datetime.date(2026, 10, 2),
        nav=Decimal("101"),
        benchmark=Decimal("451"),
    )
    db_session.add_all([tagged, legacy])
    db_session.flush()

    assert tagged.benchmark_ticker == "SPY"
    assert legacy.benchmark_ticker is None
