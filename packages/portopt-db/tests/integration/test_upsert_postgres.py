"""``RepositoryBase._upsert`` against a real PostgreSQL server.

SQLite compiles ``ON CONFLICT ON CONSTRAINT <name>`` to a bare ``ON CONFLICT DO
UPDATE`` and drops the name, so none of the SQLite-backed repository tests exercise
the conflict target, the ``updated_at`` re-stamp or the SAVEPOINT isolation.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import FredObservation
from portopt_db.repository import RepositoryBase

pytestmark = pytest.mark.integration

_CONSTRAINT = "uq_fred_observation_series_date"
_OBSERVATION_DATE = datetime.date(2026, 11, 26)


def _stored(session: Session, series_id: str) -> FredObservation:
    session.expire_all()
    return session.execute(
        select(FredObservation).where(FredObservation.series_id == series_id)
    ).scalar_one()


def test_upsert_by_constraint_updates_the_existing_row(pg_session: Session) -> None:
    """A second write to the same series and date updates in place and re-stamps."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [{"series_id": "TESTSERIES", "date": _OBSERVATION_DATE, "value": 1.0}],
        constraint_name=_CONSTRAINT,
    )
    first = _stored(pg_session, "TESTSERIES")
    first_stamp = first.updated_at

    repo._upsert(
        FredObservation,
        [{"series_id": "TESTSERIES", "date": _OBSERVATION_DATE, "value": 2.5}],
        constraint_name=_CONSTRAINT,
    )

    row = _stored(pg_session, "TESTSERIES")
    count = len(
        pg_session.execute(
            select(FredObservation).where(FredObservation.series_id == "TESTSERIES")
        ).all()
    )
    assert count == 1
    assert row.value == 2.5
    assert row.updated_at > first_stamp


def test_upsert_by_index_elements_updates_the_existing_row(
    pg_session: Session,
) -> None:
    """The column-based conflict target resolves the same way as a named constraint."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [{"series_id": "INDEXSERIES", "date": _OBSERVATION_DATE, "value": 1.0}],
        index_elements=["series_id", "date"],
    )
    repo._upsert(
        FredObservation,
        [{"series_id": "INDEXSERIES", "date": _OBSERVATION_DATE, "value": 4.0}],
        index_elements=["series_id", "date"],
    )

    assert _stored(pg_session, "INDEXSERIES").value == 4.0


def test_failed_upsert_does_not_discard_the_earlier_write(
    pg_session: Session,
) -> None:
    """The SAVEPOINT confines a failed statement so an earlier write survives."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [{"series_id": "SURVIVOR", "date": _OBSERVATION_DATE, "value": 1.0}],
        constraint_name=_CONSTRAINT,
    )

    with pytest.raises(IntegrityError):
        repo._upsert(
            FredObservation,
            [{"series_id": None, "date": _OBSERVATION_DATE, "value": 9.9}],
            constraint_name=_CONSTRAINT,
        )

    assert _stored(pg_session, "SURVIVOR").value == 1.0
