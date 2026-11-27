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
_CONSTRAINT_SERIES = "TESTSERIES"
_INDEX_SERIES = "INDEXSERIES"
_SURVIVOR_SERIES = "SURVIVOR"
_FIRST_VALUE = 1.0
_CONSTRAINT_REVISED_VALUE = 2.5
_INDEX_REVISED_VALUE = 4.0
_REJECTED_VALUE = 9.9


def _row(series_id: str | None, value: float) -> dict[str, object]:
    return {"series_id": series_id, "date": _OBSERVATION_DATE, "value": value}


def _stored(session: Session, series_id: str) -> FredObservation:
    session.expire_all()
    return session.execute(
        select(FredObservation).where(FredObservation.series_id == series_id)
    ).scalar_one()


@pytest.mark.criterion("REQ-DB-011")
def test_upsert_by_constraint_updates_the_existing_row(pg_session: Session) -> None:
    """Validates RepositoryBase._upsert with constraint_name: a second write to the
    same series and date updates in place and re-stamps updated_at."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [_row(_CONSTRAINT_SERIES, _FIRST_VALUE)],
        constraint_name=_CONSTRAINT,
    )
    first_stamp = _stored(pg_session, _CONSTRAINT_SERIES).updated_at

    repo._upsert(
        FredObservation,
        [_row(_CONSTRAINT_SERIES, _CONSTRAINT_REVISED_VALUE)],
        constraint_name=_CONSTRAINT,
    )

    row = _stored(pg_session, _CONSTRAINT_SERIES)
    count = len(
        pg_session.execute(
            select(FredObservation).where(
                FredObservation.series_id == _CONSTRAINT_SERIES
            )
        ).all()
    )
    assert count == 1
    assert row.value == _CONSTRAINT_REVISED_VALUE
    assert row.updated_at > first_stamp


@pytest.mark.criterion("REQ-DB-011")
def test_upsert_by_index_elements_updates_the_existing_row(
    pg_session: Session,
) -> None:
    """Validates RepositoryBase._upsert with index_elements: the column-based
    conflict target resolves the same way as a named constraint."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [_row(_INDEX_SERIES, _FIRST_VALUE)],
        index_elements=["series_id", "date"],
    )
    repo._upsert(
        FredObservation,
        [_row(_INDEX_SERIES, _INDEX_REVISED_VALUE)],
        index_elements=["series_id", "date"],
    )

    assert _stored(pg_session, _INDEX_SERIES).value == _INDEX_REVISED_VALUE


@pytest.mark.criterion("REQ-DB-011")
def test_failed_upsert_does_not_discard_the_earlier_write(
    pg_session: Session,
) -> None:
    """Validates RepositoryBase._upsert: the SAVEPOINT confines a failed statement so
    an earlier write in the same transaction survives."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        FredObservation,
        [_row(_SURVIVOR_SERIES, _FIRST_VALUE)],
        constraint_name=_CONSTRAINT,
    )

    with pytest.raises(IntegrityError):
        repo._upsert(
            FredObservation,
            [_row(None, _REJECTED_VALUE)],
            constraint_name=_CONSTRAINT,
        )

    assert _stored(pg_session, _SURVIVOR_SERIES).value == _FIRST_VALUE
