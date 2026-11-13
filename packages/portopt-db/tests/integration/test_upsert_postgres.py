"""``RepositoryBase._upsert`` against a real PostgreSQL server.

SQLite compiles ``ON CONFLICT ON CONSTRAINT <name>`` to a bare ``ON CONFLICT DO
UPDATE`` and drops the name, so none of the SQLite-backed repository tests exercise
the conflict target, the ``updated_at`` re-stamp or the SAVEPOINT isolation.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import EconomicIndicator
from portopt_db.repository import RepositoryBase

pytestmark = pytest.mark.integration

_CONSTRAINT = "uq_economic_indicator_country"


def _stored(session: Session, country: str) -> EconomicIndicator:
    session.expire_all()
    return session.execute(
        select(EconomicIndicator).where(EconomicIndicator.country == country)
    ).scalar_one()


def test_upsert_by_constraint_updates_the_existing_row(pg_session: Session) -> None:
    """A second write to the same country updates in place and re-stamps updated_at."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        EconomicIndicator,
        [{"country": "Testland", "last_inflation": 1.0}],
        constraint_name=_CONSTRAINT,
    )
    first = _stored(pg_session, "Testland")
    first_stamp = first.updated_at

    repo._upsert(
        EconomicIndicator,
        [{"country": "Testland", "last_inflation": 2.5}],
        constraint_name=_CONSTRAINT,
    )

    row = _stored(pg_session, "Testland")
    count = len(
        pg_session.execute(
            select(EconomicIndicator).where(EconomicIndicator.country == "Testland")
        ).all()
    )
    assert count == 1
    assert row.last_inflation == 2.5
    assert row.updated_at > first_stamp


def test_upsert_by_index_elements_updates_the_existing_row(
    pg_session: Session,
) -> None:
    """The column-based conflict target resolves the same way as a named constraint."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        EconomicIndicator,
        [{"country": "Indexland", "last_inflation": 1.0}],
        index_elements=["country"],
    )
    repo._upsert(
        EconomicIndicator,
        [{"country": "Indexland", "last_inflation": 4.0}],
        index_elements=["country"],
    )

    assert _stored(pg_session, "Indexland").last_inflation == 4.0


def test_failed_upsert_does_not_discard_the_earlier_write(
    pg_session: Session,
) -> None:
    """The SAVEPOINT confines a failed statement so an earlier write survives."""
    repo = RepositoryBase(pg_session)
    repo._upsert(
        EconomicIndicator,
        [{"country": "Survivor", "last_inflation": 1.0}],
        constraint_name=_CONSTRAINT,
    )

    with pytest.raises(IntegrityError):
        repo._upsert(
            EconomicIndicator,
            [{"country": None, "last_inflation": 9.9}],
            constraint_name=_CONSTRAINT,
        )

    assert _stored(pg_session, "Survivor").last_inflation == 1.0
