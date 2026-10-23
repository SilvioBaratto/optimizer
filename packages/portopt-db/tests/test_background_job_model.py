"""``BackgroundJob`` ORM model: the active-slot partial unique index.

``claim_or_create`` is an ``INSERT ... SELECT ... WHERE NOT EXISTS`` which, under
PostgreSQL's READ COMMITTED isolation, lets two concurrent claimers both pass the
check. A partial unique index over the active statuses is what actually bounds a
job type to one pending/running row; SQLite supports partial indexes, so it fires
in the in-memory test harness as well.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex

from portopt_db.models import BackgroundJob

INDEX_NAME = "uq_background_jobs_active_type"


def _row(job_type: str, status: str) -> BackgroundJob:
    return BackgroundJob(job_type=job_type, status=status)


def test_active_slot_index_is_a_partial_unique_index_on_postgresql() -> None:
    """The DDL emitted for PostgreSQL is unique, over job_type, active-only."""
    index = next(i for i in BackgroundJob.__table__.indexes if i.name == INDEX_NAME)

    ddl = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert "CREATE UNIQUE INDEX" in ddl
    assert "(job_type)" in ddl
    assert "WHERE status IN ('pending', 'running')" in ddl


@pytest.mark.parametrize("first", ["pending", "running"])
@pytest.mark.parametrize("second", ["pending", "running"])
def test_second_active_row_of_one_job_type_is_rejected(
    db_session, first: str, second: str
) -> None:
    db_session.add(_row("slot_dup", first))
    db_session.flush()

    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.add(_row("slot_dup", second))
        db_session.flush()


def test_terminal_rows_do_not_occupy_the_slot(db_session) -> None:
    """Any number of finished rows may coexist with one active row."""
    db_session.add_all(
        [
            _row("slot_hist", "completed"),
            _row("slot_hist", "completed"),
            _row("slot_hist", "failed"),
            _row("slot_hist", "running"),
        ]
    )
    db_session.flush()


def test_distinct_job_types_each_get_their_own_active_row(db_session) -> None:
    db_session.add_all([_row("slot_a", "running"), _row("slot_b", "running")])
    db_session.flush()
