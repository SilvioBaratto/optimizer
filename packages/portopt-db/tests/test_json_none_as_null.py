"""A Python ``None`` in a nullable JSON column must persist as SQL NULL.

With the default JSON type a ``None`` is stored as the JSON literal ``null``, which
``IS NULL`` filters miss. The agent tables already opt out; the job and journal
tables must as well.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from portopt_db.models.jobs.background_job import BackgroundJob
from portopt_db.models.orders.portfolio_journal import PortfolioJournal


@pytest.mark.parametrize("column", ["extra", "result"])
def test_background_job_none_is_sql_null(db_session: Session, column: str) -> None:
    """Store an unset job column as SQL NULL."""
    job = BackgroundJob(job_type="json_null_probe", status="running")
    setattr(job, column, None)
    db_session.add(job)
    db_session.flush()

    empty = db_session.execute(
        text(f"SELECT COUNT(*) FROM background_jobs WHERE {column} IS NULL")
    ).scalar_one()

    assert empty == 1


@pytest.mark.parametrize("column", ["trades", "allocation", "drift"])
def test_portfolio_journal_none_is_sql_null(db_session: Session, column: str) -> None:
    """Store an unset journal column as SQL NULL."""
    row = PortfolioJournal(portfolio_id=uuid.uuid4(), as_of=date(2026, 1, 2))
    setattr(row, column, None)
    db_session.add(row)
    db_session.flush()

    empty = db_session.execute(
        text(f"SELECT COUNT(*) FROM portfolio_journal WHERE {column} IS NULL")
    ).scalar_one()

    assert empty == 1


def test_real_values_still_round_trip(db_session: Session) -> None:
    """Keep storing populated documents unchanged."""
    row = PortfolioJournal(
        portfolio_id=uuid.uuid4(), as_of=date(2026, 1, 2), trades={"AAA": 1}
    )
    db_session.add(row)
    db_session.flush()
    db_session.expire_all()

    assert db_session.get(PortfolioJournal, row.id).trades == {"AAA": 1}  # type: ignore[union-attr]
