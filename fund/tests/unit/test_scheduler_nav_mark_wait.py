"""The scheduled NAV mark waits for the day's ingestion price sweep.

The 08:00 UTC mark used to run whether or not the ingestion daemon had stored the
day's closes, so the newest NAV point carried yesterday's prices forward until the
next day's re-mark. ``run_scheduled_nav_mark`` now marks only once today's
``yfinance_fetch`` sweep has settled (completed or failed), re-marks when a sweep
settles after an earlier mark, and past ``fund_nav_mark_wait_seconds`` marks once
anyway with a warning. It decides from ``background_jobs`` alone, so the
single-worker scheduler is never blocked while it waits.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pytest
from portopt_db.models import BackgroundJob, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from fund.config import settings
from fund.scheduler import JOB_NAV_MARK, run_scheduled_nav_mark

SessionFactory = Callable[[], "contextmanager[Session]"]

_CONFIG = dataclasses.replace(
    settings, fund_nav_mark_cron="0 8 * * *", fund_nav_mark_wait_seconds=4 * 3600
)
_SWEEP = "yfinance_fetch"


def _at(hour: int, minute: int = 0, *, day: int = 26) -> dt.datetime:
    return dt.datetime(2026, 10, day, hour, minute, tzinfo=dt.UTC)


@pytest.fixture
def session_factory(tmp_path) -> Generator[SessionFactory, None, None]:
    """A function-scoped file-based SQLite engine + session-context factory."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'navwait.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _factory() -> Generator[Session, None, None]:
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    try:
        yield _factory
    finally:
        engine.dispose()


@pytest.fixture
def marks(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record each NAV mark the policy releases instead of computing one."""
    calls: list[dict[str, Any]] = []

    def _mark(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("fund.scheduler.run_nav_mark", _mark)
    return calls


def _job(
    factory: SessionFactory,
    job_type: str,
    status: str,
    *,
    started: dt.datetime,
    finished: dt.datetime | None = None,
) -> None:
    with factory() as session:
        session.add(
            BackgroundJob(
                job_type=job_type,
                status=status,
                started_at=started,
                finished_at=finished,
            )
        )
        session.commit()


def _run(factory: SessionFactory, now: dt.datetime, config=_CONFIG) -> bool:
    return run_scheduled_nav_mark(session_factory=factory, config=config, now=now)


def test_waits_while_todays_price_sweep_is_running(session_factory, marks) -> None:
    """A sweep still running today holds the mark back."""
    _job(session_factory, _SWEEP, "running", started=_at(7, 5))

    assert _run(session_factory, _at(8)) is False
    assert marks == []


def test_marks_once_todays_price_sweep_has_finished(session_factory, marks) -> None:
    """A completed sweep releases the mark on its fresh closes."""
    _job(session_factory, _SWEEP, "completed", started=_at(7, 5), finished=_at(9, 40))

    assert _run(session_factory, _at(9, 45)) is True
    assert len(marks) == 1


def test_a_failed_sweep_still_releases_the_mark(session_factory, marks) -> None:
    """A failed sweep stored what it could; waiting longer would not add more."""
    _job(session_factory, _SWEEP, "failed", started=_at(7, 5), finished=_at(8, 30))

    assert _run(session_factory, _at(8, 45)) is True
    assert len(marks) == 1


def test_does_not_remark_once_marked_on_the_finished_sweep(
    session_factory, marks
) -> None:
    """A mark already taken after the sweep finished is not repeated."""
    _job(session_factory, _SWEEP, "completed", started=_at(7, 5), finished=_at(9, 40))
    _job(
        session_factory,
        JOB_NAV_MARK,
        "completed",
        started=_at(9, 45),
        finished=_at(9, 46),
    )

    assert _run(session_factory, _at(10)) is False
    assert marks == []


def test_remarks_when_the_sweep_finishes_after_the_fallback_mark(
    session_factory, marks
) -> None:
    """A sweep settling after the fallback mark triggers a fresh one."""
    _job(
        session_factory, JOB_NAV_MARK, "completed", started=_at(12), finished=_at(12, 1)
    )
    _job(session_factory, _SWEEP, "completed", started=_at(7, 5), finished=_at(13, 10))

    assert _run(session_factory, _at(13, 15)) is True
    assert len(marks) == 1


def test_marks_anyway_past_the_deadline_and_warns(
    session_factory, marks, caplog: pytest.LogCaptureFixture
) -> None:
    """Past the wait budget the mark runs on stored closes and says so."""
    _job(session_factory, _SWEEP, "running", started=_at(7, 5))

    with caplog.at_level(logging.WARNING, logger="fund.scheduler"):
        assert _run(session_factory, _at(12)) is True

    assert len(marks) == 1
    assert "has not finished" in caplog.text


def test_marks_past_the_deadline_when_ingestion_never_ran_today(
    session_factory, marks
) -> None:
    """No ingestion today at all still yields one mark after the wait budget."""
    assert _run(session_factory, _at(12, 5)) is True
    assert len(marks) == 1


def test_the_fallback_mark_runs_once_a_day(session_factory, marks) -> None:
    """The deadline fallback is taken at most once a day."""
    _job(session_factory, _SWEEP, "running", started=_at(7, 5))
    _job(
        session_factory, JOB_NAV_MARK, "completed", started=_at(12), finished=_at(12, 1)
    )

    assert _run(session_factory, _at(12, 15)) is False
    assert marks == []


def test_nothing_runs_before_todays_scheduled_mark(session_factory, marks) -> None:
    """Nothing is marked before the cron's time today, even on a finished sweep."""
    _job(session_factory, _SWEEP, "completed", started=_at(7, 5), finished=_at(7, 50))

    assert _run(session_factory, _at(7, 55)) is False
    assert marks == []


def test_yesterdays_sweep_does_not_release_todays_mark(session_factory, marks) -> None:
    """Only a sweep started today counts as today's closes."""
    _job(
        session_factory,
        _SWEEP,
        "completed",
        started=_at(7, 5, day=25),
        finished=_at(9, 0, day=25),
    )

    assert _run(session_factory, _at(9)) is False
    assert marks == []


def test_a_second_sweep_still_running_holds_the_mark(session_factory, marks) -> None:
    """Any sweep still active today holds the mark, even after another finished."""
    _job(session_factory, _SWEEP, "completed", started=_at(7, 5), finished=_at(7, 30))
    _job(session_factory, _SWEEP, "running", started=_at(7, 31))

    assert _run(session_factory, _at(9)) is False
    assert marks == []


def test_waiting_disabled_marks_at_the_scheduled_time(session_factory, marks) -> None:
    """A zero wait budget marks at the scheduled time without waiting."""
    _job(session_factory, _SWEEP, "running", started=_at(7, 5))
    config = dataclasses.replace(_CONFIG, fund_nav_mark_wait_seconds=0)

    assert _run(session_factory, _at(8), config=config) is True
    assert len(marks) == 1
