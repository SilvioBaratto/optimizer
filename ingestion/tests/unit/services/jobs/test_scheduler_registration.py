"""Scheduler registration tests (issue #824).

Asserts ``create_scheduler()`` registers exactly the 7 expected job ids with
the expected trigger types, without starting the scheduler or invoking any
registered function (no live fetch).  The DB engine is mocked non-None and the
SQLAlchemy jobstore is swapped for an in-memory one so no real database is
touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.services.jobs.scheduler import create_scheduler

_DM = "app.services.jobs.scheduler.database_manager"
_JOBSTORE = "app.services.jobs.scheduler.SQLAlchemyJobStore"

_CRON_JOBS = {
    "daily_pipeline",
    "midday_news",
    "universe_build",
    "weekly_refetch",
    "fred_monthly",
    "weekly_market_wide",
}
_INTERVAL_JOBS = {"news_refresh", "orphan_reaper"}
_ALL_JOBS = _CRON_JOBS | _INTERVAL_JOBS


def _build_scheduler():
    """Create a scheduler with a mocked engine + in-memory jobstore."""
    dm = MagicMock()
    dm.engine = MagicMock()  # non-None
    with (
        patch(_DM, dm),
        patch(_JOBSTORE, lambda **_kwargs: MemoryJobStore()),
    ):
        return create_scheduler()


class TestSchedulerRegistration:
    def test_when_created_then_all_jobs_registered(self) -> None:
        scheduler = _build_scheduler()
        ids = {job.id for job in scheduler.get_jobs()}
        assert ids == _ALL_JOBS

    def test_when_created_then_cron_jobs_use_cron_triggers(self) -> None:
        scheduler = _build_scheduler()
        for job in scheduler.get_jobs():
            if job.id in _CRON_JOBS:
                assert isinstance(job.trigger, CronTrigger)

    def test_when_created_then_interval_jobs_use_interval_triggers(self) -> None:
        scheduler = _build_scheduler()
        for job in scheduler.get_jobs():
            if job.id in _INTERVAL_JOBS:
                assert isinstance(job.trigger, IntervalTrigger)

    def test_when_created_then_scheduler_not_started(self) -> None:
        scheduler = _build_scheduler()
        assert scheduler.running is False

    def test_when_engine_none_then_raises_runtime_error(self) -> None:
        dm = MagicMock()
        dm.engine = None
        with patch(_DM, dm), pytest.raises(RuntimeError, match="before"):
            create_scheduler()
