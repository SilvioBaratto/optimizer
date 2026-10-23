"""Scheduler registration tests.

Asserts ``create_scheduler()`` registers exactly the 7 expected job ids with
the expected trigger types, without starting the scheduler or invoking any
registered function (no live fetch).  The DB engine is mocked non-None and the
SQLAlchemy jobstore is swapped for an in-memory one so no real database is
touched.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.services.jobs.scheduler import create_scheduler

_DM = "app.services.jobs.scheduler.database_manager"
_JOBSTORE = "app.services.jobs.scheduler.SQLAlchemyJobStore"
_BOUND_SCHEDULER = "app.services.jobs.scheduler._scheduler"

_CRON_JOBS = {
    "daily_pipeline",
    "midday_news",
    "universe_build",
    "weekly_refetch",
    "fred_monthly",
    "weekly_market_wide",
}
_INTERVAL_JOBS = {"orphan_reaper"}
_ALL_JOBS = _CRON_JOBS | _INTERVAL_JOBS


def _build_scheduler():
    """Create a scheduler with a mocked engine + in-memory jobstore.

    The module-level scheduler binding is patched so a scheduler these tests
    start and stop does not leak into later tests as a draining one.
    """
    dm = MagicMock()
    dm.engine = MagicMock()  # non-None
    with (
        patch(_BOUND_SCHEDULER, None),
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


class TestSchedulerLoopLiveness:
    def test_when_job_store_raises_in_the_loop_then_loop_alive_turns_false(
        self,
    ) -> None:
        """APScheduler's loop thread dies on a job-store error while ``running``
        stays True; the owner needs a signal that does not depend on that flag."""
        scheduler = _build_scheduler()
        with patch.object(
            MemoryJobStore, "get_next_run_time", side_effect=RuntimeError("db gone")
        ):
            scheduler.start()
            try:
                deadline = time.monotonic() + 5
                while scheduler.loop_alive() and time.monotonic() < deadline:
                    time.sleep(0.01)
                dead = not scheduler.loop_alive()
                still_running = scheduler.running
            finally:
                scheduler.shutdown(wait=False)

        assert dead
        assert still_running  # the stock flag cannot be trusted

    def test_when_loop_is_healthy_then_loop_alive_is_true(self) -> None:
        scheduler = _build_scheduler()
        scheduler.start(paused=True)
        try:
            assert scheduler.loop_alive()
        finally:
            scheduler.shutdown(wait=False)
