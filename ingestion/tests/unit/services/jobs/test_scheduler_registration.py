"""Scheduler registration tests.

Asserts ``create_scheduler()`` registers exactly the 7 expected job ids with
the expected trigger types, without starting the scheduler or invoking any
registered function (no live fetch).  The DB engine is mocked non-None and the
SQLAlchemy jobstore is swapped for an in-memory one so no real database is
touched.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.executors.base import BaseExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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


@pytest.fixture
def jobstore_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    yield engine
    engine.dispose()


class _RecordingExecutor(BaseExecutor):
    """Records submissions instead of running them, so no real step ever fires."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def _do_submit_job(self, job, run_times) -> None:
        self.submitted.append(job.id)


def _boot(engine: Engine, *, paused: bool = True):
    """Build the scheduler over a real SQLAlchemy jobstore and start it.

    Returns the scheduler and the recording executor. Paused by default so the
    persisted schedules are flushed and readable without anything firing.
    """
    dm = MagicMock()
    dm.engine = engine
    executors: list[_RecordingExecutor] = []

    def _executor(*args, **kwargs) -> _RecordingExecutor:
        executors.append(_RecordingExecutor(*args, **kwargs))
        return executors[-1]

    with (
        patch(_BOUND_SCHEDULER, None),
        patch(_DM, dm),
        patch("app.services.jobs.scheduler.ThreadPoolExecutor", _executor),
    ):
        scheduler = create_scheduler()
    scheduler.start(paused=paused)
    return scheduler, executors[0]


def _persist_next_run_time(engine: Engine, job_id: str, when: datetime) -> None:
    """Rewrite a stored job's fire time, as if the daemon had been down since."""
    store = SQLAlchemyJobStore(engine=engine)
    job = store.lookup_job(job_id)
    job.next_run_time = when
    store.update_job(job)


def _seed_persisted_schedules(engine: Engine, when: datetime, *job_ids: str) -> None:
    scheduler, _ = _boot(engine)
    scheduler.shutdown(wait=False)
    for job_id in job_ids:
        _persist_next_run_time(engine, job_id, when)


class TestRestartPreservesPersistedSchedule:
    """A fire missed while the daemon was down must survive the restart.

    Re-registering every job on boot recomputes ``next_run_time`` from now, so
    the persisted missed fire was discarded and the misfire grace window never
    applied.
    """

    @staticmethod
    def _minutes_ago(minutes: int) -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
            minutes=minutes
        )

    def test_missed_fire_time_is_kept_so_the_job_catches_up(
        self, jobstore_engine: Engine
    ) -> None:
        missed = self._minutes_ago(30)
        _seed_persisted_schedules(jobstore_engine, missed, "daily_pipeline")

        scheduler, _ = _boot(jobstore_engine)
        try:
            job = scheduler.get_job("daily_pipeline")
            assert job is not None
            kept = job.next_run_time
        finally:
            scheduler.shutdown(wait=False)

        assert kept == missed

    def test_every_job_keeps_its_persisted_fire_time(
        self, jobstore_engine: Engine
    ) -> None:
        stale = self._minutes_ago(10)
        _seed_persisted_schedules(jobstore_engine, stale, *_ALL_JOBS)

        scheduler, _ = _boot(jobstore_engine)
        try:
            times = {j.id: j.next_run_time for j in scheduler.get_jobs()}
        finally:
            scheduler.shutdown(wait=False)

        assert times == dict.fromkeys(_ALL_JOBS, stale)

    def test_missed_run_inside_the_grace_window_fires_at_startup(
        self, jobstore_engine: Engine
    ) -> None:
        _seed_persisted_schedules(
            jobstore_engine, self._minutes_ago(30), "daily_pipeline"
        )

        scheduler, executor = _boot(jobstore_engine, paused=False)
        try:
            deadline = time.monotonic() + 3
            while "daily_pipeline" not in executor.submitted:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.02)
        finally:
            scheduler.shutdown(wait=False)

        assert "daily_pipeline" in executor.submitted

    def test_changed_cron_expression_is_rescheduled_from_now(
        self, jobstore_engine: Engine
    ) -> None:
        _seed_persisted_schedules(
            jobstore_engine, self._minutes_ago(30), "daily_pipeline"
        )

        with patch(
            "app.services.jobs.scheduler.settings.scheduler_daily_pipeline_cron",
            "30 9 * * *",
        ):
            scheduler, _ = _boot(jobstore_engine)
        try:
            job = scheduler.get_job("daily_pipeline")
            assert job is not None
            rescheduled = job.next_run_time
        finally:
            scheduler.shutdown(wait=False)

        assert rescheduled > datetime.now(timezone.utc)
        assert (rescheduled.hour, rescheduled.minute) == (9, 30)

    def test_first_boot_schedules_every_job_in_the_future(
        self, jobstore_engine: Engine
    ) -> None:
        scheduler, _ = _boot(jobstore_engine)
        try:
            times = [j.next_run_time for j in scheduler.get_jobs()]
        finally:
            scheduler.shutdown(wait=False)

        now = datetime.now(timezone.utc)
        assert len(times) == len(_ALL_JOBS)
        assert all(t > now for t in times)

    def test_unreadable_persisted_job_falls_back_to_a_fresh_schedule(self) -> None:
        class _BrokenStore(MemoryJobStore):
            def lookup_job(self, job_id):
                raise RuntimeError("cannot unpickle persisted job")

        dm = MagicMock()
        dm.engine = MagicMock()
        with (
            patch(_BOUND_SCHEDULER, None),
            patch(_DM, dm),
            patch(_JOBSTORE, lambda **_kwargs: _BrokenStore()),
        ):
            scheduler = create_scheduler()
        scheduler.start(paused=True)
        try:
            times = {j.id: j.next_run_time for j in scheduler.get_jobs()}
        finally:
            scheduler.shutdown(wait=False)

        now = datetime.now(timezone.utc)
        assert set(times) == _ALL_JOBS
        assert all(t > now for t in times.values())
