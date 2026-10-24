"""``create_scheduler`` registration + the agent-stack-free import guard.

Asserts ``create_scheduler()`` registers exactly the five fund jobs with the
right trigger types, on a single-worker executor, a **distinct** jobstore table
(``fund_apscheduler_jobs``), UTC, and the Saturday cron (weekday name — a bare
``0`` would fire Monday) — **without starting** the scheduler; and that it raises
``RuntimeError`` when the DB engine is unset.

The engine is mocked non-None and ``SQLAlchemyJobStore`` is swapped for an
in-memory jobstore so no real database is touched (mirrors
``ingestion/.../test_scheduler_registration.py`` — a DB-less test otherwise builds
``SQLAlchemyJobStore(engine=None)`` and fails).

A subprocess guard (fresh interpreter) proves ``import fund.scheduler`` drags in
none of the agent stack (deepagents/langchain/langgraph) nor the ingestion daemon.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from fund.config import settings
from fund.scheduler import (
    JOB_DRIFT_MONITOR,
    JOB_NAV_MARK,
    JOB_NAV_MARK_FOLLOWUP,
    JOB_ORPHAN_REAPER,
    JOB_REBALANCE_SWEEP,
    create_scheduler,
    run_scheduled_nav_mark,
)

_DM = "fund.scheduler.database_manager"
_JOBSTORE = "fund.scheduler.SQLAlchemyJobStore"
_EXECUTOR = "fund.scheduler.ThreadPoolExecutor"

_CRON_JOBS = {JOB_REBALANCE_SWEEP, JOB_NAV_MARK}
_INTERVAL_JOBS = {JOB_DRIFT_MONITOR, JOB_ORPHAN_REAPER, JOB_NAV_MARK_FOLLOWUP}
_ALL_JOBS = _CRON_JOBS | _INTERVAL_JOBS


def _build_scheduler(jobstore_calls: list[dict] | None = None, config=settings):
    dm = MagicMock()
    dm.engine = MagicMock()  # non-None

    def _fake_jobstore(**kwargs):
        if jobstore_calls is not None:
            jobstore_calls.append(kwargs)
        return MemoryJobStore()

    with patch(_DM, dm), patch(_JOBSTORE, _fake_jobstore):
        return create_scheduler(config)


def test_registers_exactly_the_five_fund_jobs():
    scheduler = _build_scheduler()
    ids = {job.id for job in scheduler.get_jobs()}
    assert ids == _ALL_JOBS


def test_nav_mark_waits_for_ingestion_by_default():
    """The cron and a periodic re-check both go through the waiting policy."""
    scheduler = _build_scheduler()
    assert scheduler.get_job(JOB_NAV_MARK).func is run_scheduled_nav_mark
    assert scheduler.get_job(JOB_NAV_MARK_FOLLOWUP).func is run_scheduled_nav_mark


def test_the_nav_mark_recheck_is_registered_even_without_a_wait_budget():
    """The jobstore persists jobs, so one registered only sometimes would outlive a
    config change; with no budget the re-check still catches a late sweep."""
    config = dataclasses.replace(settings, fund_nav_mark_wait_seconds=0)
    scheduler = _build_scheduler(config=config)
    assert scheduler.get_job(JOB_NAV_MARK_FOLLOWUP).func is run_scheduled_nav_mark


def test_rebalance_sweep_uses_a_cron_trigger():
    scheduler = _build_scheduler()
    job = scheduler.get_job(JOB_REBALANCE_SWEEP)
    assert isinstance(job.trigger, CronTrigger)


def test_nav_mark_uses_a_cron_trigger():
    scheduler = _build_scheduler()
    job = scheduler.get_job(JOB_NAV_MARK)
    assert isinstance(job.trigger, CronTrigger)


def test_interval_jobs_use_interval_triggers():
    scheduler = _build_scheduler()
    for job_id in _INTERVAL_JOBS:
        job = scheduler.get_job(job_id)
        assert isinstance(job.trigger, IntervalTrigger)


def test_rebalance_cron_fires_saturday_not_monday():
    # the default cron uses the weekday NAME `sat`; a bare `0` fires Monday.
    scheduler = _build_scheduler()
    trigger = scheduler.get_job(JOB_REBALANCE_SWEEP).trigger
    assert "day_of_week='sat'" in str(trigger)


def test_scheduler_is_not_started():
    scheduler = _build_scheduler()
    assert scheduler.running is False


def test_jobstore_uses_the_distinct_fund_table():
    calls: list[dict] = []
    _build_scheduler(jobstore_calls=calls)
    assert calls, "SQLAlchemyJobStore was never constructed"
    assert all(c.get("tablename") == "fund_apscheduler_jobs" for c in calls)


def test_executor_is_single_worker():
    dm = MagicMock()
    dm.engine = MagicMock()
    recorded: dict[str, int] = {}

    def _spy_executor(*, max_workers):
        recorded["max_workers"] = max_workers
        return ThreadPoolExecutor(max_workers=max_workers)

    with (
        patch(_DM, dm),
        patch(_JOBSTORE, lambda **_kwargs: MemoryJobStore()),
        patch(_EXECUTOR, _spy_executor),
    ):
        create_scheduler()

    assert recorded["max_workers"] == 1


def test_jobs_wait_behind_the_single_worker_instead_of_being_dropped_as_missed():
    # One worker runs one step at a time, so a job queued behind a multi-minute LLM
    # round starts late. APScheduler's default 1s misfire grace would discard it as
    # "missed" (a skipped weekly sweep, NAV mark or reaper tick), so every job must
    # carry its own generous grace.
    scheduler = _build_scheduler()
    for job in scheduler.get_jobs():
        grace = getattr(job, "misfire_grace_time", None)
        assert grace is not None and grace >= 3600, (
            f"{job.id} falls back to APScheduler's 1s misfire grace"
        )


def test_misfire_grace_comes_from_the_config_knob():
    config = dataclasses.replace(settings, fund_misfire_grace_time_seconds=7200)
    scheduler = _build_scheduler(config=config)
    assert {job.misfire_grace_time for job in scheduler.get_jobs()} == {7200}


def test_raises_when_engine_is_none():
    dm = MagicMock()
    dm.engine = None
    with patch(_DM, dm), pytest.raises(RuntimeError, match="before"):
        create_scheduler()


def test_import_fund_scheduler_is_agent_stack_free():
    # The agent stack + model builders live lazily inside functions; a bare
    # ``import fund.scheduler`` must drag none of it (nor the ingestion daemon)
    # into a fresh interpreter's sys.modules. Fresh process ⇒ order-independent.
    code = textwrap.dedent(
        """
        import sys
        import fund.scheduler  # noqa: F401
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
