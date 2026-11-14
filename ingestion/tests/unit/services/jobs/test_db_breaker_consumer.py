"""The ingestion daemon's handling of an open database circuit breaker.

Validates ``app.services.jobs.scheduler._run_step``, ``_mark_job_failed`` and
``run_orphan_reaper``, the ``portopt_db_breaker_state`` gauge and the
``jobs_db_unavailable_total`` counter in ``app.metrics``, and the binding in
``app.worker._serve_metrics``. The engines are real SQLite engines: one whose
every connect fails, one that works until the test opens its breaker. The
breakers run on a fake clock, so no test sleeps.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from portopt_db.models import BackgroundJob, BackgroundJobError
from portopt_db.resilience import (
    BreakerConfig,
    DatabaseUnavailableError,
    DbCircuitBreaker,
)
from prometheus_client import REGISTRY
from sqlalchemy.exc import DisconnectionError, OperationalError

from app import metrics
from app import worker as worker_module
from app.repositories.jobs.background_job_repository import BackgroundJobRepository
from app.services.jobs import scheduler
from app.services.jobs.background_job import BackgroundJobService

_THRESHOLD = 2
_TIMEOUT_S = 30.0
_SCHEDULER_LOGGER = "app.services.jobs.scheduler"
_JOBS_LOGGER = "app.services.jobs.background_job"


class _Clock:
    """Monotonic fake clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


class _SqliteConfig(DbConfig):
    """SQLite config: the psycopg2 connect args would be rejected."""

    def connect_args(self) -> dict:
        """Return no driver arguments, since sqlite3 accepts none of psycopg2's."""
        return {}


def _manager(url: str, clock: _Clock) -> DatabaseManager:
    breaker = DbCircuitBreaker(
        BreakerConfig(failure_threshold=_THRESHOLD, reset_timeout_s=_TIMEOUT_S),
        clock=clock,
    )
    return DatabaseManager(_SqliteConfig(url=url), breaker=breaker)


def _counter(name: str, domain: str) -> float:
    value = REGISTRY.get_sample_value(name, {"domain": domain})
    return value or 0.0


def _unavailable_count(domain: str) -> float:
    return _counter("jobs_db_unavailable_total", domain)


def _refused_status_count(domain: str) -> float:
    return _counter("jobs_failed_status_refused_total", domain)


def _scheduler_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _SCHEDULER_LOGGER]


@pytest.fixture
def clock() -> _Clock:
    """A fresh fake clock per test."""
    return _Clock()


@pytest.fixture
def tracked_breaker_gauge():
    """Give the test the process-wide breaker gauge, then put the old binding back."""
    previous = metrics.track_db_breaker(None)
    yield
    metrics.track_db_breaker(previous)


@pytest.fixture
def unreachable(tmp_path, clock: _Clock):
    """A manager whose every connect fails: its SQLite file's folder is missing."""
    mgr = _manager(f"sqlite:///{tmp_path / 'missing' / 'jobs.db'}", clock)
    yield mgr
    mgr.close()


@pytest.fixture
def reachable(tmp_path, clock: _Clock):
    """A working manager holding only the job tables."""
    mgr = _manager(f"sqlite:///{tmp_path / 'jobs.db'}", clock)
    mgr.initialize()
    assert mgr.engine is not None
    BackgroundJob.metadata.create_all(
        mgr.engine, tables=[BackgroundJob.__table__, BackgroundJobError.__table__]
    )
    yield mgr
    mgr.close()


def test_open_breaker_skips_the_step_without_touching_the_database(
    unreachable: DatabaseManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A dead database fails the step in one line: no traceback, no engine, no work."""
    domain = "breaker_probe_claim"
    jobs = BackgroundJobService(
        job_type=domain, session_factory=unreachable.get_session
    )
    work = MagicMock()
    for _ in range(_THRESHOLD):
        assert scheduler._run_step("probe", jobs, work) is False
    before = _unavailable_count(domain)
    caplog.clear()

    with (
        caplog.at_level(logging.INFO, logger=_SCHEDULER_LOGGER),
        patch.object(unreachable, "initialize") as initialize,
    ):
        completed = scheduler._run_step("probe", jobs, work)

    assert completed is False
    work.assert_not_called()
    initialize.assert_not_called()
    records = _scheduler_records(caplog)
    assert [r.levelname for r in records] == ["ERROR"]
    assert "database unavailable" in records[0].getMessage()
    assert records[0].exc_info is None
    assert _unavailable_count(domain) == before + 1


def test_breaker_opening_mid_step_leaves_the_row_for_the_orphan_reaper(
    reachable: DatabaseManager,
    clock: _Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failed status cannot be written through an open breaker; the reaper
    records the failure once the database answers again."""
    domain = "breaker_probe_midstep"
    jobs = BackgroundJobService(job_type=domain, session_factory=reachable.get_session)

    def database_dies(*, on_progress) -> None:
        """Lose the connection mid-step, then try one more session."""
        for _ in range(_THRESHOLD):
            reachable.breaker.record_failure(DisconnectionError("server closed"))
        with reachable.get_session():
            pass

    before = _unavailable_count(domain)
    refused_before = _refused_status_count(domain)
    with caplog.at_level(logging.INFO, logger=_SCHEDULER_LOGGER):
        completed = scheduler._run_step("probe", jobs, database_dies)

    assert completed is False
    records = _scheduler_records(caplog)
    assert all(r.exc_info is None for r in records)
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 2
    assert "database unavailable" in errors[0]
    assert "orphan reaper" in errors[1]
    assert _unavailable_count(domain) == before + 1
    assert _refused_status_count(domain) == refused_before + 1

    clock.advance(_TIMEOUT_S)
    with reachable.get_session() as session:
        repo = BackgroundJobRepository(session)
        (row,) = session.query(BackgroundJob).filter_by(job_type=domain).all()
        assert row.status == "running"
        repo.update(
            row.id, last_heartbeat_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        reaped = repo.reap_orphans("orphaned - heartbeat stale")
        session.commit()
        session.refresh(row)

    assert [entry["job_type"] for entry in reaped] == [domain]
    assert row.status == "failed"


def test_failed_status_carries_the_db_unavailable_reason_when_it_lands() -> None:
    """When the write does get through, the row names why the step stopped."""
    jobs = MagicMock()

    scheduler._mark_job_failed(
        jobs,
        str(uuid.uuid4()),
        "probe",
        DatabaseUnavailableError("database unavailable: breaker open"),
    )

    error = jobs.update_job.call_args.kwargs["error"]
    assert error.startswith("db_unavailable: ")
    assert "breaker open" in error


def test_a_refused_failed_status_write_is_logged_and_counted_when_it_happens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whatever stopped the step, a refused status write is visible at once."""
    domain = "breaker_probe_refused_status"
    jobs = MagicMock()
    jobs._job_type = domain
    jobs.update_job.side_effect = DatabaseUnavailableError("database unavailable: x")
    before = _refused_status_count(domain)

    with caplog.at_level(logging.INFO, logger=_SCHEDULER_LOGGER):
        scheduler._mark_job_failed(
            jobs, str(uuid.uuid4()), "probe", RuntimeError("the step's own bug")
        )

    records = _scheduler_records(caplog)
    assert len(records) == 1
    assert "orphan reaper" in records[0].getMessage()
    assert records[0].exc_info is None
    assert _refused_status_count(domain) == before + 1


def test_other_step_failures_keep_their_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only the breaker refusal is shortened; a real bug still logs its stack."""
    jobs = MagicMock()
    jobs.create_job.return_value = str(uuid.uuid4())
    jobs._heartbeat_cadence = 30
    jobs._job_type = "breaker_probe_bug"
    jobs.get_job.return_value = {"status": "running"}
    work = MagicMock(side_effect=OperationalError("SELECT 1", {}, Exception("boom")))

    with (
        caplog.at_level(logging.INFO, logger=_SCHEDULER_LOGGER),
        patch.object(scheduler.threading, "Thread"),
    ):
        assert scheduler._run_step("probe", jobs, work) is False

    raised = [r for r in _scheduler_records(caplog) if "raised" in r.getMessage()]
    assert len(raised) == 1
    assert raised[0].exc_info is not None


def test_refused_progress_write_logs_one_line_without_a_stack(
    unreachable: DatabaseManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep reports progress per item; a refusal must not log a stack each time."""
    jobs = BackgroundJobService(
        job_type="breaker_probe_progress", session_factory=unreachable.get_session
    )
    for _ in range(_THRESHOLD):
        unreachable.breaker.record_failure(DisconnectionError("server closed"))

    with caplog.at_level(logging.INFO, logger=_JOBS_LOGGER):
        jobs.update_job(str(uuid.uuid4()), current=1, total=2)

    records = [r for r in caplog.records if r.name == _JOBS_LOGGER]
    assert len(records) == 1
    assert "database unavailable" in records[0].getMessage()
    assert records[0].exc_info is None


def test_refused_heartbeat_logs_one_line_and_keeps_the_loop_alive(
    unreachable: DatabaseManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A heartbeat tick refused by the breaker keeps ticking and logs no stack."""
    jobs = BackgroundJobService(
        job_type="breaker_probe_heartbeat", session_factory=unreachable.get_session
    )
    for _ in range(_THRESHOLD):
        unreachable.breaker.record_failure(DisconnectionError("server closed"))

    with caplog.at_level(logging.INFO, logger=_JOBS_LOGGER):
        keep_ticking = jobs._tick_heartbeat(uuid.uuid4(), threading.Event())

    assert keep_ticking is True
    records = [r for r in caplog.records if r.name == _JOBS_LOGGER]
    assert len(records) == 1
    assert "database unavailable" in records[0].getMessage()
    assert records[0].exc_info is None


def test_reaper_tick_skips_quietly_while_the_breaker_is_open(
    unreachable: DatabaseManager, caplog: pytest.LogCaptureFixture
) -> None:
    """The periodic reaper logs one line, not a traceback per tick."""
    for _ in range(_THRESHOLD):
        unreachable.breaker.record_failure(DisconnectionError("server closed"))

    with (
        caplog.at_level(logging.INFO, logger=_SCHEDULER_LOGGER),
        patch.object(scheduler, "database_manager", unreachable),
    ):
        scheduler.run_orphan_reaper()

    records = _scheduler_records(caplog)
    assert len(records) == 1
    assert "database unavailable" in records[0].getMessage()
    assert records[0].exc_info is None


def test_breaker_state_gauge_reports_zero_one_two(
    clock: _Clock, tracked_breaker_gauge: None
) -> None:
    """The gauge reads 0 closed, 1 open and 2 half-open at each scrape."""
    breaker = DbCircuitBreaker(
        BreakerConfig(failure_threshold=1, reset_timeout_s=_TIMEOUT_S), clock=clock
    )
    metrics.track_db_breaker(breaker)

    def scraped() -> float | None:
        """Read the gauge the way a Prometheus scrape would."""
        return REGISTRY.get_sample_value("portopt_db_breaker_state")

    assert scraped() == 0
    breaker.record_failure(DisconnectionError("server closed"))
    assert scraped() == 1
    clock.advance(_TIMEOUT_S)
    ticket = breaker.before_call()
    assert scraped() == 2
    breaker.record_success(ticket)
    assert scraped() == 0


def test_serving_metrics_binds_the_gauge_to_the_daemon_breaker() -> None:
    """The worker reports the breaker of the manager every step goes through."""
    with (
        patch.object(worker_module.settings, "enable_metrics", True),
        patch.object(worker_module, "start_http_server"),
        patch.object(metrics, "track_db_breaker") as track,
    ):
        worker_module._serve_metrics()

    track.assert_called_once_with(worker_module.database_manager.breaker)


def test_tracking_a_breaker_returns_the_one_it_replaces(
    clock: _Clock, tracked_breaker_gauge: None
) -> None:
    """A caller can put the previous binding back; none reads as closed."""
    breaker = DbCircuitBreaker(BreakerConfig(failure_threshold=1), clock=clock)
    breaker.record_failure(DisconnectionError("server closed"))

    assert metrics.track_db_breaker(breaker) is None
    assert REGISTRY.get_sample_value("portopt_db_breaker_state") == 1
    assert metrics.track_db_breaker(None) is breaker
    assert REGISTRY.get_sample_value("portopt_db_breaker_state") == 0


@pytest.mark.parametrize(
    "counter", ["jobs_db_unavailable_total", "jobs_failed_status_refused_total"]
)
def test_registering_a_domain_creates_its_breaker_series_at_zero(counter: str) -> None:
    """The counters exist before the first refusal, so ``increase()`` sees them."""
    metrics.register_domain("breaker_probe_registered")

    assert (
        REGISTRY.get_sample_value(counter, {"domain": "breaker_probe_registered"})
        == 0.0
    )
