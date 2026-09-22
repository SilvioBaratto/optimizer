"""T2 — ``FundJobRepository``: atomic job slot + heartbeat-lease reaper.

Boundary-clean job mechanics over the shared ``background_jobs`` model, with two
fund-specific ``job_type``s (:data:`FUND_JOB_TYPES`). Mirrors the ingestion
``BackgroundJobRepository`` but lives in ``fund.audit`` (fund must never import
``ingestion.app``).

Contracts under test:
- ``claim_or_create`` is atomic — at most one active row per ``job_type`` (a
  double claim returns ``None``); the slot re-opens once finalised.
- ``mark_running`` / ``update_heartbeat`` / ``mark_done`` are status-guarded.
- ``reap_orphans`` is a pure **heartbeat lease**: it fails NULL/stale-heartbeat
  ``(pending|running)`` rows with **no** ``worker_host``/``worker_pid`` clause
  (a stale row stamped with this very process's live PID is still reaped),
  scoped to ``FUND_JOB_TYPES`` (a stale ingestion job is left alone), leaving
  fresh rows untouched.

Because this repo owns its own transactions (each lifecycle mutator
``commit()``s so the slot state lands for a reaper in another process), it cannot
use the shared SAVEPOINT ``db_session`` fixture — an internal ``commit()`` there
escapes the outer rollback and leaks rows across the session-scoped engine. These
tests use the private, function-scoped :func:`job_session` fixture instead: a
pristine in-memory DB per test, discarded on teardown, where commits are free.
``reap_orphans`` does not commit (its scheduler caller owns the commit), so the
reap tests ``expire_all()`` before re-reading.
"""

from __future__ import annotations

import inspect
import os
import socket
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from portopt_db.models import BackgroundJob, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fund.audit import FUND_JOB_TYPES, FundJobRepository


@pytest.fixture
def job_session() -> Generator[Session, None, None]:
    """A fresh in-memory DB per test whose real ``commit()`` calls are contained.

    ``FundJobRepository`` is the one fund repo that owns its transactions, so its
    tests get a private, function-scoped engine (pristine DB per test) rather than
    the shared SAVEPOINT ``db_session`` — where an internal ``commit()`` would
    escape the outer rollback and leak rows across the session-scoped engine.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _insert_job(
    session,
    *,
    job_type: str,
    status: str = "pending",
    heartbeat: str = "fresh",
) -> BackgroundJob:
    """Add a controlled ``background_jobs`` row (bypassing the atomic claim)."""
    now = datetime.now(UTC)
    hb = {"fresh": now, "stale": now - timedelta(seconds=10_000), "none": None}[
        heartbeat
    ]
    job = BackgroundJob(
        job_type=job_type,
        status=status,
        started_at=now,
        last_heartbeat_at=hb,
    )
    session.add(job)
    session.flush()
    return job


def _naive(dt: datetime) -> datetime:
    """Drop tz so SQLite's naive ``func.now()`` reads compare cleanly."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


# --- exports + constants ----------------------------------------------------


def test_fund_job_types_are_the_two_daemon_slots():
    assert FUND_JOB_TYPES == ("fund_rebalance_sweep", "fund_drift_monitor")


def test_fund_job_repository_exported_from_audit():
    from fund.audit import FundJobRepository as Exported

    assert Exported is FundJobRepository


# --- claim_or_create (atomic slot) ------------------------------------------


def test_claim_or_create_returns_uuid_then_none_for_double_claim(job_session):
    repo = FundJobRepository(job_session)

    first = repo.claim_or_create("fund_rebalance_sweep")
    assert isinstance(first, uuid.UUID)

    second = repo.claim_or_create("fund_rebalance_sweep")
    assert second is None

    rows = (
        job_session.query(BackgroundJob)
        .filter_by(job_type="fund_rebalance_sweep")
        .all()
    )
    assert len(rows) == 1  # the second claim never inserted a row


def test_claim_stamps_pending_heartbeat_and_worker_identity(job_session):
    repo = FundJobRepository(job_session)

    job_id = repo.claim_or_create("fund_drift_monitor")
    assert job_id is not None

    row = repo.get(job_id)
    assert row is not None
    assert row.status == "pending"
    assert row.last_heartbeat_at is not None
    assert row.worker_pid == os.getpid()
    assert row.worker_host == socket.gethostname()


def test_distinct_job_types_get_independent_slots(job_session):
    repo = FundJobRepository(job_session)

    sweep = repo.claim_or_create("fund_rebalance_sweep")
    monitor = repo.claim_or_create("fund_drift_monitor")

    assert sweep is not None
    assert monitor is not None
    assert sweep != monitor


def test_claim_or_create_reopens_slot_after_finalize(job_session):
    repo = FundJobRepository(job_session)

    first = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(first)
    repo.mark_done(first, status="completed")

    second = repo.claim_or_create("fund_rebalance_sweep")
    assert second is not None
    assert second != first


# --- mark_running -----------------------------------------------------------


def test_mark_running_flips_pending_to_running(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")

    assert repo.mark_running(job_id) is True
    assert repo.get(job_id).status == "running"


def test_mark_running_does_not_resurrect_terminal_row(job_session):
    repo = FundJobRepository(job_session)
    job = _insert_job(job_session, job_type="fund_rebalance_sweep", status="completed")

    assert repo.mark_running(job.id) is False
    assert repo.get(job.id).status == "completed"


# --- update_heartbeat -------------------------------------------------------


def test_update_heartbeat_renews_only_running_rows(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_drift_monitor")

    # still pending → the lease is not renewed
    assert repo.update_heartbeat(job_id) is False

    repo.mark_running(job_id)
    row = repo.get(job_id)
    row.last_heartbeat_at = datetime(2000, 1, 1)  # unmistakably old sentinel
    job_session.flush()

    assert repo.update_heartbeat(job_id) is True
    renewed = repo.get(job_id).last_heartbeat_at
    assert renewed is not None
    assert _naive(renewed) > datetime(2000, 1, 1)


def test_update_heartbeat_returns_false_for_unknown_job(job_session):
    repo = FundJobRepository(job_session)
    assert repo.update_heartbeat(uuid.uuid4()) is False


# --- mark_done --------------------------------------------------------------


def test_mark_done_finalizes_active_slot(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(job_id)

    assert repo.mark_done(job_id, status="completed") is True
    row = repo.get(job_id)
    assert row.status == "completed"
    assert row.finished_at is not None
    assert row.error is None


def test_mark_done_records_error_message(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_drift_monitor")
    repo.mark_running(job_id)

    assert repo.mark_done(job_id, status="failed", error="kaboom") is True
    row = repo.get(job_id)
    assert row.status == "failed"
    assert row.error == "kaboom"


def test_mark_done_is_status_guarded_against_double_finalize(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(job_id)
    assert repo.mark_done(job_id, status="completed") is True

    # a second (failure) finalize must not overwrite the terminal row
    assert repo.mark_done(job_id, status="failed", error="boom") is False
    row = repo.get(job_id)
    assert row.status == "completed"
    assert row.error is None


def test_mark_done_returns_false_for_unknown_job(job_session):
    repo = FundJobRepository(job_session)
    assert repo.mark_done(uuid.uuid4(), status="completed") is False


# --- reap_orphans (heartbeat lease) -----------------------------------------


def test_reap_orphans_fails_stale_and_null_leaves_fresh(job_session):
    repo = FundJobRepository(job_session)
    fresh = _insert_job(
        job_session,
        job_type="fund_rebalance_sweep",
        status="running",
        heartbeat="fresh",
    )
    stale = _insert_job(
        job_session, job_type="fund_drift_monitor", status="running", heartbeat="stale"
    )
    null_hb = _insert_job(
        job_session, job_type="fund_drift_monitor", status="pending", heartbeat="none"
    )

    reaped = repo.reap_orphans("lease expired", heartbeat_timeout_seconds=300)
    job_session.expire_all()  # reap_orphans does not commit; refresh the reads

    assert reaped == 2
    assert repo.get(stale.id).status == "failed"
    assert repo.get(stale.id).error == "lease expired"
    assert repo.get(stale.id).finished_at is not None
    assert repo.get(null_hb.id).status == "failed"
    # the fresh-heartbeat row is untouched
    assert repo.get(fresh.id).status == "running"
    assert repo.get(fresh.id).finished_at is None


def test_reap_orphans_ignores_non_fund_jobs(job_session):
    repo = FundJobRepository(job_session)
    ingestion_job = _insert_job(
        job_session, job_type="yfinance_fetch", status="running", heartbeat="stale"
    )
    fund_job = _insert_job(
        job_session,
        job_type="fund_rebalance_sweep",
        status="running",
        heartbeat="stale",
    )

    reaped = repo.reap_orphans("lease expired", heartbeat_timeout_seconds=300)
    job_session.expire_all()

    assert reaped == 1
    assert repo.get(fund_job.id).status == "failed"
    assert repo.get(ingestion_job.id).status == "running"  # left alone


def test_reap_orphans_reaps_stale_row_stamped_with_this_live_process(job_session):
    """A stale row carrying THIS process's live pid/host is still reaped —
    proving liveness is a pure lease, never a pid/host identity check."""
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")  # stamps our live pid/host
    row = repo.get(job_id)
    row.last_heartbeat_at = datetime(2000, 1, 1)
    job_session.flush()

    reaped = repo.reap_orphans("lease expired", heartbeat_timeout_seconds=300)
    job_session.expire_all()

    assert reaped == 1
    reaped_row = repo.get(job_id)
    assert reaped_row.status == "failed"
    assert reaped_row.worker_pid == os.getpid()  # our own live pid, reaped anyway


def test_reap_orphans_predicate_has_no_host_or_pid_clause():
    src = inspect.getsource(FundJobRepository.reap_orphans)
    assert "worker_host" not in src
    assert "worker_pid" not in src
