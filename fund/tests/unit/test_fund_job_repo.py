"""``FundJobRepository``: atomic job slot + heartbeat-lease reaper.

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
from unittest.mock import MagicMock, patch

import pytest
from portopt_db.models import BackgroundJob, Base
from sqlalchemy import create_engine, event, false
from sqlalchemy.exc import IntegrityError
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


def test_fund_job_types_are_the_daemon_slots():
    assert FUND_JOB_TYPES == (
        "fund_rebalance_sweep",
        "fund_drift_monitor",
        "fund_nav_mark",
    )


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


def test_claim_returns_none_when_a_concurrent_claimer_won_the_race(job_session):
    """The unique index, not NOT EXISTS, is the atomicity guarantee.

    Under READ COMMITTED a racing claimer cannot see the winner's uncommitted
    row, so its NOT EXISTS passes; blinding the check reproduces that and leaves
    only the database to refuse the second row. The slot must read as busy
    (``None``), not raise, and the session must stay usable.
    """
    repo = FundJobRepository(job_session)
    first = repo.claim_or_create("fund_rebalance_sweep")
    assert first is not None

    blind_check = MagicMock()
    blind_check.where.return_value = false()
    with patch("fund.audit.fund_job_repository.exists", return_value=blind_check):
        second = repo.claim_or_create("fund_rebalance_sweep")

    assert second is None
    rows = (
        job_session.query(BackgroundJob)
        .filter_by(job_type="fund_rebalance_sweep")
        .all()
    )
    assert [r.id for r in rows] == [first]
    assert repo.claim_or_create("fund_drift_monitor") is not None


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
        job_session, job_type="fund_nav_mark", status="pending", heartbeat="none"
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


# --- shared work session: a failed flush must not wedge the slot -------------


def _poison_session(session, existing_job_id: uuid.UUID) -> None:
    """Fail a flush (duplicate primary key) so the session needs a rollback."""
    session.add(BackgroundJob(id=existing_job_id, job_type="fund_nav_mark"))
    with pytest.raises(IntegrityError):
        session.flush()
    assert session.is_active is False


def test_mark_done_recovers_a_session_poisoned_by_a_failed_flush(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(job_id)
    _poison_session(job_session, job_id)

    assert repo.mark_done(job_id, status="failed", error="boom") is True

    job_session.expire_all()
    row = repo.get(job_id)
    assert row.status == "failed"
    assert row.error == "boom"


def test_mark_running_recovers_a_session_poisoned_by_a_failed_flush(job_session):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_drift_monitor")
    _poison_session(job_session, job_id)

    assert repo.mark_running(job_id) is True

    job_session.expire_all()
    assert repo.get(job_id).status == "running"


def test_claim_or_create_recovers_a_session_poisoned_by_a_failed_flush(job_session):
    repo = FundJobRepository(job_session)
    first = repo.claim_or_create("fund_rebalance_sweep")
    _poison_session(job_session, first)

    assert repo.claim_or_create("fund_drift_monitor") is not None


# --- claim_or_create under concurrent PostgreSQL claimants ------------------


@pytest.fixture
def pg_claim_trace() -> Generator[tuple[Session, list[tuple[str, str]]], None, None]:
    """A SQLite session that poses as PostgreSQL and records slot-claim SQL.

    SQLite serialises writers so it cannot show the READ COMMITTED race itself;
    what *is* observable is the statement order. ``pg_advisory_xact_lock`` and
    ``hashtext`` are registered as recording stubs so the PostgreSQL-only guard
    can run, and the dialect name is flipped so the repository takes that branch.
    """
    trace: list[tuple[str, str]] = []
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _register(dbapi_conn, _record):
        dbapi_conn.create_function("hashtext", 1, lambda key: key)
        dbapi_conn.create_function(
            "pg_advisory_xact_lock",
            2,
            lambda ns, key: trace.append(("lock", str(key))),
        )

    @event.listens_for(engine, "before_cursor_execute")
    def _record(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith("INSERT INTO background_jobs"):
            trace.append(("insert", ""))

    Base.metadata.create_all(bind=engine)
    engine.dialect.name = "postgresql"
    session = Session(bind=engine)
    try:
        yield session, trace
    finally:
        session.close()
        engine.dispose()


def test_claim_or_create_takes_a_per_job_type_lock_before_the_guarded_insert(
    pg_claim_trace,
):
    session, trace = pg_claim_trace

    job_id = FundJobRepository(session).claim_or_create("fund_rebalance_sweep")

    assert job_id is not None
    # Without the lock, two READ COMMITTED claimants both pass NOT EXISTS and insert.
    assert trace == [("lock", "fund_rebalance_sweep"), ("insert", "")]


# --- one clock for the heartbeat lease --------------------------------------


class _AppClockAheadOfDb(datetime):
    """``datetime`` whose ``now()`` runs an hour ahead of the database clock."""

    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + timedelta(hours=1)


def test_heartbeating_slot_survives_the_reaper_when_app_and_db_clocks_skew(
    job_session, monkeypatch
):
    monkeypatch.setattr("fund.audit.fund_job_repository.datetime", _AppClockAheadOfDb)
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(job_id)
    assert repo.update_heartbeat(job_id) is True

    reaped = repo.reap_orphans("lease expired", heartbeat_timeout_seconds=300)
    job_session.expire_all()

    # The lease is stamped and compared on one clock, so skew cannot expire it.
    assert reaped == 0
    assert repo.get(job_id).status == "running"


# --- finished_at is the wall-clock moment of the transition -----------------

_WALL_CLOCK = datetime(2031, 1, 1, 12, 0, tzinfo=UTC)


class _FrozenDatetime(datetime):
    """``datetime`` pinned to :data:`_WALL_CLOCK` (a fixed point, not a txn start)."""

    @classmethod
    def now(cls, tz=None):
        return _WALL_CLOCK.astimezone(tz)


def test_mark_done_stamps_the_wall_clock_moment_not_the_transaction_start(
    job_session, monkeypatch
):
    repo = FundJobRepository(job_session)
    job_id = repo.claim_or_create("fund_rebalance_sweep")
    repo.mark_running(job_id)

    monkeypatch.setattr("fund.audit.fund_job_repository.datetime", _FrozenDatetime)
    assert repo.mark_done(job_id, status="completed") is True

    job_session.expire_all()
    row = repo.get(job_id)
    assert _naive(row.finished_at) == _naive(_WALL_CLOCK)
    assert _naive(row.updated_at) == _naive(_WALL_CLOCK)


def test_reap_orphans_stamps_the_wall_clock_moment_on_failed_slots(
    job_session, monkeypatch
):
    repo = FundJobRepository(job_session)
    stale = _insert_job(
        job_session, job_type="fund_drift_monitor", status="running", heartbeat="stale"
    )

    monkeypatch.setattr("fund.audit.fund_job_repository.datetime", _FrozenDatetime)
    assert repo.reap_orphans("lease expired", heartbeat_timeout_seconds=300) == 1
    job_session.expire_all()

    assert _naive(repo.get(stale.id).finished_at) == _naive(_WALL_CLOCK)
