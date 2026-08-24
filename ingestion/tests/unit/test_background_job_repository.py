"""Tests for BackgroundJobRepository — dialect-agnostic SQL (issue #314).

Covers claim_or_create atomicity and cleanup_expired cutoff arithmetic
on SQLite (the test dialect).  Regression guards against reintroduction
of PostgreSQL-only raw SQL in these two methods.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.repositories.jobs.background_job_repository import BackgroundJobRepository


def _make_stale(
    repo: BackgroundJobRepository, jid: uuid.UUID, session: Session
) -> None:
    """Force the row's heartbeat past the default 5-minute timeout."""
    stale = datetime.now(timezone.utc) - timedelta(minutes=10)
    repo.update(jid, last_heartbeat_at=stale)
    session.flush()


class TestClaimOrCreate:
    """claim_or_create must be atomic and dialect-agnostic."""

    def test_returns_uuid_when_no_active_job(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        result = repo.claim_or_create("test_job_type_314_a")
        assert isinstance(result, uuid.UUID)

    def test_returns_none_when_pending_job_exists(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        first = repo.claim_or_create("test_job_type_314_b")
        assert first is not None
        second = repo.claim_or_create("test_job_type_314_b")
        assert second is None

    def test_returns_none_when_running_job_exists(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_c")
        assert job_id is not None
        repo.update(job_id, status="running")
        db_session.flush()

        second = repo.claim_or_create("test_job_type_314_c")
        assert second is None

    def test_allows_new_job_after_completion(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_d")
        assert job_id is not None
        repo.update(job_id, status="completed", finished_at=datetime.now(timezone.utc))
        db_session.flush()

        second = repo.claim_or_create("test_job_type_314_d")
        assert second is not None
        assert second != job_id

    def test_allows_new_job_after_failure(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_e")
        assert job_id is not None
        repo.update(job_id, status="failed", finished_at=datetime.now(timezone.utc))
        db_session.flush()

        second = repo.claim_or_create("test_job_type_314_e")
        assert second is not None

    def test_different_job_types_do_not_conflict(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        first = repo.claim_or_create("test_job_type_314_f_alpha")
        second = repo.claim_or_create("test_job_type_314_f_beta")
        assert first is not None
        assert second is not None

    def test_initial_extra_stored_on_created_job(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_g", country="USA", run=42)
        assert job_id is not None
        db_session.flush()
        row = repo.get(job_id)
        assert row is not None
        assert row.extra == {"country": "USA", "run": 42}

    def test_created_job_has_pending_status(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_h")
        assert job_id is not None
        row = repo.get(job_id)
        assert row is not None
        assert row.status == "pending"


class TestCleanupExpired:
    """cleanup_expired must use dialect-agnostic datetime arithmetic."""

    def test_deletes_completed_job_past_ttl(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_cleanup_a")
        assert job_id is not None
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=7200)
        repo.update(job_id, status="completed", finished_at=old_ts)
        db_session.flush()

        deleted = repo.cleanup_expired(ttl_seconds=3600)
        assert deleted == 1

    def test_deletes_failed_job_past_ttl(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_cleanup_b")
        assert job_id is not None
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=7200)
        repo.update(job_id, status="failed", finished_at=old_ts)
        db_session.flush()

        deleted = repo.cleanup_expired(ttl_seconds=3600)
        assert deleted == 1

    def test_does_not_delete_job_within_ttl(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_cleanup_c")
        assert job_id is not None
        recent_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
        repo.update(job_id, status="completed", finished_at=recent_ts)
        db_session.flush()

        deleted = repo.cleanup_expired(ttl_seconds=3600)
        assert deleted == 0

    def test_does_not_delete_active_jobs(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_cleanup_d")
        assert job_id is not None

        deleted = repo.cleanup_expired(ttl_seconds=0)
        assert deleted == 0

    def test_does_not_delete_job_without_finished_at(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_cleanup_e")
        assert job_id is not None
        repo.update(job_id, status="completed")  # no finished_at
        db_session.flush()

        deleted = repo.cleanup_expired(ttl_seconds=0)
        assert deleted == 0

    def test_returns_count_of_deleted_rows(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=7200)
        for i in range(3):
            jid = repo.claim_or_create(f"test_job_type_314_cleanup_count_{i}")
            assert jid is not None
            repo.update(jid, status="completed", finished_at=old_ts)
        db_session.flush()

        deleted = repo.cleanup_expired(ttl_seconds=3600)
        assert deleted == 3


class TestGetLatestByType:
    """get_latest_by_type returns the most recently finished job."""

    def test_returns_none_when_no_jobs(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        result = repo.get_latest_by_type("no_such_type_375")
        assert result is None

    def test_returns_none_when_only_pending_jobs(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        repo.claim_or_create("test_375_pending_only")
        db_session.flush()

        result = repo.get_latest_by_type("test_375_pending_only")
        assert result is None

    def test_returns_completed_job(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_375_completed")
        assert job_id is not None
        finished = datetime.now(timezone.utc)
        repo.update(job_id, status="completed", finished_at=finished)
        db_session.flush()

        result = repo.get_latest_by_type("test_375_completed")
        assert result is not None
        assert result.status == "completed"

    def test_returns_failed_job(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_375_failed")
        assert job_id is not None
        finished = datetime.now(timezone.utc)
        repo.update(job_id, status="failed", finished_at=finished)
        db_session.flush()

        result = repo.get_latest_by_type("test_375_failed")
        assert result is not None
        assert result.status == "failed"

    def test_returns_most_recent_when_multiple(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        older = datetime.now(timezone.utc) - timedelta(hours=2)
        newer = datetime.now(timezone.utc)

        jid_old = repo.claim_or_create("test_375_multi")
        assert jid_old is not None
        repo.update(jid_old, status="failed", finished_at=older)
        db_session.flush()

        jid_new = repo.claim_or_create("test_375_multi")
        assert jid_new is not None
        repo.update(jid_new, status="completed", finished_at=newer)
        db_session.flush()

        result = repo.get_latest_by_type("test_375_multi")
        assert result is not None
        assert result.id == jid_new
        assert result.status == "completed"

    def test_does_not_cross_contaminate_job_types(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        finished = datetime.now(timezone.utc)

        jid = repo.claim_or_create("test_375_type_a")
        assert jid is not None
        repo.update(jid, status="completed", finished_at=finished)
        db_session.flush()

        result = repo.get_latest_by_type("test_375_type_b")
        assert result is None


class TestReconcileOrphans:
    """reconcile_orphans marks pending/running rows as failed at startup."""

    def test_when_pending_job_exists_marks_failed(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("test_557_pending")
        assert jid is not None
        _make_stale(repo, jid, db_session)

        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 1

        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"
        assert row.error == "orphaned at startup"
        assert row.finished_at is not None

    def test_when_running_job_exists_marks_failed(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("test_557_running")
        assert jid is not None
        repo.update(jid, status="running")
        _make_stale(repo, jid, db_session)

        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 1

        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"

    def test_when_only_terminal_jobs_returns_zero(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        finished = datetime.now(timezone.utc)
        for status_value in ("completed", "failed"):
            jid = repo.claim_or_create(f"test_557_terminal_{status_value}")
            assert jid is not None
            repo.update(jid, status=status_value, finished_at=finished)
        db_session.flush()

        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 0

    def test_when_table_empty_returns_zero(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 0

    def test_when_called_twice_second_call_is_noop(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("test_557_idempotent")
        assert jid is not None
        _make_stale(repo, jid, db_session)

        first = repo.reconcile_orphans("orphaned at startup")
        second = repo.reconcile_orphans("orphaned at startup")
        assert first == 1
        assert second == 0

    def test_when_multiple_active_jobs_returns_count(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        for i in range(3):
            jid = repo.claim_or_create(f"test_557_multi_{i}")
            assert jid is not None
            _make_stale(repo, jid, db_session)

        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 3

    def test_when_terminal_jobs_present_does_not_touch_them(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        completed_id = repo.claim_or_create("test_557_mixed_completed")
        assert completed_id is not None
        original_finished = datetime.now(timezone.utc) - timedelta(hours=1)
        repo.update(
            completed_id,
            status="completed",
            finished_at=original_finished,
        )
        pending_id = repo.claim_or_create("test_557_mixed_pending")
        assert pending_id is not None
        _make_stale(repo, pending_id, db_session)

        count = repo.reconcile_orphans("orphaned at startup")
        assert count == 1

        completed_row = repo.get(completed_id)
        assert completed_row is not None
        assert completed_row.status == "completed"
        assert completed_row.error is None

    def test_marks_pending_and_running_as_failed(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        pending_id = repo.claim_or_create("test_558_pending")
        assert pending_id is not None
        running_id = repo.claim_or_create("test_558_running")
        assert running_id is not None
        repo.update(running_id, status="running")
        _make_stale(repo, pending_id, db_session)
        _make_stale(repo, running_id, db_session)

        count = repo.reconcile_orphans("test msg")
        assert count == 2

        pending_row = repo.get(pending_id)
        running_row = repo.get(running_id)
        assert pending_row is not None
        assert running_row is not None
        assert pending_row.status == "failed"
        assert running_row.status == "failed"
        assert pending_row.finished_at is not None
        assert running_row.finished_at is not None
        assert pending_row.error == "test msg"
        assert running_row.error == "test msg"

    def test_returns_zero_when_no_active_jobs(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        count = repo.reconcile_orphans("test msg")
        assert count == 0

    def test_does_not_touch_completed_jobs(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("test_558_completed")
        assert jid is not None
        finished = datetime.now(timezone.utc) - timedelta(hours=1)
        repo.update(jid, status="completed", finished_at=finished)
        db_session.flush()

        count = repo.reconcile_orphans("test msg")
        assert count == 0

        row = repo.get(jid)
        assert row is not None
        assert row.status == "completed"
        assert row.error is None
        assert row.finished_at is not None
        assert row.finished_at.replace(tzinfo=timezone.utc) == finished

    def test_does_not_touch_already_failed_jobs(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("test_558_already_failed")
        assert jid is not None
        finished = datetime.now(timezone.utc) - timedelta(hours=1)
        repo.update(
            jid,
            status="failed",
            error="original error",
            finished_at=finished,
        )
        db_session.flush()

        count = repo.reconcile_orphans("test msg")
        assert count == 0

        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"
        assert row.error == "original error"
        assert row.finished_at is not None
        assert row.finished_at.replace(tzinfo=timezone.utc) == finished


class TestReapOrphans:
    """R3/§5.3 — reap_orphans fails stale orphans and returns them to re-run."""

    def test_reaps_stale_and_returns_job_type_and_attempt(
        self, db_session: Session
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("yfinance_fetch", attempt=1)
        assert jid is not None
        _make_stale(repo, jid, db_session)

        reaped = repo.reap_orphans("orphan retry")

        assert reaped == [{"job_type": "yfinance_fetch", "attempt": 1}]
        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"

    def test_fresh_lease_is_not_reaped(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("macro_fetch")
        assert jid is not None
        db_session.flush()

        assert repo.reap_orphans("orphan") == []
        row = repo.get(jid)
        assert row is not None
        assert row.status == "pending"

    def test_claim_or_create_stores_attempt(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("fred_fetch", attempt=2)
        assert jid is not None
        row = repo.get(jid)
        assert row is not None
        assert row.attempt == 2

    def test_claim_or_create_defaults_attempt_to_zero(
        self, db_session: Session
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("fred_fetch")
        assert jid is not None
        row = repo.get(jid)
        assert row is not None
        assert row.attempt == 0


class TestLivenessPredicate:
    """T2.1 / §5.3 — lease-based liveness: only a stale heartbeat reaps a claim.

    Host and PID no longer participate (they false-reaped across a host/
    container change and were inert off Linux). A fresh lease is never reaped,
    regardless of which host/PID holds it.
    """

    def test_reconcile_skips_fresh_lease(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_fresh")  # stamps a fresh heartbeat
        assert jid is not None
        db_session.flush()

        count = repo.reconcile_orphans("orphan")

        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.status == "pending"

    def test_reconcile_ignores_dead_pid_when_lease_fresh(
        self, db_session: Session
    ) -> None:
        # A dead PID no longer triggers reaping; only lease expiry does. A
        # worker that died less than a TTL ago keeps its claim until the lease
        # lapses — then the stale-heartbeat path reaps it.
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_dead_pid_fresh_lease")
        assert jid is not None
        repo.update(jid, worker_pid=2147483000)  # a PID that is not running
        db_session.flush()

        count = repo.reconcile_orphans("orphan")

        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.status == "pending"

    def test_reconcile_ignores_other_host_when_lease_fresh(
        self, db_session: Session
    ) -> None:
        # Host-scope removed: a fresh-lease row is never reaped regardless of
        # which host claimed it — two daemons on one DB no longer reap each
        # other's live jobs (single-daemon remains the design; this is safety).
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_other_host")
        assert jid is not None
        repo.update(jid, worker_host="OTHER_HOST_NAME")
        db_session.flush()

        count = repo.reconcile_orphans("orphan")

        assert count == 0
        row = repo.get(jid)
        assert row is not None
        assert row.status == "pending"

    def test_reconcile_marks_stale_heartbeat(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_stale_hb")
        assert jid is not None
        _make_stale(repo, jid, db_session)

        count = repo.reconcile_orphans("orphan")

        assert count == 1
        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"

    def test_reconcile_marks_stale_lease_on_any_host(self, db_session: Session) -> None:
        # Even a row from another host is reaped once its lease goes stale —
        # liveness is purely the heartbeat, independent of host/PID.
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_stale_other_host")
        assert jid is not None
        repo.update(jid, worker_host="OTHER_HOST_NAME")
        _make_stale(repo, jid, db_session)

        count = repo.reconcile_orphans("orphan")

        assert count == 1
        row = repo.get(jid)
        assert row is not None
        assert row.status == "failed"

    def test_claim_or_create_writes_pid_host_heartbeat(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_claim_writes")
        assert jid is not None
        db_session.flush()

        row = repo.get(jid)
        assert row is not None
        assert row.worker_pid == os.getpid()
        assert row.worker_host == socket.gethostname()
        assert row.last_heartbeat_at is not None

    def test_update_heartbeat_returns_false_for_failed_row(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_hb_failed")
        assert jid is not None
        repo.update(
            jid,
            status="failed",
            finished_at=datetime.now(timezone.utc),
        )
        db_session.flush()

        ok = repo.update_heartbeat(jid)
        assert ok is False

    def test_update_job_noop_when_reaper_has_failed_row(
        self,
        db_session: Session,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        jid = repo.claim_or_create("liveness_race")
        assert jid is not None
        repo.update(jid, status="running", current=10)
        _make_stale(repo, jid, db_session)

        reaped = repo.reconcile_orphans("orphan")
        assert reaped == 1

        rowcount = repo.update(jid, current=50)
        assert rowcount == 0

        row = repo.get(jid)
        assert row is not None
        assert row.current == 10
        assert row.status == "failed"
