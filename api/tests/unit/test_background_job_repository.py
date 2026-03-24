"""Tests for BackgroundJobRepository — dialect-agnostic SQL (issue #314).

Covers claim_or_create atomicity and cleanup_expired cutoff arithmetic
on SQLite (the test dialect).  Regression guards against reintroduction
of PostgreSQL-only raw SQL in these two methods.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from app.repositories.background_job_repository import BackgroundJobRepository


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
        repo.update(job_id, status="completed",
                    finished_at=datetime.now(timezone.utc))
        db_session.flush()

        second = repo.claim_or_create("test_job_type_314_d")
        assert second is not None
        assert second != job_id

    def test_allows_new_job_after_failure(self, db_session: Session) -> None:
        repo = BackgroundJobRepository(db_session)
        job_id = repo.claim_or_create("test_job_type_314_e")
        assert job_id is not None
        repo.update(job_id, status="failed",
                    finished_at=datetime.now(timezone.utc))
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
