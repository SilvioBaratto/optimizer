"""Repository for persistent background job operations."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.background_job import BackgroundJob, BackgroundJobError
from app.repositories.base import RepositoryBase


class BackgroundJobRepository(RepositoryBase):
    """Database operations for the ``background_jobs`` table."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def create(self, job_type: str, **initial_extra: Any) -> BackgroundJob:
        """Insert a new job row and return the ORM instance."""
        row = BackgroundJob(
            id=uuid.uuid4(),
            job_type=job_type,
            status="pending",
            current=0,
            total=0,
            extra=initial_extra or None,
            started_at=datetime.now(timezone.utc),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get(self, job_id: uuid.UUID) -> BackgroundJob | None:
        """Return a single job by primary key."""
        stmt = select(BackgroundJob).where(BackgroundJob.id == job_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def update(self, job_id: uuid.UUID, **kwargs: Any) -> None:
        """Update columns on an existing job row.

        Core columns (status, current, total, error, finished_at, result)
        are mapped directly.  ``errors`` is handled via child table rows.
        Everything else is merged into the JSONB ``extra`` column.
        """
        row = self.get(job_id)
        if row is None:
            return

        _CORE = frozenset({
            "status", "current", "total", "error",
            "finished_at", "result",
        })

        errors_value = kwargs.pop("errors", None)

        extra_updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _CORE:
                if key == "finished_at" and isinstance(value, str):
                    value = datetime.fromisoformat(value)
                setattr(row, key, value)
            else:
                extra_updates[key] = value

        if extra_updates:
            merged = dict(row.extra) if row.extra else {}
            merged.update(extra_updates)
            row.extra = merged

        # Handle errors via child table
        if errors_value is not None:
            row.error_entries.clear()
            self.session.flush()
            if isinstance(errors_value, list):
                for idx, msg in enumerate(errors_value):
                    self.session.add(BackgroundJobError(
                        job_id=job_id, error_index=idx, message=str(msg),
                    ))

        self.session.flush()

    def is_any_running(self, job_type: str) -> tuple[bool, str | None]:
        """Check if any job of *job_type* is pending or running.

        Returns ``(True, job_id_str)`` or ``(False, None)``.
        """
        stmt = (
            select(BackgroundJob.id)
            .where(
                BackgroundJob.job_type == job_type,
                BackgroundJob.status.in_(("pending", "running")),
            )
            .limit(1)
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        if row is not None:
            return True, str(row)
        return False, None

    def claim_or_create(
        self, job_type: str, **initial_extra: Any,
    ) -> uuid.UUID | None:
        """Atomically create a job only if none is already active.

        Uses a single INSERT ... WHERE NOT EXISTS to close the TOCTOU
        window.  Returns the new job's UUID on success, or ``None`` if
        a pending/running job already exists for *job_type*.
        """
        new_id = uuid.uuid4()
        extra_str = json.dumps(initial_extra) if initial_extra else None

        result = self.session.execute(
            text("""
                INSERT INTO background_jobs
                    (id, job_type, status, current, total, extra, started_at,
                     created_at, updated_at)
                SELECT
                    :new_id, :job_type, 'pending', 0, 0,
                    CAST(:extra AS jsonb), NOW(),
                    NOW(), NOW()
                WHERE NOT EXISTS (
                    SELECT 1 FROM background_jobs
                    WHERE job_type = :job_type
                      AND status IN ('pending', 'running')
                )
                RETURNING id
            """),
            {
                "new_id": new_id,
                "job_type": job_type,
                "extra": extra_str,
            },
        )
        row = result.fetchone()
        self.session.flush()
        if row is not None:
            return row[0]
        return None

    @staticmethod
    def _apply_filters(stmt, domain: str | None, status: str | None):
        if domain is not None:
            stmt = stmt.where(BackgroundJob.job_type == domain)
        if status is not None:
            stmt = stmt.where(BackgroundJob.status == status)
        return stmt

    def list_jobs(
        self,
        domain: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[BackgroundJob]:
        """Return paginated job rows, most recent first."""
        stmt = select(BackgroundJob)
        stmt = self._apply_filters(stmt, domain, status)
        stmt = stmt.order_by(BackgroundJob.started_at.desc())
        stmt = stmt.offset(offset).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_jobs(
        self, domain: str | None = None, status: str | None = None,
    ) -> int:
        """Count jobs matching the given filters."""
        stmt = select(func.count()).select_from(BackgroundJob)
        stmt = self._apply_filters(stmt, domain, status)
        return self.session.execute(stmt).scalar_one()

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Delete completed/failed jobs older than *ttl_seconds*.

        Returns the number of rows deleted.
        """
        result = self.session.execute(
            text("""
                DELETE FROM background_jobs
                WHERE status IN ('completed', 'failed')
                  AND finished_at IS NOT NULL
                  AND finished_at < NOW() - INTERVAL '1 second' * :ttl
            """),
            {"ttl": ttl_seconds},
        )
        self.session.flush()
        return result.rowcount  # type: ignore[return-value]
