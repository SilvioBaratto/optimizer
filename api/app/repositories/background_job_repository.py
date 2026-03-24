"""Repository for persistent background job operations."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, exists, func, insert, literal, select
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

        Uses a single INSERT ... SELECT WHERE NOT EXISTS (SQLAlchemy Core)
        to close the TOCTOU window.  Dialect-agnostic: works on both
        PostgreSQL (production) and SQLite (tests).

        Returns the new job's UUID on success, or ``None`` if a
        pending/running job already exists for *job_type*.
        """
        new_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        extra_value = initial_extra if initial_extra else None

        tbl = BackgroundJob.__table__

        conflict_exists = exists().where(
            BackgroundJob.job_type == job_type,
            BackgroundJob.status.in_(("pending", "running")),
        )

        source = select(
            literal(new_id, type_=tbl.c.id.type).label("id"),
            literal(job_type, type_=tbl.c.job_type.type).label("job_type"),
            literal("pending", type_=tbl.c.status.type).label("status"),
            literal(0, type_=tbl.c.current.type).label("current"),
            literal(0, type_=tbl.c.total.type).label("total"),
            literal(extra_value, type_=tbl.c.extra.type).label("extra"),
            literal(now, type_=tbl.c.started_at.type).label("started_at"),
            literal(now, type_=tbl.c.created_at.type).label("created_at"),
            literal(now, type_=tbl.c.updated_at.type).label("updated_at"),
        ).where(~conflict_exists)

        stmt = insert(tbl).from_select(
            ["id", "job_type", "status", "current", "total", "extra",
             "started_at", "created_at", "updated_at"],
            source,
        )

        result = self.session.execute(stmt)
        self.session.flush()
        if result.rowcount == 1:
            return new_id
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
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        stmt = (
            delete(BackgroundJob)
            .where(
                BackgroundJob.status.in_(("completed", "failed")),
                BackgroundJob.finished_at.is_not(None),
                BackgroundJob.finished_at < cutoff,
            )
        )
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount  # type: ignore[return-value]
