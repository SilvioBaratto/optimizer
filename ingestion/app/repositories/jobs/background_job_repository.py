"""Repository for persistent background job operations."""

import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from sqlalchemy import (
    delete,
    exists,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from app.models.jobs.background_job import BackgroundJob, BackgroundJobError
from app.repositories._shared import RepositoryBase

_ACTIVE_STATUSES: tuple[str, ...] = ("pending", "running")
_TERMINAL_STATUSES: tuple[str, ...] = ("failed", "completed")
_DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 300

_CORE_COLUMNS: frozenset[str] = frozenset(
    {
        "status",
        "current",
        "total",
        "error",
        "finished_at",
        "result",
        "worker_pid",
        "worker_host",
        "last_heartbeat_at",
    }
)


class BackgroundJobRepository(RepositoryBase):
    """Database operations for the ``background_jobs`` table."""

    def _ensure_tables_exist(self) -> None:
        """Ensure background_jobs table exists (for test isolation)."""
        from app.models._shared import Base

        try:
            # Try to check if table exists
            inspector = __import__(
                "sqlalchemy.inspection", fromlist=["inspect"]
            ).inspect(self.session.bind)
            if "background_jobs" not in inspector.get_table_names():
                # Create tables if missing
                Base.metadata.create_all(bind=self.session.bind)  # type: ignore[arg-type]
        except Exception:
            # Inspector unavailable (e.g. dialect quirk): not fatal — fall back
            # to create-anyway. The terminal failure of that fallback is logged
            # at debug below, so the drop here is justified, not silent.
            try:
                from app.models._shared import Base

                Base.metadata.create_all(bind=self.session.bind)  # type: ignore[arg-type]
            except Exception as e:
                logger.debug("Could not ensure background_jobs table exists: %s", e)

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

    def update(self, job_id: uuid.UUID, **kwargs: Any) -> int:
        """Update columns on an existing job row, status-guarded.

        Core columns map directly to the table; non-core kwargs merge into
        the JSONB ``extra`` column. The UPDATE only fires when either the
        target row is still in (pending, running) OR the requested status
        is terminal (failed/completed) — this closes the worker-vs-reaper
        race so a reaped row's daemon worker cannot silently overwrite it.

        Returns the affected rowcount (0 or 1). The ``errors`` child-table
        replacement only runs when the main UPDATE matched a row.
        """
        errors_value = kwargs.pop("errors", None)
        new_status = kwargs.get("status")

        core, extras = self._partition_kwargs(kwargs)
        values = self._build_values(core, extras, job_id)

        rowcount = self._exec_status_guarded_update(job_id, new_status, values)

        if rowcount == 1 and errors_value is not None:
            self._replace_error_entries(job_id, errors_value)

        self.session.flush()
        return rowcount

    @staticmethod
    def _partition_kwargs(
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        core, extras = {}, {}
        for key, value in kwargs.items():
            if key in _CORE_COLUMNS:
                if key == "finished_at" and isinstance(value, str):
                    value = datetime.fromisoformat(value)
                core[key] = value
            else:
                extras[key] = value
        return core, extras

    def _build_values(
        self,
        core: dict[str, Any],
        extras: dict[str, Any],
        job_id: uuid.UUID,
    ) -> dict[str, Any]:
        values = dict(core)
        if extras:
            existing = self.session.execute(
                select(BackgroundJob.extra).where(BackgroundJob.id == job_id)
            ).scalar_one_or_none()
            merged = dict(existing) if existing else {}
            merged.update(extras)
            values["extra"] = merged
        return values

    def _exec_status_guarded_update(
        self,
        job_id: uuid.UUID,
        new_status: str | None,
        values: dict[str, Any],
    ) -> int:
        if not values:
            return 0
        clauses = [BackgroundJob.id == job_id]
        if new_status not in _TERMINAL_STATUSES:
            clauses.append(BackgroundJob.status.in_(_ACTIVE_STATUSES))
        stmt = update(BackgroundJob).where(*clauses).values(**values)
        result: CursorResult[Any] = self.session.execute(stmt)  # type: ignore[assignment]
        return result.rowcount or 0

    def _replace_error_entries(
        self,
        job_id: uuid.UUID,
        errors_value: Any,
    ) -> None:
        self.session.execute(
            delete(BackgroundJobError).where(
                BackgroundJobError.job_id == job_id,
            )
        )
        self.session.flush()
        if not isinstance(errors_value, list):
            return
        for idx, msg in enumerate(errors_value):
            self.session.add(
                BackgroundJobError(
                    job_id=job_id,
                    error_index=idx,
                    message=str(msg),
                )
            )

    def update_heartbeat(self, job_id: uuid.UUID) -> bool:
        """Stamp ``last_heartbeat_at = NOW()`` iff row is still running.

        Returns ``True`` when a running row was touched, ``False`` otherwise
        (already terminal, never started, or unknown ``job_id``).
        """
        stmt = (
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status == "running",
            )
            .values(last_heartbeat_at=func.now())
        )
        result: CursorResult[Any] = self.session.execute(stmt)  # type: ignore[assignment]
        self.session.flush()
        return (result.rowcount or 0) == 1

    def is_any_running(self, job_type: str) -> tuple[bool, str | None]:
        """Check if any job of *job_type* is pending or running."""
        stmt = (
            select(BackgroundJob.id)
            .where(
                BackgroundJob.job_type == job_type,
                BackgroundJob.status.in_(_ACTIVE_STATUSES),
            )
            .limit(1)
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        if row is not None:
            return True, str(row)
        return False, None

    def claim_or_create(
        self,
        job_type: str,
        attempt: int = 0,
        **initial_extra: Any,
    ) -> uuid.UUID | None:
        """Atomically create a job only if none is already active.

        Stamps ``worker_pid``, ``worker_host`` and ``last_heartbeat_at`` so
        the liveness-aware reaper can distinguish a live worker from a dead
        one. ``attempt`` records the reclaim-retry generation (R3/§5.3) — 0 for
        a normal job, ``prev + 1`` when the reaper re-dispatches an orphan.
        Dialect-agnostic: works on PostgreSQL and SQLite (tests).
        """
        self._ensure_tables_exist()
        new_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        pid = os.getpid()
        host = socket.gethostname()
        extra_value = initial_extra if initial_extra else None

        tbl = BackgroundJob.__table__  # type: ignore[attr-defined]

        conflict_exists = exists().where(
            BackgroundJob.job_type == job_type,
            BackgroundJob.status.in_(_ACTIVE_STATUSES),
        )

        source = select(
            literal(new_id, type_=tbl.c.id.type).label("id"),
            literal(job_type, type_=tbl.c.job_type.type).label("job_type"),
            literal("pending", type_=tbl.c.status.type).label("status"),
            literal(0, type_=tbl.c.current.type).label("current"),
            literal(0, type_=tbl.c.total.type).label("total"),
            literal(extra_value, type_=tbl.c.extra.type).label("extra"),
            literal(now, type_=tbl.c.started_at.type).label("started_at"),
            literal(pid, type_=tbl.c.worker_pid.type).label("worker_pid"),
            literal(host, type_=tbl.c.worker_host.type).label("worker_host"),
            literal(now, type_=tbl.c.last_heartbeat_at.type).label(
                "last_heartbeat_at",
            ),
            literal(attempt, type_=tbl.c.attempt.type).label("attempt"),
            literal(now, type_=tbl.c.created_at.type).label("created_at"),
            literal(now, type_=tbl.c.updated_at.type).label("updated_at"),
        ).where(~conflict_exists)

        stmt = insert(tbl).from_select(  # type: ignore[arg-type]
            [
                "id",
                "job_type",
                "status",
                "current",
                "total",
                "extra",
                "started_at",
                "worker_pid",
                "worker_host",
                "last_heartbeat_at",
                "attempt",
                "created_at",
                "updated_at",
            ],
            source,
        )

        result: CursorResult[Any] = self.session.execute(stmt)  # type: ignore[assignment]  # type: ignore[assignment]
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
        self,
        domain: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count jobs matching the given filters."""
        stmt = select(func.count()).select_from(BackgroundJob)
        stmt = self._apply_filters(stmt, domain, status)
        return self.session.execute(stmt).scalar_one()

    def get_latest_by_type(self, job_type: str) -> BackgroundJob | None:
        """Return the most-recently-finished job of *job_type*, or ``None``."""
        stmt = (
            select(BackgroundJob)
            .where(
                BackgroundJob.job_type == job_type,
                BackgroundJob.finished_at.is_not(None),
            )
            .order_by(BackgroundJob.finished_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Delete completed/failed jobs older than *ttl_seconds*."""
        self._ensure_tables_exist()

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        stmt = delete(BackgroundJob).where(
            BackgroundJob.status.in_(_TERMINAL_STATUSES),
            BackgroundJob.finished_at.is_not(None),
            BackgroundJob.finished_at < cutoff,
        )
        try:
            result: CursorResult[Any] = self.session.execute(stmt)  # type: ignore[assignment]  # type: ignore[assignment]
            self.session.flush()
            return result.rowcount or 0
        except Exception as e:
            # If the table doesn't exist, skip cleanup
            if "no such table" in str(e).lower():
                return 0
            raise

    def reap_orphans(
        self,
        error_msg: str,
        heartbeat_timeout_seconds: int = _DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Fail lease-expired pending/running rows; return one entry per row.

        T2.1 / ARCHITECTURE.md §5.3: liveness is a **heartbeat lease**, not a
        host/PID identity check. A worker renews ``last_heartbeat_at`` every
        ``SCHEDULER_HEARTBEAT_CADENCE_SECONDS`` (default 30s); a claim is reaped
        only once the lease TTL (``heartbeat_timeout_seconds``, a multiple of the
        cadence — default 300s = 10x) has elapsed with no renewal. It is the
        lease *renewal* that protects long synchronous steps (the multi-minute
        yfinance fetch, the reference-index seed) from being falsely reaped while
        alive. Host-scope and the Linux-``/proc`` PID probe were removed: they
        false-reaped across a container/host change and were inert off Linux
        (``worker_host`` / ``worker_pid`` are still recorded for observability).

        Returns ``{"job_type": ..., "attempt": ...}`` per reaped row so the
        caller can emit terminal metrics and (under RECLAIM) re-dispatch the
        step. The single source of the reap rule — :meth:`reconcile_orphans`
        wraps this for callers that need only a count.

        Note: SELECT then UPDATE re-apply the same predicate; a heartbeat landing
        between them could leave a returned row un-failed. The one-daemon-per-DB
        invariant makes that window negligible, and a spurious re-dispatch is
        safe because fetch writes are idempotent (§5.4). Caller owns the commit.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=heartbeat_timeout_seconds,
        )
        stale_active = (
            BackgroundJob.status.in_(_ACTIVE_STATUSES),
            or_(
                BackgroundJob.last_heartbeat_at.is_(None),
                BackgroundJob.last_heartbeat_at < cutoff,
            ),
        )
        rows = self.session.execute(
            select(BackgroundJob.job_type, BackgroundJob.attempt).where(*stale_active)
        ).all()
        reaped = [{"job_type": r.job_type, "attempt": r.attempt} for r in rows]
        if reaped:
            self.session.execute(
                update(BackgroundJob)
                .where(*stale_active)
                .values(
                    status="failed",
                    error=error_msg,
                    finished_at=func.now(),
                )
            )
            self.session.flush()
        return reaped

    def reconcile_orphans(
        self,
        error_msg: str,
        heartbeat_timeout_seconds: int = _DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    ) -> int:
        """Fail lease-expired orphans; return the count.

        Thin wrapper over :meth:`reap_orphans` for the startup path
        (``app.worker._reconcile_orphans``), which needs only a count and emits
        no metrics (this process's ``jobs_in_progress`` gauge is 0 at boot).
        """
        return len(self.reap_orphans(error_msg, heartbeat_timeout_seconds))
