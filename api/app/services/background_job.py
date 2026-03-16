"""PostgreSQL-backed background job service.

Provides thread-safe job lifecycle management (create, get, update) with
persistent storage in the ``background_jobs`` table.  Designed to be
instantiated once per job domain (``job_type``) at module level.

The public interface is backwards-compatible with the previous in-memory
implementation so that route files require minimal changes.
"""

import logging
import threading
import time
import uuid
from typing import Any

from app.config import settings
from app.repositories.background_job_repository import BackgroundJobRepository

logger = logging.getLogger(__name__)


class JobAlreadyRunningError(Exception):
    """Raised when a job of the same type is already pending or running."""

    def __init__(self, existing_job_id: str) -> None:
        self.existing_job_id = existing_job_id
        super().__init__(
            f"A job is already in progress (id={existing_job_id})"
        )


class BackgroundJobService:
    """DB-backed job store scoped to a single ``job_type``.

    Args:
        job_type: Domain identifier (e.g. ``"yfinance_fetch"``).
        session_factory: Callable returning a context-manager that yields
            a SQLAlchemy ``Session`` (typically ``database_manager.get_session``).
        ttl_seconds: Seconds after which completed/failed jobs are eligible
            for cleanup.  Defaults to 86400 (24 hours).
    """

    def __init__(
        self,
        job_type: str,
        session_factory: Any,
        ttl_seconds: int = 86400,
    ) -> None:
        self._job_type = job_type
        self._session_factory = session_factory
        self._ttl_seconds = ttl_seconds
        self._start_times: dict[str, float] = {}
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface (backwards-compatible)
    # ------------------------------------------------------------------

    def create_job(self, **initial_data: Any) -> str:
        """Atomically create a job if none is active for this job type.

        Returns the new job ID as a string.

        Raises:
            JobAlreadyRunningError: If a pending/running job already exists.
        """
        # Serialise any non-primitive initial_data values for JSONB
        extra = {
            k: self._jsonb_safe(v) for k, v in initial_data.items()
        } if initial_data else {}

        with self._session_factory() as session:
            repo = BackgroundJobRepository(session)
            # Periodic cleanup of old rows
            repo.cleanup_expired(self._ttl_seconds)
            new_id = repo.claim_or_create(self._job_type, **extra)
            if new_id is None:
                # A job is already running — find its id for the error
                _, existing_id = repo.is_any_running(self._job_type)
                session.commit()
                # Capture for raising outside session context
                _conflict_id = existing_id or "unknown"
            else:
                session.commit()
                job_id = str(new_id)
                self._emit_started(job_id)
                return job_id

        # Raise outside the session context manager to avoid noisy logging
        raise JobAlreadyRunningError(_conflict_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a dict representation of the job, or ``None``.

        The returned dict keeps the ``job_id`` key (string) for
        backwards compatibility with existing response schemas.
        """
        try:
            uid = uuid.UUID(job_id)
        except ValueError:
            return None

        with self._session_factory() as session:
            repo = BackgroundJobRepository(session)
            row = repo.get(uid)
            if row is None:
                return None
            return self._row_to_dict(row)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update an existing job.  No-op if the job does not exist."""
        try:
            uid = uuid.UUID(job_id)
        except ValueError:
            return

        self._emit_transition(job_id, kwargs)

        with self._session_factory() as session:
            repo = BackgroundJobRepository(session)
            repo.update(uid, **kwargs)
            session.commit()

    def is_any_running(self) -> tuple[bool, str | None]:
        """Check whether any job of this type is pending or running.

        Returns ``(True, job_id_str)`` or ``(False, None)``.
        """
        with self._session_factory() as session:
            repo = BackgroundJobRepository(session)
            return repo.is_any_running(self._job_type)

    def start_background(
        self,
        target: Any,
        args: tuple[Any, ...] = (),
    ) -> threading.Thread:
        """Launch *target* in a daemon thread and return the Thread."""
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_started(self, job_id: str) -> None:
        """Fire started counter and increment in-progress gauge."""
        if not settings.enable_metrics:
            return
        from app import metrics

        metrics.jobs_started_total.labels(domain=self._job_type).inc()
        metrics.jobs_in_progress.labels(domain=self._job_type).inc()
        with self._metrics_lock:
            self._start_times[job_id] = time.monotonic()

    def _emit_transition(self, job_id: str, kwargs: dict[str, Any]) -> None:
        """Fire completion/failure counters, duration histogram, gauge, and webhook on terminal status."""
        new_status = kwargs.get("status")
        if new_status not in ("running", "completed", "failed"):
            return

        if new_status == "running":
            if settings.enable_metrics:
                from app import metrics  # noqa: F811 — lazy import

                with self._metrics_lock:
                    self._start_times[job_id] = time.monotonic()
            return

        # Terminal status: completed or failed
        if settings.enable_metrics:
            from app import metrics

            with self._metrics_lock:
                start = self._start_times.pop(job_id, None)
            if start is not None:
                duration = time.monotonic() - start
                metrics.job_duration_seconds.labels(domain=self._job_type).observe(duration)
            metrics.jobs_in_progress.labels(domain=self._job_type).dec()
            if new_status == "completed":
                metrics.jobs_completed_total.labels(domain=self._job_type).inc()
            else:
                metrics.jobs_failed_total.labels(domain=self._job_type).inc()

        # Webhook notification (independent of metrics toggle)
        if new_status == "failed":
            webhook_url = settings.notification_webhook_url
            if webhook_url:
                from app.services.notifications import notify_failure

                error_text = str(kwargs.get("error") or "")
                notify_failure(webhook_url, self._job_type, job_id, error_text)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert a ``BackgroundJob`` ORM row to the legacy dict shape."""
        d: dict[str, Any] = {
            "job_id": str(row.id),
            "status": row.status,
            "current": row.current,
            "total": row.total,
            "errors": row.errors or [],
            "result": row.result,
            "error": row.error,
            "started_at": (
                row.started_at.isoformat() if row.started_at else None
            ),
            "finished_at": (
                row.finished_at.isoformat() if row.finished_at else None
            ),
        }
        # Merge extra fields into the top-level dict for compat
        if row.extra:
            d.update(row.extra)
        return d

    @staticmethod
    def _jsonb_safe(value: Any) -> Any:
        """Ensure a value is JSON-serialisable for JSONB storage."""
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            return value
        return str(value)
