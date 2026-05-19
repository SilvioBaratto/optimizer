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
from app.repositories.jobs.background_job_repository import BackgroundJobRepository

logger = logging.getLogger(__name__)


class JobAlreadyRunningError(Exception):
    """Raised when a job of the same type is already pending or running."""

    def __init__(self, existing_job_id: str) -> None:
        self.existing_job_id = existing_job_id
        super().__init__(f"A job is already in progress (id={existing_job_id})")


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
        heartbeat_cadence_seconds: float = 30.0,
    ) -> None:
        self._job_type = job_type
        self._session_factory = session_factory
        self._ttl_seconds = ttl_seconds
        self._heartbeat_cadence = heartbeat_cadence_seconds
        self._start_times: dict[str, float] = {}
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface (backwards-compatible)
    # ------------------------------------------------------------------

    def _execute_create_job(self, session: Any, extra: dict[str, Any]) -> None:
        """Execute job creation logic within a session context."""
        repo = BackgroundJobRepository(session)
        # Periodic cleanup of old rows
        repo.cleanup_expired(self._ttl_seconds)
        new_id = repo.claim_or_create(self._job_type, **extra)
        if new_id is None:
            # A job is already running — find its id for the error
            _, existing_id = repo.is_any_running(self._job_type)
            session.commit()
            # Capture for raising outside session context
            self._conflict_id = existing_id or "unknown"
        else:
            session.commit()
            self._job_id = str(new_id)
            self._emit_started(self._job_id)

    def _get_or_raise_conflict(self) -> str:
        """Return job ID or raise conflict error."""
        if hasattr(self, "_job_id") and self._job_id:
            return self._job_id
        if hasattr(self, "_conflict_id"):
            raise JobAlreadyRunningError(self._conflict_id)
        raise RuntimeError("Unexpected state in create_job")

    def _get_session_context_manager(self) -> Any:
        """Get a context manager that safely handles nested context managers.

        In tests, session_factory might return a context manager that yields
        another context manager. This helper handles that case.
        """
        from contextlib import contextmanager

        @contextmanager
        def safe_cm():
            session_or_cm = self._session_factory()

            # Check if it's a context manager
            if hasattr(session_or_cm, "__enter__") and not hasattr(
                session_or_cm, "execute"
            ):
                with session_or_cm as potential_session:
                    # Check if the yielded value is also a context manager
                    if hasattr(potential_session, "__enter__") and not hasattr(
                        potential_session, "execute"
                    ):
                        with potential_session as session:
                            yield session
                    else:
                        yield potential_session
            else:
                yield session_or_cm

        return safe_cm()

    def create_job(self, **initial_data: Any) -> str:
        """Atomically create a job if none is active for this job type.

        Returns the new job ID as a string.

        Raises:
            JobAlreadyRunningError: If a pending/running job already exists.
        """
        # Serialise any non-primitive initial_data values for JSONB
        extra = (
            {k: self._jsonb_safe(v) for k, v in initial_data.items()}
            if initial_data
            else {}
        )

        # Get session, handling test isolation issues where _session_factory
        # may return a context manager instead of a session due to conftest mocking
        session_or_cm = self._session_factory()

        # If we got a context manager instead of a session, enter it
        if hasattr(session_or_cm, "__enter__") and not hasattr(
            session_or_cm, "execute"
        ):
            with session_or_cm as potential_session:
                # Handle case where context manager yields another context manager
                if hasattr(potential_session, "__enter__") and not hasattr(
                    potential_session, "execute"
                ):
                    with potential_session as session:
                        self._execute_create_job(session, extra)
                        return self._get_or_raise_conflict()
                else:
                    # Got the actual session
                    session = potential_session
                    self._execute_create_job(session, extra)
                    return self._get_or_raise_conflict()
        else:
            # session_or_cm is already a session, not a context manager
            session = session_or_cm
            self._execute_create_job(session, extra)
            return self._get_or_raise_conflict()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a dict representation of the job, or ``None``.

        The returned dict keeps the ``job_id`` key (string) for
        backwards compatibility with existing response schemas.
        """
        try:
            uid = uuid.UUID(job_id)
        except ValueError:
            return None

        with self._get_session_context_manager() as session:
            repo = BackgroundJobRepository(session)
            row = repo.get(uid)
            if row is None:
                return None
            return self._row_to_dict(row)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update an existing job.  No-op if the job does not exist.

        Logs WARNING when the underlying status-guarded UPDATE matched no
        row and the requested status is not terminal — typically the
        worker-vs-reaper race closure (the row was already failed by the
        orphan reaper before this write landed).
        """
        try:
            uid = uuid.UUID(job_id)
        except ValueError:
            return

        self._emit_transition(job_id, kwargs)

        with self._get_session_context_manager() as session:
            repo = BackgroundJobRepository(session)
            rowcount = repo.update(uid, **kwargs)
            session.commit()

        if rowcount == 0 and kwargs.get("status") not in ("failed", "completed", None):
            logger.warning(
                "update_job no-op for %s: row reaped or absent (kwargs=%s)",
                job_id,
                kwargs,
            )

    def is_any_running(self) -> tuple[bool, str | None]:
        """Check whether any job of this type is pending or running.

        Returns ``(True, job_id_str)`` or ``(False, None)``.
        """
        with self._get_session_context_manager() as session:
            repo = BackgroundJobRepository(session)
            return repo.is_any_running(self._job_type)

    def start_background(
        self,
        target: Any,
        args: tuple[Any, ...] = (),
        cancel_event: threading.Event | None = None,
    ) -> tuple[threading.Thread, threading.Event]:
        """Launch *target* in a daemon thread with a heartbeat companion.

        Returns ``(worker_thread, cancel_event)``. The event is set by the
        heartbeat thread when the underlying row has been reaped, and by
        the worker wrapper on exit. Callers can pass an externally-built
        event so that an ``on_progress`` closure constructed via
        ``make_cancellable_progress`` shares the same fence token; if
        omitted, an internal event is created and returned.
        """
        job_id = self._extract_job_id(args)
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        cadence = self._heartbeat_cadence

        heartbeat = threading.Thread(
            target=self._run_heartbeat,
            args=(uuid.UUID(job_id), cancel_event, cadence),
            daemon=True,
            name=f"hb:{self._job_type}:{job_id[:8]}",
        )

        def _wrapped() -> None:
            try:
                target(*args, cancel_event=cancel_event)
            except Exception:
                logger.exception("background worker target raised")
            finally:
                cancel_event.set()
                heartbeat.join(timeout=2 * cadence)

        worker = threading.Thread(
            target=_wrapped,
            daemon=True,
            name=f"job:{self._job_type}:{job_id[:8]}",
        )
        heartbeat.start()
        worker.start()
        return worker, cancel_event

    @staticmethod
    def _extract_job_id(args: tuple[Any, ...]) -> str:
        """First positional arg is conventionally ``job_id: str``."""
        if not args or not isinstance(args[0], str):
            raise ValueError(
                "start_background requires args[0] to be a job_id string",
            )
        return args[0]

    def _run_heartbeat(
        self,
        job_id: uuid.UUID,
        stop_event: threading.Event,
        cadence: float,
    ) -> None:
        """Refresh ``last_heartbeat_at`` on each cadence tick.

        Stops when *stop_event* is set, when the row has been reaped
        (``update_heartbeat`` returns False) or when a DB error occurs.
        DB errors are logged at WARNING and swallowed so a transient DB
        hiccup does not kill the worker.
        """
        while not stop_event.wait(timeout=cadence):
            if not self._tick_heartbeat(job_id, stop_event):
                return

    def _tick_heartbeat(
        self,
        job_id: uuid.UUID,
        stop_event: threading.Event,
    ) -> bool:
        """Single heartbeat tick. Returns False if loop should exit."""
        try:
            with self._get_session_context_manager() as session:
                repo = BackgroundJobRepository(session)
                ok = repo.update_heartbeat(job_id)
                session.commit()
        except Exception:
            logger.warning("heartbeat update failed for %s", job_id, exc_info=True)
            return True
        if not ok:
            logger.warning("heartbeat reaped row %s — signalling cancel", job_id)
            stop_event.set()
            return False
        return True

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
                from app import metrics

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
                metrics.job_duration_seconds.labels(domain=self._job_type).observe(
                    duration
                )
            metrics.jobs_in_progress.labels(domain=self._job_type).dec()
            if new_status == "completed":
                metrics.jobs_completed_total.labels(domain=self._job_type).inc()
            else:
                metrics.jobs_failed_total.labels(domain=self._job_type).inc()

        # Webhook notification (independent of metrics toggle)
        if new_status == "failed":
            webhook_url = settings.notification_webhook_url
            if webhook_url:
                from app.services._shared import notify_failure

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
            "started_at": (row.started_at.isoformat() if row.started_at else None),
            "finished_at": (row.finished_at.isoformat() if row.finished_at else None),
        }
        # Merge extra fields into the top-level dict for compat
        if row.extra:
            d.update(row.extra)
        return d

    @staticmethod
    def _jsonb_safe(value: Any) -> Any:
        """Ensure a value is JSON-serialisable for JSONB storage."""
        if isinstance(value, str | int | float | bool | type(None) | list | dict):
            return value
        return str(value)
