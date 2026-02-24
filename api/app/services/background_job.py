"""Generic in-memory background job service.

Provides thread-safe job lifecycle management (create, get, update) with
TTL-based cleanup of completed/failed jobs. Designed to be instantiated
once per job domain and reused across requests.
"""

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any


class BackgroundJobService:
    """Thread-safe in-memory job store with TTL cleanup.

    Args:
        ttl_seconds: Seconds after which completed/failed jobs are evicted.
            Defaults to 3600 (1 hour).
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def create_job(self, **initial_data: Any) -> str:
        """Create a new job with status ``pending``.

        Any keyword arguments are merged into the job dict.

        Returns:
            The generated job ID (UUID4 string).
        """
        self._cleanup_expired()
        job_id = str(uuid.uuid4())
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "pending",
            "current": 0,
            "total": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "errors": [],
            "result": None,
            "error": None,
            **initial_data,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a shallow copy of the job dict, or ``None`` if not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Merge *kwargs* into the job dict (no-op if job not found)."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def is_any_running(self) -> tuple[bool, str | None]:
        """Check whether any job is pending or running.

        Returns:
            ``(True, job_id)`` if a running job exists, else ``(False, None)``.
        """
        with self._lock:
            for job in self._jobs.values():
                if job.get("status") in ("pending", "running"):
                    return True, job["job_id"]
        return False, None

    def start_background(
        self,
        target: Any,
        args: tuple[Any, ...] = (),
    ) -> threading.Thread:
        """Launch *target* in a daemon thread and return the Thread object."""
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_expired(self) -> None:
        """Remove completed/failed jobs older than *ttl_seconds*."""
        now = time.time()
        with self._lock:
            to_remove: list[str] = []
            for jid, job in self._jobs.items():
                if job.get("status") not in ("completed", "failed"):
                    continue
                finished = job.get("finished_at")
                if finished is None:
                    continue
                try:
                    finished_dt = datetime.fromisoformat(finished)
                    age = now - finished_dt.timestamp()
                    if age > self._ttl_seconds:
                        to_remove.append(jid)
                except (ValueError, TypeError):
                    pass
            for jid in to_remove:
                del self._jobs[jid]
