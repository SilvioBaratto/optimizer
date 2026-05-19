"""In-process session store for the ``/portfolio-builder`` wizard.

Holds frozen ``_PipelineSession`` value objects keyed by session id.
Mutation flows through ``dataclasses.replace`` swapped under
``_sessions_lock``. CRUD lives here; the per-session
``BackgroundJobService`` factory and TTL eviction land in later issues
(#691-#693).

Lock-acquisition order rule
---------------------------
Always acquire ``_sessions_lock`` BEFORE ``_job_services_lock``. Never
the reverse — doing so would risk deadlock with ``delete_session`` and
the eviction sweep (issue #692), both of which follow the same order.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.database import database_manager
from app.services.jobs.background_job import BackgroundJobService

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CONCURRENT_SESSIONS",
    "SESSION_TTL_SECONDS",
    "STEP_IDS",
    "PipelineSessionCapacityError",
    "_PipelineSession",
    "create_session",
    "delete_session",
    "get_job_service",
    "get_session",
    "update_session",
]


STEP_IDS: tuple[str, ...] = (
    "load",
    "screen",
    "clean_returns",
    "build_history",
    "validate_is",
    "validate_oos",
    "coverage_gate",
    "regime",
    "optimize",
    "rebalance_decision",
    "cost",
    "report",
    "persist",
)

SESSION_TTL_SECONDS: int = 86400
MAX_CONCURRENT_SESSIONS: int = 3


class PipelineSessionCapacityError(Exception):
    """Raised when ``create_session`` would exceed ``MAX_CONCURRENT_SESSIONS``."""

    def __init__(self, current: int, cap: int) -> None:
        self.current = current
        self.cap = cap
        super().__init__(f"Pipeline session cap reached ({current}/{cap})")


@dataclass(frozen=True)
class _PipelineSession:
    """Immutable snapshot of one in-flight ``/portfolio-builder`` run.

    Frozen so concurrent readers see a consistent view; mutation goes
    through ``dataclasses.replace`` under ``_sessions_lock``.
    ``slots=True`` is intentionally NOT used: it would prevent
    ``replace`` from copying mutable container fields such as
    ``step_status``.
    """

    session_id: str
    created_at: datetime
    run_config: dict[str, Any]
    assembly: Any
    investable: Any
    clean_returns: Any
    factor_scores_dict: Any
    returns_history: Any
    is_result: Any
    oos_result: Any
    coverage_result: Any
    regime_result: Any
    optimize_result: Any
    rebalance_result: Any
    cost_result: Any
    report_result: Any
    current_step: str
    step_status: dict[str, str]
    step_results: dict[str, Any]


_sessions: dict[str, _PipelineSession] = {}
_job_services: dict[str, BackgroundJobService] = {}
_sessions_lock = threading.Lock()
_job_services_lock = threading.Lock()
_daemon_started: bool = False


_FIELD_NAMES: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(_PipelineSession)
)


def create_session(run_config: dict[str, Any]) -> str:
    """Insert a new session and return its hex id."""
    _ensure_daemon_running()
    sid = uuid.uuid4().hex
    session = _PipelineSession(
        session_id=sid,
        created_at=datetime.now(timezone.utc),
        run_config=run_config,
        assembly=None,
        investable=None,
        clean_returns=None,
        factor_scores_dict=None,
        returns_history=None,
        is_result=None,
        oos_result=None,
        coverage_result=None,
        regime_result=None,
        optimize_result=None,
        rebalance_result=None,
        cost_result=None,
        report_result=None,
        current_step="idle",
        step_status=dict.fromkeys(STEP_IDS, "pending"),
        step_results={},
    )
    with _sessions_lock:
        if len(_sessions) >= MAX_CONCURRENT_SESSIONS:
            raise PipelineSessionCapacityError(
                current=len(_sessions), cap=MAX_CONCURRENT_SESSIONS
            )
        _sessions[sid] = session
    return sid


def get_session(session_id: str) -> _PipelineSession | None:
    """Return the session snapshot, or ``None`` if not present."""
    with _sessions_lock:
        return _sessions.get(session_id)


def update_session(session_id: str, **fields: Any) -> None:
    """Atomically replace ``session_id`` with the patched dataclass."""
    unknown = set(fields) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"Unknown _PipelineSession field: {sorted(unknown)[0]}")
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            return
        _sessions[session_id] = dataclasses.replace(session, **fields)


def delete_session(session_id: str) -> None:
    """Remove the session and any cached job service. Idempotent."""
    with _sessions_lock:
        _sessions.pop(session_id, None)
    with _job_services_lock:
        _job_services.pop(session_id, None)


def get_job_service(session_id: str) -> BackgroundJobService:
    """Return the cached job service for ``session_id``, creating it on first call."""
    with _job_services_lock:
        svc = _job_services.get(session_id)
        if svc is None:
            svc = BackgroundJobService(
                job_type=f"pipeline_{session_id[:8]}",
                session_factory=database_manager.get_session,
                heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
            )
            _job_services[session_id] = svc
        return svc


def _evict_stale(
    now: datetime,
    snapshot: dict[str, datetime],
    ttl_seconds: int,
) -> list[str]:
    """Return ids whose age is ``>= ttl_seconds``. Pure, takes no lock."""
    return [
        sid
        for sid, created in snapshot.items()
        if (now - created).total_seconds() >= ttl_seconds
    ]


def _eviction_loop(interval_seconds: int = 60) -> None:
    """Sweep stale sessions every ``interval_seconds``. Runs forever."""
    while True:
        time.sleep(interval_seconds)
        try:
            with _sessions_lock:
                snapshot = {sid: s.created_at for sid, s in _sessions.items()}
            stale = _evict_stale(
                datetime.now(timezone.utc), snapshot, SESSION_TTL_SECONDS
            )
            for sid in stale:
                age = (datetime.now(timezone.utc) - snapshot[sid]).total_seconds()
                logger.info("Evicting stale pipeline session %s (age=%.1fs)", sid, age)
                delete_session(sid)
        except Exception:
            logger.warning("pipeline_builder eviction iteration failed", exc_info=True)


def _ensure_daemon_running() -> None:
    """Spawn the eviction daemon on first call. Subsequent calls no-op."""
    global _daemon_started
    with _sessions_lock:
        if _daemon_started:
            return
        thread = threading.Thread(
            target=_eviction_loop,
            daemon=True,
            name="pipeline-session-evictor",
        )
        _daemon_started = True
    thread.start()
    logger.info("Pipeline session evictor daemon started")
