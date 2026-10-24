"""Fund daemon entrypoint: ``python -m fund.worker``.

The wiring counterpart to ``fund.scheduler`` (which holds the testable steps).
This module builds and drives the long-lived process that owns the fund's
scheduled work — the weekly rebalance sweep, the 15-minute drift monitor, and the
orphan reaper — mirroring ``ingestion/app/worker.py`` with the fund deltas.

Boot sequence (``main``):

1. ``load_project_env()`` **first**, before any import reads the environment.
2. ``_configure_logging()``.
3. ``init_db()`` + ``health_check()`` (a failed check warns but does not abort —
   the engine is lazy and a transient blip must not stop a deploy).
4. Build the lifetime runtime **once** — ``build_primary``/``build_fallback``
   (needs ``OLLAMA_API_KEY``) + one ``setup_langgraph`` pool held for the whole
   process (unlike the CLI's per-command open/close) — and install it via
   ``configure_runtime`` so the scheduler's zero-arg steps reuse it.
5. ``_reconcile_orphans()`` — a startup heartbeat-lease reap so a slot orphaned by
   a crashed predecessor does not block its type forever.
6. ``create_scheduler()`` → ``start()``.
7. ``_install_signal_handlers()`` (SIGTERM **and** SIGINT set a module
   ``threading.Event``) → ``_wait_for_shutdown()``, which beats the heartbeat file the
   container healthcheck reads and exits non-zero if APScheduler's thread has died.
8. Drain: ``scheduler.pause()`` → ``scheduler.shutdown(wait=True)`` bounded by
   ``fund_shutdown_drain_timeout_seconds``. A clean drain then runs ``pool.close()``
   → ``close_db()``. A timed-out drain leaves the pool and engine open (the step
   still owns them), fails the in-flight slots, and ends the process at once with
   ``os._exit`` — executor threads are non-daemon, so a normal return would block
   until the abandoned step finished.

There is no HTTP API and no Prometheus (out of scope for the fund daemon). Run
**exactly one** fund daemon per DB — the atomic ``claim_or_create`` slot + the
host-agnostic heartbeat lease assume a single writer.

Import discipline (SPEC §5, hygiene): ``fund.config``/``fund.database``/
``fund.scheduler`` (APScheduler + ``portopt_db``-adjacent, all agent-stack-free)
are top-level; the **agent stack** (model builders + ``setup_langgraph`` →
langgraph) and the ``fund.audit`` repositories are imported **inside** functions,
so a bare ``import fund.worker`` stays agent-stack-free and imports no
``ingestion.app``.
"""

from __future__ import annotations

# Load environment variables FIRST, before any other import reads them. An exported
# variable keeps precedence over the .env (see ``load_project_env``).
from fund.config import load_project_env

load_project_env()

import logging
import os
import signal
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import FrameType
from typing import TYPE_CHECKING

from fund.config import settings
from fund.database import close_db, database_manager, init_db
from fund.healthcheck import touch_heartbeat
from fund.scheduler import SchedulerRuntime, configure_runtime, create_scheduler

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# A callable yielding a session context manager — ``database_manager.get_session``
# in production; a test injects one bound to an in-memory engine.
SessionFactory = Callable[[], AbstractContextManager["Session"]]

# Set by the SIGTERM/SIGINT handlers; ``main`` blocks on it until shutdown.
_shutdown = threading.Event()

# Zero, like the ingestion daemon: the shutdown was *requested* (SIGTERM) and the
# abandoned step is safe to re-run, so a drain timeout is a degraded-but-expected
# outcome. A non-zero code would misclassify a normal deploy-during-sweep to
# exit-code alerting; the warning log and the failed slot carry the signal instead.
_FORCED_SHUTDOWN_EXIT_CODE = 0

# Non-zero, unlike a requested shutdown: a dead scheduler is a fault, and the exit
# status is what lets the restart policy bring the daemon back.
_SCHEDULER_DEAD_EXIT_CODE = 1

# How often the main thread checks the scheduler thread and beats the heartbeat.
_LIVENESS_POLL_SECONDS = 30.0


def _configure_logging() -> None:
    """Basic stdout logging at INFO (no JSON/env knobs — fund keeps it simple)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _build_runtime() -> SchedulerRuntime:  # pragma: no cover - needs OLLAMA + DB
    """Build the lifetime model + fallback + LangGraph pool the steps reuse.

    The agent stack is imported here (lazily) so ``import fund.worker`` stays
    agent-stack-free. ``build_primary``/``build_fallback`` require
    ``OLLAMA_API_KEY``; ``setup_langgraph`` requires ``DATABASE_URL`` — both are
    validated at build time. The caller owns the pool's lifetime (closed on drain).
    """
    from fund.agents.model import build_fallback, build_primary
    from fund.audit.persistence import setup_langgraph

    persistence = setup_langgraph(settings)
    model = build_primary(settings)
    fallback = build_fallback(settings)
    return SchedulerRuntime(model=model, fallback=fallback, persistence=persistence)


def _reconcile_orphans(*, session_factory: SessionFactory | None = None) -> None:
    """Fail fund slots whose worker died without writing a terminal status.

    A startup heartbeat-lease reap (NULL/stale ``last_heartbeat_at``, no host/PID
    clause), scoped to the fund job types. **Non-fatal**: a failure here logs and
    returns so it never stops the scheduler from starting. The repo flushes but
    does not commit — this caller owns the commit.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = session_factory or database_manager.get_session
    try:
        with factory() as session:
            n = FundJobRepository(session).reap_orphans(
                "orphaned at startup — fund daemon restarted",
                heartbeat_timeout_seconds=settings.fund_orphan_timeout_seconds,
            )
            session.commit()
        logger.info("Reconciled %d orphan fund job(s) on startup", n)
    except Exception as exc:
        logger.warning("Fund orphan reconciliation failed: %s", exc)


def _fail_in_flight_slots(*, session_factory: SessionFactory | None = None) -> None:
    """Fail every active fund slot so an abandoned step is not left ``running``.

    Run only after a timed-out drain, when the process is about to exit and the
    heartbeat thread dies with it. Without this the slot would stay ``running`` until
    its lease expired, blocking the next scheduled run of that type. Lease timeout 0
    matches every active row: with exactly one daemon per DB they are all this
    process's. Status-guarded transitions mean the abandoned step's own late
    ``mark_done`` can no longer flip the row back to ``completed``. **Non-fatal**: a
    DB failure logs and returns so the exit is never blocked.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = session_factory or database_manager.get_session
    try:
        with factory() as session:
            n = FundJobRepository(session).reap_orphans(
                "abandoned at shutdown — drain timeout exceeded",
                heartbeat_timeout_seconds=0,
            )
            session.commit()
        logger.warning("Failed %d in-flight fund job(s) abandoned at shutdown", n)
    except Exception as exc:
        logger.warning("Could not fail in-flight fund jobs at shutdown: %s", exc)


def _install_signal_handlers() -> None:
    """Wire SIGTERM + SIGINT to set the shutdown event (graceful drain)."""

    def _handle(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %d — shutting down", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _scheduler_alive(scheduler: object) -> bool:
    """Whether APScheduler's loop thread is still running.

    The loop calls the job store's ``update_job`` unguarded, so one store error
    kills the thread while ``scheduler.running`` stays True. The thread itself is
    the only reliable signal.
    """
    if not getattr(scheduler, "running", False):
        return False
    thread = getattr(scheduler, "_thread", None)
    return thread is None or thread.is_alive()


def _wait_for_shutdown(scheduler: object) -> bool:
    """Block until a shutdown signal, beating the heartbeat while the scheduler lives.

    Returns:
        ``True`` once shutdown was requested; ``False`` if the scheduler thread died
        first.
    """
    while not _shutdown.wait(_LIVENESS_POLL_SECONDS):
        if not _scheduler_alive(scheduler):
            return False
        touch_heartbeat()
    return True


def _drain_and_shutdown(scheduler: object, drain_timeout_seconds: float) -> bool:
    """Stop claiming new work, then drain in-flight jobs within a time bound.

    ``scheduler.pause()`` stops new triggers from firing (no new claims), then
    ``scheduler.shutdown(wait=True)`` blocks until the running step finishes. That
    call has no timeout of its own, so it runs in a helper thread joined with a
    deadline. Returns ``True`` if in-flight work drained cleanly, ``False`` if the
    deadline elapsed first. On ``False`` the step is still running and still using
    the pool and engine, so the caller must not close them; the abandoned run is
    safe to re-run (it only ever paused at the HITL gate, never approved).
    """
    try:
        scheduler.pause()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("scheduler.pause() failed during shutdown: %s", exc)

    drained = threading.Event()

    def _shutdown_scheduler() -> None:
        try:
            scheduler.shutdown(wait=True)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("scheduler.shutdown(wait=True) raised: %s", exc)
        finally:
            drained.set()

    threading.Thread(
        target=_shutdown_scheduler, daemon=True, name="fund-scheduler-drain"
    ).start()

    if drained.wait(timeout=drain_timeout_seconds):
        logger.info("In-flight fund jobs drained cleanly")
        return True

    logger.warning(
        "Drain timeout (%ss) exceeded — in-flight work will be abandoned",
        drain_timeout_seconds,
    )
    return False


def main() -> None:
    """Run the fund daemon until a shutdown signal arrives."""
    _configure_logging()
    logger.info("Starting fund daemon")

    init_db()
    logger.info("Database initialized")
    if not database_manager.health_check():
        logger.warning("Database health check failed — continuing startup")

    runtime = _build_runtime()
    configure_runtime(runtime)

    _reconcile_orphans()

    scheduler = create_scheduler()
    scheduler.start()
    logger.info(
        "APScheduler started — rebalance_cron=%s, drift_interval=%ds",
        settings.fund_rebalance_cron,
        settings.fund_drift_interval_seconds,
    )

    _install_signal_handlers()
    touch_heartbeat()
    if not _wait_for_shutdown(scheduler):
        logger.critical("APScheduler thread died; exiting so the container restarts")
        _fail_in_flight_slots()
        os._exit(_SCHEDULER_DEAD_EXIT_CODE)

    logger.info("Shutting down fund daemon...")
    drained = _drain_and_shutdown(
        scheduler, settings.fund_shutdown_drain_timeout_seconds
    )
    if drained:
        configure_runtime(None)
        runtime.persistence.pool.close()
        close_db()
        logger.info("Shutdown complete")
    else:
        _fail_in_flight_slots()
        logger.warning("Forced shutdown — in-flight work abandoned")
        os._exit(_FORCED_SHUTDOWN_EXIT_CODE)


if __name__ == "__main__":
    main()
