"""Fund daemon entrypoint: ``python -m fund.worker`` (Phase 9D, Task 7).

The wiring counterpart to ``fund.scheduler`` (which holds the testable steps).
This module builds and drives the long-lived process that owns the fund's
scheduled work — the weekly rebalance sweep, the 15-minute drift monitor, and the
orphan reaper — mirroring ``ingestion/app/worker.py`` with the fund deltas.

Boot sequence (``main``):

1. ``load_dotenv()`` **first**, before any import reads the environment.
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
   ``threading.Event``) → ``_shutdown.wait()``.
8. Drain: ``scheduler.pause()`` → ``scheduler.shutdown(wait=True)`` bounded by
   ``fund_shutdown_drain_timeout_seconds`` → ``pool.close()`` → ``close_db()``.

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

# Load environment variables FIRST, before any other import reads them.
from dotenv import load_dotenv

load_dotenv()

import logging
import signal
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import FrameType
from typing import TYPE_CHECKING

from fund.config import settings
from fund.database import close_db, database_manager, init_db
from fund.scheduler import SchedulerRuntime, configure_runtime, create_scheduler

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# A callable yielding a session context manager — ``database_manager.get_session``
# in production; a test injects one bound to an in-memory engine.
SessionFactory = Callable[[], AbstractContextManager["Session"]]

# Set by the SIGTERM/SIGINT handlers; ``main`` blocks on it until shutdown.
_shutdown = threading.Event()


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


def _install_signal_handlers() -> None:
    """Wire SIGTERM + SIGINT to set the shutdown event (graceful drain)."""

    def _handle(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %d — shutting down", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _drain_and_shutdown(scheduler: object, drain_timeout_seconds: float) -> bool:
    """Stop claiming new work, then drain in-flight jobs within a time bound.

    ``scheduler.pause()`` stops new triggers from firing (no new claims), then
    ``scheduler.shutdown(wait=True)`` blocks until the running step finishes. That
    call has no timeout of its own, so it runs in a helper thread joined with a
    deadline. Returns ``True`` if in-flight work drained cleanly, ``False`` if the
    deadline elapsed first (the caller still closes its resources — the abandoned
    run is safe to re-run: it only ever paused at the HITL gate, never committed).
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
        "Drain timeout (%ss) exceeded — forcing shutdown, in-flight work abandoned",
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
    _shutdown.wait()

    logger.info("Shutting down fund daemon...")
    try:
        _drain_and_shutdown(scheduler, settings.fund_shutdown_drain_timeout_seconds)
    finally:
        configure_runtime(None)
        runtime.persistence.pool.close()
        close_db()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
