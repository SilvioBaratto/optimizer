"""Fund daemon step functions + APScheduler assembly (Phase 9C, Task 6).

The testable daemon logic, split from the wiring (Task 7's ``fund.worker`` builds
and drives the scheduler; this module holds the plain step functions that are
unit-tested headless). Three scheduled steps — the weekly rebalance sweep, the
15-minute drift monitor, and the orphan reaper — plus the shared per-portfolio
driver :func:`_drive_rebalance` and the :func:`create_scheduler` factory. **None
of these start a scheduler.**

Mirrors ``ingestion/app/services/jobs/scheduler.py`` with the fund deltas:

* **Sequential (D1).** ``ThreadPoolExecutor(max_workers=1)`` — one run at a time —
  vs ingestion's four.
* **Distinct slots.** The two fund ``job_type``s (:data:`FUND_JOB_TYPES`) never
  collide with ingestion's in the shared ``background_jobs`` table, and a
  **distinct** jobstore table (``fund_apscheduler_jobs``) keeps the persisted jobs
  apart. Atomic ``claim_or_create`` keeps ≤1 active row per type; run exactly one
  fund daemon per DB.
* **Heartbeat lease.** Every step runs under :func:`_heartbeat` — only that daemon
  thread stamps ``last_heartbeat_at``, so a multi-minute sweep is not false-reaped
  after ``fund_orphan_timeout_seconds`` (300s) while it is still alive.
* **Drift → gate, never approve (Q2).** A drift breach drives one rebalance run to
  its ``place_orders`` HITL pause and stops; a human commits via the Phase-8
  CLI/TUI. Portfolios already awaiting HITL are skipped to avoid stacking.
* **Cron weekday name.** ``fund_rebalance_cron`` defaults to ``sat`` — a bare ``0``
  would fire Monday under APScheduler ``from_crontab`` (0=Mon..6=Sun).

Import discipline (SPEC §5, hygiene): APScheduler + ``portopt_db``-adjacent
imports (config, database, the boundary-clean ``fund.audit`` repos, ``fund.observe``)
are top-level; the **agent stack** (``run_fund`` / model builders / deepagents /
langgraph) is imported **inside** functions so a bare ``import fund.scheduler``
stays agent-stack-free and imports no ``ingestion.app``.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import threading
import uuid
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from fund import observe
from fund.config import FundConfig, settings
from fund.database import database_manager
from fund.schemas.mandate import PortfolioMandate

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from fund.audit.persistence import LangGraphPersistence

logger = logging.getLogger(__name__)

# A callable yielding a session context manager — ``database_manager.get_session``
# in production; a test injects one bound to an in-memory/temp engine.
SessionFactory = Callable[[], AbstractContextManager["Session"]]

# The three scheduled fund jobs (job ids == APScheduler job ids).
JOB_REBALANCE_SWEEP = "fund_rebalance_sweep"
JOB_DRIFT_MONITOR = "fund_drift_monitor"
JOB_ORPHAN_REAPER = "fund_orphan_reaper"


@dataclass
class SchedulerRuntime:
    """The chat model + fallback + persistence a run needs.

    The daemon (Task 7) builds these **once** for its lifetime and installs them
    via :func:`configure_runtime`; the scheduled zero-arg steps then reuse them.
    A manual single run (``python -m fund.scheduler ...``) builds a throwaway one
    and closes its pool afterwards. Tests inject a scripted model + an in-memory
    ``LangGraphPersistence`` stand-in.
    """

    model: Any
    fallback: Any | None
    persistence: LangGraphPersistence


# Module-level runtime the worker installs once; the zero-arg scheduled steps read
# it. ``None`` until configured (a manual run/tests pass ``runtime=`` explicitly).
_runtime: SchedulerRuntime | None = None


def configure_runtime(runtime: SchedulerRuntime | None) -> None:
    """Install (or clear) the shared runtime the scheduled zero-arg steps read."""
    global _runtime
    _runtime = runtime


# ---------------------------------------------------------------------------
# Session + heartbeat plumbing
# ---------------------------------------------------------------------------


def _resolve_factory(session_factory: SessionFactory | None) -> SessionFactory:
    """The injected factory, or the module ``database_manager`` session opener."""
    return session_factory or database_manager.get_session


def _resolve_runtime(runtime: SchedulerRuntime | None) -> SchedulerRuntime | None:
    """The injected runtime, else the module-level one the worker installed."""
    return runtime if runtime is not None else _runtime


@contextmanager
def _heartbeat(
    job_id: uuid.UUID,
    *,
    cadence: float,
    session_factory: SessionFactory | None = None,
) -> Generator[None, None, None]:
    """Stamp ``last_heartbeat_at`` from a daemon thread for the wrapped work.

    Steps run synchronously in the scheduler thread, so no heartbeat thread
    exists for them otherwise. Only this thread stamps ``last_heartbeat_at``
    (``update_heartbeat`` is a no-op unless the row is ``running``), so without
    this a step outliving the orphan-reaper lease (``fund_orphan_timeout_seconds``,
    300s) is falsely reaped mid-run and its slot flips to ``failed`` while the
    work is still going.

    The thread opens its **own** session (never sharing the step's working
    ``Session`` across threads), fires one immediate pulse (so a slot is renewed
    the instant it starts running, and a heartbeat is observable even for a fast
    step), then re-pulses every ``cadence`` seconds until the block exits. A pulse
    failure is logged and swallowed — a heartbeat hiccup must never crash the step.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    stop = threading.Event()

    def _pulse() -> None:
        try:
            with factory() as hb_session:
                repo = FundJobRepository(hb_session)
                repo.update_heartbeat(job_id)  # immediate pulse
                while not stop.wait(cadence):
                    repo.update_heartbeat(job_id)
        except Exception:  # pragma: no cover - defensive; a hiccup must not crash
            logger.exception("heartbeat[%s]: pulse thread failed", job_id)

    thread = threading.Thread(
        target=_pulse, daemon=True, name=f"hb:fund:{str(job_id)[:8]}"
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(2.0, 2.0 * float(cadence)))


# ---------------------------------------------------------------------------
# Shared per-portfolio driver
# ---------------------------------------------------------------------------


def _drive_rebalance(
    portfolio_id: uuid.UUID,
    asof: dt.date,
    *,
    session: Session,
    persistence: Any,
    model: Any,
    fallback: Any | None,
) -> None:
    """Drive one paper rebalance for ``portfolio_id`` to the HITL gate — no resume.

    Resolves the portfolio's mandate and active ``ConstraintSet``, then calls
    :func:`run_fund` so the pipeline pauses at the ``place_orders`` gate; logs the
    run id + paused status and **returns without resuming** (a human commits via
    the Phase-8 CLI/TUI). Guards that never abort the sweep, each logged + skipped:

    * an existing paused run for the portfolio (avoid stacking a second run),
    * no persisted mandate,
    * no active ``ConstraintSet`` (run the profiler first — ``run_fund`` would
      raise ``RuntimeError``; pre-resolving keeps the miss a clean skip).

    The agent stack is imported here (lazily) so ``import fund.scheduler`` stays
    agent-stack-free.
    """
    from fund.agents.graph import run_fund
    from fund.audit.mifid_repository import resolve_constraint_set
    from fund.audit.repository import AgentRunRepository
    from fund.schemas import ConstraintSetRef

    pid_str = str(portfolio_id)

    if AgentRunRepository(session).list_paused_runs(portfolio_id):
        logger.info("drive[%s]: skipped — a paused run already awaits HITL", pid_str)
        return

    mandate = observe.get_mandate(session, portfolio_id)
    if mandate is None:
        logger.warning("drive[%s]: skipped — no mandate persisted", pid_str)
        return

    if persistence is None or persistence.store is None:
        logger.warning("drive[%s]: skipped — no persistence store available", pid_str)
        return
    constraint_set = resolve_constraint_set(
        persistence.store,
        ConstraintSetRef(
            portfolio_id=pid_str, store_key=settings.constraint_set_store_key
        ),
    )
    if constraint_set is None:
        logger.warning(
            "drive[%s]: skipped — no active ConstraintSet; run the profiler first",
            pid_str,
        )
        return

    run = run_fund(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof,
        session=session,
        checkpointer=persistence.saver,
        store=persistence.store,
        fallback=fallback,
    )
    logger.info(
        "drive[%s]: run %s status=%s (paused at HITL gate; not resuming)",
        pid_str,
        run.run_id,
        run.status,
    )


# ---------------------------------------------------------------------------
# Step functions (each claims a slot + heartbeats; APScheduler calls them 0-arg)
# ---------------------------------------------------------------------------


def _resolved_asof(asof: dt.date | None) -> dt.date:
    """The decision bar for a scheduled run: the caller's date, else today (UTC)."""
    return asof if asof is not None else datetime.now(UTC).date()


def _runtime_parts(
    runtime: SchedulerRuntime | None,
) -> tuple[Any, Any | None, Any]:
    """Unpack ``(model, fallback, persistence)`` from a runtime, or ``(None,)*3``."""
    if runtime is None:
        return None, None, None
    return runtime.model, runtime.fallback, runtime.persistence


def run_rebalance_sweep(
    *,
    session_factory: SessionFactory | None = None,
    runtime: SchedulerRuntime | None = None,
    asof: dt.date | None = None,
    config: FundConfig = settings,
) -> bool:
    """Claim the sweep slot and drive every ``triggers.cron`` mandate to its gate.

    Enumerates ``MandateRepository.list_active()`` in deterministic order,
    rehydrates each ``PortfolioMandate``, and for every one with ``triggers.cron``
    calls :func:`_drive_rebalance` **sequentially**. One portfolio erroring is
    logged and skipped — it never aborts the sweep. **Never approves.** Returns
    ``True`` when the sweep completed (``False`` if the slot was busy or the sweep
    failed at the infrastructure level).
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    resolved_runtime = _resolve_runtime(runtime)
    asof_date = _resolved_asof(asof)
    with factory() as session:
        job_repo = FundJobRepository(session)
        job_id = job_repo.claim_or_create(JOB_REBALANCE_SWEEP)
        if job_id is None:
            logger.info("rebalance sweep: slot busy; skipping")
            return False
        job_repo.mark_running(job_id)
        try:
            with _heartbeat(
                job_id,
                cadence=config.fund_heartbeat_cadence_seconds,
                session_factory=factory,
            ):
                _sweep_cron_mandates(session, resolved_runtime, asof_date)
        except Exception as exc:
            logger.exception("rebalance sweep: aborted")
            job_repo.mark_done(job_id, status="failed", error=str(exc))
            return False
        job_repo.mark_done(job_id, status="completed")
        logger.info("rebalance sweep: completed")
        return True


def _sweep_cron_mandates(
    session: Session, runtime: SchedulerRuntime | None, asof: dt.date
) -> None:
    """Drive each active ``triggers.cron`` mandate; one error is skipped, not fatal."""
    from fund.audit.mandate_repository import MandateRepository

    model, fallback, persistence = _runtime_parts(runtime)
    for row in MandateRepository(session).list_active():
        mandate = PortfolioMandate.model_validate(row.mandate)
        if not mandate.triggers.cron:
            continue
        pid = row.portfolio_id
        try:
            _drive_rebalance(
                pid,
                asof,
                session=session,
                persistence=persistence,
                model=model,
                fallback=fallback,
            )
        except Exception:
            logger.exception("rebalance sweep: portfolio %s errored; skipping", pid)


def run_drift_monitor(
    *,
    session_factory: SessionFactory | None = None,
    runtime: SchedulerRuntime | None = None,
    asof: dt.date | None = None,
    config: FundConfig = settings,
) -> bool:
    """Claim the drift slot; auto-enqueue a rebalance for each breached portfolio.

    For every active mandate with ``triggers.drift`` compares
    ``observe.portfolio_state(pid).drift_l1`` against
    ``mandate.drift_l1_threshold``. On a **breach** (and no paused run already),
    drives one rebalance to the HITL gate via :func:`_drive_rebalance` (Q2) —
    **never approves**. Also logs a re-profiling marker when
    ``observe.reprofile_status`` reports the portfolio due (flag-only, no gate).
    One portfolio erroring is logged and skipped.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    resolved_runtime = _resolve_runtime(runtime)
    asof_date = _resolved_asof(asof)
    with factory() as session:
        job_repo = FundJobRepository(session)
        job_id = job_repo.claim_or_create(JOB_DRIFT_MONITOR)
        if job_id is None:
            logger.info("drift monitor: slot busy; skipping")
            return False
        job_repo.mark_running(job_id)
        try:
            with _heartbeat(
                job_id,
                cadence=config.fund_heartbeat_cadence_seconds,
                session_factory=factory,
            ):
                _monitor_drift_mandates(session, resolved_runtime, asof_date, config)
        except Exception as exc:
            logger.exception("drift monitor: aborted")
            job_repo.mark_done(job_id, status="failed", error=str(exc))
            return False
        job_repo.mark_done(job_id, status="completed")
        logger.info("drift monitor: completed")
        return True


def _monitor_drift_mandates(
    session: Session,
    runtime: SchedulerRuntime | None,
    asof: dt.date,
    config: FundConfig,
) -> None:
    """Check drift for each active ``triggers.drift`` mandate; one error is skipped."""
    from fund.audit.mandate_repository import MandateRepository

    model, fallback, persistence = _runtime_parts(runtime)
    for row in MandateRepository(session).list_active():
        mandate = PortfolioMandate.model_validate(row.mandate)
        if not mandate.triggers.drift:
            continue
        pid = row.portfolio_id
        try:
            _check_drift(
                pid,
                mandate,
                asof,
                session=session,
                persistence=persistence,
                model=model,
                fallback=fallback,
                config=config,
            )
        except Exception:
            logger.exception("drift monitor: portfolio %s errored; skipping", pid)


def _check_drift(
    portfolio_id: uuid.UUID,
    mandate: PortfolioMandate,
    asof: dt.date,
    *,
    session: Session,
    persistence: Any,
    model: Any,
    fallback: Any | None,
    config: FundConfig,
) -> None:
    """Log a due re-profiling marker, then enqueue a run iff the drift band breached."""
    from fund.audit.repository import AgentRunRepository

    pid_str = str(portfolio_id)

    reprofile = observe.reprofile_status(
        session, portfolio_id, reprofile_interval_days=config.reprofile_interval_days
    )
    if reprofile.due:
        logger.warning(
            "drift monitor: portfolio %s due for re-profiling (%s); continuing on "
            "the existing ConstraintSet",
            pid_str,
            ",".join(reprofile.reasons) or "annual",
        )

    state = observe.portfolio_state(session, portfolio_id)
    if state.drift_l1 <= mandate.drift_l1_threshold:
        logger.info(
            "drift monitor: portfolio %s in band (%.6f <= %.6f)",
            pid_str,
            state.drift_l1,
            mandate.drift_l1_threshold,
        )
        return

    if AgentRunRepository(session).list_paused_runs(portfolio_id):
        logger.info(
            "drift monitor: portfolio %s breached but already awaits HITL; skipping",
            pid_str,
        )
        return

    logger.warning(
        "drift monitor: portfolio %s BREACH (%.6f > %.6f); enqueuing a rebalance",
        pid_str,
        state.drift_l1,
        mandate.drift_l1_threshold,
    )
    _drive_rebalance(
        portfolio_id,
        asof,
        session=session,
        persistence=persistence,
        model=model,
        fallback=fallback,
    )


def run_orphan_reaper(
    *,
    session_factory: SessionFactory | None = None,
    config: FundConfig = settings,
) -> bool:
    """Fail every lease-expired fund slot; commit. Non-fatal on error.

    A pure heartbeat lease (no host/PID clause), scoped to ``FUND_JOB_TYPES``.
    The reaper repository does not commit — this caller owns the commit.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    try:
        with factory() as session:
            reaped = FundJobRepository(session).reap_orphans(
                "orphaned — heartbeat lease expired, reaped by fund reaper",
                heartbeat_timeout_seconds=config.fund_orphan_timeout_seconds,
            )
            session.commit()
        if reaped:
            logger.warning("orphan reaper: reaped %d stale fund slot(s)", reaped)
        return True
    except Exception:
        logger.exception("orphan reaper: tick failed (non-fatal)")
        return False


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(config: FundConfig = settings) -> BackgroundScheduler:
    """Assemble the fund ``BackgroundScheduler`` (not started).

    Must be called **after** ``database_manager.initialize()`` so the engine
    exists. Registers exactly three jobs — the cron rebalance sweep, the interval
    drift monitor, and the interval orphan reaper — on a **distinct** jobstore
    table (``fund_apscheduler_jobs``) with a single-worker executor (D1
    sequential) and UTC timezone.
    """
    engine = database_manager.engine
    if engine is None:
        raise RuntimeError(
            "create_scheduler() called before database_manager.initialize()"
        )

    jobstores = {
        "default": SQLAlchemyJobStore(engine=engine, tablename="fund_apscheduler_jobs"),
    }
    executors = {
        "default": ThreadPoolExecutor(max_workers=1),
    }
    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        timezone="UTC",
    )

    scheduler.add_job(
        run_rebalance_sweep,
        trigger=CronTrigger.from_crontab(config.fund_rebalance_cron, timezone="UTC"),
        id=JOB_REBALANCE_SWEEP,
        name="Weekly fund rebalance sweep",
        replace_existing=True,
        coalesce=True,
    )
    scheduler.add_job(
        run_drift_monitor,
        trigger=IntervalTrigger(seconds=config.fund_drift_interval_seconds),
        id=JOB_DRIFT_MONITOR,
        name="Portfolio drift monitor",
        replace_existing=True,
        coalesce=True,
    )
    scheduler.add_job(
        run_orphan_reaper,
        trigger=IntervalTrigger(seconds=config.fund_orphan_timeout_seconds),
        id=JOB_ORPHAN_REAPER,
        name="Fund orphan job reaper",
        replace_existing=True,
        coalesce=True,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Manual single-run dispatch (CLI-parity — the same functions the scheduler calls)
# ---------------------------------------------------------------------------


def _build_runtime(
    config: FundConfig = settings,
) -> SchedulerRuntime:  # pragma: no cover - needs OLLAMA_API_KEY + DB
    """Build a throwaway runtime for a manual single run (caller closes the pool)."""
    from fund.agents.model import build_fallback, build_primary
    from fund.audit.persistence import setup_langgraph

    model = build_primary(config)
    fallback = build_fallback(config)
    persistence = setup_langgraph(config)
    return SchedulerRuntime(model=model, fallback=fallback, persistence=persistence)


def _run_once(step: str) -> bool:  # pragma: no cover - manual entrypoint wiring
    """Build a runtime, run one step to its gate, and close the pool."""
    runtime = _build_runtime()
    try:
        if step == "rebalance-sweep":
            return run_rebalance_sweep(runtime=runtime)
        if step == "drift-monitor":
            return run_drift_monitor(runtime=runtime)
        raise SystemExit(
            f"unknown step {step!r}; expected 'rebalance-sweep' or 'drift-monitor'"
        )
    finally:
        runtime.persistence.pool.close()


def main(
    argv: list[str] | None = None,
) -> int:  # pragma: no cover - manual entrypoint wiring
    """``python -m fund.scheduler {rebalance-sweep|drift-monitor}`` — one manual run."""
    from dotenv import load_dotenv

    # override=True so the project .env wins over ambient shell pollution (e.g. a
    # conda env exporting SSL_CERT_FILE without the corporate CA).
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(
            "usage: python -m fund.scheduler {rebalance-sweep|drift-monitor}"
        )
    return 0 if _run_once(args[0]) else 1


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())


__all__ = [
    "JOB_DRIFT_MONITOR",
    "JOB_ORPHAN_REAPER",
    "JOB_REBALANCE_SWEEP",
    "SchedulerRuntime",
    "configure_runtime",
    "create_scheduler",
    "run_drift_monitor",
    "run_orphan_reaper",
    "run_rebalance_sweep",
]
