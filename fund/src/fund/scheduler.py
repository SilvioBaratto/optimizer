"""Fund daemon step functions + APScheduler assembly.

The testable daemon logic, split from the wiring (``fund.worker`` builds and drives
the scheduler; this module holds the plain step functions that are unit-tested
headless). Four scheduled steps — the weekly rebalance sweep, the
15-minute drift monitor, the daily NAV mark, and the orphan reaper — plus the
shared per-portfolio driver :func:`_drive_rebalance` and the
:func:`create_scheduler` factory. **None of these start a scheduler.**

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
import math
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
from pydantic import ValidationError

from fund import observe
from fund.config import FundConfig, load_project_env, settings
from fund.database import database_manager
from fund.schemas.mandate import PortfolioMandate

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from fund.audit.persistence import LangGraphPersistence
    from fund.performance import NavPoint

logger = logging.getLogger(__name__)

# A callable yielding a session context manager — ``database_manager.get_session``
# in production; a test injects one bound to an in-memory/temp engine.
SessionFactory = Callable[[], AbstractContextManager["Session"]]

# The scheduled fund jobs (job ids == APScheduler job ids).
JOB_REBALANCE_SWEEP = "fund_rebalance_sweep"
JOB_DRIFT_MONITOR = "fund_drift_monitor"
JOB_NAV_MARK = "fund_nav_mark"
JOB_ORPHAN_REAPER = "fund_orphan_reaper"


@dataclass
class SchedulerRuntime:
    """The chat model + fallback + persistence a run needs.

    The daemon builds these **once** for its lifetime and installs them
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

    The thread opens its **own** session per pulse (never sharing the step's working
    ``Session`` across threads), fires one immediate pulse (so a slot is renewed
    the instant it starts running, and a heartbeat is observable even for a fast
    step), then re-pulses every ``cadence`` seconds until the block exits. A pulse
    failure is logged and swallowed and the next tick retries on a fresh session —
    a heartbeat hiccup must neither crash the step nor stop renewing the lease.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    stop = threading.Event()

    def _pulse_once() -> None:
        try:
            with factory() as hb_session:
                FundJobRepository(hb_session).update_heartbeat(job_id)
        except Exception:
            # A session that failed mid-commit is unusable, so each tick opens a
            # new one rather than reusing it.
            logger.warning(
                "heartbeat[%s]: pulse failed; retrying next tick",
                job_id,
                exc_info=True,
            )

    def _pulse() -> None:
        _pulse_once()  # immediate pulse
        while not stop.wait(cadence):
            _pulse_once()

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


def _autonomy_blocks_round(session: Session, portfolio_id: uuid.UUID) -> bool:
    """Whether a FROZEN fund or portfolio should block a rebalance round.

    Reads the autonomy posture, lazy-importing ``AutonomyRepository`` from its
    **submodule** so a bare ``import fund.scheduler`` stays agent-stack-free
    (``fund.audit`` loads its langgraph bootstrap on demand, so the lazy import is
    defence in depth). Fund scope wins over portfolio scope — a FROZEN fund blocks every
    portfolio.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio whose round is being considered.

    Returns:
        ``True`` when the fund row or the portfolio's row is FROZEN, else ``False``.
    """
    from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

    autonomy = AutonomyRepository(session)
    if autonomy.fund_state().state == AutonomyState.FROZEN:
        return True
    return autonomy.portfolio_state(portfolio_id).state == AutonomyState.FROZEN


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

    * an existing paused run for the portfolio (avoid stacking a second run; one past
      the approval window is flagged, since only a reject can clear it),
    * a FROZEN fund or portfolio (autonomy halted — re-arm to resume; Q3),
    * no persisted mandate,
    * no active ``ConstraintSet`` (run the profiler first — ``run_fund`` would
      raise ``RuntimeError``; pre-resolving keeps the miss a clean skip).

    The agent stack is imported here (lazily) so ``import fund.scheduler`` stays
    agent-stack-free.
    """
    from fund.agents.graph import run_fund
    from fund.audit.mifid_repository import resolve_constraint_set
    from fund.audit.repository import AgentRunRepository, run_age
    from fund.schemas import ConstraintSetRef

    pid_str = str(portfolio_id)

    paused = AgentRunRepository(session).list_paused_runs(portfolio_id)
    if paused:
        window = dt.timedelta(days=settings.approval_max_age_days)
        for run in paused:
            if run_age(run) > window:
                # Approving it is refused, so it blocks every later round until an
                # adviser rejects it; nothing else would surface that.
                logger.warning(
                    "drive[%s]: skipped — paused run %s is stale (past the %d-day "
                    "approval window); reject it to unblock scheduled rebalances",
                    pid_str,
                    run.id,
                    settings.approval_max_age_days,
                )
        logger.info("drive[%s]: skipped — a paused run already awaits HITL", pid_str)
        return

    if _autonomy_blocks_round(session, portfolio_id):
        logger.info("drive[%s]: skipped — autonomy frozen; re-arm to resume", pid_str)
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
        session=session,
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
    if run.status == "paused":
        logger.info(
            "drive[%s]: run %s status=paused (at HITL gate; not resuming)",
            pid_str,
            run.run_id,
        )
    else:
        # A freeze that landed mid-round or a round-cap end finalises the run without
        # a gate, so claiming one would send the operator to an empty queue.
        logger.warning(
            "drive[%s]: run %s status=%s (no approval gate)",
            pid_str,
            run.run_id,
            run.status,
        )


# ---------------------------------------------------------------------------
# Step functions (each claims a slot + heartbeats; APScheduler calls them 0-arg)
# ---------------------------------------------------------------------------


def _resolved_asof(asof: dt.date | None) -> dt.date:
    """The caller's date, else today (UTC) — the upper bound a NAV mark runs to."""
    return asof if asof is not None else datetime.now(UTC).date()


def _decision_asof(session: Session, asof: dt.date | None) -> dt.date:
    """The decision bar for a rebalance run: the caller's date, else a fillable bar.

    Today's date is no decision bar — ``place_orders`` needs a stored close strictly
    after it, which only a later ingest provides — so an unset ``asof`` resolves to
    :func:`fund.observe.default_decision_bar` instead.
    """
    if asof is not None:
        return asof
    return observe.default_decision_bar(session, today=_resolved_asof(None))


def _runtime_parts(
    runtime: SchedulerRuntime | None,
) -> tuple[Any, Any | None, Any]:
    """Unpack ``(model, fallback, persistence)`` from a runtime, or ``(None,)*3``."""
    if runtime is None:
        return None, None, None
    return runtime.model, runtime.fallback, runtime.persistence


@dataclass(frozen=True)
class _PassOutcome:
    """How many portfolios a sweep/monitor pass processed, and which of them errored."""

    attempted: int
    failed: tuple[uuid.UUID, ...]

    @property
    def all_failed(self) -> bool:
        """Whether every processed portfolio errored (vacuously ``False`` if none)."""
        return self.attempted > 0 and len(self.failed) == self.attempted

    @property
    def error(self) -> str | None:
        """The slot's error text naming the errored portfolios, or ``None``."""
        if not self.failed:
            return None
        ids = ", ".join(str(pid) for pid in self.failed)
        return f"{len(self.failed)} of {self.attempted} portfolio(s) errored: {ids}"


def _close_slot(
    label: str, job_repo: Any, job_id: uuid.UUID, outcome: _PassOutcome
) -> bool:
    """Finalise a pass's slot: ``failed`` iff every portfolio errored, else completed.

    One portfolio erroring never aborts a pass, so a partial failure completes the
    slot but records which portfolios errored; a pass in which *nothing* succeeded is
    ``failed`` (and returns ``False``) so operators and the manual CLI exit code do
    not read an outage as success.
    """
    status = "failed" if outcome.all_failed else "completed"
    job_repo.mark_done(job_id, status=status, error=outcome.error)
    if outcome.error is not None:
        logger.warning("%s: %s — %s", label, status, outcome.error)
    else:
        logger.info("%s: %s", label, status)
    return status == "completed"


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
    ``True`` when the sweep completed, with the errored portfolios (if any) recorded
    on its slot; ``False`` if the slot was busy, the sweep failed at the
    infrastructure level, or every portfolio errored.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    resolved_runtime = _resolve_runtime(runtime)
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
                outcome = _sweep_cron_mandates(
                    session, resolved_runtime, _decision_asof(session, asof)
                )
        except Exception as exc:
            logger.exception("rebalance sweep: aborted")
            # Discard the half-finished work: mark_done commits the shared session.
            session.rollback()
            job_repo.mark_done(job_id, status="failed", error=str(exc))
            return False
        return _close_slot("rebalance sweep", job_repo, job_id, outcome)


def _active_mandates(
    session: Session,
) -> list[tuple[uuid.UUID, PortfolioMandate | None]]:
    """``(portfolio_id, mandate)`` for every active mandate, in deterministic order.

    Plain values rather than ORM rows: the per-portfolio commit/rollback expires every
    loaded instance, so reading a row afterwards would re-query (or fail on a row
    deleted meanwhile) in the middle of the loop.

    A row whose stored JSON no longer validates (hand-edited, or a schema change
    added a required field) yields ``None`` and is logged here, so it is one
    portfolio's failure rather than an exception that starves every portfolio sorted
    after it on every run.
    """
    from fund.audit.mandate_repository import MandateRepository

    rows: list[tuple[uuid.UUID, PortfolioMandate | None]] = []
    for row in MandateRepository(session).list_active():
        mandate: PortfolioMandate | None
        try:
            mandate = PortfolioMandate.from_stored(row.mandate)
        except ValidationError:
            logger.exception(
                "portfolio %s: stored mandate is invalid; skipping", row.portfolio_id
            )
            mandate = None
        rows.append((row.portfolio_id, mandate))
    return rows


def _sweep_cron_mandates(
    session: Session, runtime: SchedulerRuntime | None, asof: dt.date
) -> _PassOutcome:
    """Drive each active ``triggers.cron`` mandate; one error is skipped, not fatal.

    Each portfolio is its own transaction: committed as soon as it reaches its
    gate, rolled back if it fails. A paused run is therefore visible to the adviser
    immediately and survives a crash later in the sweep, and a failed portfolio
    can neither leave a pending orphan row nor poison the session for the rest.
    Returns which portfolios errored so the slot can report them.
    """
    model, fallback, persistence = _runtime_parts(runtime)
    attempted = 0
    failed: list[uuid.UUID] = []
    for pid, mandate in _active_mandates(session):
        if mandate is not None and not mandate.triggers.cron:
            continue
        attempted += 1
        if mandate is None:
            failed.append(pid)
            continue
        try:
            _drive_rebalance(
                pid,
                asof,
                session=session,
                persistence=persistence,
                model=model,
                fallback=fallback,
            )
            session.commit()
        except Exception:
            logger.exception("rebalance sweep: portfolio %s errored; skipping", pid)
            session.rollback()
            failed.append(pid)
    return _PassOutcome(attempted, tuple(failed))


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
    One portfolio erroring is logged, skipped and recorded on the slot; the call
    returns ``False`` only when every processed portfolio errored.
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    resolved_runtime = _resolve_runtime(runtime)
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
                outcome = _monitor_drift_mandates(
                    session, resolved_runtime, _decision_asof(session, asof), config
                )
        except Exception as exc:
            logger.exception("drift monitor: aborted")
            # Discard the half-finished work: mark_done commits the shared session.
            session.rollback()
            job_repo.mark_done(job_id, status="failed", error=str(exc))
            return False
        return _close_slot("drift monitor", job_repo, job_id, outcome)


def _monitor_drift_mandates(
    session: Session,
    runtime: SchedulerRuntime | None,
    asof: dt.date,
    config: FundConfig,
) -> _PassOutcome:
    """Check drift for each active ``triggers.drift`` mandate; one error is skipped.

    Transaction boundaries match :func:`_sweep_cron_mandates`: one commit per
    portfolio, a rollback for the one that errored, and the errored portfolios
    returned for the slot to report.
    """
    model, fallback, persistence = _runtime_parts(runtime)
    attempted = 0
    failed: list[uuid.UUID] = []
    for pid, mandate in _active_mandates(session):
        if mandate is not None and not mandate.triggers.drift:
            continue
        attempted += 1
        if mandate is None:
            failed.append(pid)
            continue
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
            session.commit()
        except Exception:
            logger.exception("drift monitor: portfolio %s errored; skipping", pid)
            session.rollback()
            failed.append(pid)
    return _PassOutcome(attempted, tuple(failed))


# Run outcomes that mean "a rebalance was attempted and nobody acted on it".
_DISCARDED_RUN_STATUSES = frozenset({"rejected", "incomplete", "blocked", "failed"})


def _in_discard_cooldown(
    session: Session, portfolio_id: uuid.UUID, *, cooldown_seconds: int
) -> bool:
    """Whether the portfolio's newest run was discarded less than the cooldown ago.

    Drift persists after a rejected or order-less round, so without this the
    monitor would re-drive the full LLM pipeline on every tick and override the
    adviser's rejection within one interval.
    """
    runs = observe.list_portfolio_runs(session, portfolio_id)
    if not runs or runs[0].status not in _DISCARDED_RUN_STATUSES:
        return False
    ended = runs[0].finished_at or runs[0].created_at
    if ended.tzinfo is None:
        # SQLite drops the offset; stored timestamps are UTC.
        ended = ended.replace(tzinfo=UTC)
    return datetime.now(UTC) - ended < dt.timedelta(seconds=cooldown_seconds)


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
    if not state.target_approved:
        logger.info(
            "drift monitor: portfolio %s has no approved target yet; "
            "drift does not apply",
            pid_str,
        )
        return

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

    if _in_discard_cooldown(
        session, portfolio_id, cooldown_seconds=config.fund_drift_cooldown_seconds
    ):
        logger.info(
            "drift monitor: portfolio %s breached but its last rebalance was "
            "discarded within the cooldown; skipping",
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


def run_nav_mark(
    *,
    session_factory: SessionFactory | None = None,
    asof: dt.date | None = None,
    config: FundConfig = settings,
    full: bool = False,
) -> bool:
    """Claim the NAV-mark slot and persist every active mandate's NAV series.

    Pure compute — no agent stack, no :class:`SchedulerRuntime`. For each active
    mandate it rebuilds the FX-normalized NAV series via
    :func:`fund.performance.build_nav_series` (which runs from the earliest fill, so
    the first run backfills) and upserts it into ``nav_history`` — idempotent on
    ``(portfolio_id, date)``, so a re-run leaves the same rows. One portfolio
    erroring is logged and rolled back, never aborting the sweep. Runs under the
    shared heartbeat lease. ``full=True`` re-marks every bar instead of the recent
    window — the explicit hook for a one-time repair (e.g. after a mandate benchmark
    change, which the incremental mark does not back-propagate). Returns ``True``
    when the slot completed (``False`` if the slot was busy or the sweep failed at
    the infrastructure level).
    """
    from fund.audit.fund_job_repository import FundJobRepository

    factory = _resolve_factory(session_factory)
    asof_date = _resolved_asof(asof)
    with factory() as session:
        job_repo = FundJobRepository(session)
        job_id = job_repo.claim_or_create(JOB_NAV_MARK)
        if job_id is None:
            logger.info("nav mark: slot busy; skipping")
            return False
        job_repo.mark_running(job_id)
        try:
            with _heartbeat(
                job_id,
                cadence=config.fund_heartbeat_cadence_seconds,
                session_factory=factory,
            ):
                _mark_all_navs(session, asof_date, full=full)
        except Exception as exc:
            logger.exception("nav mark: aborted")
            job_repo.mark_done(job_id, status="failed", error=str(exc))
            return False
        job_repo.mark_done(job_id, status="completed")
        logger.info("nav mark: completed")
        return True


# On an incremental run, re-mark this many calendar days before the last persisted
# NAV so a late price revision / backfill inside the window is picked up, while the
# bulk of the history is not rewritten every day.
_NAV_MARK_LOOKBACK_DAYS = 7

# ``nav_history.nav`` / ``benchmark`` are ``Numeric(20, 6)``: a recomputed value
# within half a unit of the sixth decimal is the stored one, not a change.
_NAV_ROW_ABS_TOL = 5e-7


def _nav_row_changed(
    point: NavPoint,
    stored: tuple[float, float | None, str | None] | None,
    benchmark_ticker: str | None,
) -> bool:
    """Whether a recomputed NAV point differs from its persisted values.

    A row whose level is unchanged but is tagged with another benchmark (or none) still
    counts as changed: the tag is what lets readers and :func:`_nav_points_to_persist`
    trust the level, so it has to be written.
    """
    if stored is None:
        return True
    nav, benchmark, stored_ticker = stored
    if not math.isclose(nav, point.nav, abs_tol=_NAV_ROW_ABS_TOL):
        return True
    if benchmark is None or point.benchmark is None:
        if benchmark is not point.benchmark:
            return True
    elif not math.isclose(benchmark, point.benchmark, abs_tol=_NAV_ROW_ABS_TOL):
        return True
    return point.benchmark is not None and stored_ticker != benchmark_ticker


# Persisted NAV is Numeric(20, 6); this only has to absorb that rounding.
_NAV_ANCHOR_REL_TOL = 1e-6


def _levels_match(recomputed: float, persisted: Any) -> bool:
    return math.isclose(
        recomputed,
        float(persisted),
        rel_tol=_NAV_ANCHOR_REL_TOL,
        abs_tol=_NAV_ANCHOR_REL_TOL,
    )


def _anchor_drifted(anchor: Any, points: list[Any]) -> bool:
    """Whether the freshly computed series disagrees with the persisted anchor row.

    Compares the NAV and, when both sides hold one, the benchmark level. The
    benchmark is a price level, so a distribution that restates the stored history
    changes it for every older bar while the incremental mark rewrites only the
    recent ones, and the seam would read as a day of benchmark return.

    A recomputed series that has no bar on the anchor's date cannot be compared and
    is not treated as drift: a stale row would otherwise force a full re-mark on every
    run, since the upsert never deletes it.
    """
    for point in points:
        if point.date != anchor.date:
            continue
        if not _levels_match(float(point.nav), anchor.nav):
            return True
        return (
            point.benchmark is not None
            and anchor.benchmark is not None
            and not _levels_match(float(point.benchmark), anchor.benchmark)
        )
    return False


def _nav_points_to_persist(
    session: Session,
    nav_repo: Any,
    portfolio_id: uuid.UUID,
    mandate: PortfolioMandate,
    asof: dt.date,
    *,
    full: bool,
) -> list[Any]:
    """The NAV points one mandate's mark should upsert.

    Incremental marking assumes every bar before the lookback window is already
    right. The newest persisted bar before the window is therefore recomputed too and
    checked against what is stored: when they disagree the stored history was written
    under another NAV convention (e.g. flat-notional marks from before rebalances were
    chain-linked), and the whole series is rebuilt rather than leaving a seam at the
    window boundary.

    The anchor check cannot tell which benchmark a level came from, so a mandate
    benchmark edit would go unnoticed and leave older rows on the previous
    benchmark's level, stitching two benchmarks into one series with a phantom
    return at the join. Each row therefore records which benchmark its level came
    from, and any persisted level that is not the mandate's current benchmark's
    (including an untagged legacy one) rebuilds the whole series.
    """
    from fund.performance import build_nav_series

    def _build(since: dt.date | None) -> list[Any]:
        return build_nav_series(
            portfolio_id,
            session=session,
            base_currency=mandate.base_currency,
            benchmark=mandate.benchmark,
            asof=asof,
            since=since,
            capital=float(mandate.capital),
        )

    last = None if full else nav_repo.last_nav_date(portfolio_id)
    if last is None:
        return _build(None)
    if nav_repo.has_foreign_benchmark(portfolio_id, mandate.benchmark):
        logger.info(
            "nav mark: portfolio %s holds benchmark levels that are not %s's; "
            "re-marking in full",
            portfolio_id,
            mandate.benchmark,
        )
        return _build(None)
    window_start = last - dt.timedelta(days=_NAV_MARK_LOOKBACK_DAYS)
    anchor = nav_repo.last_nav_before(portfolio_id, window_start)
    points = _build(window_start if anchor is None else anchor.date)
    if anchor is not None and _anchor_drifted(anchor, points):
        logger.warning(
            "nav mark: portfolio %s history before %s no longer matches the "
            "recomputed series; re-marking in full",
            portfolio_id,
            window_start,
        )
        return _build(None)
    return points


def _mark_all_navs(session: Session, asof: dt.date, *, full: bool = False) -> None:
    """Build + upsert each active mandate's NAV series; one error is skipped.

    Incremental after the first run: only bars from ``_NAV_MARK_LOOKBACK_DAYS``
    before the last persisted NAV are re-emitted, so the whole history is not
    rewritten daily — unless the persisted history no longer matches the recomputed
    series or holds a benchmark level that is not the mandate's current benchmark's
    (see :func:`_nav_points_to_persist`), or ``full`` is set. A mandate whose
    base currency has no FX coverage (e.g. a non-EUR mandate against the EUR-only
    feed) is skipped loudly, not mis-marked.
    """
    from fund.audit.nav_history_repository import NavHistoryRepository
    from fund.performance import FxCoverageError

    nav_repo = NavHistoryRepository(session)
    for pid, mandate in _active_mandates(session):
        if mandate is None:
            continue
        try:
            points = _nav_points_to_persist(
                session, nav_repo, pid, mandate, asof, full=full
            )
            persisted = {
                row.date: (
                    float(row.nav),
                    None if row.benchmark is None else float(row.benchmark),
                    row.benchmark_ticker,
                )
                for row in nav_repo.get_navs(pid)
            }
            changed = [
                point
                for point in points
                if _nav_row_changed(point, persisted.get(point.date), mandate.benchmark)
            ]
            if not changed:
                continue
            nav_repo.upsert_navs(
                [
                    {
                        "portfolio_id": pid,
                        "date": point.date,
                        "nav": point.nav,
                        "benchmark": point.benchmark,
                        "benchmark_ticker": mandate.benchmark,
                    }
                    for point in changed
                ]
            )
            session.commit()
        except FxCoverageError as exc:
            logger.error("nav mark: portfolio %s skipped — %s", pid, exc)
            session.rollback()
        except Exception:
            logger.exception("nav mark: portfolio %s errored; skipping", pid)
            session.rollback()


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(config: FundConfig = settings) -> BackgroundScheduler:
    """Assemble the fund ``BackgroundScheduler`` (not started).

    Must be called **after** ``database_manager.initialize()`` so the engine
    exists. Registers four jobs — the cron rebalance sweep, the interval drift
    monitor, the daily cron NAV mark, and the interval orphan reaper — on a
    **distinct** jobstore table (``fund_apscheduler_jobs``) with a single-worker
    executor (D1 sequential) and UTC timezone. Every job carries
    ``fund_misfire_grace_time_seconds`` so one queued behind a long step still runs.
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
    grace = config.fund_misfire_grace_time_seconds

    scheduler.add_job(
        run_rebalance_sweep,
        trigger=CronTrigger.from_crontab(config.fund_rebalance_cron, timezone="UTC"),
        id=JOB_REBALANCE_SWEEP,
        name="Weekly fund rebalance sweep",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=grace,
    )
    scheduler.add_job(
        run_drift_monitor,
        trigger=IntervalTrigger(seconds=config.fund_drift_interval_seconds),
        id=JOB_DRIFT_MONITOR,
        name="Portfolio drift monitor",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=grace,
    )
    scheduler.add_job(
        run_nav_mark,
        trigger=CronTrigger.from_crontab(config.fund_nav_mark_cron, timezone="UTC"),
        id=JOB_NAV_MARK,
        name="Daily NAV mark",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=grace,
    )
    scheduler.add_job(
        run_orphan_reaper,
        trigger=IntervalTrigger(seconds=config.fund_orphan_timeout_seconds),
        id=JOB_ORPHAN_REAPER,
        name="Fund orphan job reaper",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=grace,
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


def _run_once(
    step: str, *, full: bool = False
) -> bool:  # pragma: no cover - manual entrypoint wiring
    """Run one step manually.

    The NAV mark is pure compute (no model/persistence), so it runs without
    building a runtime; the rebalance/drift steps build a throwaway runtime and
    close its pool afterwards. ``full`` applies to the NAV mark only.
    """
    if step == "nav-mark":
        return run_nav_mark(full=full)
    runtime = _build_runtime()
    try:
        if step == "rebalance-sweep":
            return run_rebalance_sweep(runtime=runtime)
        if step == "drift-monitor":
            return run_drift_monitor(runtime=runtime)
        raise SystemExit(
            f"unknown step {step!r}; expected 'rebalance-sweep', 'drift-monitor', "
            "or 'nav-mark'"
        )
    finally:
        runtime.persistence.pool.close()


def main(
    argv: list[str] | None = None,
) -> int:  # pragma: no cover - manual entrypoint wiring
    """``python -m fund.scheduler {rebalance-sweep|drift-monitor|nav-mark} [--full]``.

    One manual run. ``--full`` makes ``nav-mark`` re-mark every bar rather than the
    recent window.
    """
    load_project_env()
    logging.basicConfig(level=logging.INFO)
    raw = argv if argv is not None else sys.argv[1:]
    full = "--full" in raw
    args = [a for a in raw if a != "--full"]
    if not args:
        raise SystemExit(
            "usage: python -m fund.scheduler "
            "{rebalance-sweep|drift-monitor|nav-mark} [--full]"
        )
    return 0 if _run_once(args[0], full=full) else 1


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())


__all__ = [
    "JOB_DRIFT_MONITOR",
    "JOB_NAV_MARK",
    "JOB_ORPHAN_REAPER",
    "JOB_REBALANCE_SWEEP",
    "SchedulerRuntime",
    "configure_runtime",
    "create_scheduler",
    "run_drift_monitor",
    "run_nav_mark",
    "run_orphan_reaper",
    "run_rebalance_sweep",
]
