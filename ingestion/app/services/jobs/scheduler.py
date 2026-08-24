"""APScheduler wiring for the ingestion daemon.

Scheduled jobs:
  1. **daily_pipeline** (CronTrigger, default 07:00 UTC) — sequential data
     pipeline: ref-indices → yfinance → macro → news → summarize → calibrate.
  2. **midday_news** (CronTrigger, default 14:00 UTC) — news fetch + summarize
     only (catch afternoon market news).
  3. **universe_build** (CronTrigger, default Sunday 02:00 UTC) — Trading 212
     instrument-universe rebuild. Runs *before* ``weekly_refetch`` so the
     yfinance rebuild sees the fresh instrument set.
  4. **weekly_refetch** (CronTrigger, default Sunday 03:00 UTC) — full yfinance
     + macro rebuild.
  5. **fred_monthly** (CronTrigger, default 1st of month 08:00 UTC) — FRED
     economic data.
  6. **news_refresh** (IntervalTrigger, default every 30 min) — incremental
     news re-summarization for countries with new articles.
  7. **orphan_reaper** (IntervalTrigger) — fails jobs whose worker died without
     a terminal status, so a wedged run does not block its type forever.

All cron schedules are configurable via environment variables.
Jobs are persisted in PostgreSQL via ``SQLAlchemyJobStore`` so misfired runs
(e.g. the worker was down at 07:00) execute at next startup within the grace
window.

Every job body here is also reachable from the CLI (``python -m app.cli``) for
manual runs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.database import database_manager
from app.services._shared import make_progress
from app.services.jobs.background_job import (
    BackgroundJobService,
    JobAlreadyRunningError,
)
from app.services.macro.macro_calibration import run_bulk_calibrate
from app.services.macro.macro_news_summary import run_news_summarize
from app.services.macro.macro_regime_service import (
    run_bulk_fred_fetch,
    run_bulk_macro_fetch,
    run_macro_news_fetch,
)
from app.services.market_data.yfinance_data_service import run_bulk_yfinance_fetch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job-service instances (one per pipeline step)
# ---------------------------------------------------------------------------

_yfinance_jobs = BackgroundJobService(
    job_type="yfinance_fetch",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_macro_jobs = BackgroundJobService(
    job_type="macro_fetch",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_news_fetch_jobs = BackgroundJobService(
    job_type="macro_news_fetch",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_summarize_jobs = BackgroundJobService(
    job_type="news_summarize",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_calibrate_jobs = BackgroundJobService(
    job_type="macro_calibrate",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_fred_jobs = BackgroundJobService(
    job_type="fred_fetch",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_ref_index_jobs = BackgroundJobService(
    job_type="reference_index_seed",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)
_universe_jobs = BackgroundJobService(
    job_type="universe_build",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)


# ---------------------------------------------------------------------------
# Reference-index refresh helpers
# ---------------------------------------------------------------------------


def _resolve_benchmark_tickers() -> list[str]:
    """Return the configured benchmark tickers.

    Override with the ``BENCHMARK_TICKERS`` env var (comma-separated).
    """
    return sorted(set(settings.benchmark_tickers))


@contextmanager
def _heartbeat(
    job_svc: BackgroundJobService, job_id: str, label: str
) -> Iterator[None]:
    """Run the heartbeat companion for a job executed in the scheduler thread.

    Steps invoked here run synchronously rather than through
    ``start_background``, so no heartbeat thread exists for them. Only the
    heartbeat thread stamps ``last_heartbeat_at`` — ``update_job`` never does —
    so without this any step outliving the orphan-reaper timeout (300s) is
    falsely reaped mid-run and its row flips to ``failed`` while the work is
    still going. Both the multi-minute yfinance fetch and the reference-index
    seed exceed that.
    """
    cadence = job_svc._heartbeat_cadence
    stop = threading.Event()
    thread = threading.Thread(
        target=job_svc._run_heartbeat,
        args=(uuid.UUID(job_id), stop, cadence),
        daemon=True,
        name=f"hb:sched:{label}:{job_id[:8]}",
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2 * cadence)


def refresh_reference_indices(
    label: str,
    *,
    period: str = "5y",
    mode: str = "incremental",
    attempt: int = 0,
) -> bool:
    """Run ``seed_reference_indices`` as a tracked background job.

    Args:
        label: Log-prefix label (e.g. ``"daily_pipeline"``).
        period: yfinance lookback window (``"5y"`` for full rebuilds).
        mode: ``"incremental"`` (default) or ``"full"`` — currently the seeder
            always uses incremental fetch_and_store under the hood, but ``mode``
            is forwarded for future expansion / observability.

    Returns ``True`` on success, ``False`` on skip or failure.
    """
    from app.services.market_data.reference_index_seeder import seed_reference_indices
    from app.services.market_data.yfinance import get_yfinance_client

    tickers = _resolve_benchmark_tickers()
    if not tickers:
        logger.info(
            "%s: reference_index_refresh skipped — no benchmarks configured", label
        )
        return False

    try:
        job_id = _ref_index_jobs.create_job(current_ticker="", attempt=attempt)
    except JobAlreadyRunningError as exc:
        logger.warning(
            "%s: reference_index_refresh skipped — already running (job %s)",
            label,
            exc.existing_job_id,
        )
        return False

    logger.info(
        "%s: reference_index_refresh started (job %s, %d tickers, mode=%s)",
        label,
        job_id,
        len(tickers),
        mode,
    )
    _ref_index_jobs.update_job(job_id, status="running")

    on_progress = make_progress(job_id, _ref_index_jobs)
    try:
        with _heartbeat(_ref_index_jobs, job_id, "refidx"):
            seed_reference_indices(
                tickers,
                get_yfinance_client(),
                on_progress=on_progress,
            )
        job = _ref_index_jobs.get_job(job_id) or {}
        if job.get("status") != "completed":
            _ref_index_jobs.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        logger.info("%s: reference_index_refresh completed (period=%s)", label, period)
        return True
    except Exception:
        logger.exception("%s: reference_index_refresh raised", label)
        _ref_index_jobs.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return False


# ---------------------------------------------------------------------------
# Daily pipeline
# ---------------------------------------------------------------------------


def _run_step(
    label: str,
    job_svc: BackgroundJobService,
    fn,
    *args,
    attempt: int = 0,
) -> bool:
    """Run a single pipeline step synchronously.

    The service function ``fn`` must accept ``on_progress`` as a keyword
    argument.  Job lifecycle (create / running / failed) is managed here.
    If ``fn`` returns without raising and without calling
    ``on_progress(status='completed')``, the job is marked completed
    defensively so downstream steps are not silently skipped.

    ``attempt`` records the reclaim generation (R3/§5.3); it is 0 on the normal
    cron/CLI path and ``prev + 1`` when the orphan reaper re-dispatches.

    Returns ``True`` if the step completed, ``False`` on skip or failure.
    """
    try:
        job_id = job_svc.create_job(attempt=attempt)
    except JobAlreadyRunningError as exc:
        logger.warning(
            "%s: skipped — already running (job %s)",
            label,
            exc.existing_job_id,
        )
        return False

    logger.info("%s: started (job %s)", label, job_id)
    job_svc.update_job(job_id, status="running")

    on_progress = make_progress(job_id, job_svc)
    try:
        with _heartbeat(job_svc, job_id, label):
            fn(*args, on_progress=on_progress)
        job = job_svc.get_job(job_id) or {}
        if job.get("status") != "completed":
            logger.warning(
                "%s: returned without calling on_progress(status='completed')"
                " (status=%s) — marking completed defensively",
                label,
                job.get("status"),
            )
            job_svc.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        logger.info("%s: completed", label)
        return True
    except Exception:
        logger.exception("%s: raised an exception", label)
        job_svc.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return False


# ---------------------------------------------------------------------------
# Individual steps
#
# Each returns True only when the step completed. They are the unit both the
# scheduled pipelines below and the CLI (``python -m app.cli``) compose, so a
# manual run takes the identical job-slot / heartbeat / progress path.
# ---------------------------------------------------------------------------


def run_yfinance_step(
    *,
    mode: Literal["incremental", "full"] = "incremental",
    period: str = "5y",
    workers: int | None = None,
    attempt: int = 0,
) -> bool:
    """Fetch prices, fundamentals, holders, and news for every instrument."""
    from app.schemas.market_data.yfinance_data import YFinanceFetchRequest
    from app.services.market_data.yfinance import get_yfinance_client

    if workers is None:
        workers = settings.yfinance_fetch_workers
    workers = max(1, min(workers, 16))

    return _run_step(
        "yfinance",
        _yfinance_jobs,
        run_bulk_yfinance_fetch,
        YFinanceFetchRequest(mode=mode, period=period, workers=workers),
        get_yfinance_client(),
        attempt=attempt,
    )


def run_macro_step(*, attempt: int = 0) -> bool:
    """Scrape Il Sole 24 Ore + Trading Economics into bond yields / indicators."""
    from app.schemas.macro.macro_regime import MacroFetchRequest

    return _run_step(
        "macro",
        _macro_jobs,
        run_bulk_macro_fetch,
        MacroFetchRequest(),
        attempt=attempt,
    )


def run_fred_step(*, incremental: bool = True, attempt: int = 0) -> bool:
    """Fetch FRED economic series."""
    from app.schemas.macro.macro_regime import FredFetchRequest

    return _run_step(
        "fred",
        _fred_jobs,
        run_bulk_fred_fetch,
        FredFetchRequest(incremental=incremental),
        attempt=attempt,
    )


def run_news_step(*, attempt: int = 0) -> bool:
    """Fetch macro news articles into ``macro_news``."""
    from app.schemas.macro.macro_regime import MacroNewsFetchRequest

    return _run_step(
        "news",
        _news_fetch_jobs,
        run_macro_news_fetch,
        MacroNewsFetchRequest(),
        attempt=attempt,
    )


def run_summarize_step(*, force_refresh: bool = True, attempt: int = 0) -> bool:
    """LLM-summarize macro news per country."""
    from app.schemas.macro.macro_regime import MacroNewsSummarizeRequest

    return _run_step(
        "summarize",
        _summarize_jobs,
        run_news_summarize,
        MacroNewsSummarizeRequest(force_refresh=force_refresh),
        attempt=attempt,
    )


def run_calibrate_step(*, attempt: int = 0) -> bool:
    """LLM-calibrate the macro regime (delta / tau) per country."""
    from app.services.macro.scrapers import PORTFOLIO_COUNTRIES

    return _run_step(
        "calibrate",
        _calibrate_jobs,
        run_bulk_calibrate,
        list(PORTFOLIO_COUNTRIES),
        True,
        attempt=attempt,
    )


def run_universe_step(*, attempt: int = 0) -> bool:
    """Rebuild the Trading 212 instrument universe.

    A missing ``TRADING_212_API_KEY`` is a configuration state, not a failure:
    the step logs, returns ``False``, and never claims a job slot.
    """
    from app.schemas.universe.trading212 import UniverseBuildRequest
    from app.services.universe.universe_build_service import (
        Trading212NotConfiguredError,
        build_trading212_client,
    )
    from app.services.universe.universe_build_service import (
        run_universe_build as _build,
    )

    try:
        client = build_trading212_client()
    except Trading212NotConfiguredError as exc:
        logger.warning("universe: skipped — %s", exc)
        return False

    return _run_step(
        "universe",
        _universe_jobs,
        _build,
        UniverseBuildRequest(),
        client,
        attempt=attempt,
    )


# ---------------------------------------------------------------------------
# Daily pipeline
# ---------------------------------------------------------------------------


def run_daily_pipeline() -> None:
    """Sequential pipeline: ref-indices → yfinance → macro → news → summarize → calibrate.

    News, summarize, and calibrate form a dependency chain: each consumes what
    the previous one wrote, so a failure upstream skips the rest rather than
    summarizing stale articles or calibrating off a stale summary. Macro is
    independent of yfinance and always runs.
    """
    logger.info("daily_pipeline: starting")

    refresh_reference_indices("daily_pipeline", mode="incremental")

    yf_ok = run_yfinance_step(mode="incremental")

    run_macro_step()

    if not yf_ok:
        logger.warning("daily_pipeline: news skipped — yfinance did not complete")
    elif not run_news_step():
        logger.warning("daily_pipeline: summarize skipped — news did not complete")
    elif not run_summarize_step():
        logger.warning("daily_pipeline: calibrate skipped — summarize did not complete")
    else:
        run_calibrate_step()

    logger.info("daily_pipeline: finished")


# ---------------------------------------------------------------------------
# Midday news refresh (news + summarize only)
# ---------------------------------------------------------------------------


def run_midday_news_refresh() -> None:
    """News fetch + summarize — catches afternoon market news."""
    logger.info("midday_news: starting")

    if run_news_step():
        run_summarize_step()
    else:
        logger.warning("midday_news: summarize skipped — news did not complete")

    logger.info("midday_news: finished")


# ---------------------------------------------------------------------------
# Weekly full refetch (data rebuild)
# ---------------------------------------------------------------------------


def run_weekly_refetch() -> None:
    """Full yfinance + macro data rebuild (no news/summarize/calibrate)."""
    logger.info("weekly_refetch: starting")

    run_yfinance_step(mode="full", period="5y", workers=settings.yfinance_fetch_workers)
    run_macro_step()

    # Reference-index full 5y rebuild — keep benchmarks aligned with universe cadence
    refresh_reference_indices("weekly_refetch", period="5y", mode="full")

    logger.info("weekly_refetch: finished")


# ---------------------------------------------------------------------------
# Weekly universe build
# ---------------------------------------------------------------------------


def run_universe_build() -> None:
    """Weekly Trading 212 instrument-universe rebuild.

    Scheduled ahead of ``weekly_refetch`` so the yfinance rebuild fetches the
    fresh instrument set rather than last week's.
    """
    logger.info("universe_build: starting")
    run_universe_step()
    logger.info("universe_build: finished")


# ---------------------------------------------------------------------------
# FRED monthly
# ---------------------------------------------------------------------------


def run_fred_monthly() -> None:
    """Monthly FRED economic data fetch."""
    logger.info("fred_monthly: starting")
    run_fred_step(incremental=True)
    logger.info("fred_monthly: finished")


# ---------------------------------------------------------------------------
# News refresh (replaces MacroNewsSummaryScheduler)
# ---------------------------------------------------------------------------

_RETENTION_DAYS = 90


def run_news_refresh() -> None:
    """Incremental 30-minute news re-summarization.

    Stateless: derives "last run" from the most recent ``macro_news_summaries``
    row so the job is restart-safe without in-memory state.
    """
    from app.repositories.macro.macro_regime_repository import MacroRegimeRepository
    from app.services.macro.macro_news_summary import (
        _find_countries_with_new_articles,
        _is_morning_pipeline_complete,
        _summarize_country_safe,
    )

    now = datetime.now(timezone.utc)
    today = now.date()

    try:
        with database_manager.get_session() as session:
            repo = MacroRegimeRepository(session)

            # Guard: skip if morning pipeline has not run yet
            if not _is_morning_pipeline_complete(repo, today):
                logger.info(
                    "news_refresh: morning pipeline not yet complete for %s, skipping",
                    today,
                )
                return

            # Derive last-run from DB (restart-safe)
            last_refresh = _get_last_refresh_time(session)
            logger.info("news_refresh: checking articles since %s", last_refresh)

            countries = _find_countries_with_new_articles(repo, last_refresh)
            if not countries:
                logger.info("news_refresh: no new articles, skipping")
            else:
                logger.info(
                    "news_refresh: %d countries with new articles: %s",
                    len(countries),
                    countries,
                )
                for country in countries:
                    _summarize_country_safe(session, country)

            # Retention purge
            cutoff = today - timedelta(days=_RETENTION_DAYS)
            pruned = repo.delete_old_news_summaries(cutoff)
            if pruned:
                logger.info("news_refresh: pruned %d old summaries", pruned)
            session.commit()

    except Exception:
        logger.exception("news_refresh: tick failed (non-fatal)")


def _get_last_refresh_time(session) -> datetime:
    """Return the most recent ``macro_news_summaries.updated_at``, or fallback."""
    from sqlalchemy import text

    row = session.execute(
        text("SELECT MAX(updated_at) FROM macro_news_summaries")
    ).scalar_one_or_none()

    if row is None:
        return datetime.now(timezone.utc) - timedelta(
            minutes=settings.scheduler_news_refresh_interval_minutes,
        )
    return row


# job_type -> re-dispatch thunk for the RECLAIM strategy (R3/§5.3). Each re-runs
# the step's default (incremental / idempotent) variant, tagged with the retry
# attempt. A weekly-full orphan re-runs as incremental here; the next weekly cron
# reconciles, and idempotent upserts (§5.4) make the partial re-run safe.
_RECLAIM_DISPATCH: dict[str, Callable[[int], bool]] = {
    "yfinance_fetch": lambda a: run_yfinance_step(attempt=a),
    "macro_fetch": lambda a: run_macro_step(attempt=a),
    "fred_fetch": lambda a: run_fred_step(attempt=a),
    "macro_news_fetch": lambda a: run_news_step(attempt=a),
    "news_summarize": lambda a: run_summarize_step(attempt=a),
    "macro_calibrate": lambda a: run_calibrate_step(attempt=a),
    "universe_build": lambda a: run_universe_step(attempt=a),
    "reference_index_seed": lambda a: refresh_reference_indices(
        "orphan_reclaim", attempt=a
    ),
}


def _redispatch_reclaimed(reclaimed: list[dict[str, Any]]) -> None:
    """Re-run each reclaimed orphan's step in a daemon thread, capped by attempt.

    The cap (``SCHEDULER_ORPHAN_MAX_RECLAIM_ATTEMPTS``) stops a job whose worker
    keeps dying from being re-dispatched forever. Each re-dispatch runs off the
    reaper thread so a multi-minute fetch does not block the next reaper tick.
    """
    max_attempts = settings.scheduler_orphan_max_reclaim_attempts
    for entry in reclaimed:
        job_type = entry["job_type"]
        next_attempt = int(entry["attempt"]) + 1
        thunk = _RECLAIM_DISPATCH.get(job_type)
        if thunk is None:
            logger.warning(
                "orphan_reaper: no reclaim dispatcher for %r — left failed",
                job_type,
            )
            continue
        if next_attempt > max_attempts:
            logger.error(
                "orphan_reaper: %s hit max reclaim attempts (%d) — giving up",
                job_type,
                max_attempts,
            )
            continue
        logger.warning(
            "orphan_reaper: re-dispatching %s (reclaim attempt %d/%d)",
            job_type,
            next_attempt,
            max_attempts,
        )
        threading.Thread(
            target=thunk,
            args=(next_attempt,),
            daemon=True,
            name=f"reclaim:{job_type}",
        ).start()


def run_orphan_reaper() -> None:
    """Reap background jobs whose worker died without a terminal status.

    Orphan reconciliation otherwise runs only at process startup
    (``app.worker.main``), so a job that hangs while the process stays alive
    (dead daemon thread, wedged network socket) is never reaped and blocks
    future runs of its type via ``JobAlreadyRunningError``. This periodic tick
    closes that gap without requiring a restart.

    R3/§5.3 — behaviour depends on ``SCHEDULER_ORPHAN_STRATEGY``:
    - ``fail`` (default): mark stale orphans failed; the next cron re-runs them.
    - ``reclaim``: mark them failed *and* immediately re-dispatch the step
      (at-least-once self-healing), capped by attempt. Safe only because every
      fetch write is an idempotent upsert (§5.4) — that ordering is the C1 gate.
    """
    try:
        from app.repositories.jobs.background_job_repository import (
            BackgroundJobRepository,
        )

        timeout = settings.scheduler_orphan_heartbeat_timeout_seconds

        if settings.scheduler_orphan_strategy == "reclaim":
            with database_manager.get_session() as session:
                reclaimed = BackgroundJobRepository(session).reclaim_orphans(
                    "orphaned — reclaimed for retry by periodic reaper",
                    heartbeat_timeout_seconds=timeout,
                )
                session.commit()
            if reclaimed:
                logger.warning(
                    "orphan_reaper: reclaimed %d stale job(s)", len(reclaimed)
                )
                _redispatch_reclaimed(reclaimed)
        else:
            with database_manager.get_session() as session:
                n = BackgroundJobRepository(session).reconcile_orphans(
                    "orphaned — heartbeat stale, reaped by periodic reaper",
                    heartbeat_timeout_seconds=timeout,
                )
                session.commit()
            if n:
                logger.warning("orphan_reaper: reaped %d stale job(s)", n)
    except Exception:
        logger.exception("orphan_reaper: tick failed (non-fatal)")


# ---------------------------------------------------------------------------
# Schedule registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleDefinition:
    """A single scheduler entry — maps a job function to its trigger."""

    job_id: str
    name: str
    func: Callable[[], None]
    trigger: Any  # CronTrigger | IntervalTrigger
    misfire_grace_time: int
    coalesce: bool = True
    replace_existing: bool = True


def _build_schedule_registry() -> list[ScheduleDefinition]:
    """Build the full list of schedules from current settings."""
    grace = settings.scheduler_misfire_grace_time_seconds

    return [
        ScheduleDefinition(
            job_id="daily_pipeline",
            name="Daily data pipeline",
            func=run_daily_pipeline,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_daily_pipeline_cron,
                timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="midday_news",
            name="Midday news refresh",
            func=run_midday_news_refresh,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_midday_news_cron,
                timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="universe_build",
            name="Weekly Trading 212 universe rebuild",
            func=run_universe_build,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_universe_build_cron,
                timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="weekly_refetch",
            name="Weekly full data refetch",
            func=run_weekly_refetch,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_weekly_refetch_cron,
                timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="fred_monthly",
            name="Monthly FRED data fetch",
            func=run_fred_monthly,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_fred_monthly_cron,
                timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="news_refresh",
            name="Incremental news summary refresh",
            func=run_news_refresh,
            trigger=IntervalTrigger(
                minutes=settings.scheduler_news_refresh_interval_minutes,
            ),
            misfire_grace_time=grace // 4,
        ),
        ScheduleDefinition(
            job_id="orphan_reaper",
            name="Periodic orphan job reaper",
            func=run_orphan_reaper,
            trigger=IntervalTrigger(
                seconds=settings.scheduler_orphan_heartbeat_timeout_seconds,
            ),
            misfire_grace_time=grace // 4,
        ),
    ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_scheduler() -> BackgroundScheduler:
    """Create a fully configured APScheduler instance (not yet started).

    Must be called **after** ``init_db()`` so that ``database_manager.engine``
    is available.
    """
    engine = database_manager.engine
    if engine is None:
        raise RuntimeError(
            "create_scheduler() called before database_manager.initialize()"
        )

    jobstores = {
        "default": SQLAlchemyJobStore(engine=engine),
    }
    executors = {
        "default": ThreadPoolExecutor(max_workers=4),
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        timezone="UTC",
    )

    for entry in _build_schedule_registry():
        scheduler.add_job(
            entry.func,
            trigger=entry.trigger,
            id=entry.job_id,
            name=entry.name,
            replace_existing=entry.replace_existing,
            misfire_grace_time=entry.misfire_grace_time,
            coalesce=entry.coalesce,
        )

    return scheduler
