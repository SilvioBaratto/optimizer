"""Unified APScheduler service replacing supercronic + MacroNewsSummaryScheduler.

Provides configurable scheduled jobs:
  1. **daily_pipeline** (CronTrigger, default 07:00 UTC) — sequential data
     pipeline: yfinance → macro → news → summarize → calibrate.
  2. **midday_news** (CronTrigger, default 14:00 UTC) — news fetch + summarize
     only (catch afternoon market news).
  3. **weekly_refetch** (CronTrigger, default Sunday 03:00 UTC) — full yfinance
     + macro rebuild.
  4. **fred_monthly** (CronTrigger, default 1st of month 08:00 UTC) — FRED
     economic data.
  5. **news_refresh** (IntervalTrigger, default every 30 min) — incremental
     news re-summarization for countries with new articles.

All cron schedules are configurable via environment variables.
Jobs are persisted in PostgreSQL via ``SQLAlchemyJobStore`` so misfired runs
(e.g. API was down at 07:00) execute at next startup within the grace window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.database import database_manager
from app.services._progress import make_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.macro_calibration import run_bulk_calibrate
from app.services.macro_news_summary import run_news_summarize
from app.services.macro_regime_service import (
    run_bulk_fred_fetch,
    run_bulk_macro_fetch,
    run_macro_news_fetch,
)
from app.services.yfinance_data_service import run_bulk_yfinance_fetch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job-service instances (one per pipeline step)
# ---------------------------------------------------------------------------

_yfinance_jobs = BackgroundJobService(
    job_type="yfinance_fetch",
    session_factory=database_manager.get_session,
)
_macro_jobs = BackgroundJobService(
    job_type="macro_fetch",
    session_factory=database_manager.get_session,
)
_news_fetch_jobs = BackgroundJobService(
    job_type="macro_news_fetch",
    session_factory=database_manager.get_session,
)
_summarize_jobs = BackgroundJobService(
    job_type="news_summarize",
    session_factory=database_manager.get_session,
)
_calibrate_jobs = BackgroundJobService(
    job_type="macro_calibrate",
    session_factory=database_manager.get_session,
)
_fred_jobs = BackgroundJobService(
    job_type="fred_fetch",
    session_factory=database_manager.get_session,
)


# ---------------------------------------------------------------------------
# Daily pipeline
# ---------------------------------------------------------------------------


def _run_step(label: str, job_svc: BackgroundJobService, fn, *args) -> bool:
    """Run a single pipeline step synchronously.

    The service function ``fn`` must accept ``on_progress`` as a keyword
    argument.  Job lifecycle (create / running / failed) is managed here;
    the service function reports completion via the ``on_progress`` callback.

    Returns ``True`` if the step completed, ``False`` on skip or failure.
    """
    try:
        job_id = job_svc.create_job()
    except JobAlreadyRunningError as exc:
        logger.warning(
            "daily_pipeline: %s skipped — already running (job %s)",
            label,
            exc.existing_job_id,
        )
        return False

    logger.info("daily_pipeline: %s started (job %s)", label, job_id)
    job_svc.update_job(job_id, status="running")

    on_progress = make_progress(job_id, job_svc)
    try:
        fn(*args, on_progress=on_progress)
        job = job_svc.get_job(job_id) or {}
        ok = job.get("status") == "completed"
        if ok:
            logger.info("daily_pipeline: %s completed", label)
        else:
            logger.warning(
                "daily_pipeline: %s finished with status=%s",
                label,
                job.get("status"),
            )
        return ok
    except Exception:
        logger.exception("daily_pipeline: %s raised an exception", label)
        job_svc.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return False


def run_daily_pipeline() -> None:
    """Sequential pipeline: yfinance → macro → news → summarize → calibrate."""
    logger.info("daily_pipeline: starting")

    from app.schemas.macro_regime import (
        MacroFetchRequest,
        MacroNewsFetchRequest,
        MacroNewsSummarizeRequest,
    )
    from app.schemas.yfinance_data import YFinanceFetchRequest
    from app.services.scrapers import PORTFOLIO_COUNTRIES
    from app.services.yfinance import get_yfinance_client

    # Step 1: yfinance
    yf_ok = _run_step(
        "yfinance",
        _yfinance_jobs,
        run_bulk_yfinance_fetch,
        YFinanceFetchRequest(mode="incremental"),
        get_yfinance_client(),
    )

    # Step 2: macro (independent — always runs)
    _run_step("macro", _macro_jobs, run_bulk_macro_fetch, MacroFetchRequest())

    # Step 3: news (gated on yfinance)
    if yf_ok:
        news_ok = _run_step(
            "news",
            _news_fetch_jobs,
            run_macro_news_fetch,
            MacroNewsFetchRequest(),
        )
    else:
        logger.warning("daily_pipeline: news skipped — yfinance did not complete")
        news_ok = False

    # Step 4: summarize (gated on news)
    if news_ok:
        summarize_ok = _run_step(
            "summarize",
            _summarize_jobs,
            run_news_summarize,
            MacroNewsSummarizeRequest(force_refresh=True),
        )
    else:
        logger.warning("daily_pipeline: summarize skipped — news did not complete")
        summarize_ok = False

    # Step 5: calibrate (gated on summarize)
    if summarize_ok:
        _run_step(
            "calibrate",
            _calibrate_jobs,
            run_bulk_calibrate,
            list(PORTFOLIO_COUNTRIES),
            True,
        )
    else:
        logger.warning("daily_pipeline: calibrate skipped — summarize did not complete")

    logger.info("daily_pipeline: finished")


# ---------------------------------------------------------------------------
# Midday news refresh (news + summarize only)
# ---------------------------------------------------------------------------


def run_midday_news_refresh() -> None:
    """News fetch + summarize — catches afternoon market news."""
    logger.info("midday_news: starting")

    from app.schemas.macro_regime import MacroNewsFetchRequest, MacroNewsSummarizeRequest

    news_ok = _run_step(
        "news",
        _news_fetch_jobs,
        run_macro_news_fetch,
        MacroNewsFetchRequest(),
    )

    if news_ok:
        _run_step(
            "summarize",
            _summarize_jobs,
            run_news_summarize,
            MacroNewsSummarizeRequest(force_refresh=True),
        )
    else:
        logger.warning("midday_news: summarize skipped — news did not complete")

    logger.info("midday_news: finished")


# ---------------------------------------------------------------------------
# Weekly full refetch (data rebuild)
# ---------------------------------------------------------------------------


def run_weekly_refetch() -> None:
    """Full yfinance + macro data rebuild (no news/summarize/calibrate)."""
    logger.info("weekly_refetch: starting")

    from app.schemas.macro_regime import MacroFetchRequest
    from app.schemas.yfinance_data import YFinanceFetchRequest
    from app.services.yfinance import get_yfinance_client

    # Full yfinance re-download
    _run_step(
        "yfinance",
        _yfinance_jobs,
        run_bulk_yfinance_fetch,
        YFinanceFetchRequest(mode="full", period="5y"),
        get_yfinance_client(),
    )

    # Full macro re-fetch (independent of yfinance outcome)
    _run_step("macro", _macro_jobs, run_bulk_macro_fetch, MacroFetchRequest())

    logger.info("weekly_refetch: finished")


# ---------------------------------------------------------------------------
# FRED monthly
# ---------------------------------------------------------------------------


def run_fred_monthly() -> None:
    """Monthly FRED economic data fetch."""
    logger.info("fred_monthly: starting")

    from app.schemas.macro_regime import FredFetchRequest

    _run_step(
        "fred",
        _fred_jobs,
        run_bulk_fred_fetch,
        FredFetchRequest(incremental=True),
    )

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
    from app.repositories.macro_regime_repository import MacroRegimeRepository
    from app.services.macro_news_summary import (
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
                settings.scheduler_daily_pipeline_cron, timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="midday_news",
            name="Midday news refresh",
            func=run_midday_news_refresh,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_midday_news_cron, timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="weekly_refetch",
            name="Weekly full data refetch",
            func=run_weekly_refetch,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_weekly_refetch_cron, timezone="UTC",
            ),
            misfire_grace_time=grace,
        ),
        ScheduleDefinition(
            job_id="fred_monthly",
            name="Monthly FRED data fetch",
            func=run_fred_monthly,
            trigger=CronTrigger.from_crontab(
                settings.scheduler_fred_monthly_cron, timezone="UTC",
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
