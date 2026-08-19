"""Ingestion daemon entrypoint: ``python -m app.worker``.

Boots the process that owns the scheduled data pipeline:

1. Serve Prometheus metrics on ``settings.metrics_port`` (also the container
   healthcheck target).
2. ``init_db()`` — build the engine and session factory.
3. Reap orphan background jobs left behind by a crashed process, so a job that
   died mid-run does not block its type forever via ``JobAlreadyRunningError``.
4. Bootstrap reference-index benchmarks in a daemon thread — a cold database
   needs a multi-minute yfinance fetch, and blocking startup on it would delay
   the first scheduled run.
5. Start APScheduler and block until SIGTERM / SIGINT.

There is no HTTP API. Manual runs go through ``python -m app.cli``; job
progress is readable from the ``background_jobs`` table and the logs.
"""

from __future__ import annotations

# Load environment variables FIRST, before any other import reads them.
from dotenv import load_dotenv

load_dotenv()

import logging
import signal
import threading
from types import FrameType

from prometheus_client import start_http_server

from app.config import settings
from app.database import close_db, database_manager, init_db
from app.services.jobs.scheduler import create_scheduler

logger = logging.getLogger(__name__)

_shutdown = threading.Event()


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if settings.log_format == "text"
            else '{"timestamp": "%(asctime)s", "name": "%(name)s",'
            ' "level": "%(levelname)s", "message": "%(message)s"}'
        ),
    )


def _serve_metrics() -> None:
    """Expose Prometheus metrics. Doubles as the container healthcheck target."""
    if not settings.enable_metrics:
        logger.info("Prometheus metrics disabled (enable_metrics=False)")
        return

    # BackgroundJobService imports app.metrics lazily, so without this the
    # jobs_* families are absent from /metrics until the first job runs — a
    # scrape started before then sees no series at all, and rate() over a
    # freshly-appearing counter is undefined. Importing here registers them
    # against the default collector at zero.
    import app.metrics  # noqa: F401

    start_http_server(settings.metrics_port)
    logger.info("Prometheus metrics served on :%d/metrics", settings.metrics_port)


def _reconcile_orphans() -> None:
    """Fail jobs whose worker died without writing a terminal status.

    Non-fatal: a failure here must not stop the scheduler from starting.
    """
    from app.repositories.jobs.background_job_repository import BackgroundJobRepository

    try:
        with database_manager.get_session() as session:
            n = BackgroundJobRepository(session).reconcile_orphans(
                "orphaned at startup — process restarted",
                heartbeat_timeout_seconds=(
                    settings.scheduler_orphan_heartbeat_timeout_seconds
                ),
            )
            session.commit()
        logger.info("Reconciled %d orphan job(s) on startup", n)
    except Exception as exc:
        logger.warning("Orphan reconciliation failed: %s", exc)


def _bootstrap_benchmarks_async() -> None:
    """Seed missing/stale reference indices in a daemon thread."""

    def _run() -> None:
        try:
            from app.services._shared._benchmark_bootstrap import bootstrap_benchmarks
            from app.services.market_data.yfinance import get_yfinance_client

            bootstrap_benchmarks(
                database_manager.get_session,
                get_yfinance_client(),
                tickers=settings.benchmark_tickers,
                stale_days=settings.scheduler_benchmark_stale_days,
            )
        except Exception as exc:
            logger.warning("Benchmark bootstrap failed (continuing): %s", exc)

    threading.Thread(target=_run, daemon=True, name="benchmark-bootstrap").start()


def _install_signal_handlers() -> None:
    def _handle(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %d — shutting down", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main() -> None:
    """Run the ingestion daemon until a shutdown signal arrives."""
    _configure_logging()
    logger.info(
        "Starting %s (environment=%s)", settings.project_name, settings.environment
    )

    _serve_metrics()

    init_db()
    logger.info("Database initialized")
    if not database_manager.health_check():
        logger.warning("Database health check failed — continuing startup")

    _reconcile_orphans()
    _bootstrap_benchmarks_async()

    scheduler = create_scheduler()
    scheduler.start()
    logger.info(
        "APScheduler started — daily=%s, midday_news=%s, universe=%s, "
        "weekly=%s, fred=%s, news_refresh=%dmin",
        settings.scheduler_daily_pipeline_cron,
        settings.scheduler_midday_news_cron,
        settings.scheduler_universe_build_cron,
        settings.scheduler_weekly_refetch_cron,
        settings.scheduler_fred_monthly_cron,
        settings.scheduler_news_refresh_interval_minutes,
    )

    _install_signal_handlers()
    _shutdown.wait()

    logger.info("Shutting down %s...", settings.project_name)
    scheduler.shutdown(wait=False)
    close_db()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
