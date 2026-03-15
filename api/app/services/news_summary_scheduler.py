"""30-minute incremental news summary scheduler.

Runs as a daemon thread. On each tick:
  1. Checks that the morning pipeline has completed for today (skips if not).
  2. Finds countries with new MacroNews articles since the last run.
  3. Re-summarizes only those countries, isolating errors per-country.
  4. Logs skipped countries (no new articles), updated countries, and errors.
  5. Purges news summaries older than 90 days.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy.orm import Session

from app.database import database_manager
from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.services.macro_news_summary import (
    QUERY_COUNTRY_MAP,
    TICKER_COUNTRY_MAP,
    _GLOBAL,
    generate_country_summaries,
)

logger = logging.getLogger(__name__)

_RETENTION_DAYS = 90

CountryOutcome = Literal["updated", "skipped", "error"]


@dataclass
class TickResult:
    """Structured outcome of a single scheduler tick."""

    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)
    pruned: int = 0


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _find_countries_with_new_articles(
    repo: MacroRegimeRepository,
    since: datetime,
) -> list[str]:
    """Return country names that have at least one new article since *since*."""
    articles = repo.get_macro_news(start_date=since, limit=500)
    countries: set[str] = set()

    for article in articles:
        if article.source_ticker and article.source_ticker in TICKER_COUNTRY_MAP:
            country = TICKER_COUNTRY_MAP[article.source_ticker]
            if country != _GLOBAL:
                countries.add(country)
        elif article.source_query and article.source_query in QUERY_COUNTRY_MAP:
            for c in QUERY_COUNTRY_MAP[article.source_query]:
                if c != _GLOBAL:
                    countries.add(c)

    return sorted(countries)


def _is_morning_pipeline_complete(
    repo: MacroRegimeRepository,
    today: date,
) -> bool:
    """Return True if at least one MacroNewsSummary row exists for today."""
    summaries = repo.get_all_news_summaries(summary_date=today)
    return len(summaries) > 0


# ---------------------------------------------------------------------------
# Per-country isolation
# ---------------------------------------------------------------------------


def _summarize_country_safe(
    session: Session,
    country: str,
) -> CountryOutcome:
    """Summarize a single country with error isolation.

    Returns "updated", "skipped", or "error".
    """
    try:
        results = generate_country_summaries(
            session,
            force_refresh=True,
            countries=[country],
        )
        session.commit()
        return "updated" if results else "skipped"
    except Exception:
        logger.exception(
            "Scheduler: summarization failed for country=%s (non-fatal)", country,
        )
        try:
            session.rollback()
        except Exception:
            logger.exception("Scheduler: rollback failed for country=%s", country)
        return "error"


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------


class MacroNewsSummaryScheduler:
    """Daemon thread scheduler for incremental 30-minute news summary refresh."""

    def __init__(self, interval_seconds: int = 1800) -> None:
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run: datetime = datetime.now(timezone.utc) - timedelta(
            seconds=interval_seconds,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="news-summary-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "MacroNewsSummaryScheduler started (interval=%ds)", self._interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("MacroNewsSummaryScheduler stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._tick()
            except Exception:
                logger.exception("Scheduler tick failed (non-fatal)")

    def _tick(self) -> TickResult:
        tick_start = datetime.now(timezone.utc)
        today = tick_start.date()
        result = TickResult()

        logger.info(
            "Scheduler tick: checking for new articles since %s", self._last_run,
        )

        try:
            with database_manager.get_session() as session:
                repo = MacroRegimeRepository(session)

                # Guard: skip if morning pipeline has not run yet
                if not _is_morning_pipeline_complete(repo, today):
                    logger.info(
                        "Scheduler tick: morning pipeline not yet complete for %s, "
                        "skipping all countries",
                        today,
                    )
                    self._last_run = tick_start
                    return result

                # Incremental detection
                countries = _find_countries_with_new_articles(repo, self._last_run)
                if not countries:
                    logger.info(
                        "Scheduler tick: no new articles since %s, skipping",
                        self._last_run,
                    )
                else:
                    logger.info(
                        "Scheduler tick: %d countries with new articles: %s",
                        len(countries),
                        countries,
                    )

                    # Per-country isolation loop
                    for country in countries:
                        outcome = _summarize_country_safe(session, country)
                        if outcome == "updated":
                            result.updated.append(country)
                        elif outcome == "skipped":
                            result.skipped.append(country)
                        else:
                            result.errored.append(country)

                    logger.info(
                        "Scheduler tick: updated=%s skipped=%s errored=%s",
                        result.updated,
                        result.skipped,
                        result.errored,
                    )

                # Retention purge (always, separate commit)
                cutoff = today - timedelta(days=_RETENTION_DAYS)
                result.pruned = repo.delete_old_news_summaries(cutoff)
                if result.pruned:
                    logger.info(
                        "Scheduler tick: pruned %d old summaries", result.pruned,
                    )
                session.commit()

        except Exception:
            logger.exception("Scheduler tick DB error (non-fatal)")

        self._last_run = tick_start
        return result
