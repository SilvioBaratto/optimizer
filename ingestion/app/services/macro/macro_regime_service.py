"""Service layer orchestrating macro regime data fetching and storage.

Bulk-ingestion error policy: the FRED refresh walks its series one by one. Each
per-series ``except`` logs and accumulates the failure into an ``errors`` list,
then continues — one failing series must not abort the whole refresh. That list
is returned to the caller, so every drop is surfaced, not silent.

Every FRED series write runs in its own SAVEPOINT. Postgres aborts the whole
transaction on a failed statement, so without that scope one bad write would make
the later writes fail and the caller's commit would silently discard the series
that had already succeeded. Counts are recorded only after the series' savepoint
is released, so they never claim rows that were rolled back.
"""

import datetime
import logging
from typing import Any

from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

from app.services._shared import ProgressCallback, _noop
from app.services.macro.scrapers.fred_scraper import FRED_SERIES, FredScraper

logger = logging.getLogger(__name__)

# Trailing window an incremental FRED fetch re-requests behind the latest stored
# date. FRED restates observations after the first print (GDP advance, second and
# third estimates plus annual revisions, monthly OECD CLI revisions, NBER
# recession dates assigned up to ~21 months late) and the upsert overwrites
# ``value`` on conflict, so re-pulling the window is what stores the restatement.
FRED_REVISION_LOOKBACK_DAYS = 730
_FRED_REVISION_LOOKBACK = datetime.timedelta(days=FRED_REVISION_LOOKBACK_DAYS)

# Errors quoted in a failed job's one-line ``error`` summary; the full list is
# persisted separately.
_ERROR_SUMMARY_LIMIT = 3


class MacroRegimeService:
    """Fetches FRED series and macro news and stores them via the repository.

    Args:
        repo: Repository for persisting FRED observations and macro news.
        fred_scraper: Override for the FRED scraper; constructed lazily from
            ``settings.fred_api_key`` when ``None``.
    """

    def __init__(
        self,
        repo: MacroRegimeRepository,
        fred_scraper: FredScraper | None = None,
    ):
        self.repo = repo
        self._fred_scraper = fred_scraper

    @property
    def fred_scraper(self) -> FredScraper | None:
        """Lazy-construct FredScraper from settings when first accessed."""
        if self._fred_scraper is None:
            from app.config import settings

            key = settings.fred_api_key
            if not key:
                logger.warning(
                    "FRED_API_KEY missing or empty (settings.fred_api_key=%r). "
                    "Check container environment, not just .env — "
                    "docker exec -e override with empty value will shadow .env.",
                    key,
                )
                return None
            self._fred_scraper = FredScraper(api_key=key)
        return self._fred_scraper

    def fetch_fred_series(
        self,
        series_ids: list[str] | None = None,
        incremental: bool = True,
        on_progress: ProgressCallback = _noop,
    ) -> dict[str, Any]:
        """Fetch FRED time-series observations and store in DB.

        Args:
            series_ids: Series to fetch. ``None`` means all ``FRED_SERIES`` keys.
            incremental: When ``True``, fetch from ``FRED_REVISION_LOOKBACK_DAYS``
                before the last stored date, so recent restatements are refreshed
                along with new observations.
            on_progress: Callback invoked once per series with ``current``,
                ``total``, and ``current_country`` (the FRED series ID).

        Returns:
            Dict with ``"counts"`` and ``"errors"`` keys.
        """
        scraper = self.fred_scraper
        if scraper is None:
            from app.config import settings

            return {
                "counts": {},
                "errors": [
                    f"FRED_API_KEY missing or empty "
                    f"(settings.fred_api_key={settings.fred_api_key!r}). "
                    f"Check container environment, not just .env — "
                    f"docker exec -e override with empty value will shadow .env."
                ],
            }

        ids = series_ids if series_ids is not None else list(FRED_SERIES.keys())
        total = len(ids)
        counts: dict[str, int] = {}
        errors: list[str] = []

        for idx, series_id in enumerate(ids, 1):
            on_progress(current=idx, total=total, current_country=series_id)
            try:
                start_date: str | None = None
                if incremental:
                    latest = self.repo.get_fred_latest_date(series_id)
                    if latest is not None:
                        start_date = (latest - _FRED_REVISION_LOOKBACK).isoformat()

                result = scraper.get_series_observations(
                    series_id, start_date=start_date
                )

                if result["status"] == "success":
                    observations = result.get("observations", [])
                    with self.repo.session.begin_nested():
                        n = self.repo.upsert_fred_observations(series_id, observations)
                    counts[series_id] = n
                    logger.info(
                        "FRED %s: upserted %d observations (start_date=%s)",
                        series_id,
                        n,
                        start_date,
                    )
                else:
                    counts[series_id] = 0
                    errors.append(f"{series_id}: {result.get('error', 'unknown')}")
                    logger.warning(
                        "FRED fetch failed for %s: %s",
                        series_id,
                        result.get("error"),
                    )

                # Per series, so a forced exit mid-sweep keeps the series already
                # fetched and the connection is not pinned for the whole run.
                self.repo.session.commit()

            except Exception as exc:
                self.repo.session.rollback()
                counts[series_id] = 0
                errors.append(f"{series_id}: {exc}")
                logger.error("FRED fetch exception for %s: %s", series_id, exc)

        return {"counts": counts, "errors": errors}

    def fetch_macro_news(
        self,
        max_articles: int = 100,
        fetch_full_content: bool = False,
        on_progress: ProgressCallback = _noop,
    ) -> dict[str, Any]:
        """Fetch macro-themed news from yfinance and store in DB.

        Args:
            max_articles: Maximum number of articles to fetch.
            fetch_full_content: Whether to scrape full article body in addition
                to the headline snippet. When ``True``, articles whose body
                could not be scraped are skipped; when ``False``, articles are
                stored with their snippet only and a body scraped by an earlier
                run for the same article is kept.
            on_progress: Receives a report per source fetched, so a long
                rate-limited run keeps its job lease.

        Returns:
            Dict with ``"count"`` and ``"errors"`` keys.
        """
        import uuid

        from app.services.market_data.yfinance import get_yfinance_client
        from app.services.market_data.yfinance.news.macro_news import MacroNewsFetcher

        errors: list[str] = []
        try:
            yf_client = get_yfinance_client()
            search_client = yf_client.search

            scraper = None
            if fetch_full_content:
                from app.services.market_data.yfinance.news.scraper import (
                    ArticleScraper,
                )

                scraper = ArticleScraper(delay=0.5)

            fetcher = MacroNewsFetcher(
                yf_client=yf_client,
                search_client=search_client,
                scraper=scraper,
            )
            articles = fetcher.fetch_all(
                max_articles=max_articles,
                fetch_full_content=fetch_full_content,
                on_progress=on_progress,
            )
        except Exception as exc:
            logger.error("MacroNewsFetcher.fetch_all failed: %s", exc)
            return {"count": 0, "errors": [str(exc)]}

        rows: list[dict[str, Any]] = []
        for article in articles:
            pub_time = None
            if article.get("publish_time"):
                import contextlib

                with contextlib.suppress(ValueError, TypeError):
                    pub_time = datetime.datetime.fromisoformat(
                        article["publish_time"],
                    )

            rows.append(
                {
                    "id": uuid.uuid4(),
                    "news_id": article["news_id"],
                    "title": article.get("title"),
                    "publisher": article.get("publisher"),
                    "link": article.get("link"),
                    "publish_time": pub_time,
                    "source_ticker": article.get("source_ticker"),
                    "source_query": article.get("source_query"),
                    "themes": article.get("themes"),
                    "snippet": article.get("snippet"),
                    "full_content": article.get("full_content"),
                }
            )

        # Drop articles with nothing to analyse. In scrape mode a failed body
        # scrape leaves only the headline, which is not worth storing; in
        # snippet-only mode the body is never fetched, so the snippet is the
        # content and requiring full_content would discard every article.
        body_field = "full_content" if fetch_full_content else "snippet"
        rows = [r for r in rows if r.get(body_field)]

        try:
            count = self.repo.upsert_macro_news(rows)
        except Exception as exc:
            logger.error("upsert_macro_news failed: %s", exc)
            return {"count": 0, "errors": [str(exc)]}

        return {"count": count, "errors": errors}


# ---------------------------------------------------------------------------
# Standalone bulk functions (called from the scheduler and CLI)
# ---------------------------------------------------------------------------


def _final_progress(errors: list[str], result: dict[str, Any]) -> dict[str, Any]:
    """Build the terminal ``on_progress`` kwargs for a bulk fetch.

    A partial failure (e.g. one FRED series failing) must not produce a
    ``completed`` status, because callers rely on the status to detect gaps in
    the dataset. ``error`` carries a one-line summary since the failure webhook
    reads it and would otherwise post a blank alert.
    """
    progress: dict[str, Any] = {
        "status": "failed" if errors else "completed",
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "errors": errors,
        "result": result,
    }
    if errors:
        shown = "; ".join(errors[:_ERROR_SUMMARY_LIMIT])
        hidden = len(errors) - _ERROR_SUMMARY_LIMIT
        progress["error"] = f"{shown} (+{hidden} more)" if hidden > 0 else shown
    return progress


def run_bulk_fred_fetch(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute bulk FRED time-series fetch.

    The job ends ``failed`` when any series errored; rows fetched before the
    failure are kept. A missing API key also fails here, but ``run_fred_step``
    skips an unconfigured install before it claims a job slot.

    Args:
        request: ``FredFetchRequest`` with ``series_ids`` and ``incremental``.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``series_fetched`` (series that returned rows),
        ``observations_upserted`` and ``error_count``.
    """
    from app.database import database_manager

    with database_manager.get_session() as session:
        repo = MacroRegimeRepository(session)
        service = MacroRegimeService(repo)

        result = service.fetch_fred_series(
            series_ids=request.series_ids,
            incremental=request.incremental,
            on_progress=on_progress,
        )
        session.commit()

        counts = result.get("counts", {})
        errors = result.get("errors", [])
        observations_upserted = sum(counts.values())
        result_dict = {
            "series_fetched": sum(1 for n in counts.values() if n),
            "observations_upserted": observations_upserted,
            "error_count": len(errors),
        }
        on_progress(**_final_progress(errors, result_dict))
        logger.info(
            "Bulk FRED fetch finished: %d observations across %d series",
            observations_upserted,
            len(counts),
        )

    return result_dict


def run_macro_news_fetch(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute macro news fetch in the service layer.

    The job ends ``failed`` when the fetch or the upsert reported an error.

    Args:
        request: ``MacroNewsFetchRequest`` with ``max_articles`` and ``fetch_full_content``.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``articles_stored`` and ``errors``.

    Raises:
        RuntimeError: When the fetch reported errors (upstream outage or a
            failed upsert), so the scheduler records the job as failed.
    """
    from app.database import database_manager

    with database_manager.get_session() as session:
        repo = MacroRegimeRepository(session)
        service = MacroRegimeService(repo)
        result = service.fetch_macro_news(
            max_articles=request.max_articles,
            fetch_full_content=request.fetch_full_content,
            on_progress=on_progress,
        )
        session.commit()

        errors = result.get("errors", [])
        result_dict = {"articles_stored": result.get("count", 0)}
        if errors:
            # Raise rather than only report status="failed": an exception reaches
            # every caller of this function and the step runner stores its text on
            # the job row, so a swallowed outage is visible in the job history.
            on_progress(errors=errors, result=result_dict)
            raise RuntimeError("Macro news fetch failed: " + "; ".join(errors))

        on_progress(
            status="completed",
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            errors=errors,
            result=result_dict,
        )
        logger.info("Macro news fetch completed: %d articles", result.get("count", 0))

    return result_dict
