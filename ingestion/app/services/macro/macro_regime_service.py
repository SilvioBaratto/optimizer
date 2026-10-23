"""Service layer orchestrating macro regime data fetching and storage.

Bulk-ingestion error policy: macro data is aggregated per country
across multiple scrapers (IlSole, Trading Economics, FRED). Each per-country /
per-source ``except`` logs and accumulates the failure into an ``errors`` /
``all_errors`` list, then continues — one failing source or country must not
abort the whole macro refresh. Those lists are returned to the caller, so every
drop is surfaced, not silent.

Every independent write group (one source, one FRED series) runs in its own
SAVEPOINT. Postgres aborts the whole transaction on a failed statement, so
without that scope one bad write would make the later writes fail and the
caller's commit would silently discard the groups that had already succeeded.
Counts are recorded only after their group's savepoint is released, so they
never claim rows that were rolled back.
"""

import datetime
import logging
from typing import Any

from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository

from app.services._shared import ProgressCallback, _noop
from app.services.macro.scrapers.fred_scraper import FRED_SERIES, FredScraper
from app.services.macro.scrapers.ilsole_scraper import (
    PORTFOLIO_COUNTRIES,
    IlSoleScraper,
)
from app.services.macro.scrapers.tradingeconomics_scraper import (
    TradingEconomicsIndicatorsScraper,
)

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

# Numeric columns of an IlSole forecast row. ``reference_date`` and ``timestamp``
# are metadata and must not count as usable data.
_ILSOLE_VALUE_FIELDS = (
    "last_inflation",
    "inflation_6m",
    "inflation_10y_avg",
    "gdp_growth_6m",
    "earnings_12m",
    "eps_expected_12m",
    "peg_ratio",
    "lt_rate_forecast",
)


class MacroRegimeService:
    """Fetches macroeconomic data from scrapers and stores via repository.

    Args:
        repo: Repository for persisting macro indicators, observations, and
            FRED series.
        ilsole_scraper: Override for the IlSole24Ore scraper; defaults to a
            fresh ``IlSoleScraper()`` if omitted.
        te_scraper: Override for the Trading Economics scraper; defaults to a
            fresh ``TradingEconomicsIndicatorsScraper()``.
        fred_scraper: Override for the FRED scraper; constructed lazily from
            ``settings.fred_api_key`` when ``None``.
    """

    def __init__(
        self,
        repo: MacroRegimeRepository,
        ilsole_scraper: IlSoleScraper | None = None,
        te_scraper: TradingEconomicsIndicatorsScraper | None = None,
        fred_scraper: FredScraper | None = None,
    ):
        self.repo = repo
        self.ilsole_scraper = ilsole_scraper or IlSoleScraper()
        self.te_scraper = te_scraper or TradingEconomicsIndicatorsScraper()
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

    def fetch_and_store(
        self,
        countries: list[str] | None = None,
        include_bonds: bool = True,
    ) -> dict[str, Any]:
        """Fetch macro data for all specified countries and store in database.

        Args:
            countries: Countries to fetch. ``None`` means ``PORTFOLIO_COUNTRIES``.
            include_bonds: Whether to include bond yield data.

        Returns:
            Dict with ``"counts"`` (per-category totals) and ``"errors"``
            (list of error strings).
        """
        if countries is None:
            countries = list(PORTFOLIO_COUNTRIES)

        total_counts: dict[str, int] = {
            "ilsole_forecast": 0,
            "ilsole_observations": 0,
            "te_indicators": 0,
            "te_observations": 0,
            "bond_yields": 0,
            "bond_yield_observations": 0,
        }
        all_errors: list[str] = []

        for country in countries:
            try:
                result = self.fetch_country(country, include_bonds=include_bonds)

                for key, count in result["counts"].items():
                    total_counts[key] = total_counts.get(key, 0) + count

                for err in result["errors"]:
                    all_errors.append(f"{country}: {err}")

            except Exception as e:
                logger.error("Failed to process country %s: %s", country, e)
                all_errors.append(f"{country}: {e}")

        return {"counts": total_counts, "errors": all_errors}

    def fetch_country(
        self,
        country: str,
        include_bonds: bool = True,
    ) -> dict[str, Any]:
        """Fetch and store macro data for a single country.

        Args:
            country: English country name (must be in ``PORTFOLIO_COUNTRIES``).
            include_bonds: Whether to include bond yield data.

        Returns:
            Dict with ``"counts"`` and ``"errors"`` for this country.
        """
        counts: dict[str, int] = {}
        errors: list[str] = []

        today = datetime.date.today()
        try:
            forecast_data = self.ilsole_scraper.get_forecasts(country)
            if forecast_data and all(
                forecast_data.get(field) is None for field in _ILSOLE_VALUE_FIELDS
            ):
                # The country row was found but no cell parsed (renamed cell
                # ids, masked paywall overlay). Upserting it would replace the
                # last good snapshot with NULLs, so skip the write and report.
                counts["ilsole_forecast"] = 0
                counts["ilsole_observations"] = 0
                errors.append(
                    "ilsole_forecast: row found but no forecast value parsed "
                    "(page structure may have drifted)"
                )
                logger.warning("IlSole forecasts for %s parsed to all-None", country)
            elif forecast_data:
                with self.repo.session.begin_nested():
                    n_forecast = self.repo.upsert_economic_indicator(
                        country=country,
                        data=forecast_data,
                    )
                    n_forecast_obs = self.repo.upsert_economic_indicator_observation(
                        country=country,
                        snapshot_date=today,
                        data=forecast_data,
                    )
                counts["ilsole_forecast"] = n_forecast
                counts["ilsole_observations"] = n_forecast_obs
            else:
                counts["ilsole_forecast"] = 0
                counts["ilsole_observations"] = 0
                logger.info("No IlSole forecasts for %s", country)
        except Exception as e:
            errors.append(f"ilsole_forecast: {e}")
            counts.setdefault("ilsole_forecast", 0)
            counts.setdefault("ilsole_observations", 0)
            logger.warning("Failed IlSole forecasts for %s: %s", country, e)

        try:
            te_data = self.te_scraper.get_country_indicators(
                country, include_bonds=include_bonds
            )

            if te_data.get("status") == "success":
                indicators = te_data.get("indicators", {})
                if indicators:
                    with self.repo.session.begin_nested():
                        n_indicators = self.repo.upsert_te_indicators(
                            country=country,
                            indicators_dict=indicators,
                        )
                        n_obs = self.repo.upsert_te_observations(
                            country=country,
                            snapshot_date=today,
                            indicators_dict=indicators,
                        )
                    counts["te_indicators"] = n_indicators
                    counts["te_observations"] = n_obs
                else:
                    counts["te_indicators"] = 0
                    counts["te_observations"] = 0
                    logger.warning(
                        "Trading Economics returned 0 matched indicators for %s "
                        "(HTML structure may have drifted — no names matched "
                        "INDICATOR_PATTERNS)",
                        country,
                    )

                if include_bonds:
                    # The indicators fetch can succeed while the separate bond page
                    # fails; without surfacing it the job would finish "completed"
                    # with the yield tables silently stale.
                    if "bond_yields_error" in te_data:
                        bond_error = te_data["bond_yields_error"] or "Unknown error"
                        errors.append(f"trading_economics_bonds: {bond_error}")
                        logger.warning(
                            "Trading Economics bond yields failed for %s: %s",
                            country,
                            bond_error,
                        )
                    bond_yields = te_data.get("bond_yields", {})
                    if bond_yields:
                        with self.repo.session.begin_nested():
                            n_bonds = self.repo.upsert_bond_yields(
                                country=country,
                                yields_dict=bond_yields,
                            )
                            n_bond_obs = self.repo.upsert_bond_yield_observations(
                                country=country,
                                snapshot_date=today,
                                yields_dict=bond_yields,
                            )
                        counts["bond_yields"] = n_bonds
                        counts["bond_yield_observations"] = n_bond_obs
                    else:
                        counts["bond_yields"] = 0
                        counts["bond_yield_observations"] = 0
                        # The scraper reports a failed bond page as an otherwise
                        # successful result carrying only this key.
                        bond_error = te_data.get("bond_yields_error")
                        if bond_error:
                            errors.append(f"trading_economics_bonds: {bond_error}")
                            logger.warning(
                                "Trading Economics bond yields failed for %s: %s",
                                country,
                                bond_error,
                            )
                else:
                    counts["bond_yields"] = 0
                    counts["bond_yield_observations"] = 0
            else:
                counts["te_indicators"] = 0
                counts["te_observations"] = 0
                counts["bond_yields"] = 0
                counts["bond_yield_observations"] = 0
                te_error = te_data.get("error", "Unknown error")
                errors.append(f"trading_economics: {te_error}")
                logger.warning("Trading Economics failed for %s: %s", country, te_error)

        except Exception as e:
            errors.append(f"trading_economics: {e}")
            counts.setdefault("te_indicators", 0)
            counts.setdefault("te_observations", 0)
            counts.setdefault("bond_yields", 0)
            counts.setdefault("bond_yield_observations", 0)
            logger.warning("Failed Trading Economics for %s: %s", country, e)

        return {"counts": counts, "errors": errors}

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

            except Exception as exc:
                counts[series_id] = 0
                errors.append(f"{series_id}: {exc}")
                logger.error("FRED fetch exception for %s: %s", series_id, exc)

        return {"counts": counts, "errors": errors}

    def fetch_macro_news(
        self,
        max_articles: int = 100,
        fetch_full_content: bool = False,
    ) -> dict[str, Any]:
        """Fetch macro-themed news from yfinance and store in DB.

        Args:
            max_articles: Maximum number of articles to fetch.
            fetch_full_content: Whether to scrape full article body in addition
                to the headline snippet. When ``True``, articles whose body
                could not be scraped are skipped; when ``False``, articles are
                stored with their snippet only and a body scraped by an earlier
                run for the same article is kept.

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

    @staticmethod
    def get_portfolio_countries() -> list[str]:
        """Return the default list of portfolio countries."""
        return list(PORTFOLIO_COUNTRIES)


# ---------------------------------------------------------------------------
# Standalone bulk functions (called from the scheduler and CLI)
# ---------------------------------------------------------------------------


def _final_progress(errors: list[str], result: dict[str, Any]) -> dict[str, Any]:
    """Build the terminal ``on_progress`` kwargs for a bulk fetch.

    A partial failure (e.g. one country's Trading Economics circuit breaker
    tripping) must not produce a ``completed`` status, because callers rely on
    the status to detect gaps in the dataset. ``error`` carries a one-line
    summary since the failure webhook reads it and would otherwise post a blank
    alert.
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


def run_bulk_macro_fetch(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute bulk macro data fetch for all portfolio countries.

    Args:
        request: ``MacroFetchRequest`` with ``countries`` and ``include_bonds``.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``countries_processed``, ``counts``, ``error_count``.
    """
    from app.database import database_manager

    with database_manager.get_session() as session:
        repo = MacroRegimeRepository(session)
        service = MacroRegimeService(repo)

        countries = (
            request.countries
            if request.countries
            else MacroRegimeService.get_portfolio_countries()
        )
        total = len(countries)
        on_progress(total=total)

        all_errors: list[str] = []
        total_counts: dict[str, int] = {}

        for idx, country in enumerate(countries, 1):
            on_progress(current=idx, current_country=country)

            try:
                result = service.fetch_country(
                    country, include_bonds=request.include_bonds
                )

                for k, v in result["counts"].items():
                    total_counts[k] = total_counts.get(k, 0) + v
                for err in result["errors"]:
                    all_errors.append(f"{country}: {err}")

                session.commit()

            except Exception as e:
                logger.error("Failed to process %s: %s", country, e)
                all_errors.append(f"{country}: {e}")
                session.rollback()

        result_dict = {
            "countries_processed": total,
            "counts": total_counts,
            "error_count": len(all_errors),
        }
        on_progress(**_final_progress(all_errors, result_dict))
        logger.info(
            "Bulk macro fetch finished: %d countries, %d error(s)",
            total,
            len(all_errors),
        )

    return result_dict


def run_bulk_fred_fetch(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute bulk FRED time-series fetch.

    Args:
        request: ``FredFetchRequest`` with ``series_ids`` and ``incremental``.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``series_fetched`` and ``error_count``.
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
        result_dict = {
            "series_fetched": sum(counts.values()),
            "error_count": len(errors),
        }
        on_progress(
            status="completed",
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            errors=errors,
            result=result_dict,
        )
        logger.info(
            "Bulk FRED fetch completed: %d observations across %d series",
            result_dict["series_fetched"],
            len(counts),
        )

    return result_dict


def run_macro_news_fetch(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute macro news fetch in the service layer.

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
        )
        session.commit()

        errors = result.get("errors", [])
        result_dict = {"articles_stored": result.get("count", 0)}
        if errors:
            # Raise rather than report status="failed": the step runner flips
            # any non-"completed" status to "completed", so only an exception
            # makes a swallowed outage visible in the job history.
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
