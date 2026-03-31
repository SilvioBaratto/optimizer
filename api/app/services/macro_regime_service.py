"""Service layer orchestrating macro regime data fetching and storage."""

import datetime
import logging
from typing import Any

from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.services._progress import ProgressCallback, _noop
from app.services.scrapers.fred_scraper import FRED_SERIES, FredScraper
from app.services.scrapers.ilsole_scraper import PORTFOLIO_COUNTRIES, IlSoleScraper
from app.services.scrapers.tradingeconomics_scraper import (
    TradingEconomicsIndicatorsScraper,
)

logger = logging.getLogger(__name__)


class MacroRegimeService:
    """Fetches macroeconomic data from scrapers and stores via repository."""

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
                return None
            self._fred_scraper = FredScraper(api_key=key)
        return self._fred_scraper

    def fetch_and_store(
        self,
        countries: list[str] | None = None,
        include_bonds: bool = True,
    ) -> dict[str, Any]:
        """
        Fetch macro data for all specified countries and store in database.

        Args:
            countries: List of countries to fetch. None means PORTFOLIO_COUNTRIES.
            include_bonds: Whether to fetch bond yield data.

        Returns:
            Dict with "counts" (per-category totals) and "errors" (list of error strings).
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

                # Accumulate counts
                for key, count in result["counts"].items():
                    total_counts[key] = total_counts.get(key, 0) + count

                # Accumulate errors with country prefix
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
        """
        Fetch and store macro data for a single country.

        Returns:
            Dict with "counts" and "errors" for this country.
        """
        counts: dict[str, int] = {}
        errors: list[str] = []

        # 1. IlSole forecasts (real indicators sourced from TradingEconomics)
        today = datetime.date.today()
        try:
            forecast_data = self.ilsole_scraper.get_forecasts(country)
            if forecast_data:
                counts["ilsole_forecast"] = self.repo.upsert_economic_indicator(
                    country=country,
                    data=forecast_data,
                )
                # Also write to time-series observation table
                counts["ilsole_observations"] = (
                    self.repo.upsert_economic_indicator_observation(
                        country=country,
                        snapshot_date=today,
                        data=forecast_data,
                    )
                )
            else:
                counts["ilsole_forecast"] = 0
                counts["ilsole_observations"] = 0
                logger.info("No IlSole forecasts for %s", country)
        except Exception as e:
            errors.append(f"ilsole_forecast: {e}")
            counts.setdefault("ilsole_observations", 0)
            logger.warning("Failed IlSole forecasts for %s: %s", country, e)

        # 3. Trading Economics indicators (+ bonds)
        try:
            te_data = self.te_scraper.get_country_indicators(
                country, include_bonds=include_bonds
            )

            if te_data.get("status") == "success":
                # Store indicators (latest-snapshot table)
                indicators = te_data.get("indicators", {})
                if indicators:
                    counts["te_indicators"] = self.repo.upsert_te_indicators(
                        country=country,
                        indicators_dict=indicators,
                    )
                    # Also write to time-series observation table
                    n_obs = self.repo.upsert_te_observations(
                        country=country,
                        snapshot_date=today,
                        indicators_dict=indicators,
                    )
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

                # Store bond yields (latest-snapshot table)
                if include_bonds:
                    bond_yields = te_data.get("bond_yields", {})
                    if bond_yields:
                        counts["bond_yields"] = self.repo.upsert_bond_yields(
                            country=country,
                            yields_dict=bond_yields,
                        )
                        # Also write to time-series observation table
                        n_bond_obs = self.repo.upsert_bond_yield_observations(
                            country=country,
                            snapshot_date=today,
                            yields_dict=bond_yields,
                        )
                        counts["bond_yield_observations"] = n_bond_obs
                    else:
                        counts["bond_yields"] = 0
                        counts["bond_yield_observations"] = 0
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
    ) -> dict[str, Any]:
        """Fetch FRED time-series observations and store in DB.

        Args:
            series_ids: Series to fetch. ``None`` means all ``FRED_SERIES`` keys.
            incremental: When ``True``, fetch only observations newer than last stored.

        Returns:
            Dict with ``"counts"`` and ``"errors"`` keys.
        """
        scraper = self.fred_scraper
        if scraper is None:
            return {
                "counts": {},
                "errors": ["FRED_API_KEY not configured in .env"],
            }

        ids = series_ids if series_ids is not None else list(FRED_SERIES.keys())
        counts: dict[str, int] = {}
        errors: list[str] = []

        for series_id in ids:
            try:
                start_date: str | None = None
                if incremental:
                    latest = self.repo.get_fred_latest_date(series_id)
                    if latest is not None:
                        start_date = (
                            latest + datetime.timedelta(days=1)
                        ).isoformat()

                result = scraper.get_series_observations(
                    series_id, start_date=start_date
                )

                if result["status"] == "success":
                    observations = result.get("observations", [])
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

        Returns:
            Dict with ``"count"`` and ``"errors"`` keys.
        """
        import uuid

        from app.services.yfinance import get_yfinance_client
        from app.services.yfinance.news.macro_news import MacroNewsFetcher

        errors: list[str] = []
        try:
            yf_client = get_yfinance_client()
            search_client = yf_client.search

            scraper = None
            if fetch_full_content:
                from app.services.yfinance.news.scraper import ArticleScraper

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

            rows.append({
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
            })

        rows = [r for r in rows if r.get("full_content")]

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
# Standalone bulk functions (callable from routes and scheduler)
# ---------------------------------------------------------------------------


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
        on_progress(
            status="completed",
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            errors=all_errors,
            result=result_dict,
        )
        logger.info("Bulk macro fetch completed: %d countries", total)

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

        result_dict = {"articles_stored": result.get("count", 0)}
        on_progress(
            status="completed",
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            errors=result.get("errors", []),
            result=result_dict,
        )
        logger.info("Macro news fetch completed: %d articles", result.get("count", 0))

    return result_dict
