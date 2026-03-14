"""FastAPI router for macroeconomic regime data fetch and read endpoints."""

import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import database_manager, get_db
from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.schemas.macro_regime import (
    BondYieldResponse,
    CountryMacroSummary,
    EconomicIndicatorObservationResponse,
    EconomicIndicatorResponse,
    FredFetchRequest,
    FredObservationResponse,
    MacroFetchJobResponse,
    MacroFetchProgress,
    MacroFetchRequest,
    MacroNewsFetchRequest,
    MacroNewsResponse,
    TradingEconomicsIndicatorResponse,
    TradingEconomicsObservationResponse,
)
from app.services.background_job import BackgroundJobService
from app.services.macro_regime_service import MacroRegimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/macro-data", tags=["Macro Data"])

_FORECAST_COLUMNS: frozenset[str] = frozenset({
    "last_inflation",
    "inflation_6m",
    "inflation_10y_avg",
    "gdp_growth_6m",
    "earnings_12m",
    "eps_expected_12m",
    "peg_ratio",
    "lt_rate_forecast",
})

# Shared job service instance for this router
_job_service = BackgroundJobService()


# ---------------------------------------------------------------------------
# Background bulk fetch worker
# ---------------------------------------------------------------------------


def _run_bulk_fetch(
    job_id: str,
    request: MacroFetchRequest,
) -> None:
    """Execute bulk macro data fetch in a background thread."""
    _job_service.update_job(job_id, status="running")

    try:
        with database_manager.get_session() as session:
            repo = MacroRegimeRepository(session)
            service = MacroRegimeService(repo)

            countries = (
                request.countries
                if request.countries
                else MacroRegimeService.get_portfolio_countries()
            )
            total = len(countries)
            _job_service.update_job(job_id, total=total)

            all_errors: list[str] = []
            total_counts: dict[str, int] = {}

            for idx, country in enumerate(countries, 1):
                _job_service.update_job(job_id, current=idx, current_country=country)

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

            _job_service.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                errors=all_errors,
                result={
                    "countries_processed": total,
                    "counts": total_counts,
                    "error_count": len(all_errors),
                },
            )
            logger.info("Macro fetch %s completed: %d countries", job_id, total)

    except Exception as e:
        logger.error("Macro fetch %s failed: %s", job_id, e)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_repo(db: Session = Depends(get_db)) -> MacroRegimeRepository:
    return MacroRegimeRepository(db)


# ---------------------------------------------------------------------------
# Bulk fetch endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/fetch",
    response_model=MacroFetchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_bulk_fetch(
    request: MacroFetchRequest = MacroFetchRequest(),
):
    """Start a background job that fetches macro data for all portfolio countries."""
    running, running_id = _job_service.is_any_running()
    if running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A macro fetch job is already in progress (id={running_id})",
        )

    job_id = _job_service.create_job(current_country="")
    _job_service.start_background(
        target=_run_bulk_fetch,
        args=(job_id, request),
    )

    return MacroFetchJobResponse(
        job_id=job_id,
        status="pending",
        message="Macro fetch started. Poll GET /macro-data/fetch/{job_id}.",
    )


@router.get("/fetch/{job_id}", response_model=MacroFetchProgress)
def get_fetch_status(job_id: str):
    """Poll the status and progress of a macro fetch job."""
    job = _job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return MacroFetchProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_country=job.get("current_country", ""),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Single-country fetch
# ---------------------------------------------------------------------------


@router.post("/fetch/{country}")
def fetch_single_country(
    country: str,
    request: MacroFetchRequest = MacroFetchRequest(),
    db: Session = Depends(get_db),
):
    """Synchronously fetch macro data for a single country."""
    repo = MacroRegimeRepository(db)
    service = MacroRegimeService(repo)
    result = service.fetch_country(country, include_bonds=request.include_bonds)
    db.commit()

    return {
        "country": country,
        "counts": result["counts"],
        "errors": result["errors"],
    }


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/countries/{country}",
    response_model=CountryMacroSummary,
)
def get_country_summary(
    country: str,
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Get all macro data for a specific country."""
    summary = repo.get_country_summary(country)

    return CountryMacroSummary(
        country=country,
        economic_indicators=summary["economic_indicators"],
        te_indicators=summary["te_indicators"],
        bond_yields=summary["bond_yields"],
    )


@router.get(
    "/economic-indicators",
    response_model=list[EconomicIndicatorResponse],
)
def get_economic_indicators(
    country: str | None = Query(default=None, description="Filter by country"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """List all economic indicators, optionally filtered by country."""
    return repo.get_economic_indicators(country=country)


@router.get(
    "/te-indicators",
    response_model=list[TradingEconomicsIndicatorResponse],
)
def get_te_indicators(
    country: str | None = Query(default=None, description="Filter by country"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """List all Trading Economics indicators, optionally filtered by country."""
    return repo.get_te_indicators(country=country)


@router.get(
    "/bond-yields",
    response_model=list[BondYieldResponse],
)
def get_bond_yields(
    country: str | None = Query(default=None, description="Filter by country"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """List all bond yields, optionally filtered by country."""
    return repo.get_bond_yields(country=country)


# ---------------------------------------------------------------------------
# FRED time-series endpoints
# ---------------------------------------------------------------------------

_fred_job_service = BackgroundJobService()


def _run_fred_fetch(job_id: str, request: FredFetchRequest) -> None:
    """Execute FRED fetch in a background thread."""
    _fred_job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            repo = MacroRegimeRepository(session)
            service = MacroRegimeService(repo)
            result = service.fetch_fred_series(
                series_ids=request.series_ids,
                incremental=request.incremental,
            )
            session.commit()
            _fred_job_service.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                errors=result.get("errors", []),
                result=result.get("counts", {}),
            )
    except Exception as exc:
        logger.error("FRED fetch %s failed: %s", job_id, exc)
        _fred_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


@router.post(
    "/fred/fetch",
    response_model=MacroFetchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_fred_fetch(request: FredFetchRequest = FredFetchRequest()):
    """Start a background job that fetches FRED time-series observations."""
    running, running_id = _fred_job_service.is_any_running()
    if running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A FRED fetch job is already running (id={running_id})",
        )

    job_id = _fred_job_service.create_job()
    _fred_job_service.start_background(
        target=_run_fred_fetch,
        args=(job_id, request),
    )

    return MacroFetchJobResponse(
        job_id=job_id,
        status="pending",
        message="FRED fetch started. Poll GET /macro-data/fred/fetch/{job_id}.",
    )


@router.get("/fred/fetch/{job_id}", response_model=MacroFetchProgress)
def get_fred_fetch_status(job_id: str):
    """Poll the status of a FRED fetch background job."""
    job = _fred_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return MacroFetchProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_country=job.get("current_country", ""),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get(
    "/fred/series",
    response_model=list[FredObservationResponse],
)
def get_fred_observations(
    series_id: str | None = Query(default=None, description="Filter by series ID"),
    start_date: date | None = Query(default=None, description="Start date YYYY-MM-DD"),
    end_date: date | None = Query(default=None, description="End date YYYY-MM-DD"),
    limit: int = Query(default=500, le=10_000, description="Max rows to return"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Query stored FRED observations with optional filters."""
    if series_id is not None:
        from app.services.scrapers.fred_scraper import FRED_SERIES

        if series_id not in FRED_SERIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown FRED series_id: '{series_id}'. "
                    f"Valid IDs: {sorted(FRED_SERIES)}"
                ),
            )
    return repo.get_fred_observations(
        series_id=series_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Trading Economics time-series endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/te-observations",
    response_model=list[TradingEconomicsObservationResponse],
)
def get_te_observations(
    country: str | None = Query(default=None, description="Filter by country (e.g. USA, UK)"),
    indicator_keys: list[str] | None = Query(
        default=None, description="Filter by indicator keys (e.g. manufacturing_pmi)"
    ),
    start_date: date | None = Query(default=None, description="Start date YYYY-MM-DD"),
    end_date: date | None = Query(default=None, description="End date YYYY-MM-DD"),
    limit: int = Query(default=500, le=10_000, description="Max rows to return"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Query stored Trading Economics observations with optional filters."""
    rows = repo.get_te_observations(
        country=country,
        indicator_keys=indicator_keys,
        start_date=start_date,
        end_date=end_date,
    )
    return rows[:limit]


# ---------------------------------------------------------------------------
# Economic indicator observations time-series endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/economic-indicator-observations",
    response_model=list[EconomicIndicatorObservationResponse],
)
def get_economic_indicator_observations(
    country: str | None = Query(
        default=None, description="Filter by country (e.g. USA, UK)"
    ),
    start_date: date | None = Query(
        default=None, description="Start date YYYY-MM-DD"
    ),
    end_date: date | None = Query(default=None, description="End date YYYY-MM-DD"),
    limit: int = Query(default=500, le=10_000, description="Max rows to return"),
    columns: list[str] | None = Query(
        default=None,
        description=(
            "Forecast columns to include in each row. Allowed values: "
            "last_inflation, inflation_6m, inflation_10y_avg, gdp_growth_6m, "
            "earnings_12m, eps_expected_12m, peg_ratio, lt_rate_forecast. "
            "Omit this parameter to return all columns."
        ),
    ),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Query IlSole24Ore forecast observations with optional column filter.

    Use the ``columns`` parameter to select a subset of the 8 forecast
    columns per row.  Identity fields (id, country, date, reference_date,
    created_at, updated_at) are always included.
    """
    if columns:
        invalid = set(columns) - _FORECAST_COLUMNS
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unknown column(s): {sorted(invalid)}. "
                    f"Allowed: {sorted(_FORECAST_COLUMNS)}"
                ),
            )

    rows = repo.get_economic_indicator_observations(
        country=country,
        start_date=start_date,
        end_date=end_date,
    )
    rows = list(rows)[:limit]

    if not columns:
        return rows

    requested: set[str] = set(columns)
    result = []
    for row in rows:
        row_dict: dict = {
            "id": row.id,
            "country": row.country,
            "date": row.date,
            "reference_date": row.reference_date,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        for col in requested:
            row_dict[col] = getattr(row, col)
        result.append(row_dict)
    return result


# ---------------------------------------------------------------------------
# Macro news endpoints
# ---------------------------------------------------------------------------

_news_job_service = BackgroundJobService()


def _run_macro_news_fetch(job_id: str, request: MacroNewsFetchRequest) -> None:
    """Execute macro news fetch in a background thread."""
    _news_job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            repo = MacroRegimeRepository(session)
            service = MacroRegimeService(repo)
            result = service.fetch_macro_news(
                max_articles=request.max_articles,
                fetch_full_content=request.fetch_full_content,
            )
            session.commit()
            _news_job_service.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                errors=result.get("errors", []),
                result={"articles_stored": result.get("count", 0)},
            )
    except Exception as exc:
        logger.error("Macro news fetch %s failed: %s", job_id, exc)
        _news_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


@router.post(
    "/news/fetch",
    response_model=MacroFetchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_macro_news_fetch(
    request: MacroNewsFetchRequest = MacroNewsFetchRequest(),
):
    """Start a background job to fetch macro-themed news from yfinance."""
    running, running_id = _news_job_service.is_any_running()
    if running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A macro news fetch job is already running (id={running_id})",
        )

    job_id = _news_job_service.create_job()
    _news_job_service.start_background(
        target=_run_macro_news_fetch,
        args=(job_id, request),
    )

    return MacroFetchJobResponse(
        job_id=job_id,
        status="pending",
        message="Macro news fetch started. Poll GET /macro-data/news/fetch/{job_id}.",
    )


@router.get("/news/fetch/{job_id}", response_model=MacroFetchProgress)
def get_macro_news_fetch_status(job_id: str):
    """Poll the status of a macro news fetch background job."""
    job = _news_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return MacroFetchProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_country=job.get("current_country", ""),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get(
    "/news",
    response_model=list[MacroNewsResponse],
)
def get_macro_news(
    theme: str | None = Query(default=None, description="Filter by macro theme"),
    start_date: date | None = Query(default=None, description="Start date YYYY-MM-DD"),
    end_date: date | None = Query(default=None, description="End date YYYY-MM-DD"),
    limit: int = Query(default=50, le=500, description="Max rows to return"),
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Query stored macro news with optional theme and date filters."""
    from datetime import datetime as dt
    from datetime import time

    start_dt = dt.combine(start_date, time.min) if start_date else None
    end_dt = dt.combine(end_date, time.max) if end_date else None

    return repo.get_macro_news(
        theme=theme,
        start_date=start_dt,
        end_date=end_dt,
        limit=limit,
    )


@router.get("/news/themes")
def get_macro_news_themes():
    """Return available macro theme enum values."""
    from app.services.yfinance.news.macro_news import MacroTheme

    return [
        {"value": t.value, "label": t.value.replace("_", " ").title()}
        for t in MacroTheme
    ]


# ---------------------------------------------------------------------------
# FRED catalog endpoint
# ---------------------------------------------------------------------------


@router.get("/fred/catalog")
def get_fred_catalog():
    """Return the static FRED series registry with group metadata."""
    from app.services.scrapers.fred_scraper import (
        FRED_CLI_SERIES,
        FRED_RECESSION_SERIES,
        FRED_SERIES,
        FRED_SPREAD_SERIES,
        FRED_VOLATILITY_SERIES,
    )

    group_map: dict[str, str] = {}
    for sid in FRED_SPREAD_SERIES:
        group_map[sid] = "spreads"
    for sid in FRED_VOLATILITY_SERIES:
        group_map[sid] = "volatility"
    for sid in FRED_CLI_SERIES:
        group_map[sid] = "cli"
    for sid in FRED_RECESSION_SERIES:
        group_map[sid] = "recession"

    return [
        {"series_id": sid, "description": desc, "group": group_map.get(sid, "other")}
        for sid, desc in FRED_SERIES.items()
    ]


# ---------------------------------------------------------------------------
# Distinct countries endpoint
# ---------------------------------------------------------------------------


@router.get("/countries", response_model=list[str])
def get_distinct_countries(
    repo: MacroRegimeRepository = Depends(_get_repo),
):
    """Return a sorted deduplicated list of all countries with stored macro data."""
    return repo.get_distinct_countries()
