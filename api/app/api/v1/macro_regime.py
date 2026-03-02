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
    EconomicIndicatorResponse,
    FredFetchRequest,
    FredObservationResponse,
    MacroFetchJobResponse,
    MacroFetchProgress,
    MacroFetchRequest,
    TradingEconomicsIndicatorResponse,
)
from app.services.background_job import BackgroundJobService
from app.services.macro_regime_service import MacroRegimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/macro-data", tags=["Macro Data"])

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
    obs = repo.get_fred_observations(
        series_id=series_id,
        start_date=start_date,
        end_date=end_date,
    )
    return list(obs)[:limit]
