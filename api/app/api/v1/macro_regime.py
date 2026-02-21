"""FastAPI router for macroeconomic regime data fetch and read endpoints."""

import logging
import threading
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import database_manager, get_db
from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.schemas.macro_regime import (
    BondYieldResponse,
    CountryMacroSummary,
    EconomicIndicatorResponse,
    MacroFetchJobResponse,
    MacroFetchProgress,
    MacroFetchRequest,
    TradingEconomicsIndicatorResponse,
)
from app.services.macro_regime_service import MacroRegimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/macro-data", tags=["Macro Data"])


# ---------------------------------------------------------------------------
# In-memory job store (same pattern as yfinance_data.py)
# ---------------------------------------------------------------------------

_fetch_jobs: dict[str, dict[str, Any]] = {}
_fetch_jobs_lock = threading.Lock()


def _set_job(job_id: str, data: dict[str, Any]) -> None:
    with _fetch_jobs_lock:
        _fetch_jobs[job_id] = data


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _fetch_jobs_lock:
        return _fetch_jobs.get(job_id, {}).copy() if job_id in _fetch_jobs else None


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _fetch_jobs_lock:
        if job_id in _fetch_jobs:
            _fetch_jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Background bulk fetch worker
# ---------------------------------------------------------------------------


def _run_bulk_fetch(
    job_id: str,
    request: MacroFetchRequest,
) -> None:
    """Execute bulk macro data fetch in a background thread."""
    _update_job(job_id, status="running")

    try:
        with database_manager.get_session() as session:
            service = MacroRegimeService(session)

            # Determine countries
            from app.services.scrapers.ilsole_scraper import PORTFOLIO_COUNTRIES

            countries = (
                request.countries if request.countries else list(PORTFOLIO_COUNTRIES)
            )
            total = len(countries)
            _update_job(job_id, total=total)

            all_errors: list[str] = []
            total_counts: dict[str, int] = {}

            for idx, country in enumerate(countries, 1):
                _update_job(job_id, current=idx, current_country=country)

                try:
                    result = service.fetch_country(
                        country, include_bonds=request.include_bonds
                    )

                    # Accumulate counts
                    for k, v in result["counts"].items():
                        total_counts[k] = total_counts.get(k, 0) + v

                    # Accumulate errors with country prefix
                    for err in result["errors"]:
                        all_errors.append(f"{country}: {err}")

                    # Commit per country
                    session.commit()

                except Exception as e:
                    logger.error("Failed to process %s: %s", country, e)
                    all_errors.append(f"{country}: {e}")
                    session.rollback()

            _update_job(
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
        _update_job(
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
    # Reject if another job is already running
    with _fetch_jobs_lock:
        for job in _fetch_jobs.values():
            if job.get("status") in ("pending", "running"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A macro fetch job is already in progress (id={job['job_id']})",
                )

    job_id = str(uuid_mod.uuid4())
    _set_job(
        job_id,
        {
            "job_id": job_id,
            "status": "pending",
            "current": 0,
            "total": 0,
            "current_country": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "errors": [],
            "result": None,
            "error": None,
        },
    )

    thread = threading.Thread(
        target=_run_bulk_fetch,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()

    return MacroFetchJobResponse(
        job_id=job_id,
        status="pending",
        message="Macro fetch started. Poll GET /macro-data/fetch/{job_id} for progress.",
    )


@router.get("/fetch/{job_id}", response_model=MacroFetchProgress)
def get_fetch_status(job_id: str):
    """Poll the status and progress of a macro fetch job."""
    job = _get_job(job_id)
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
    service = MacroRegimeService(db)
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
