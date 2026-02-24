"""FastAPI router for yfinance data fetch and read endpoints."""

import logging
from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import database_manager, get_db
from app.repositories.yfinance_repository import YFinanceRepository
from app.schemas.yfinance_data import (
    AnalystPriceTargetResponse,
    AnalystRecommendationResponse,
    DividendResponse,
    FinancialStatementResponse,
    InsiderTransactionResponse,
    InstitutionalHolderResponse,
    MutualFundHolderResponse,
    PriceHistoryResponse,
    StockSplitResponse,
    TickerNewsResponse,
    TickerProfileResponse,
    YFinanceFetchJobResponse,
    YFinanceFetchProgress,
    YFinanceFetchRequest,
    YFinanceSingleFetchRequest,
    YFinanceSingleFetchResponse,
)
from app.services.background_job import BackgroundJobService
from app.services.yfinance import YFinanceClient, get_yfinance_client
from app.services.yfinance_data_service import YFinanceDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yfinance-data", tags=["YFinance Data"])

# Shared job service instance for this router
_job_service = BackgroundJobService()


# ---------------------------------------------------------------------------
# Background bulk fetch worker
# ---------------------------------------------------------------------------


def _run_bulk_fetch(
    job_id: str,
    request: YFinanceFetchRequest,
    yf_client: YFinanceClient,
) -> None:
    """Execute bulk yfinance fetch in a background thread."""
    _job_service.update_job(job_id, status="running")

    try:
        with database_manager.get_session() as session:
            repo = YFinanceRepository(session)
            instruments = repo.get_instruments_with_yfinance_ticker()

            total = len(instruments)
            _job_service.update_job(job_id, total=total)

            all_errors: list[str] = []
            total_counts: dict[str, int] = {}
            total_skipped: int = 0

            service = YFinanceDataService(repo, yf_client)

            for idx, instrument in enumerate(instruments, 1):
                ticker = instrument.yfinance_ticker
                _job_service.update_job(job_id, current=idx, current_ticker=ticker)

                try:
                    result = service.fetch_and_store(
                        instrument_id=instrument.id,
                        yfinance_ticker=ticker,
                        period=request.period,
                        mode=request.mode,
                        exchange_name=instrument.exchange_name,
                    )

                    for k, v in result["counts"].items():
                        total_counts[k] = total_counts.get(k, 0) + v
                    total_skipped += len(result.get("skipped", []))
                    for err in result["errors"]:
                        all_errors.append(f"{ticker}: {err}")

                    session.commit()

                except Exception as e:
                    logger.error("Failed to process %s: %s", ticker, e)
                    all_errors.append(f"{ticker}: {e}")
                    session.rollback()

            _job_service.update_job(
                job_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                errors=all_errors,
                result={
                    "tickers_processed": total,
                    "counts": total_counts,
                    "error_count": len(all_errors),
                    "skipped_category_count": total_skipped,
                },
            )
            logger.info("Bulk fetch %s completed: %d tickers", job_id, total)

    except Exception as e:
        logger.error("Bulk fetch %s failed: %s", job_id, e)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_yf_client() -> YFinanceClient:
    return get_yfinance_client()


def _get_repo(db: Session = Depends(get_db)) -> YFinanceRepository:
    return YFinanceRepository(db)


# ---------------------------------------------------------------------------
# Bulk fetch endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/fetch",
    response_model=YFinanceFetchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_bulk_fetch(
    request: YFinanceFetchRequest = YFinanceFetchRequest(),
    yf_client: YFinanceClient = Depends(_get_yf_client),
):
    """Start a background job that fetches yfinance data for all instruments."""
    running, running_id = _job_service.is_any_running()
    if running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A fetch job is already in progress (id={running_id})",
        )

    job_id = _job_service.create_job(current_ticker="")
    _job_service.start_background(
        target=_run_bulk_fetch,
        args=(job_id, request, yf_client),
    )

    return YFinanceFetchJobResponse(
        job_id=job_id,
        status="pending",
        message="Fetch started. Poll GET /yfinance-data/fetch/{job_id} for progress.",
    )


@router.get("/fetch/{job_id}", response_model=YFinanceFetchProgress)
def get_fetch_status(job_id: str):
    """Poll the status and progress of a bulk fetch job."""
    job = _job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return YFinanceFetchProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_ticker=job.get("current_ticker", ""),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Single-ticker fetch
# ---------------------------------------------------------------------------


@router.post(
    "/fetch/ticker/{yfinance_ticker}",
    response_model=YFinanceSingleFetchResponse,
)
def fetch_single_ticker(
    yfinance_ticker: str,
    request: YFinanceSingleFetchRequest = YFinanceSingleFetchRequest(),
    db: Session = Depends(get_db),
    repo: YFinanceRepository = Depends(_get_repo),
    yf_client: YFinanceClient = Depends(_get_yf_client),
):
    """Synchronously fetch all yfinance data for a single ticker."""
    instrument = repo.get_instrument_by_yfinance_ticker(yfinance_ticker)
    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No instrument found with yfinance_ticker={yfinance_ticker!r}",
        )

    service = YFinanceDataService(repo, yf_client)
    result = service.fetch_and_store(
        instrument_id=instrument.id,
        yfinance_ticker=yfinance_ticker,
        period=request.period,
        mode=request.mode,
        exchange_name=instrument.exchange_name,
    )
    db.commit()

    return YFinanceSingleFetchResponse(
        ticker=yfinance_ticker,
        instrument_id=str(instrument.id),
        counts=result["counts"],
        errors=result["errors"],
        skipped=result.get("skipped", []),
    )


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/instruments/{instrument_id}/profile",
    response_model=TickerProfileResponse,
)
def get_profile(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored ticker profile for an instrument."""
    profile = repo.get_profile(instrument_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.get(
    "/instruments/{instrument_id}/prices",
    response_model=list[PriceHistoryResponse],
)
def get_prices(
    instrument_id: UUID,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=50000),
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored price history for an instrument."""
    return repo.get_price_history(
        instrument_id, start_date=start_date, end_date=end_date, limit=limit
    )


@router.get(
    "/instruments/{instrument_id}/financials",
    response_model=list[FinancialStatementResponse],
)
def get_financials(
    instrument_id: UUID,
    statement_type: str | None = Query(
        default=None,
        description="income_statement | balance_sheet | cashflow | earnings",
    ),
    period_type: str | None = Query(default=None, description="annual | quarterly"),
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored financial statements for an instrument."""
    return repo.get_financial_statements(
        instrument_id, statement_type=statement_type, period_type=period_type
    )


@router.get(
    "/instruments/{instrument_id}/dividends",
    response_model=list[DividendResponse],
)
def get_dividends(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored dividend data for an instrument."""
    return repo.get_dividends(instrument_id)


@router.get(
    "/instruments/{instrument_id}/splits",
    response_model=list[StockSplitResponse],
)
def get_splits(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored stock split data for an instrument."""
    return repo.get_splits(instrument_id)


@router.get(
    "/instruments/{instrument_id}/recommendations",
    response_model=list[AnalystRecommendationResponse],
)
def get_recommendations(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored analyst recommendations for an instrument."""
    return repo.get_recommendations(instrument_id)


@router.get(
    "/instruments/{instrument_id}/price-targets",
    response_model=AnalystPriceTargetResponse,
)
def get_price_targets(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored analyst price targets for an instrument."""
    targets = repo.get_price_targets(instrument_id)
    if not targets:
        raise HTTPException(status_code=404, detail="Price targets not found")
    return targets


@router.get(
    "/instruments/{instrument_id}/institutional-holders",
    response_model=list[InstitutionalHolderResponse],
)
def get_institutional_holders(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored institutional holders for an instrument."""
    return repo.get_institutional_holders(instrument_id)


@router.get(
    "/instruments/{instrument_id}/mutualfund-holders",
    response_model=list[MutualFundHolderResponse],
)
def get_mutualfund_holders(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored mutual fund holders for an instrument."""
    return repo.get_mutualfund_holders(instrument_id)


@router.get(
    "/instruments/{instrument_id}/insider-transactions",
    response_model=list[InsiderTransactionResponse],
)
def get_insider_transactions(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored insider transactions for an instrument."""
    return repo.get_insider_transactions(instrument_id)


@router.get(
    "/instruments/{instrument_id}/news",
    response_model=list[TickerNewsResponse],
)
def get_news(
    instrument_id: UUID,
    repo: YFinanceRepository = Depends(_get_repo),
):
    """Read stored news articles for an instrument."""
    return repo.get_news(instrument_id)
