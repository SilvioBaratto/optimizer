"""FastAPI router for portfolio dashboard endpoints."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.repositories.dashboard_repository import DashboardRepository
from app.repositories.portfolio_repository import PortfolioRepository
from app.schemas.dashboard import (
    ActivityFeedResponse,
    ActivityItem,
    AllocationResponse,
    AssetClassReturnsResponse,
    DriftResponse,
    EquityCurveResponse,
    MarketSnapshotResponse,
    PerformanceMetricsResponse,
    ReferenceIndexItem,
    ReferenceIndicesResponse,
    RollingMetricsResponse,
)
from app.services import dashboard_service
from app.services._sector_resolver import resolve_sector_map

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio-analytics", tags=["Dashboard"])
market_router = APIRouter(prefix="/market", tags=["Market"])

_PERIOD_DAYS: dict[str, int | None] = {
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "MAX": None,
}
_MAX_FLOOR_DATE = date(2000, 1, 1)


def _require_benchmark_prices(
    repo: DashboardRepository, ticker: str, n: int = 2,
) -> list[float]:
    """Load *n* most recent prices for a benchmark ticker.

    Raises 422 if the ticker is absent from the instruments table
    (seed it first) and 503 if the instrument exists but has fewer than
    *n* price rows (run yfinance fetch first). Distinguishing the two
    tells the caller which upstream step is missing (issue #461).
    """
    if not repo.instrument_exists(ticker):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Benchmark ticker '{ticker}' not found in instruments "
                "table — seed it first"
            ),
        )
    prices = repo.get_benchmark_prices(ticker, n=n)
    if len(prices) < n:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Benchmark ticker '{ticker}' exists but has no price "
                "data; run yfinance fetch first"
            ),
        )
    return prices


def _validate_benchmark(
    repo: DashboardRepository,
    benchmark: str,
    prices: pd.DataFrame,
) -> None:
    """Validate that a benchmark ticker exists and has price data.

    Raises 422 with a distinct message for:
      - Ticker not found in the instruments table (seed it first)
      - Ticker found but absent from price data (fetch needed)

    Args:
        repo: Dashboard repository for instrument lookups.
        benchmark: Benchmark ticker symbol to validate.
        prices: Price DataFrame whose columns are available tickers.
    """
    if not repo.instrument_exists(benchmark):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Benchmark ticker '{benchmark}' not found in instruments "
                "table — seed it first"
            ),
        )
    if benchmark not in prices.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Benchmark ticker '{benchmark}' has no price data",
        )


@router.get(
    "/{name}/performance-metrics",
    response_model=PerformanceMetricsResponse,
    response_model_by_alias=True,
    summary="Portfolio performance KPIs with sparklines and change deltas",
)
def get_performance_metrics(
    name: str,
    benchmark: str = Query(
        default_factory=lambda: settings.default_benchmark_ticker,
        description="Benchmark ticker",
    ),
    period: Literal["1Y", "3Y", "5Y", "MAX"] = Query(
        default="3Y",
        description="Lookback period for KPI computation",
    ),
    db: Session = Depends(get_db),
) -> PerformanceMetricsResponse:
    """Return 7 KPIs for the dashboard strip.

    Requires at least one portfolio snapshot from the optimization
    or broker sync pipeline. The ``period`` query parameter scopes the
    price history used to compute every KPI so the dashboard period
    selector can refresh KPIs alongside the equity curve (issue #433).
    """
    portfolio_repo = PortfolioRepository(db)
    dashboard_repo = DashboardRepository(db)

    # --- Resolve portfolio ---
    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    # --- Get latest snapshot (weights source) ---
    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights

    # --- Get NAV from broker account snapshot (optional) ---
    account = portfolio_repo.get_latest_account_snapshot(portfolio.id)
    nav = account.total if account else None

    # --- Fetch price history scoped to the requested period ---
    tickers = list({*weights.keys(), benchmark})
    end_date = date.today()
    days = _PERIOD_DAYS[period]
    start_date = (
        end_date - timedelta(days=days) if days is not None else _MAX_FLOOR_DATE
    )

    prices = dashboard_repo.get_multi_ticker_prices(tickers, start_date, end_date)

    if prices.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No price data available for portfolio tickers",
        )

    _validate_benchmark(dashboard_repo, benchmark, prices)

    # --- Compute metrics ---
    try:
        result = dashboard_service.compute_performance_metrics(
            weights=weights,
            prices=prices,
            nav=nav,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return PerformanceMetricsResponse(
        **result,
        currency=portfolio.currency,
    )


@router.get(
    "/{name}/equity-curve",
    response_model=EquityCurveResponse,
    response_model_by_alias=True,
    summary="Dual-line equity curve: portfolio NAV vs benchmark rebased to 100",
)
def get_equity_curve(
    name: str,
    benchmark: str = Query(
        default_factory=lambda: settings.default_benchmark_ticker,
        description="Benchmark ticker symbol",
    ),
    period: Literal["1Y", "3Y", "5Y", "MAX"] = Query(
        default="3Y",
        description="Lookback period",
    ),
    db: Session = Depends(get_db),
) -> EquityCurveResponse:
    """Return daily equity curve points for portfolio vs benchmark.

    Both series are rebased to 100 at the start date.
    """
    portfolio_repo = PortfolioRepository(db)
    dashboard_repo = DashboardRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights
    tickers = list({*weights.keys(), benchmark})

    end_date = date.today()
    days = _PERIOD_DAYS[period]
    start_date = (
        end_date - timedelta(days=days) if days is not None else _MAX_FLOOR_DATE
    )

    prices = dashboard_repo.get_multi_ticker_prices(tickers, start_date, end_date)

    if prices.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No price data available for portfolio tickers",
        )

    _validate_benchmark(dashboard_repo, benchmark, prices)

    try:
        result = dashboard_service.compute_equity_curve(
            weights=weights,
            prices=prices,
            benchmark=benchmark,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return EquityCurveResponse(**result)


_DEFAULT_ROLLING_WINDOW = 63


@router.get(
    "/{name}/rolling-metrics",
    response_model=RollingMetricsResponse,
    response_model_by_alias=True,
    summary="Rolling Sharpe / volatility / beta series for the dashboard",
)
def get_rolling_metrics(
    name: str,
    benchmark: str = Query(
        default_factory=lambda: settings.default_benchmark_ticker,
        description="Benchmark ticker symbol",
    ),
    period: Literal["1Y", "3Y", "5Y", "MAX"] = Query(
        default="3Y",
        description="Lookback period for the rolling series",
    ),
    window: int = Query(
        default=_DEFAULT_ROLLING_WINDOW,
        ge=5,
        le=252,
        description="Rolling window length in trading days (default 63 ≈ 3 months)",
    ),
    db: Session = Depends(get_db),
) -> RollingMetricsResponse:
    """Return dated rolling Sharpe, annualized volatility, and beta series."""
    portfolio_repo = PortfolioRepository(db)
    dashboard_repo = DashboardRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights
    tickers = list({*weights.keys(), benchmark})

    end_date = date.today()
    days = _PERIOD_DAYS[period]
    start_date = (
        end_date - timedelta(days=days) if days is not None else _MAX_FLOOR_DATE
    )

    prices = dashboard_repo.get_multi_ticker_prices(tickers, start_date, end_date)

    if prices.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No price data available for portfolio tickers",
        )

    _validate_benchmark(dashboard_repo, benchmark, prices)

    try:
        result = dashboard_service.compute_rolling_metrics(
            weights=weights,
            prices=prices,
            benchmark=benchmark,
            window=window,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return RollingMetricsResponse(**result)


@router.get(
    "/{name}/allocation",
    response_model=AllocationResponse,
    response_model_by_alias=True,
    summary="Sector allocation sunburst",
)
def get_allocation(
    name: str,
    db: Session = Depends(get_db),
) -> AllocationResponse:
    """Return sector → ticker allocation hierarchy for the sunburst chart."""
    portfolio_repo = PortfolioRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights
    snapshot_mapping: dict[str, str] | None = snapshot.sector_mapping

    if snapshot_mapping is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No sector mapping found in latest snapshot;"
                " run optimization with sector data"
            ),
        )

    # Shared resolver (issue #427): snapshot-first with TickerProfile fallback
    # for any ticker the snapshot didn't cover.
    sector_mapping = resolve_sector_map(
        session=db,
        tickers=weights.keys(),
        snapshot_mapping=snapshot_mapping,
    )

    try:
        result = dashboard_service.compute_allocation(
            weights=weights,
            sector_mapping=sector_mapping,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return AllocationResponse(**result)


@router.get(
    "/{name}/drift",
    response_model=DriftResponse,
    response_model_by_alias=True,
    summary="Portfolio drift analysis: target vs actual weights",
)
def get_drift(
    name: str,
    threshold: float = Query(default=0.05, description="Absolute drift threshold"),
    db: Session = Depends(get_db),
) -> DriftResponse:
    """Return per-ticker drift between target and actual weights."""
    portfolio_repo = PortfolioRepository(db)
    dashboard_repo = DashboardRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights
    positions = portfolio_repo.get_positions(portfolio.id)

    # Build position dicts for the service layer
    position_dicts = [
        {
            "yfinance_ticker": p.yfinance_ticker,
            "name": p.name,
            "quantity": p.quantity,
            "current_price": p.current_price,
        }
        for p in positions
    ]

    # Fallback: fetch prices if no broker positions available
    prices_df = None
    if not position_dicts or all(
        p.get("current_price") is None or not p.get("yfinance_ticker")
        for p in position_dicts
    ):
        tickers = list(weights.keys())
        prices_df = dashboard_repo.get_multi_ticker_prices(
            tickers, snapshot.snapshot_date, date.today(),
        )

    try:
        result = dashboard_service.compute_drift(
            target_weights=weights,
            broker_positions=position_dicts,
            threshold=threshold,
            prices_df=prices_df,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return DriftResponse(**result)


@router.get(
    "/{name}/activity",
    response_model=ActivityFeedResponse,
    response_model_by_alias=True,
    summary="Paginated portfolio activity feed",
)
def get_activity(
    name: str,
    limit: int = Query(default=20, ge=1, le=200, description="Max events to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    type: str | None = Query(default=None, description="Filter by event type"),
    db: Session = Depends(get_db),
) -> ActivityFeedResponse:
    """Return paginated activity events for a portfolio."""
    portfolio_repo = PortfolioRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    events = portfolio_repo.get_events(
        portfolio_id=portfolio.id,
        event_type=type,
        limit=limit,
        offset=offset,
    )
    total = portfolio_repo.count_events(
        portfolio_id=portfolio.id,
        event_type=type,
    )

    items = [
        ActivityItem(
            id=str(event.id),
            type=event.event_type,
            title=event.title,
            description=event.description,
            timestamp=event.created_at,
        )
        for event in events
    ]

    return ActivityFeedResponse(items=items, total=total)


# ---------------------------------------------------------------------------
# Asset class returns heatmap
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/asset-class-returns",
    response_model=AssetClassReturnsResponse,
    response_model_by_alias=True,
    summary="Asset class returns heatmap: 1D / 1W / 1M / YTD by sector",
)
def get_asset_class_returns(
    name: str,
    db: Session = Depends(get_db),
) -> AssetClassReturnsResponse:
    """Return sector-level returns over standard periods for the heatmap chart.

    Groups portfolio tickers by sector from the latest snapshot's sector_mapping.
    Returns = Σ (ticker_weight / sector_total_weight) × ticker_return.
    Sectors ordered by descending portfolio weight.
    """
    portfolio_repo = PortfolioRepository(db)
    dashboard_repo = DashboardRepository(db)

    portfolio = portfolio_repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio snapshot found; run optimization first",
        )

    weights: dict[str, float] = snapshot.weights
    snapshot_mapping: dict[str, str] | None = snapshot.sector_mapping

    if snapshot_mapping is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No sector mapping found in latest snapshot;"
                " run optimization with sector data"
            ),
        )

    # Shared resolver (issue #427): snapshot-first with TickerProfile fallback.
    sector_mapping = resolve_sector_map(
        session=db,
        tickers=weights.keys(),
        snapshot_mapping=snapshot_mapping,
    )

    # Lookback must cover from Jan 1 of current year at minimum
    tickers = list(weights.keys())
    end_date = date.today()
    start_date = date(end_date.year, 1, 1)

    prices = dashboard_repo.get_multi_ticker_prices(tickers, start_date, end_date)

    if prices.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No price data available for portfolio tickers",
        )

    try:
        result = dashboard_service.compute_asset_class_returns(
            weights=weights,
            sector_mapping=sector_mapping,
            prices=prices,
            today=end_date,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return AssetClassReturnsResponse(**result)


# ---------------------------------------------------------------------------
# Market snapshot (portfolio-independent)
# ---------------------------------------------------------------------------

_FRED_MARKET_SERIES = ["VIXCLS", "DTWEXBGS"]


@market_router.get(
    "/snapshot",
    response_model=MarketSnapshotResponse,
    response_model_by_alias=True,
    summary="Market context snapshot: VIX, S&P 500, 10Y yield, USD index",
)
def get_market_snapshot(
    db: Session = Depends(get_db),
) -> MarketSnapshotResponse:
    """Return current market indicators with daily changes.

    Independent of any portfolio — sources data from FRED, bond yields,
    and SPY price history already stored in the database.
    """
    repo = DashboardRepository(db)

    fred_data = repo.get_latest_fred_observations(_FRED_MARKET_SERIES)
    if "VIXCLS" not in fred_data or "DTWEXBGS" not in fred_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Market data not yet available; "
                "run FRED fetch first (missing VIXCLS or DTWEXBGS)"
            ),
        )

    benchmark_ticker = settings.default_benchmark_ticker
    spy_prices = _require_benchmark_prices(repo, benchmark_ticker, n=2)

    bond_yield = repo.get_ten_year_yield_usa()
    if bond_yield is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bond yield data not yet available; run macro fetch first",
        )

    # Determine as_of from the most recent data point
    fred_date = repo.get_latest_fred_observation_dates(_FRED_MARKET_SERIES)
    spy_date = repo.get_benchmark_latest_date(benchmark_ticker)
    bond_date = repo.get_ten_year_yield_reference_date()

    candidates = [d for d in (fred_date, spy_date, bond_date) if d is not None]
    as_of = max(candidates) if candidates else date.today()

    result = dashboard_service.get_market_snapshot(
        fred_data=fred_data,
        spy_prices=spy_prices,
        bond_yield=bond_yield,
        as_of=as_of,
    )

    return MarketSnapshotResponse(**result)


# ---------------------------------------------------------------------------
# Reference indices list
# ---------------------------------------------------------------------------


@market_router.get(
    "/indices",
    response_model=ReferenceIndicesResponse,
    response_model_by_alias=True,
    summary="List of benchmark reference indices (ETF-type instruments)",
)
def get_reference_indices(
    db: Session = Depends(get_db),
) -> ReferenceIndicesResponse:
    """Return benchmark reference indices for the frontend benchmark selector.

    Queries the instruments table filtered by instrument_type='ETF'.
    Only ETF-type instruments are seeded through the reference index pipeline.
    """
    repo = DashboardRepository(db)
    instruments = repo.get_etf_instruments()

    indices = [
        ReferenceIndexItem(
            ticker=inst.yfinance_ticker or inst.ticker,
            name=inst.name,
            instrument_type=inst.instrument_type,
        )
        for inst in instruments
    ]
    return ReferenceIndicesResponse(indices=indices, total=len(indices))
