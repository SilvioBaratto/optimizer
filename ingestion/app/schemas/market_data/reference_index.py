"""Pydantic schemas for reference-index seeding endpoints."""

from datetime import date

from pydantic import BaseModel, Field

from app.schemas._shared import AsyncJobCreateResponse, AsyncJobProgress, CamelCaseModel


class ReferenceIndexSeedRequest(BaseModel):
    """Request body for the reference-index seed endpoint."""

    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "yfinance tickers to seed as reference indices (e.g. ['SPY', 'QQQ'])."
        ),
    )


# Reuse base schemas; no extra fields required for this domain.
ReferenceIndexSeedJobResponse = AsyncJobCreateResponse
ReferenceIndexSeedProgress = AsyncJobProgress


class ReferenceIndexStatusItem(CamelCaseModel):
    """Per-ticker coverage diagnostic for the status endpoint."""

    ticker: str
    instrument_exists: bool
    price_rows: int
    latest_price_date: date | None
    is_stale: bool


class ReferenceIndexStatusResponse(CamelCaseModel):
    """Response for ``GET /reference-indices/status``.

    Surfaces which configured benchmarks are missing, stale, or healthy so
    operators can detect gaps without triggering a fetch.
    """

    items: list[ReferenceIndexStatusItem]
    total: int
    healthy_count: int
    missing_count: int
    stale_count: int
    stale_days: int


class ReferenceIndexConfiguredResponse(CamelCaseModel):
    """Response for ``GET /reference-indices/configured``.

    Returns the union of ``settings.benchmark_tickers`` and every distinct
    ``portfolios.benchmark_ticker`` value, so the daily scheduler refreshes
    every benchmark a user portfolio actually depends on.
    """

    tickers: list[str]
    settings_tickers: list[str]
    portfolio_tickers: list[str]
