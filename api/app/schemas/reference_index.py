"""Pydantic schemas for reference-index seeding endpoints."""

from pydantic import BaseModel, Field

from app.schemas.base_job import AsyncJobCreateResponse, AsyncJobProgress


class ReferenceIndexSeedRequest(BaseModel):
    """Request body for the reference-index seed endpoint."""

    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "yfinance tickers to seed as reference indices "
            "(e.g. ['SPY', 'QQQ'])."
        ),
    )


# Reuse base schemas; no extra fields required for this domain.
ReferenceIndexSeedJobResponse = AsyncJobCreateResponse
ReferenceIndexSeedProgress = AsyncJobProgress
