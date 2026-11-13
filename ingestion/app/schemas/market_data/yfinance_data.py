"""Pydantic v2 schemas for yfinance data endpoints."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas._shared import AsyncJobProgress

# Mirrors what yfinance accepts: a positive count in days/weeks/months/years, or
# one of Yahoo's named ranges. Anything else makes it return an empty frame
# instead of raising, which a sweep would record as a zero-row success.
PERIOD_PATTERN = r"^([1-9]\d*(d|wk|mo|y)|ytd|max)$"


class YFinanceFetchRequest(BaseModel):
    """Request body for bulk yfinance data fetch."""

    workers: int = Field(
        default=1, ge=1, le=16, description="Parallel workers (1 = serial)"
    )
    period: str = Field(
        default="5y",
        pattern=PERIOD_PATTERN,
        description=(
            "Minimum price history window; a full fetch reaches further back "
            "when older bars are stored"
        ),
    )
    mode: Literal["full", "incremental"] = Field(
        default="incremental",
        description=(
            "'incremental' skips fresh data; 'full' re-downloads every category and "
            "the whole stored price series"
        ),
    )


class YFinanceSingleFetchRequest(BaseModel):
    """Request body for single-ticker fetch."""

    period: str = Field(
        default="5y",
        pattern=PERIOD_PATTERN,
        description=(
            "Minimum price history window; a full fetch reaches further back "
            "when older bars are stored"
        ),
    )
    mode: Literal["full", "incremental"] = Field(
        default="incremental",
        description=(
            "'incremental' skips fresh data; 'full' re-downloads every category and "
            "the whole stored price series"
        ),
    )


class YFinanceFetchProgress(AsyncJobProgress):
    """Progress info for a bulk fetch job."""

    current_ticker: str = ""
