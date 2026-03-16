"""Pydantic v2 schemas for macro regime data endpoints."""

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.base_job import AsyncJobCreateResponse, AsyncJobProgress

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class MacroFetchRequest(BaseModel):
    """Request body for macro data fetch."""

    countries: list[str] | None = Field(
        default=None,
        description="List of countries to fetch. None means all portfolio countries.",
    )
    include_bonds: bool = Field(
        default=True,
        description="Whether to include bond yield data from Trading Economics.",
    )
    include_fred: bool = Field(
        default=True,
        description="Whether to fetch FRED time-series data during bulk fetch.",
    )


class FredFetchRequest(BaseModel):
    """Request body for FRED time-series fetch."""

    series_ids: list[str] | None = Field(
        default=None,
        description="FRED series IDs to fetch. None means all configured series.",
    )
    incremental: bool = Field(
        default=True,
        description="When True, only fetch observations newer than last stored date.",
    )


class MacroFetchJobResponse(AsyncJobCreateResponse):
    """Returned when a background macro fetch job is created."""


class MacroFetchProgress(AsyncJobProgress):
    """Progress info for a macro fetch background job."""

    current_country: str = ""


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class EconomicIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    last_inflation: float | None = None
    inflation_6m: float | None = None
    inflation_10y_avg: float | None = None
    gdp_growth_6m: float | None = None
    earnings_12m: float | None = None
    eps_expected_12m: float | None = None
    peg_ratio: float | None = None
    lt_rate_forecast: float | None = None
    reference_date: date | None = None
    created_at: datetime
    updated_at: datetime


class TradingEconomicsIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    indicator_key: str
    value: float | None = None
    previous: float | None = None
    unit: str | None = None
    reference: str | None = None
    raw_name: str | None = None
    created_at: datetime
    updated_at: datetime


class BondYieldResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    maturity: str
    yield_value: float | None = None
    day_change: float | None = None
    month_change: float | None = None
    year_change: float | None = None
    reference_date: date | None = None
    created_at: datetime
    updated_at: datetime


class FredObservationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    series_id: str
    date: date
    value: float | None = None
    created_at: datetime
    updated_at: datetime


class EconomicIndicatorObservationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    date: date
    last_inflation: float | None = None
    inflation_6m: float | None = None
    inflation_10y_avg: float | None = None
    gdp_growth_6m: float | None = None
    earnings_12m: float | None = None
    eps_expected_12m: float | None = None
    peg_ratio: float | None = None
    lt_rate_forecast: float | None = None
    reference_date: date | None = None
    created_at: datetime
    updated_at: datetime


class TradingEconomicsObservationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    indicator_key: str
    date: date
    value: float | None = None
    created_at: datetime
    updated_at: datetime


class BondYieldObservationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    maturity: str
    date: date
    yield_value: float | None = None
    created_at: datetime
    updated_at: datetime


class MacroNewsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    news_id: str
    title: str | None = None
    publisher: str | None = None
    link: str | None = None
    publish_time: datetime | None = None
    source_ticker: str | None = None
    source_query: str | None = None
    themes: str | None = None
    snippet: str | None = None
    full_content: str | None = None
    created_at: datetime
    updated_at: datetime


class MacroNewsFetchRequest(BaseModel):
    """Request body for macro news fetch."""

    fetch_full_content: bool = Field(
        default=False,
        description="Whether to scrape full article content.",
    )
    max_articles: int = Field(
        default=100,
        description="Maximum number of articles to fetch.",
    )


class MacroNewsSummarizeRequest(BaseModel):
    """Request body for news summary generation."""

    countries: list[str] | None = Field(
        default=None,
        description="Countries to summarize. None means all mapped countries.",
    )
    force_refresh: bool = Field(
        default=False,
        description="Bypass the daily cache and re-invoke the LLM for all countries.",
    )


class MacroNewsSummarizeJobResponse(AsyncJobCreateResponse):
    """Returned when a background news summarize job is created."""


class MacroNewsSummarizeProgress(AsyncJobProgress):
    """Progress info for a news summarize background job."""

    current_country: str = ""


class MacroNewsSummaryResponse(BaseModel):
    """Response for a single daily country news summary."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    summary_date: date
    summary: str | None = None
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"] | None = None
    sentiment_score: float | None = None
    article_count: int | None = None
    created_at: datetime
    updated_at: datetime


class MacroCalibrateBatchRequest(BaseModel):
    """Request body for batch macro calibration."""

    countries: list[str] | None = Field(
        default=None,
        description="Countries to calibrate. None means all portfolio countries.",
    )
    force_refresh: bool = Field(
        default=True,
        description="Bypass cached calibration and re-invoke the LLM.",
    )


class MacroCalibrateBatchJobResponse(AsyncJobCreateResponse):
    """Returned when a background batch calibration job is created."""


class MacroCalibrateBatchProgress(AsyncJobProgress):
    """Progress info for a batch calibration background job."""

    current_country: str = ""


class CountryMacroSummary(BaseModel):
    """Aggregated macro data for a single country."""

    country: str
    economic_indicators: list[EconomicIndicatorResponse] = Field(default_factory=list)
    te_indicators: list[TradingEconomicsIndicatorResponse] = Field(default_factory=list)
    bond_yields: list[BondYieldResponse] = Field(default_factory=list)
