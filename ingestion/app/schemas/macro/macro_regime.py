"""Pydantic v2 schemas for macro regime data endpoints."""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import AsyncJobProgress


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


class FredFetchRequest(BaseModel):
    """Request body for FRED time-series fetch."""

    series_ids: list[str] | None = Field(
        default=None,
        description="FRED series IDs to fetch. None means all configured series.",
    )
    incremental: bool = Field(
        default=True,
        description=(
            "When True, re-fetch a trailing window behind the last stored date "
            "(to pick up restated values) instead of the full history."
        ),
    )


class MacroFetchProgress(AsyncJobProgress):
    """Progress info for a macro fetch background job."""

    current_country: str = ""


class EconomicIndicatorResponse(BaseModel):
    """ORM-backed response schema for a country-level economic indicator snapshot."""

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
    """ORM-backed response schema for a single Trading Economics indicator."""

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
    """ORM-backed response schema for a country bond yield record."""

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


class MacroNewsFetchRequest(BaseModel):
    """Request body for macro news fetch."""

    fetch_full_content: bool = Field(
        default=True,
        description="Whether to scrape full article content.",
    )
    max_articles: int = Field(
        default=100,
        description="Maximum number of articles to fetch.",
    )


class CountryMacroSummary(BaseModel):
    """Aggregated macro data for a single country."""

    country: str
    economic_indicators: list[EconomicIndicatorResponse] = Field(default_factory=list)
    te_indicators: list[TradingEconomicsIndicatorResponse] = Field(default_factory=list)
    bond_yields: list[BondYieldResponse] = Field(default_factory=list)
