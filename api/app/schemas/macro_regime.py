"""Pydantic v2 schemas for macro regime data endpoints."""

import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class MacroFetchJobResponse(BaseModel):
    """Returned when a background macro fetch job is created."""

    job_id: str
    status: str
    message: str


class MacroFetchProgress(BaseModel):
    """Progress info for a macro fetch background job."""

    job_id: str
    status: str  # pending | running | completed | failed
    current: int = 0
    total: int = 0
    current_country: str = ""
    errors: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class EconomicIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    source: str
    gdp_growth_qq: float | None = None
    industrial_production: float | None = None
    unemployment: float | None = None
    consumer_prices: float | None = None
    deficit: float | None = None
    debt: float | None = None
    st_rate: float | None = None
    lt_rate: float | None = None
    last_inflation: float | None = None
    inflation_6m: float | None = None
    inflation_10y_avg: float | None = None
    gdp_growth_6m: float | None = None
    earnings_12m: float | None = None
    eps_expected_12m: float | None = None
    peg_ratio: float | None = None
    st_rate_forecast: float | None = None
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


class CountryMacroSummary(BaseModel):
    """Aggregated macro data for a single country."""

    country: str
    economic_indicators: list[EconomicIndicatorResponse] = Field(default_factory=list)
    te_indicators: list[TradingEconomicsIndicatorResponse] = Field(default_factory=list)
    bond_yields: list[BondYieldResponse] = Field(default_factory=list)
