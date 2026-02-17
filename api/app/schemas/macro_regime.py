"""Pydantic v2 schemas for macro regime data endpoints."""

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class MacroFetchRequest(BaseModel):
    """Request body for macro data fetch."""
    countries: Optional[List[str]] = Field(
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
    errors: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class EconomicIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    source: str
    gdp_growth_qq: Optional[float] = None
    industrial_production: Optional[float] = None
    unemployment: Optional[float] = None
    consumer_prices: Optional[float] = None
    deficit: Optional[float] = None
    debt: Optional[float] = None
    st_rate: Optional[float] = None
    lt_rate: Optional[float] = None
    last_inflation: Optional[float] = None
    inflation_6m: Optional[float] = None
    inflation_10y_avg: Optional[float] = None
    gdp_growth_6m: Optional[float] = None
    earnings_12m: Optional[float] = None
    eps_expected_12m: Optional[float] = None
    peg_ratio: Optional[float] = None
    st_rate_forecast: Optional[float] = None
    lt_rate_forecast: Optional[float] = None
    reference_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime


class TradingEconomicsIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    indicator_key: str
    value: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None
    reference: Optional[str] = None
    raw_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class BondYieldResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    country: str
    maturity: str
    yield_value: Optional[float] = None
    day_change: Optional[float] = None
    month_change: Optional[float] = None
    year_change: Optional[float] = None
    reference_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime


class CountryMacroSummary(BaseModel):
    """Aggregated macro data for a single country."""
    country: str
    economic_indicators: List[EconomicIndicatorResponse] = Field(default_factory=list)
    te_indicators: List[TradingEconomicsIndicatorResponse] = Field(default_factory=list)
    bond_yields: List[BondYieldResponse] = Field(default_factory=list)
