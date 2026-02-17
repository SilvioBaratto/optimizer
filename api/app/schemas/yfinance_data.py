"""Pydantic v2 schemas for yfinance data endpoints."""

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class YFinanceFetchRequest(BaseModel):
    """Request body for bulk yfinance data fetch."""
    max_workers: int = Field(default=4, ge=1, le=20, description="Parallel workers")
    period: str = Field(default="5y", description="Price history period")
    mode: Literal["full", "incremental"] = Field(
        default="incremental",
        description="'incremental' skips fresh data; 'full' re-downloads everything",
    )


class YFinanceSingleFetchRequest(BaseModel):
    """Request body for single-ticker fetch."""
    period: str = Field(default="5y", description="Price history period")
    mode: Literal["full", "incremental"] = Field(
        default="incremental",
        description="'incremental' skips fresh data; 'full' re-downloads everything",
    )


# ---------------------------------------------------------------------------
# Job progress schemas
# ---------------------------------------------------------------------------

class YFinanceFetchJobResponse(BaseModel):
    """Returned when a bulk fetch job is created."""
    job_id: str
    status: str
    message: str


class YFinanceFetchProgress(BaseModel):
    """Progress info for a bulk fetch job."""
    job_id: str
    status: str  # pending | running | completed | failed
    current: int = 0
    total: int = 0
    current_ticker: str = ""
    errors: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class YFinanceSingleFetchResponse(BaseModel):
    """Response for a single-ticker fetch."""
    ticker: str
    instrument_id: str
    counts: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Read response schemas
# ---------------------------------------------------------------------------

class TickerProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    symbol: Optional[str] = None
    short_name: Optional[str] = None
    long_name: Optional[str] = None
    isin: Optional[str] = None
    exchange: Optional[str] = None
    quote_type: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None
    long_business_summary: Optional[str] = None
    market_cap: Optional[int] = None
    enterprise_value: Optional[int] = None
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_day_average: Optional[float] = None
    two_hundred_day_average: Optional[float] = None
    average_volume: Optional[int] = None
    beta: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    price_to_sales_trailing_12months: Optional[float] = None
    price_to_book: Optional[float] = None
    enterprise_to_revenue: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    peg_ratio: Optional[float] = None
    book_value: Optional[float] = None
    profit_margins: Optional[float] = None
    operating_margins: Optional[float] = None
    gross_margins: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    total_revenue: Optional[int] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    ebitda: Optional[int] = None
    free_cashflow: Optional[int] = None
    operating_cashflow: Optional[int] = None
    total_debt: Optional[int] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    recommendation_key: Optional[str] = None
    recommendation_mean: Optional[float] = None
    full_time_employees: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class PriceHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    dividends: Optional[float] = None
    stock_splits: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class FinancialStatementResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    statement_type: str
    period_type: str
    period_date: date
    line_item: str
    value: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class DividendResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    date: date
    amount: float
    created_at: datetime
    updated_at: datetime


class StockSplitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    date: date
    ratio: float
    created_at: datetime
    updated_at: datetime


class AnalystRecommendationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    period: str
    strong_buy: Optional[int] = None
    buy: Optional[int] = None
    hold: Optional[int] = None
    sell: Optional[int] = None
    strong_sell: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class AnalystPriceTargetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    current: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class InstitutionalHolderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    holder_name: str
    date_reported: Optional[date] = None
    shares: Optional[int] = None
    value: Optional[int] = None
    pct_held: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class MutualFundHolderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    holder_name: str
    date_reported: Optional[date] = None
    shares: Optional[int] = None
    value: Optional[int] = None
    pct_held: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class InsiderTransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    insider_name: str
    position: Optional[str] = None
    transaction_type: str
    shares: Optional[int] = None
    value: Optional[int] = None
    start_date: date
    ownership: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TickerNewsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    news_uuid: str
    title: Optional[str] = None
    publisher: Optional[str] = None
    link: Optional[str] = None
    publish_time: Optional[datetime] = None
    news_type: Optional[str] = None
    related_tickers: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("related_tickers", mode="before")
    @classmethod
    def _parse_related_tickers(cls, v: Any) -> Optional[List[str]]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v
