"""Pydantic v2 schemas for yfinance data endpoints."""

import uuid
from datetime import date, datetime
from typing import Any, Literal

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
    errors: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class YFinanceSingleFetchResponse(BaseModel):
    """Response for a single-ticker fetch."""

    ticker: str
    instrument_id: str
    counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Read response schemas
# ---------------------------------------------------------------------------


class TickerProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    symbol: str | None = None
    short_name: str | None = None
    long_name: str | None = None
    isin: str | None = None
    exchange: str | None = None
    quote_type: str | None = None
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    website: str | None = None
    long_business_summary: str | None = None
    market_cap: int | None = None
    enterprise_value: int | None = None
    shares_outstanding: int | None = None
    float_shares: int | None = None
    current_price: float | None = None
    previous_close: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    fifty_day_average: float | None = None
    two_hundred_day_average: float | None = None
    average_volume: int | None = None
    beta: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    trailing_eps: float | None = None
    forward_eps: float | None = None
    price_to_sales_trailing_12months: float | None = None
    price_to_book: float | None = None
    enterprise_to_revenue: float | None = None
    enterprise_to_ebitda: float | None = None
    peg_ratio: float | None = None
    book_value: float | None = None
    profit_margins: float | None = None
    operating_margins: float | None = None
    gross_margins: float | None = None
    return_on_assets: float | None = None
    return_on_equity: float | None = None
    total_revenue: int | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    ebitda: int | None = None
    free_cashflow: int | None = None
    operating_cashflow: int | None = None
    total_debt: int | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    dividend_rate: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    recommendation_key: str | None = None
    recommendation_mean: float | None = None
    full_time_employees: int | None = None
    created_at: datetime
    updated_at: datetime


class PriceHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None
    dividends: float | None = None
    stock_splits: float | None = None
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
    value: float | None = None
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
    strong_buy: int | None = None
    buy: int | None = None
    hold: int | None = None
    sell: int | None = None
    strong_sell: int | None = None
    created_at: datetime
    updated_at: datetime


class AnalystPriceTargetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    current: float | None = None
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    median: float | None = None
    created_at: datetime
    updated_at: datetime


class InstitutionalHolderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    holder_name: str
    date_reported: date | None = None
    shares: int | None = None
    value: int | None = None
    pct_held: float | None = None
    created_at: datetime
    updated_at: datetime


class MutualFundHolderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    holder_name: str
    date_reported: date | None = None
    shares: int | None = None
    value: int | None = None
    pct_held: float | None = None
    created_at: datetime
    updated_at: datetime


class InsiderTransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    insider_name: str
    position: str | None = None
    transaction_type: str
    shares: int | None = None
    value: int | None = None
    start_date: date
    ownership: str | None = None
    created_at: datetime
    updated_at: datetime


class TickerNewsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    instrument_id: uuid.UUID
    news_uuid: str
    title: str | None = None
    publisher: str | None = None
    link: str | None = None
    publish_time: datetime | None = None
    news_type: str | None = None
    related_tickers: list[str] | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("related_tickers", mode="before")
    @classmethod
    def _parse_related_tickers(cls, v: Any) -> list[str] | None:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v
