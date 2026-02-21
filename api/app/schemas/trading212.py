"""Pydantic v2 schemas for Trading212 universe endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UniverseBuildRequest(BaseModel):
    """Request to trigger a universe build."""

    exchanges: list[str] | None = Field(
        default=None,
        description="Filter to specific exchanges (e.g. ['NYSE', 'NASDAQ']). None = all configured.",
    )
    skip_filters: bool = Field(
        default=False,
        description="Skip quality filters and include all discovered instruments.",
    )
    max_workers: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of concurrent workers for ticker discovery.",
    )


class ExchangeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Exchange UUID")
    name: str = Field(..., description="Exchange name")
    t212_id: int | None = Field(None, description="Trading212 exchange ID")
    created_at: datetime
    updated_at: datetime


class InstrumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Instrument UUID")
    ticker: str = Field(..., description="Trading212 ticker")
    short_name: str = Field(..., description="Short ticker name")
    name: str | None = Field(None, description="Full company name")
    isin: str | None = Field(None, description="ISIN code")
    instrument_type: str | None = Field(
        None, description="Instrument type (STOCK, ETF, etc.)"
    )
    currency_code: str | None = Field(None, description="Trading currency")
    yfinance_ticker: str | None = Field(None, description="Yahoo Finance ticker")
    exchange_name: str | None = Field(None, description="Exchange name")
    created_at: datetime
    updated_at: datetime


class InstrumentListResponse(BaseModel):
    items: list[InstrumentResponse]
    total: int = Field(..., description="Total number of instruments")


class BuildResultResponse(BaseModel):
    exchanges_saved: int = Field(..., description="Number of exchanges saved")
    instruments_saved: int = Field(..., description="Number of instruments saved")
    total_processed: int = Field(0, description="Total instruments processed")
    filter_stats: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Per-filter pass/fail counts"
    )
    errors: list[str] = Field(default_factory=list, description="Error messages")


class BuildJobResponse(BaseModel):
    build_id: str = Field(..., description="Unique build job ID")
    status: str = Field(..., description="pending | running | completed | failed")
    message: str = Field("", description="Status message")


class BuildProgressResponse(BaseModel):
    build_id: str = Field(..., description="Unique build job ID")
    status: str = Field(..., description="pending | running | completed | failed")
    current: int = Field(0, description="Instruments processed so far")
    total: int = Field(0, description="Total instruments to process")
    current_exchange: str = Field("", description="Exchange currently being processed")
    current_stock: str = Field("", description="Stock currently being processed")
    result: BuildResultResponse | None = Field(
        None, description="Final result (when completed)"
    )
    error: str | None = Field(None, description="Error message (when failed)")


class CacheStatsResponse(BaseModel):
    total: int = Field(..., description="Total cached mappings")
    fresh: int = Field(..., description="Fresh (non-expired) mappings")
    expired: int = Field(..., description="Expired mappings")
    cache_file: str = Field(..., description="Cache file path")
    cache_file_size: int = Field(..., description="Cache file size in bytes")


class UniverseStatsResponse(BaseModel):
    exchange_count: int = Field(..., description="Number of exchanges")
    instrument_count: int = Field(..., description="Number of instruments")
