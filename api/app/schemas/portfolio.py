"""Pydantic schemas for portfolio CRUD and broker sync endpoints."""

import uuid
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _coerce_uuid(v: Any) -> str:
    """Coerce uuid.UUID to str for Pydantic v2 from_attributes mode."""
    if isinstance(v, uuid.UUID):
        return str(v)
    return v


StrFromUUID = Annotated[str, BeforeValidator(_coerce_uuid)]


# ------------------------------------------------------------------
# Portfolio CRUD
# ------------------------------------------------------------------


class PortfolioCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: str | None = None
    currency: str = Field(default="EUR", max_length=10)
    benchmark_ticker: str = Field(default="SPY", max_length=50)


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrFromUUID
    name: str
    description: str | None
    currency: str
    benchmark_ticker: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class PortfolioListResponse(BaseModel):
    items: list[PortfolioResponse]
    total: int


# ------------------------------------------------------------------
# Snapshots
# ------------------------------------------------------------------


class SnapshotCreate(BaseModel):
    snapshot_date: date
    snapshot_type: str = Field(default="manual", pattern=r"^(optimization|rebalance|manual)$")
    weights: dict[str, float]
    sector_mapping: dict[str, str] | None = None
    summary: dict[str, Any] | None = None
    optimizer_config: dict[str, Any] | None = None
    turnover: float | None = None


class SnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrFromUUID
    portfolio_id: StrFromUUID
    snapshot_date: date
    snapshot_type: str
    weights: dict[str, float]
    sector_mapping: dict[str, str] | None
    summary: dict[str, Any] | None
    optimizer_config: dict[str, Any] | None
    turnover: float | None
    holding_count: int
    created_at: datetime


class SnapshotListResponse(BaseModel):
    items: list[SnapshotResponse]
    total: int


# ------------------------------------------------------------------
# Broker positions
# ------------------------------------------------------------------


class BrokerPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrFromUUID
    ticker: str
    yfinance_ticker: str | None
    name: str | None
    quantity: float
    average_price: float
    current_price: float | None
    ppl: float | None
    fx_ppl: float | None
    initial_fill_date: date | None
    synced_at: datetime


# ------------------------------------------------------------------
# Broker account
# ------------------------------------------------------------------


class BrokerAccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrFromUUID
    total: float
    free: float
    invested: float
    blocked: float | None
    result: float | None
    currency: str
    synced_at: datetime


# ------------------------------------------------------------------
# Sync job
# ------------------------------------------------------------------


class SyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class SyncProgressResponse(BaseModel):
    job_id: str
    status: str
    current: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
