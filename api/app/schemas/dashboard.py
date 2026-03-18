"""Pydantic v2 schemas for dashboard endpoints (camelCase serialization)."""

from datetime import date as Date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    """Base model that serializes field names to camelCase."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class KpiItem(CamelCaseModel):
    """Single KPI card for the dashboard strip."""

    label: str
    value: float
    format: Literal["percent", "currency", "ratio", "number"]
    change: float
    change_label: str
    sparkline: list[float]


class PerformanceMetricsResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/performance-metrics."""

    kpis: list[KpiItem]
    nav: float
    nav_change_pct: float


class EquityCurvePoint(CamelCaseModel):
    """Single daily point on the equity curve chart."""

    date: Date
    portfolio: float
    benchmark: float


class EquityCurveResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/equity-curve."""

    points: list[EquityCurvePoint]
    portfolio_total_return: float
    benchmark_total_return: float


class AllocationChild(CamelCaseModel):
    """Leaf node in the sunburst allocation tree (single ticker)."""

    name: str
    value: float


class AllocationNode(CamelCaseModel):
    """Sector-level node with ticker children for sunburst chart."""

    name: str
    value: float
    children: list[AllocationChild]


class AllocationResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/allocation."""

    nodes: list[AllocationNode]
    total_positions: int
    total_sectors: int


class DriftEntry(CamelCaseModel):
    """Single holding in the drift analysis table."""

    ticker: str
    name: str | None
    target: float
    actual: float
    drift: float
    breached: bool


class DriftResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/drift."""

    entries: list[DriftEntry]
    total_drift: float
    breached_count: int
    threshold: float


class ActivityItem(CamelCaseModel):
    """Single event in the portfolio activity feed."""

    id: str
    type: str
    title: str
    description: str | None
    timestamp: datetime


class ActivityFeedResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/activity."""

    items: list[ActivityItem]
    total: int


class MarketSnapshotResponse(CamelCaseModel):
    """Response for GET /market/snapshot."""

    vix: float
    vix_change: float
    sp500_return: float
    ten_year_yield: float
    yield_change: float
    usd_index: float
    usd_change: float
    as_of: datetime


class HmmStateItem(CamelCaseModel):
    """Single HMM state in the regime probability distribution."""

    regime: str
    probability: float


class RegimeModelInfo(CamelCaseModel):
    """Metadata about the fitted HMM model."""

    n_states: int
    last_fitted: datetime


class MarketRegimeResponse(CamelCaseModel):
    """Response for GET /market/regime."""

    current: str
    probability: float
    since: Date
    hmm_states: list[HmmStateItem]
    model_info: RegimeModelInfo


class AssetClassReturnRow(CamelCaseModel):
    """Single sector row for the asset-class returns heatmap."""

    name: str
    one_d: float = Field(alias="1D")
    one_w: float = Field(alias="1W")
    one_m: float = Field(alias="1M")
    ytd: float = Field(alias="YTD")


class AssetClassReturnsResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/asset-class-returns."""

    returns: list[AssetClassReturnRow]
    as_of: Date
