"""
Schemas Package
===============

This package contains Pydantic models for request/response validation and serialization.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# Import base schemas
from app.schemas._shared import CamelCaseModel, StrFromUUID

# Import backtest schemas
from app.schemas.backtest.backtest import (
    BacktestRequest as BacktestRequest,
)
from app.schemas.backtest.backtest import (
    BacktestRunListResponse as BacktestRunListResponse,
)
from app.schemas.backtest.backtest import (
    BacktestRunResponse as BacktestRunResponse,
)

# Import factor schemas
from app.schemas.factors.factors import (
    FactorComputeRequest as FactorComputeRequest,
)
from app.schemas.factors.factors import (
    FactorScoreListResponse as FactorScoreListResponse,
)
from app.schemas.factors.factors import (
    FactorScoreResponse as FactorScoreResponse,
)
from app.schemas.factors.factors import (
    FactorValidateRequest as FactorValidateRequest,
)
from app.schemas.factors.factors import (
    FactorValidationReportResponse as FactorValidationReportResponse,
)

# Import macro regime schemas
from app.schemas.macro.macro_regime import (
    MacroCalibrationResponse as MacroCalibrationResponse,
)
from app.schemas.macro.macro_regime import (
    MacroNewsSummarizeJobResponse as MacroNewsSummarizeJobResponse,
)
from app.schemas.macro.macro_regime import (
    MacroNewsSummarizeProgress as MacroNewsSummarizeProgress,
)
from app.schemas.macro.macro_regime import (
    MacroNewsSummarizeRequest as MacroNewsSummarizeRequest,
)
from app.schemas.macro.macro_regime import (
    MacroNewsSummaryResponse as MacroNewsSummaryResponse,
)

# Import yfinance data schemas
from app.schemas.market_data.yfinance_data import (
    AnalystPriceTargetResponse as AnalystPriceTargetResponse,
)
from app.schemas.market_data.yfinance_data import (
    AnalystRecommendationResponse as AnalystRecommendationResponse,
)
from app.schemas.market_data.yfinance_data import (
    DividendResponse as DividendResponse,
)
from app.schemas.market_data.yfinance_data import (
    FinancialStatementResponse as FinancialStatementResponse,
)
from app.schemas.market_data.yfinance_data import (
    InsiderTransactionResponse as InsiderTransactionResponse,
)
from app.schemas.market_data.yfinance_data import (
    InstitutionalHolderResponse as InstitutionalHolderResponse,
)
from app.schemas.market_data.yfinance_data import (
    MutualFundHolderResponse as MutualFundHolderResponse,
)
from app.schemas.market_data.yfinance_data import (
    PriceHistoryResponse as PriceHistoryResponse,
)
from app.schemas.market_data.yfinance_data import (
    StockSplitResponse as StockSplitResponse,
)
from app.schemas.market_data.yfinance_data import (
    TickerNewsResponse as TickerNewsResponse,
)
from app.schemas.market_data.yfinance_data import (
    TickerProfileResponse as TickerProfileResponse,
)
from app.schemas.market_data.yfinance_data import (
    YFinanceFetchJobResponse as YFinanceFetchJobResponse,
)
from app.schemas.market_data.yfinance_data import (
    YFinanceFetchProgress as YFinanceFetchProgress,
)
from app.schemas.market_data.yfinance_data import (
    YFinanceFetchRequest as YFinanceFetchRequest,
)
from app.schemas.market_data.yfinance_data import (
    YFinanceSingleFetchRequest as YFinanceSingleFetchRequest,
)
from app.schemas.market_data.yfinance_data import (
    YFinanceSingleFetchResponse as YFinanceSingleFetchResponse,
)

# Import optimization schemas
from app.schemas.optimization.optimization import (
    OptimizationRunListResponse as OptimizationRunListResponse,
)
from app.schemas.optimization.optimization import (
    OptimizationRunResponse as OptimizationRunResponse,
)
from app.schemas.optimization.optimization import (
    OptimizeRequest as OptimizeRequest,
)

# Import portfolio schemas
from app.schemas.portfolio.portfolio import (
    BrokerAccountResponse as BrokerAccountResponse,
)
from app.schemas.portfolio.portfolio import (
    BrokerPositionResponse as BrokerPositionResponse,
)
from app.schemas.portfolio.portfolio import (
    PortfolioCreate as PortfolioCreate,
)
from app.schemas.portfolio.portfolio import (
    PortfolioListResponse as PortfolioListResponse,
)
from app.schemas.portfolio.portfolio import (
    PortfolioResponse as PortfolioResponse,
)
from app.schemas.portfolio.portfolio import (
    SnapshotCreate as SnapshotCreate,
)
from app.schemas.portfolio.portfolio import (
    SnapshotListResponse as SnapshotListResponse,
)
from app.schemas.portfolio.portfolio import (
    SnapshotResponse as SnapshotResponse,
)
from app.schemas.portfolio.portfolio import (
    SyncJobResponse as SyncJobResponse,
)
from app.schemas.portfolio.portfolio import (
    SyncProgressResponse as SyncProgressResponse,
)

# Import rebalancing schemas
from app.schemas.rebalancing.rebalancing import (
    RebalancingPolicyCreate as RebalancingPolicyCreate,
)
from app.schemas.rebalancing.rebalancing import (
    RebalancingPolicyListResponse as RebalancingPolicyListResponse,
)
from app.schemas.rebalancing.rebalancing import (
    RebalancingPolicyResponse as RebalancingPolicyResponse,
)

# Import risk schemas
from app.schemas.risk.risk import (
    RiskLimitCreate as RiskLimitCreate,
)
from app.schemas.risk.risk import (
    RiskLimitListResponse as RiskLimitListResponse,
)
from app.schemas.risk.risk import (
    RiskLimitResponse as RiskLimitResponse,
)

# Import stress scenarios schemas
from app.schemas.scenarios.stress_scenarios import (
    StressScenarioItem as StressScenarioItem,
)
from app.schemas.scenarios.stress_scenarios import (
    StressScenarioRequest as StressScenarioRequest,
)
from app.schemas.scenarios.stress_scenarios import (
    StressScenarioResponse as StressScenarioResponse,
)

# Import trading212 schemas
from app.schemas.universe.trading212 import (
    BuildJobResponse as BuildJobResponse,
)
from app.schemas.universe.trading212 import (
    BuildProgressResponse as BuildProgressResponse,
)
from app.schemas.universe.trading212 import (
    BuildResultResponse as BuildResultResponse,
)
from app.schemas.universe.trading212 import (
    CacheStatsResponse as CacheStatsResponse,
)
from app.schemas.universe.trading212 import (
    ExchangeResponse as ExchangeResponse,
)
from app.schemas.universe.trading212 import (
    InstrumentListResponse as InstrumentListResponse,
)
from app.schemas.universe.trading212 import (
    InstrumentResponse as InstrumentResponse,
)
from app.schemas.universe.trading212 import (
    UniverseBuildRequest as UniverseBuildRequest,
)
from app.schemas.universe.trading212 import (
    UniverseStatsResponse as UniverseStatsResponse,
)

# Import LLM moments schemas
from app.schemas.views.llm_moments import (
    AdaptFactorWeightsRequest as AdaptFactorWeightsRequest,
)
from app.schemas.views.llm_moments import (
    AdaptFactorWeightsResponse as AdaptFactorWeightsResponse,
)
from app.schemas.views.llm_moments import (
    CalibrateDeltaRequest as CalibrateDeltaRequest,
)
from app.schemas.views.llm_moments import (
    CalibrateDeltaResponse as CalibrateDeltaResponse,
)
from app.schemas.views.llm_moments import (
    SelectCovRegimeRequest as SelectCovRegimeRequest,
)
from app.schemas.views.llm_moments import (
    SelectCovRegimeResponse as SelectCovRegimeResponse,
)

# Import views schemas
from app.schemas.views.views import (
    AssetViewResponse as AssetViewResponse,
)
from app.schemas.views.views import (
    ExpertViewSummary as ExpertViewSummary,
)
from app.schemas.views.views import (
    GenerateViewsRequest as GenerateViewsRequest,
)
from app.schemas.views.views import (
    GenerateViewsResponse as GenerateViewsResponse,
)
from app.schemas.views.views import (
    ICHistory as ICHistory,
)
from app.schemas.views.views import (
    OpinionPoolRequest as OpinionPoolRequest,
)
from app.schemas.views.views import (
    OpinionPoolResponse as OpinionPoolResponse,
)

# ===========================
# Utility
# ===========================


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# ===========================
# Message/Echo Schemas
# ===========================


class MessageResponse(BaseModel):
    """Simple message response"""

    message: str


class EchoRequest(BaseModel):
    """Echo request body"""

    message: str = Field(..., description="Message to echo back")
    metadata: dict | None = Field(default=None, description="Optional metadata")


class EchoResponse(BaseModel):
    """Echo response"""

    echo: str
    received_at: datetime = Field(default_factory=utc_now)
    metadata: dict | None = None


# ===========================
# Item CRUD Schemas
# ===========================


class ItemBase(BaseModel):
    """Base item schema"""

    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    description: str | None = Field(
        default=None, max_length=500, description="Item description"
    )
    price: float | None = Field(default=None, ge=0, description="Item price")
    is_active: bool = Field(default=True, description="Whether item is active")


class ItemCreate(ItemBase):
    """Schema for creating an item"""

    pass


class ItemUpdate(BaseModel):
    """Schema for updating an item (all fields optional)"""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    price: float | None = Field(default=None, ge=0)
    is_active: bool | None = None


class ItemResponse(ItemBase):
    """Schema for item response"""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Item ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ItemListResponse(BaseModel):
    """Schema for list of items response"""

    items: list[ItemResponse]
    total: int = Field(..., description="Total number of items")


# ===========================
# Generic API Schemas
# ===========================


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str
    detail: str | None = None
    code: str | None = None


class SuccessResponse(BaseModel):
    """Standard success response"""

    success: bool = True
    message: str | None = None
    data: Any | None = None


# Export all schemas
__all__ = [
    # llm moments
    "AdaptFactorWeightsRequest",
    "AdaptFactorWeightsResponse",
    # yfinance
    "AnalystPriceTargetResponse",
    "AnalystRecommendationResponse",
    # views
    "AssetViewResponse",
    # backtest
    "BacktestRequest",
    "BacktestRunListResponse",
    "BacktestRunResponse",
    # portfolio
    "BrokerAccountResponse",
    "BrokerPositionResponse",
    # trading212
    "BuildJobResponse",
    "BuildProgressResponse",
    "BuildResultResponse",
    "CacheStatsResponse",
    "CalibrateDeltaRequest",
    "CalibrateDeltaResponse",
    # base
    "CamelCaseModel",
    "DividendResponse",
    "EchoRequest",
    "EchoResponse",
    "ErrorResponse",
    "ExchangeResponse",
    "ExpertViewSummary",
    # factors
    "FactorComputeRequest",
    "FactorScoreListResponse",
    "FactorScoreResponse",
    "FactorValidateRequest",
    "FactorValidationReportResponse",
    "FinancialStatementResponse",
    "GenerateViewsRequest",
    "GenerateViewsResponse",
    "ICHistory",
    "InsiderTransactionResponse",
    "InstitutionalHolderResponse",
    "InstrumentListResponse",
    "InstrumentResponse",
    "ItemBase",
    "ItemCreate",
    "ItemListResponse",
    "ItemResponse",
    "ItemUpdate",
    # macro regime
    "MacroCalibrationResponse",
    "MacroNewsSummarizeJobResponse",
    "MacroNewsSummarizeProgress",
    "MacroNewsSummarizeRequest",
    "MacroNewsSummaryResponse",
    # inline schemas
    "MessageResponse",
    "MutualFundHolderResponse",
    "OpinionPoolRequest",
    "OpinionPoolResponse",
    "OptimizationRunListResponse",
    "OptimizationRunResponse",
    # optimization
    "OptimizeRequest",
    "PortfolioCreate",
    "PortfolioListResponse",
    "PortfolioResponse",
    "PriceHistoryResponse",
    # rebalancing
    "RebalancingPolicyCreate",
    "RebalancingPolicyListResponse",
    "RebalancingPolicyResponse",
    # risk
    "RiskLimitCreate",
    "RiskLimitListResponse",
    "RiskLimitResponse",
    "SelectCovRegimeRequest",
    "SelectCovRegimeResponse",
    "SnapshotCreate",
    "SnapshotListResponse",
    "SnapshotResponse",
    "StockSplitResponse",
    "StrFromUUID",
    # stress scenarios
    "StressScenarioItem",
    "StressScenarioRequest",
    "StressScenarioResponse",
    "SuccessResponse",
    "SyncJobResponse",
    "SyncProgressResponse",
    "TickerNewsResponse",
    "TickerProfileResponse",
    "UniverseBuildRequest",
    "UniverseStatsResponse",
    "YFinanceFetchJobResponse",
    "YFinanceFetchProgress",
    "YFinanceFetchRequest",
    "YFinanceSingleFetchRequest",
    "YFinanceSingleFetchResponse",
    "utc_now",
]
