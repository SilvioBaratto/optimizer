"""Emit deterministic JSON-schema snapshots for API response models.

Each domain maps to its primary read-response schemas. We call
``model_json_schema(by_alias=True)`` *uniformly*: ``CamelCaseModel`` subclasses
emit camelCase property keys via their alias generator, while plain
``BaseModel`` subclasses (portfolio, jobs, reports, market_data, and the
snake_case responses in macro/universe) have ``alias == field-name`` and so emit
snake_case. No per-domain casing branch is needed -- the alias generator decides.

The committed snapshots are the structural contract both stacks consume:
``api/tests/contract/snapshots/`` (Python parity test) and
``frontend/src/testing/contract-snapshots/`` (frontend parity specs).

Run standalone to regenerate both locations::

    python api/tests/contract/emit_schemas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``import app`` resolve when run as a standalone script from any cwd.
sys.path.insert(0, str(Path(__file__).parents[2]))

from pydantic import BaseModel

from app.schemas.attribution.attribution import (
    BrinsonResponse,
    FactorAttributionResponse,
)
from app.schemas.backtest.backtest import (
    BacktestProgressResponse,
    BacktestRunListResponse,
    BacktestRunResponse,
)
from app.schemas.dashboard.dashboard import (
    AllocationResponse,
    AssetClassReturnsResponse,
    DriftResponse,
    EquityCurveResponse,
    MarketSnapshotResponse,
    PerformanceMetricsResponse,
)
from app.schemas.factors.factors import (
    FactorCompositeScoreResponse,
    FactorScoreListResponse,
    FactorScoreResponse,
    FactorSelectResponse,
    FactorValidationReportResponse,
)
from app.schemas.jobs.jobs import JobListResponse, JobSummary
from app.schemas.macro.macro_regime import MacroCalibrationResponse
from app.schemas.market_data.yfinance_data import (
    PriceHistoryResponse,
    TickerProfileResponse,
)
from app.schemas.optimization.optimization import (
    OptimizationRunListResponse,
    OptimizationRunResponse,
)
from app.schemas.pipeline_builder.pipeline_builder import (
    CreateSessionResponse,
    StepPollResponse,
    StepRunResponse,
)
from app.schemas.portfolio.drift import (
    DriftResponse as PortfolioDriftResponse,
)
from app.schemas.portfolio.portfolio import (
    PortfolioListResponse,
    SnapshotListResponse,
)
from app.schemas.rebalancing.rebalancing import (
    RebalanceDecideResponse,
    RebalancePreviewResponse,
    RebalancingPolicyListResponse,
)
from app.schemas.reports.reports import ReportJobCreateResponse
from app.schemas.risk.risk import RiskLimitListResponse
from app.schemas.risk.risk_analytics import (
    ConcentrationResponse,
    CorrelationResponse,
    LiquidityResponse,
    VarResponse,
)
from app.schemas.scenarios.stress_scenarios import (
    StressScenarioItem,
    StressScenarioResponse,
)
from app.schemas.universe.trading212 import (
    InstrumentListResponse,
    UniverseStatsResponse,
)
from app.schemas.universe.universe_screen import UniverseScreenResponse
from app.schemas.views.views import (
    EntropyPoolingResponse,
    GenerateViewsResponse,
    OpinionPoolResponse,
)

# Single source of truth: domain -> response model classes. Reused by the test.
DOMAIN_SCHEMAS: dict[str, list[type[BaseModel]]] = {
    "dashboard": [
        DriftResponse,
        MarketSnapshotResponse,
        PerformanceMetricsResponse,
        EquityCurveResponse,
        AllocationResponse,
        AssetClassReturnsResponse,
    ],
    "optimization": [OptimizationRunResponse, OptimizationRunListResponse],
    "backtest": [
        BacktestRunResponse,
        BacktestRunListResponse,
        BacktestProgressResponse,
    ],
    "risk": [
        VarResponse,
        CorrelationResponse,
        ConcentrationResponse,
        LiquidityResponse,
        RiskLimitListResponse,
    ],
    "factors": [
        FactorScoreResponse,
        FactorScoreListResponse,
        FactorValidationReportResponse,
        FactorCompositeScoreResponse,
        FactorSelectResponse,
    ],
    "rebalancing": [
        RebalanceDecideResponse,
        RebalancePreviewResponse,
        RebalancingPolicyListResponse,
    ],
    "portfolio": [
        PortfolioListResponse,
        SnapshotListResponse,
        PortfolioDriftResponse,
    ],
    "universe": [
        UniverseScreenResponse,
        InstrumentListResponse,
        UniverseStatsResponse,
    ],
    "attribution": [BrinsonResponse, FactorAttributionResponse],
    "views": [GenerateViewsResponse, OpinionPoolResponse, EntropyPoolingResponse],
    "macro": [MacroCalibrationResponse],
    "market_data": [TickerProfileResponse, PriceHistoryResponse],
    "scenarios": [StressScenarioResponse, StressScenarioItem],
    "reports": [ReportJobCreateResponse],
    "jobs": [JobListResponse, JobSummary],
    "pipeline_builder": [
        CreateSessionResponse,
        StepPollResponse,
        StepRunResponse,
    ],
}

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
FRONTEND_SNAPSHOT_DIR = (
    Path(__file__).parents[3] / "frontend" / "src" / "testing" / "contract-snapshots"
)


def schema_for(model: type[BaseModel]) -> dict:
    """Return the by-alias JSON schema for one model (casing from its base)."""
    model.model_rebuild()  # resolve PEP 563 forward refs; no-op if already built
    return model.model_json_schema(by_alias=True)


def snapshot_for_domain(models: list[type[BaseModel]]) -> dict:
    """Map each model's class name to its JSON schema."""
    return {model.__name__: schema_for(model) for model in models}


def render(snapshot: dict) -> str:
    """Serialise a snapshot deterministically (sorted keys, trailing newline)."""
    return json.dumps(snapshot, sort_keys=True, indent=2) + "\n"


def write_snapshots(target_dirs: list[Path]) -> None:
    """Write every domain snapshot into each target directory."""
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    for domain, models in DOMAIN_SCHEMAS.items():
        content = render(snapshot_for_domain(models))
        for directory in target_dirs:
            (directory / f"{domain}.json").write_text(content)


def main() -> None:
    write_snapshots([SNAPSHOT_DIR, FRONTEND_SNAPSHOT_DIR])
    print(f"Wrote {len(DOMAIN_SCHEMAS)} snapshots to {SNAPSHOT_DIR} and {FRONTEND_SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
