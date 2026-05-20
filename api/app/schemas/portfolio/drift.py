"""Drift API Pydantic v2 contracts — Cycle 2 (issue #779).

Schemas consumed by the Drift API endpoint and the Angular frontend. All
schemas are frozen value objects matching the cycle-1 convention used in
`normalized_position.py`.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.portfolio.diagnostic_entry import DiagnosticEntry
from app.schemas.portfolio.normalized_position import FlagList, NormalizedPosition


class BaseToggle(str, Enum):
    """Deployable-EUR base selector for drift computation."""

    INVESTED = "invested"
    TOTAL = "total"


class TradeAction(str, Enum):
    """Direction of a suggested trade row."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def _validate_unit_interval(value: float, *, field: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must lie in [0.0, 1.0], got {value}")
    return value


def _validate_signed_unit_interval(value: float, *, field: str) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{field} must lie in [-1.0, 1.0], got {value}")
    return value


class TargetWeight(BaseModel):
    """A single (ticker, target_weight) pair from the latest snapshot."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    weight: float

    @field_validator("weight")
    @classmethod
    def _check_weight(cls, value: float) -> float:
        return _validate_unit_interval(value, field="weight")


class DriftRow(BaseModel):
    """Per-ticker drift between current holdings and target."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    current_weight: float
    target_weight: float
    delta_weight: float
    eur_value: float
    flags: FlagList = Field(default_factory=list)

    @field_validator("current_weight")
    @classmethod
    def _check_current_weight(cls, value: float) -> float:
        return _validate_unit_interval(value, field="current_weight")

    @field_validator("target_weight")
    @classmethod
    def _check_target_weight(cls, value: float) -> float:
        return _validate_unit_interval(value, field="target_weight")

    @field_validator("delta_weight")
    @classmethod
    def _check_delta_weight(cls, value: float) -> float:
        return _validate_signed_unit_interval(value, field="delta_weight")


class TradeRow(BaseModel):
    """A single suggested trade in EUR notional + estimated shares."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    action: TradeAction
    delta_eur: float
    est_shares: float
    est_cost_eur: float


class DriftTotals(BaseModel):
    """Aggregate totals across all drift rows."""

    model_config = ConfigDict(frozen=True)

    deployable_eur: float
    total_holdings_eur: float
    total_drift_abs: float
    buy_eur: float
    sell_eur: float


class DriftDiagnostics(BaseModel):
    """Aggregate diagnostic flag counts + reconciliation status + entries."""

    model_config = ConfigDict(frozen=True)

    reconciliation_ok: bool
    reconciliation_delta_pct: float | None
    unmapped_count: int
    fx_missing_count: int
    target_not_on_broker_count: int
    base_currency: str
    sum_eur: float = 0.0
    invested_eur: float = 0.0
    delta_eur: float = 0.0
    tolerance_pct: float = 0.0
    stale_price_count: int = 0
    entries: list[DiagnosticEntry] = Field(default_factory=list)


class DriftResponse(BaseModel):
    """Top-level Drift API response envelope."""

    model_config = ConfigDict(frozen=True)

    holdings: list[NormalizedPosition]
    target: list[TargetWeight]
    drift: list[DriftRow]
    trades: list[TradeRow]
    totals: DriftTotals
    diagnostics: DriftDiagnostics
    request_id: int
