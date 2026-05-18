"""Pydantic v2 schemas for the ``/pipeline-builder`` HTTP domain (issue #711).

Contract layer between the FastAPI router (Cycle 3) and the Angular wizard
client (Cycles 4-5). Mirrors the 9 CLI flags of ``python -m research.cli``
plus the configurable parameters of each step handler.

Schemas are self-contained — no imports from ``optimizer/`` or ``research/``.
Enum values are duplicated by design to keep the schema layer decoupled.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import AsyncJobCreateResponse, CamelCaseModel

__all__ = [
    "BaseCurrencyEnum",
    "CostStepRequest",
    "CreateSessionResponse",
    "EmptyStepRequest",
    "LoadStepRequest",
    "RebalanceDecisionStepRequest",
    "RegimeStepRequest",
    "RunLevelConfig",
    "ScreenPresetEnum",
    "ScreenStepRequest",
    "StepPollResponse",
    "StepRunResponse",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BaseCurrencyEnum(str, Enum):
    """Reporting base currency for the pipeline run."""

    EUR = "EUR"
    GBP = "GBP"
    USD = "USD"


class ScreenPresetEnum(str, Enum):
    """Universe-screening preset for ``step_screen``."""

    DEVELOPED_MARKETS = "developed_markets"
    BROAD_UNIVERSE = "broad_universe"
    SMALL_CAP = "small_cap"
    LARGE_CAP = "large_cap"


# ---------------------------------------------------------------------------
# Run-level config (POST /sessions body)
# ---------------------------------------------------------------------------


class RunLevelConfig(BaseModel):
    """Run-level configuration posted at session creation.

    Maps 1:1 to the CLI flags of ``python -m research.cli`` and is stored
    on ``_PipelineSession.run_config`` as a plain dict via ``model_dump()``.
    """

    model_config = ConfigDict(extra="forbid")

    rebalance_freq: int = 63
    n_selected: int = Field(20, ge=15, le=30)
    cost_bps: float = 10.0
    tax_rate: float = 0.26
    base_currency: BaseCurrencyEnum = BaseCurrencyEnum.EUR
    robust: bool = False
    persist: bool = False
    start_date: date | None = None
    end_date: date | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Per-step request models (POST /sessions/{id}/steps/{step_id} body)
# ---------------------------------------------------------------------------


class _StrictRequest(BaseModel):
    """Shared base: forbid unknown fields on every step request."""

    model_config = ConfigDict(extra="forbid")


class LoadStepRequest(_StrictRequest):
    """Body for ``step_load``."""

    include_delisted: bool = True
    macro_country: str = "USA"


class ScreenStepRequest(_StrictRequest):
    """Body for ``step_screen``."""

    preset: ScreenPresetEnum = ScreenPresetEnum.DEVELOPED_MARKETS


class RegimeStepRequest(_StrictRequest):
    """Body for ``step_regime``."""

    enable_tilts: bool = True
    persist_regime: bool = False


class RebalanceDecisionStepRequest(_StrictRequest):
    """Body for ``step_rebalance_decision`` — no configurable params."""


class CostStepRequest(_StrictRequest):
    """Body for ``step_cost`` — no configurable params."""


class EmptyStepRequest(_StrictRequest):
    """Shared empty body for the 8 non-configurable steps."""


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class CreateSessionResponse(CamelCaseModel):
    """Returned by ``POST /sessions``."""

    session_id: str = Field(..., description="UUID hex of the new session")


class StepRunResponse(AsyncJobCreateResponse):
    """Returned by ``POST /sessions/{id}/steps/{step_id}`` (202 ACCEPTED)."""


class StepPollResponse(CamelCaseModel):
    """Returned by ``GET /sessions/{id}/steps/{step_id}``."""

    status: str = Field(..., description="pending | running | completed | failed")
    progress: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    gate_reason: str | None = Field(
        default=None,
        description=(
            "Set by ``step_coverage_gate`` when the IS ∩ OOS factor set "
            "falls below ``min_factors``."
        ),
    )
