"""Pydantic schemas for factor research endpoints.

Covers factor_scores and factor_validation_reports ORM models.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import CamelCaseModel, StrFromUUID

# ---------------------------------------------------------------------------
# Factor Score
# ---------------------------------------------------------------------------


class FactorScoreResponse(CamelCaseModel):
    """Response schema for a single factor score row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    ticker: str = Field(..., description="Asset ticker symbol")
    score_date: datetime.date = Field(..., description="Date the score was computed")
    factor_type: str = Field(..., description="Factor type (e.g. momentum, quality)")
    factor_group: str = Field(..., description="Factor group the type belongs to")
    raw_score: float = Field(..., description="Raw un-standardized factor score")
    standardized_score: float | None = Field(
        default=None, description="Z-score normalized factor score"
    )
    composite_score: float | None = Field(
        default=None, description="Composite score across factors"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class FactorScoreListResponse(CamelCaseModel):
    """Paginated list of factor score responses."""

    items: list[FactorScoreResponse] = Field(
        default_factory=list, description="Factor score rows"
    )
    total: int = Field(..., ge=0, description="Total number of rows")


# ---------------------------------------------------------------------------
# Factor Validation Report
# ---------------------------------------------------------------------------


class FactorValidationReportResponse(CamelCaseModel):
    """Response schema for a factor validation report row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    report_date: datetime.date = Field(..., description="Date the report was generated")
    factor_type: str | None = Field(
        default=None, description="Factor type; null for aggregate reports"
    )
    validation_type: str = Field(
        ..., description="Validation methodology (in_sample / out_of_sample)"
    )
    ic_mean: float | None = Field(
        default=None, description="Mean information coefficient"
    )
    ic_std: float | None = Field(default=None, description="Standard deviation of IC")
    icir: float | None = Field(default=None, description="IC information ratio")
    t_stat: float | None = Field(default=None, description="Newey-West t-statistic")
    p_value: float | None = Field(
        default=None, description="p-value for the t-statistic"
    )
    vif: float | None = Field(default=None, description="Variance inflation factor")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional validation details"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class FactorComputeRequest(BaseModel):
    """Request body for POST /factors/compute."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of asset tickers"
    )
    start_date: datetime.date = Field(..., description="Start of computation window")
    end_date: datetime.date = Field(..., description="End of computation window")
    factor_config: dict[str, Any] | None = Field(
        default=None, description="Optional factor computation configuration"
    )


class FactorValidateRequest(BaseModel):
    """Request body for POST /factors/validate."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of asset tickers"
    )
    start_date: datetime.date = Field(..., description="Start of validation window")
    end_date: datetime.date = Field(..., description="End of validation window")
    factor_type: str = Field(..., description="Factor type to validate")
    validation_type: str = Field(
        default="in_sample",
        description="Validation methodology (in_sample / out_of_sample)",
    )


class FactorScoreRequest(BaseModel):
    """Request body for POST /factors/score."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of asset tickers"
    )
    score_date: datetime.date = Field(
        ..., description="Date for which to compute composite scores"
    )
    composite_method: str = Field(
        ...,
        description="Composite scoring method (equal_weight, ic_weighted, icir_weighted, ridge_weighted, gbt_weighted)",
    )
    training_start_date: datetime.date | None = Field(
        default=None,
        description="Training window start (required for ridge_weighted / gbt_weighted)",
    )
    training_end_date: datetime.date | None = Field(
        default=None,
        description="Training window end (required for ridge_weighted / gbt_weighted)",
    )
    group_weights: dict[str, float] | None = Field(
        default=None, description="Optional per-group weight overrides"
    )


class FactorCompositeScoreResponse(BaseModel):
    """Response schema for POST /factors/score."""

    score_date: datetime.date = Field(..., description="Date scores were computed for")
    scores: dict[str, float] = Field(..., description="Per-ticker composite scores")
    group_contributions: dict[str, float] = Field(
        default_factory=dict, description="Per-group contribution to composite score"
    )


class FactorValidateResponse(BaseModel):
    """Response schema for POST /factors/validate."""

    report_date: datetime.date = Field(..., description="Date the report was generated")
    factor_type: str | None = Field(default=None, description="Factor type validated")
    validation_type: str = Field(..., description="Validation methodology used")
    ic_mean: float | None = Field(
        default=None, description="Mean information coefficient"
    )
    ic_std: float | None = Field(default=None, description="Standard deviation of IC")
    icir: float | None = Field(default=None, description="IC information ratio")
    t_stat: float | None = Field(default=None, description="Newey-West t-statistic")
    p_value: float | None = Field(
        default=None, description="p-value for the t-statistic"
    )
    vif: float | None = Field(default=None, description="Variance inflation factor")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional details"
    )


# ---------------------------------------------------------------------------
# Factor utility request/response schemas (select, exposure-constraints,
# quintile-spread, regime-tilt)
# ---------------------------------------------------------------------------


class BufferZone(BaseModel):
    """Tickers that entered or exited the selection due to the buffer zone."""

    entered: list[str] = Field(
        default_factory=list, description="Tickers that newly entered"
    )
    exited: list[str] = Field(
        default_factory=list, description="Tickers that were dropped"
    )


class FactorSelectRequest(BaseModel):
    """Request body for POST /factors/select."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Candidate universe ticker symbols"
    )
    start_date: datetime.date = Field(..., description="Start of scoring window")
    end_date: datetime.date = Field(..., description="End of scoring window")
    current_members: list[str] | None = Field(
        default=None, description="Currently selected tickers for hysteresis"
    )
    method: str | None = Field(
        default=None, description="Selection method: fixed_count or quantile"
    )
    target_count: int | None = Field(
        default=None, description="Target number of stocks (fixed_count method)"
    )
    target_quantile: float | None = Field(
        default=None, description="Top quantile threshold (quantile method)"
    )
    buffer_fraction: float | None = Field(
        default=None, description="Buffer fraction for hysteresis"
    )
    sector_balance: bool = Field(
        default=False, description="Apply sector-proportional balancing"
    )


class FactorSelectResponse(BaseModel):
    """Response schema for POST /factors/select."""

    selected_tickers: list[str] = Field(..., description="Selected ticker symbols")
    count: int = Field(..., ge=0, description="Number of selected tickers")
    turnover: float | None = Field(
        default=None, description="Fraction of universe changed vs. previous selection"
    )
    buffer_zone: BufferZone = Field(
        ..., description="Tickers entering/exiting via buffer"
    )


class FactorExposureConstraintsRequest(BaseModel):
    """Request body for POST /factors/exposure-constraints."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Asset tickers to build constraints for"
    )
    start_date: datetime.date = Field(..., description="Start of scoring window")
    end_date: datetime.date = Field(..., description="End of scoring window")
    bounds: list[float] | dict[str, list[float]] = Field(
        ...,
        description=(
            "Uniform bounds as [lower, upper] list, or per-factor dict mapping "
            "factor_name → [lower, upper]"
        ),
    )


class FactorExposureConstraintsResponse(BaseModel):
    """Response schema for POST /factors/exposure-constraints."""

    left_inequality: list[list[float]] = Field(
        ..., description="Inequality matrix A (2*n_factors × n_assets) as nested lists"
    )
    right_inequality: list[float] = Field(
        ..., description="Bound vector b (2*n_factors,) as a flat list"
    )


class FactorQuintileSpreadRequest(BaseModel):
    """Request body for POST /factors/quintile-spread."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Asset tickers for the analysis"
    )
    factor_name: str = Field(
        ..., description="Factor type to analyse (e.g. momentum_12_1)"
    )
    start_date: datetime.date = Field(..., description="Start of analysis window")
    end_date: datetime.date = Field(..., description="End of analysis window")
    n_quantiles: int = Field(default=5, ge=2, description="Number of quantile buckets")


class FactorQuintileSpreadResponse(BaseModel):
    """Response schema for POST /factors/quintile-spread."""

    quintile_cumulative_returns: dict[str, list[float]] = Field(
        ..., description="Cumulative return series per quantile bucket (Q1..Qn)"
    )
    spread_cumulative_return: list[float] = Field(
        ..., description="Cumulative return series for the Qn−Q1 long-short spread"
    )
    annualized_spread: float = Field(
        ..., description="Annualised mean of the long-short spread"
    )


class FactorRegimeTiltRequest(BaseModel):
    """Request body for POST /factors/regime-tilt."""

    group_weights: dict[str, float] = Field(
        ...,
        min_length=1,
        description="Base group weights keyed by FactorGroupType value",
    )
    enable: bool = Field(default=True, description="Whether to apply tilts")
    max_tilt_multiplier: float | None = Field(
        default=None, description="Maximum allowed tilt multiplier"
    )
    min_post_tilt_weight: float | None = Field(
        default=None, description="Floor for any group weight post-tilt"
    )


class FactorRegimeTiltResponse(BaseModel):
    """Response schema for POST /factors/regime-tilt."""

    regime: str = Field(..., description="Detected macro regime (e.g. expansion)")
    tilted_weights: dict[str, float] = Field(
        ..., description="Group weights after regime tilt"
    )
    tilt_multipliers: dict[str, float] = Field(
        ..., description="Effective tilt multiplier per group"
    )
