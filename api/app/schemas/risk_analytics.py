"""Pydantic response schemas for risk analytics endpoints (issues #368, #369).

Separate from ``risk.py`` (risk limits) to avoid file bloat.
All schemas extend CamelCaseModel for Angular frontend compatibility.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.base import CamelCaseModel


class VarResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/risk/var."""

    var: dict[str, float] = Field(
        ...,
        description="VaR keyed by confidence level string ('90', '95', '99')",
    )
    cvar: dict[str, float] = Field(
        ...,
        description="CVaR (expected shortfall) keyed by confidence level string",
    )
    method: str = Field(
        ...,
        description="Computation method: 'historical' or 'parametric'",
    )
    lookback: int = Field(..., description="Lookback window in trading days")
    n_observations: int = Field(
        ..., description="Actual number of observations used"
    )


class CorrelationResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/risk/correlation."""

    assets: list[str] = Field(
        ...,
        description="Asset tickers ordered by hierarchical cluster leaf position",
    )
    matrix: list[list[float]] = Field(
        ..., description="NxN correlation matrix aligned to assets order"
    )
    cluster_labels: list[int] = Field(
        ...,
        description="Cluster ID per asset (same index as assets list)",
    )


class FactorExposureResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/risk/factor-exposure."""

    exposures: dict[str, float] = Field(
        ...,
        description="Portfolio-weighted exposure per factor type",
    )
    asset_exposures: dict[str, dict[str, float]] = Field(
        ...,
        description="Per-asset factor exposure breakdown",
    )


# ---------------------------------------------------------------------------
# Concentration schemas (issue #369)
# ---------------------------------------------------------------------------


class ConcentrationAsset(CamelCaseModel):
    """Per-asset row for the concentration endpoint."""

    ticker: str = Field(..., description="Asset ticker symbol")
    name: str = Field(..., description="Asset display name")
    weight: float = Field(..., description="Portfolio weight (0-1)")
    # TODO: add riskContribution/componentVar once VaR service exists (#368)


class ConcentrationSummary(CamelCaseModel):
    """Portfolio-level concentration aggregates."""

    hhi: float = Field(..., description="Herfindahl-Hirschman Index = Σ(wᵢ²)")
    effective_n: float = Field(
        ..., description="Effective number of positions = 1/HHI"
    )
    top_n_ratio: float = Field(
        ..., description="Sum of top-N weights by descending weight"
    )


class ConcentrationResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/risk/concentration."""

    assets: list[ConcentrationAsset] = Field(
        ...,
        description="Per-asset breakdown sorted by descending weight",
    )
    summary: ConcentrationSummary = Field(
        ..., description="Portfolio-level concentration aggregates"
    )


# ---------------------------------------------------------------------------
# Liquidity schemas (issue #369)
# ---------------------------------------------------------------------------


class LiquidityAsset(CamelCaseModel):
    """Per-asset row for the liquidity endpoint."""

    ticker: str = Field(..., description="Asset ticker symbol")
    name: str = Field(..., description="Asset display name")
    weight: float = Field(..., description="Portfolio weight (0-1)")
    avg_daily_volume: float | None = Field(
        None,
        description="Average daily dollar volume over lookback window; null if no price history",
    )
    days_to_liquidate: float | None = Field(
        None,
        description="Estimated days to liquidate: weight / (ADDV × participation_rate)",
    )
    liquidity_cost: float | None = Field(
        None,
        description="Linear market impact estimate: days_to_liquidate × 0.0005",
    )


class LiquiditySummary(CamelCaseModel):
    """Portfolio-level liquidity aggregate."""

    weighted_avg_days_to_liquidate: float = Field(
        ...,
        description="Weighted-average days to liquidate across assets with valid ADDV",
    )


class LiquidityResponse(CamelCaseModel):
    """Response for GET /portfolio/{name}/risk/liquidity."""

    assets: list[LiquidityAsset] = Field(
        ..., description="Per-asset liquidity breakdown"
    )
    summary: LiquiditySummary = Field(
        ..., description="Portfolio-level liquidity aggregate"
    )
