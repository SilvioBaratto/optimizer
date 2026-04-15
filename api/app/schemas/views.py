"""Pydantic schemas for Black-Litterman view generation and opinion pooling."""

from __future__ import annotations

from baml_client.types import ExpertPersona
from pydantic import BaseModel, Field, field_validator

from app.schemas.base import CamelCaseModel


# ---------------------------------------------------------------------------
# View generation schemas (from api/v1/views.py)
# ---------------------------------------------------------------------------


class GenerateViewsRequest(BaseModel):
    tickers: list[str] = Field(
        ...,
        min_length=2,
        description="Ordered list of asset tickers in the portfolio universe.",
    )

    @field_validator("tickers")
    @classmethod
    def non_empty_tickers(cls, v: list[str]) -> list[str]:
        stripped = [t.strip().upper() for t in v]
        if not all(stripped):
            raise ValueError("tickers must not contain empty strings")
        return stripped


class AssetViewResponse(CamelCaseModel):
    asset: str
    direction: int
    magnitude_bps: float
    confidence: float
    reasoning: str


class GenerateViewsResponse(CamelCaseModel):
    """BL-ready view components.

    All arrays are serialised as nested lists for JSON transport.
    """

    n_views: int = Field(..., description="Number of views generated.")
    n_assets: int = Field(..., description="Number of assets in the universe.")
    view_strings: list[str] = Field(
        ..., description="skfolio-compatible view expressions (e.g. 'AAPL == 0.02')."
    )
    P: list[list[float]] = Field(
        ...,
        description="Pick matrix, shape (n_views, n_assets). Row i is the i-th view.",
    )
    Q: list[float] = Field(
        ...,
        description="Expected excess returns vector, shape (n_views,). Decimal units.",
    )
    view_confidences: list[float] = Field(
        ...,
        description="Idzorek alpha_k per view, in (0, 1). Same order as view_strings.",
    )
    idzorek_alphas: dict[str, float] = Field(
        ..., description="Asset ticker → Idzorek alpha_k. All values in (0, 1)."
    )
    views: list[AssetViewResponse] = Field(
        ..., description="Structured per-asset views."
    )
    rationale: str = Field(..., description="LLM narrative explaining view generation.")
    tickers_with_data: list[str] = Field(
        ..., description="Tickers for which factor data was found in the DB."
    )
    tickers_missing_data: list[str] = Field(
        ..., description="Tickers not found in the DB (excluded from view generation)."
    )


# ---------------------------------------------------------------------------
# Opinion pooling schemas (from api/v1/opinion_pooling.py)
# ---------------------------------------------------------------------------


class ICHistory(BaseModel):
    """Historical IC series for one expert (ordered chronologically)."""

    persona: str = Field(
        ...,
        description="Expert persona: VALUE_INVESTOR | MOMENTUM_TRADER | MACRO_ANALYST.",
    )
    ic_values: list[float] = Field(
        ..., description="Chronological IC values (at least 3 for meaningful ICIR)."
    )


class ExpertViewSummary(CamelCaseModel):
    persona: str
    name: str
    n_views: int
    view_strings: list[str]
    idzorek_alphas: dict[str, float]
    ic_weight: float


class OpinionPoolRequest(BaseModel):
    tickers: list[str] = Field(
        ..., min_length=2, description="Portfolio universe tickers."
    )
    personas: list[str] | None = Field(
        default=None,
        description="Subset of personas to run. Defaults to all three.",
    )
    ic_histories: list[ICHistory] | None = Field(
        default=None,
        description="Historical IC series per expert for IC-weighted pooling. If omitted, equal weights are used.",
    )
    tau: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="BL uncertainty scaling for each expert prior.",
    )
    is_linear_pooling: bool = Field(
        default=True, description="True = arithmetic pooling; False = geometric."
    )
    divergence_penalty: float = Field(
        default=0.0, ge=0.0, description="KL divergence penalty for robust pooling."
    )

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        stripped = [t.strip().upper() for t in v]
        if not all(stripped):
            raise ValueError("tickers must not contain empty strings")
        return stripped

    @field_validator("personas")
    @classmethod
    def validate_personas(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        valid = {p.value for p in ExpertPersona}
        for p in v:
            if p not in valid:
                raise ValueError(
                    f"Unknown persona '{p}'. Must be one of: {sorted(valid)}"
                )
        return v


class OpinionPoolResponse(CamelCaseModel):
    n_experts: int
    tickers: list[str]
    tickers_with_data: list[str]
    tickers_missing_data: list[str]
    experts: list[ExpertViewSummary]
    ic_weights: list[float] = Field(
        ..., description="IC-calibrated weights per expert, summing to 1.0."
    )
    pooling_type: str = Field(..., description="'linear' or 'geometric'.")
    total_views: int = Field(..., description="Total unique views across all experts.")
