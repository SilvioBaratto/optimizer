"""Pydantic schemas for LLM-driven risk budget calibration endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.base import CamelCaseModel


class RiskBudgetRequest(BaseModel):
    sector_outlook: str = Field(
        ...,
        min_length=10,
        description=(
            "Qualitative sector outlook text (e.g. 'Overweight Technology "
            "and Healthcare; underweight Energy and Utilities.')."
        ),
    )
    sector_universe: list[str] = Field(
        ...,
        min_length=1,
        description="Exhaustive list of sector names present in the portfolio.",
    )
    asset_sector_map: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Asset ticker → sector name mapping for the full portfolio universe.",
    )

    @field_validator("asset_sector_map")
    @classmethod
    def tickers_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        cleaned = {k.strip().upper(): s.strip() for k, s in v.items() if k.strip()}
        if not cleaned:
            raise ValueError("asset_sector_map must contain at least one asset")
        return cleaned

    @field_validator("sector_universe")
    @classmethod
    def sectors_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("sector_universe must contain at least one sector")
        return cleaned


class RiskBudgetResponse(CamelCaseModel):
    n_assets: int = Field(..., description="Number of assets in the budget vector.")
    assets: list[str] = Field(..., description="Asset tickers in budget vector order.")
    budget_vector: list[float] = Field(
        ...,
        description=(
            "Risk budget weights, shape (n_assets,). Non-negative and sum to 1.0. "
            "Pass directly as risk_budget= to build_risk_budgeting()."
        ),
    )
    budget_sum: float = Field(
        ..., description="Sum of budget_vector (should be ≈ 1.0)."
    )
