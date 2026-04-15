"""Pydantic schemas for POST /synthetic/scenario endpoint (issue #366)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScenarioRequest(BaseModel):
    """Request body for POST /synthetic/scenario."""

    tickers: list[str] = Field(..., min_length=1)
    n_samples: int = Field(default=1000, gt=0, le=100_000)
    conditioning: dict[str, float] | None = None
    preset: Literal["scenario_generation", "stress_test"] | None = None

    @field_validator("conditioning")
    @classmethod
    def validate_conditioning_values(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        """All conditioning return values must be strictly inside (-1, 1)."""
        if v is None:
            return v
        for ticker, value in v.items():
            if not (-1.0 < value < 1.0):
                raise ValueError(
                    f"Conditioning value for {ticker!r} must be in (-1, 1), got {value}"
                )
        return v


class ScenarioInlineResponse(BaseModel):
    """Returned when n_samples <= _LARGE_THRESHOLD."""

    scenarios: list[list[float]]
    columns: list[str]


class ScenarioStoredResponse(BaseModel):
    """Returned when n_samples > _LARGE_THRESHOLD; caller fetches via download_id."""

    download_id: str
    expires_at: str  # ISO 8601 datetime string
