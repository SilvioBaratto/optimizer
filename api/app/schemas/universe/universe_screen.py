"""Pydantic schemas for POST /universe/screen endpoint."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas._shared import CamelCaseModel


class ScreenPreset(str, Enum):
    """Named investability screen configurations."""

    DEVELOPED_MARKETS = "developed_markets"
    BROAD_UNIVERSE = "broad_universe"
    SMALL_CAP = "small_cap"
    LARGE_CAP = "large_cap"


class HysteresisRequest(BaseModel):
    """Entry/exit threshold pair for a single screen."""

    entry: float = Field(..., description="Entry threshold.")
    exit_: float = Field(
        ..., alias="exit", description="Exit threshold (must be ≤ entry)."
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def exit_le_entry(self) -> HysteresisRequest:
        if self.exit_ > self.entry:
            raise ValueError(f"exit ({self.exit_}) must be <= entry ({self.entry})")
        return self


_CUSTOM_FIELDS = frozenset(
    {
        "market_cap",
        "addv_12m",
        "addv_3m",
        "trading_frequency",
        "price_us",
        "price_europe",
        "min_trading_history",
        "min_ipo_seasoning",
        "min_annual_reports",
        "min_quarterly_reports",
        "exchange_region",
        "mcap_percentile_entry",
        "mcap_percentile_exit",
    }
)


class UniverseScreenRequest(BaseModel):
    """Request body for the universe screening endpoint.

    Either ``preset`` or individual config fields may be provided — not both.
    When neither is provided, defaults to developed-markets thresholds.
    """

    preset: ScreenPreset | None = Field(
        default=None,
        description="Named preset. Mutually exclusive with custom config fields.",
    )
    market_cap: HysteresisRequest | None = None
    addv_12m: HysteresisRequest | None = None
    addv_3m: HysteresisRequest | None = None
    trading_frequency: HysteresisRequest | None = None
    price_us: HysteresisRequest | None = None
    price_europe: HysteresisRequest | None = None
    min_trading_history: int | None = None
    min_ipo_seasoning: int | None = None
    min_annual_reports: int | None = None
    min_quarterly_reports: int | None = None
    exchange_region: str | None = None
    mcap_percentile_entry: float | None = None
    mcap_percentile_exit: float | None = None
    current_members: list[str] | None = Field(
        default=None,
        description="Tickers currently in the universe for hysteresis flip-flop prevention.",
    )

    @model_validator(mode="before")
    @classmethod
    def preset_and_custom_mutually_exclusive(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        has_preset = data.get("preset") is not None
        has_custom = any(data.get(f) is not None for f in _CUSTOM_FIELDS)
        if has_preset and has_custom:
            raise ValueError(
                "preset and custom config fields are mutually exclusive. "
                "Provide either a preset or individual threshold overrides."
            )
        return data


class UniverseScreenResponse(CamelCaseModel):
    """Response from the universe screening endpoint."""

    passing_tickers: list[str]
    total_screened: int
    diagnostics: dict[str, dict[str, bool]]
