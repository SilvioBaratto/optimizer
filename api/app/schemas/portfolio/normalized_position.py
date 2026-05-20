"""NormalizedPosition + PositionFlag + FlagInstance.

Transient (non-persisted) Pydantic v2 contract consumed by every downstream
component in the T212 live-position normalization pipeline. Cycle 4 (#792)
adds `FlagInstance` and retypes `NormalizedPosition.flags` to carry per-flag
human reason copy.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

_PENCE_CURRENCIES = frozenset({"GBX", "GBp"})


class PositionFlag(str, Enum):
    """Diagnostic flags emitted alongside a normalized position."""

    UNMAPPED = "unmapped"
    FX_MISSING = "fx_missing"
    STALE_PRICE = "stale_price"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    TARGET_NOT_ON_BROKER = "target_not_on_broker"


class FlagInstance(BaseModel):
    """A `PositionFlag` code with human reason copy + optional reference."""

    model_config = ConfigDict(frozen=True)

    code: PositionFlag
    reason: str = ""
    reference: str | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FlagInstance):
            return (
                self.code is other.code
                and self.reason == other.reason
                and self.reference == other.reference
            )
        if isinstance(other, PositionFlag):
            return self.code is other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.code, self.reason, self.reference))


def _coerce_flag(value: object) -> object:
    if isinstance(value, PositionFlag):
        return FlagInstance(code=value)
    return value


def _coerce_flag_list(value: object) -> object:
    if isinstance(value, list):
        return [_coerce_flag(item) for item in value]
    return value


FlagList = Annotated[list[FlagInstance], BeforeValidator(_coerce_flag_list)]


class NormalizedPosition(BaseModel):
    """Live T212 position normalized to EUR, keyed by yfinance ticker."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    t212_ticker: str
    qty: float
    price_native: float
    currency: str
    eur_value: float
    eur_weight: float
    ppl_eur: float | None = None
    flags: FlagList = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        if value in _PENCE_CURRENCIES:
            raise ValueError(
                f"currency must be the major-unit ISO code, got pence '{value}' "
                "(use 'GBP' after rescaling price by 1/100)"
            )
        if len(value) != 3 or not value.isupper() or not value.isalpha():
            raise ValueError(
                f"currency must be a 3-letter uppercase ISO code, got '{value}'"
            )
        return value

    @field_validator("eur_weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"eur_weight must lie in [0.0, 1.0], got {value}"
            )
        return value
