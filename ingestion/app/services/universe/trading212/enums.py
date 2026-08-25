"""Asset-class taxonomy for the stock + bond investable universe.

Stored as plain strings on ``instruments`` (mirrors ``instrument_type``); these
enums are the canonical value set used by the classifier, builder and repository.
"""

from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    """Top-level sleeve. Non-null on every instrument (STOCK ⇒ EQUITY)."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    MULTI_ASSET = "multi_asset"


class FiSubclass(str, Enum):
    """Fixed-income sub-class (null unless ``asset_class == fixed_income``)."""

    GOVERNMENT = "government"
    CORPORATE = "corporate"
    AGGREGATE = "aggregate"
    EM = "em"
    INFLATION_LINKED = "inflation_linked"
    HIGH_YIELD = "high_yield"


class DurationBucket(str, Enum):
    """Effective-maturity bucket (null unless fixed income)."""

    SHORT = "short"
    INTERMEDIATE = "intermediate"
    LONG = "long"
