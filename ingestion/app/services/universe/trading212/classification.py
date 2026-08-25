"""Classify an instrument into the asset-class taxonomy from its name (and,
optionally, yfinance fund metadata).

STOCK ⇒ equity. ETFs are classified as multi_asset (single-fund allocation),
fixed_income (bond funds, with a sub-class + duration bucket), or rejected
(``None``) when they are equity/thematic — equity exposure comes from the STOCK
universe, not ETFs.

Pattern tables are module-level data (reviewable/tunable without touching the
logic). Fixed-income sub-class precedence is inflation → em → high-yield →
government → corporate → aggregate (the catch-all for broad/mixed bond funds).
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from app.services.universe.trading212.enums import (
    AssetClass,
    DurationBucket,
    FiSubclass,
)


class Classification(NamedTuple):
    asset_class: str
    fi_subclass: str | None
    duration_bucket: str | None


# Leveraged / inverse derivatives (e.g. "Leverage Shares -5x Short 7-10 Year
# Treasury Bond", "GraniteShares 3x Long"). These are daily-reset trading
# products, not investable buy-and-hold exposure — reject regardless of any
# bond marker in the name.
_LEVERAGED_INVERSE = re.compile(
    r"\bleverage\b|leveraged|\binverse\b|[-+]?\d+\s?x\b|\bdaily\s+(short|long|leveraged)\b",
    re.I,
)

# Single-fund allocation products (checked before bond markers).
_MULTI_ASSET = re.compile(
    r"life\s*strategy|multi[\s-]?asset|allocation|\bbalanced\b"
    r"|\b\d{2}/\d{2}\b|\b\d{2}-\d{2}-\d{2}\b",
    re.I,
)

# A fund is fixed income if any bond marker appears in its name.
_BOND_MARKER = re.compile(
    r"\bbond\b|treasur|\bgilt|\bbund\b|sovereign|government|\bgov(t|ies)?\b"
    r"|corporate|\bcredit\b|aggregate|fixed income|inflation|linker|\btips\b"
    r"|high[\s-]?yield|\bagg\b|\bbtp\b",
    re.I,
)

# Fixed-income sub-class, most-specific first.
_FI_SUBCLASS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"inflation|linker|\btips\b|index[\s-]?linked", re.I),
        FiSubclass.INFLATION_LINKED.value,
    ),
    (re.compile(r"emerging|\bem\b|\bemd\b", re.I), FiSubclass.EM.value),
    (re.compile(r"high[\s-]?yield|\bhy\b", re.I), FiSubclass.HIGH_YIELD.value),
    (
        re.compile(
            r"treasur|\bgilt|\bbund\b|sovereign|government|\bgov(t|ies)?\b|\bbtp\b",
            re.I,
        ),
        FiSubclass.GOVERNMENT.value,
    ),
    (re.compile(r"corporate|\bcorp\b|\bcredit\b", re.I), FiSubclass.CORPORATE.value),
]

# Duration buckets.
_DURATION_SHORT = re.compile(
    r"ultra\s*short|short[\s-]?term|short[\s-]?duration|floating|money market"
    r"|\b0[\s-]?1(?=\D|$)|\b1[\s-]?3(?=\D|$)|\b0[\s-]?3(?=\D|$)|\b0[\s-]?5(?=\D|$)",
    re.I,
)
_DURATION_LONG = re.compile(
    r"\b10\+|\b15\+|\b20\+|\b25\+|long[\s-]?term|long[\s-]?duration|ultra\s*long"
    r"|\b7[\s-]?10(?=\D|$)|\b10[\s-]?25(?=\D|$)|\b15[\s-]?25(?=\D|$)",
    re.I,
)


def _duration(name: str) -> str:
    if _DURATION_SHORT.search(name):
        return DurationBucket.SHORT.value
    if _DURATION_LONG.search(name):
        return DurationBucket.LONG.value
    return DurationBucket.INTERMEDIATE.value


def _fi_subclass(name: str) -> str:
    for pattern, value in _FI_SUBCLASS:
        if pattern.search(name):
            return value
    return FiSubclass.AGGREGATE.value  # broad/mixed bond fund


# Metadata heuristics (used when a caller supplies yfinance funds_data): a fund
# with material stock AND bond weight is multi-asset; a bond-dominant fund is
# fixed income. NOTE: the universe builder classifies from the name only, so
# these paths fire only for callers that pass yf_metadata.
_MULTI_ASSET_MIN_LEG = 0.15  # min stock and bond weight to call it multi-asset
_FIXED_INCOME_MIN_BOND = 0.70  # bond weight above which a fund is fixed income


def _looks_multi_asset(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    ac = meta.get("asset_classes") or {}
    stock = float(ac.get("stockPosition") or 0.0)
    bond = float(ac.get("bondPosition") or 0.0)
    return stock >= _MULTI_ASSET_MIN_LEG and bond >= _MULTI_ASSET_MIN_LEG


def _looks_fixed_income(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    ac = meta.get("asset_classes") or {}
    return float(ac.get("bondPosition") or 0.0) >= _FIXED_INCOME_MIN_BOND


def classify_instrument(
    name: str | None,
    instrument_type: str | None,
    yf_metadata: dict[str, Any] | None = None,
) -> Classification | None:
    """Return the asset-class tags, or ``None`` if the instrument is rejected
    (equity/thematic ETF, or a non-STOCK/non-ETF type)."""
    itype = (instrument_type or "").upper()
    if itype == "STOCK":
        return Classification(AssetClass.EQUITY.value, None, None)
    if itype != "ETF":
        return None

    name = name or ""
    if _LEVERAGED_INVERSE.search(name):
        return None  # daily-reset leveraged/inverse product — not investable

    if _MULTI_ASSET.search(name) or _looks_multi_asset(yf_metadata):
        return Classification(AssetClass.MULTI_ASSET.value, None, None)

    if _BOND_MARKER.search(name) or _looks_fixed_income(yf_metadata):
        return Classification(
            AssetClass.FIXED_INCOME.value, _fi_subclass(name), _duration(name)
        )

    return None  # equity / thematic ETF — excluded from the universe
