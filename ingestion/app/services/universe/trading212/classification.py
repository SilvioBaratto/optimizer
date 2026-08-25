"""Classify an instrument into the asset-class taxonomy from its name.

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
from typing import NamedTuple

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

# Single-fund allocation products (checked before bond markers). The numeric
# form matches only equity/bond split ratios that sum to ~100 (60/40, 80/20 …)
# or an explicit triple (60-30-10); a bare "\d{2}/\d{2}" is deliberately NOT
# used because UCITS concentration-cap names like "MSCI Switzerland 20/35"
# (an equity ETF) would false-match.
_MULTI_ASSET = re.compile(
    r"life\s*strategy|multi[\s-]?asset|allocation|\bbalanced\b"
    r"|\b(?:90/10|80/20|75/25|70/30|65/35|60/40|55/45|50/50"
    r"|45/55|40/60|35/65|30/70|25/75|20/80|10/90)\b"
    r"|\b\d{2}-\d{2}-\d{2}\b",
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


def classify_instrument(
    name: str | None,
    instrument_type: str | None,
) -> Classification | None:
    """Return the asset-class tags from the instrument NAME, or ``None`` if the
    instrument is rejected (equity/thematic/leveraged ETF, or a non-STOCK/non-ETF
    type). Classification is name-only: the universe builder resolves it before
    any yfinance fetch, so there is no funds_data to consult."""
    itype = (instrument_type or "").upper()
    if itype == "STOCK":
        return Classification(AssetClass.EQUITY.value, None, None)
    if itype != "ETF":
        return None

    name = name or ""
    if _LEVERAGED_INVERSE.search(name):
        return None  # daily-reset leveraged/inverse product — not investable

    if _MULTI_ASSET.search(name):
        return Classification(AssetClass.MULTI_ASSET.value, None, None)

    if _BOND_MARKER.search(name):
        return Classification(
            AssetClass.FIXED_INCOME.value, _fi_subclass(name), _duration(name)
        )

    return None  # equity / thematic ETF — excluded from the universe
