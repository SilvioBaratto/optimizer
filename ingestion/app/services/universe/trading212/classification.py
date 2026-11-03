"""Classify an instrument into the asset-class taxonomy from its name.

STOCK ⇒ equity. ETFs are classified as multi_asset (single-fund allocation),
fixed_income (bond, T-bill, money-market and loan funds, with a sub-class + duration
bucket), commodity (physical metal, futures and broad-commodity funds) or equity.
Equity is the fallback: a fund no other marker claims, miners and sector or factor
funds included, holds equities. Only products that are not investable sleeves
come back as ``None``: leveraged/inverse daily-reset funds, and crypto, currency
and volatility funds, which would otherwise be mislabelled equity.

The fallback is name-based, so an unusual name can land in the wrong class; an
operator corrects such a row with ``add-instrument --pin``, which the build leaves
alone.

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
# Treasury Bond", "GraniteShares 3x Long", "ProShares UltraShort 20+ Year
# Treasury"). These are daily-reset trading products, not investable
# buy-and-hold exposure — reject regardless of any bond marker in the name.
# "Ultra" alone is not a marker: ultra-short *duration* bond funds ("Ultrashort
# Bond", "Ultra-Short Income", "Ultra Short 0-1 Year Treasury", "Ultra Short
# Investment Grade") are ordinary fixed income. The Ultra product line is
# recognised by "Ultra" followed directly by a maturity or credit segment,
# by the one-word "UltraShort" followed by a non-zero maturity (a 0-N range is
# the duration fund), or by the ProShares issuer prefix.
_LEVERAGED_INVERSE = re.compile(
    r"\bleverage\b|leveraged|\binverse\b|[-+]?\d+\s?x\b|\bx\d+\b"
    r"|\bdaily\s+(short|long|leveraged)\b|\bshort\s+daily\b"
    r"|\b(?:bull|bear)\b"
    r"|\bultra[\s-]?pro\b"
    r"|\bultra\s+(?=\d|investment[\s-]grade|high[\s-]yield)"
    r"|\bultrashort\s+(?=[1-9])"
    r"|\bproshares\s+(?:ultra|short\b(?![\s-]*(?:term|duration|maturity)))",
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

# Words that only ever describe debt: enough on their own to call a fund a bond
# fund.
_BOND_MARKER = re.compile(
    r"\bbond\b|treasur|\bgilt|\bbund\b|sovereign|\bgov(t|ies)?\b"
    r"|fixed income|linker|\btips\b|\bagg\b|\bbtp\b"
    r"|\bt[\s-]?bills?\b|\bmoney market\b|\b(?:senior|bank) loans?\b",
    re.I,
)

# Funds that hold physical metal, commodity futures or a commodity index. A name
# with equity vocabulary describes the companies instead (gold miners, copper
# miners, oil and gas exploration), so it stays in the equity fallback.
_COMMODITY_MARKER = re.compile(
    r"\b(?:gold|silver|platinum|palladium|bullion)\b|precious metals?"
    r"|commodit(?:y|ies)|\bcrude\b|\bbrent\b|\bwti\b|\boil fund\b|natural gas"
    r"|\bcopper\b|agricultur"
    r"|\b(?:wheat|corn|soybeans?|sugar|coffee|cotton|cocoa)\b",
    re.I,
)
_COMMODITY_COMPANY_VOCABULARY = re.compile(
    r"\bminers?\b|\bmining\b|\bproducers?\b|royalt|streaming|exploration"
    r"|\bequit(?:y|ies)\b|\bstocks?\b|\bcompanies\b|\bservices\b|\bequipment\b"
    r"|\bsector\b|\brefiners?\b|\bmidstream\b|\bdividend",
    re.I,
)

# Funds that hold no equity, debt or commodity sleeve we can name: spot crypto
# trusts, currency trusts, dollar-index and volatility funds. Left unclassified
# rather than mislabelled by the equity fallback.
_NO_SLEEVE = re.compile(
    r"bitcoin|ethereum|\bcrypto|currencyshares|currency trust|dollar index"
    r"|\bvix\b|volatility",
    re.I,
)

# Segment words that bond funds use ("Euro Corporates", "High Yield") but equity
# strategies use too ("High Yield Equity Dividend", "Inflation Beneficiaries"),
# so they count only when the name carries no equity vocabulary.
_AMBIGUOUS_BOND_MARKER = re.compile(
    r"government|corporate|\bcredit\b|aggregate|inflation|high[\s-]?yield",
    re.I,
)
_EQUITY_VOCABULARY = re.compile(
    r"\bequit(?:y|ies)\b|\bstocks?\b|\bdividend|\bbeneficiar", re.I
)

# Fixed-income sub-class, most-specific first.
_FI_SUBCLASS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"inflation|linker|\btips\b|index[\s-]?linked", re.I),
        FiSubclass.INFLATION_LINKED.value,
    ),
    (re.compile(r"emerging|\bem\b|\bemd\b", re.I), FiSubclass.EM.value),
    (
        re.compile(r"high[\s-]?yield|\bhy\b|\b(?:senior|bank) loans?\b", re.I),
        FiSubclass.HIGH_YIELD.value,
    ),
    (
        re.compile(
            r"treasur|\bgilt|\bbund\b|sovereign|government|\bgov(t|ies)?\b|\bbtp\b"
            r"|\bt[\s-]?bills?\b",
            re.I,
        ),
        FiSubclass.GOVERNMENT.value,
    ),
    (re.compile(r"corporate|\bcorp\b|\bcredit\b", re.I), FiSubclass.CORPORATE.value),
]

_DURATION_SHORT = re.compile(
    r"ultra[\s-]?short|short[\s-]?(?:term|duration|maturity)|short\s+treasur"
    r"|floating|money market"
    r"|\b0[\s-]?1(?=\D|$)|\b1[\s-]?3(?=\D|$)|\b0[\s-]?3(?=\D|$)|\b0[\s-]?5(?=\D|$)",
    re.I,
)
_DURATION_LONG = re.compile(
    r"\b10\+|\b15\+|\b20\+|\b25\+|long[\s-]?term|long[\s-]?duration|ultra[\s-]?long"
    r"|\b7[\s-]?10(?=\D|$)|\b10[\s-]?20(?=\D|$)|\b10[\s-]?25(?=\D|$)"
    r"|\b10[\s-]?30(?=\D|$)|\b15[\s-]?25(?=\D|$)",
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


def _is_bond_fund(name: str) -> bool:
    if _BOND_MARKER.search(name):
        return True
    return bool(
        _AMBIGUOUS_BOND_MARKER.search(name) and not _EQUITY_VOCABULARY.search(name)
    )


def classify_instrument(
    name: str | None,
    instrument_type: str | None,
) -> Classification | None:
    """Return the asset-class tags from the instrument NAME, or ``None`` if the
    instrument is rejected (a leveraged/inverse ETF, a crypto/currency/volatility
    fund, or a non-STOCK/non-ETF type). Classification is name-only: the universe
    builder resolves it before any yfinance fetch, so there is no funds_data to
    consult."""
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

    if _is_bond_fund(name):
        return Classification(
            AssetClass.FIXED_INCOME.value, _fi_subclass(name), _duration(name)
        )

    if _NO_SLEEVE.search(name):
        return None

    if _COMMODITY_MARKER.search(name) and not _COMMODITY_COMPANY_VOCABULARY.search(
        name
    ):
        return Classification(AssetClass.COMMODITY.value, None, None)

    return Classification(AssetClass.EQUITY.value, None, None)
