"""Dedup-first universe: name normalization, canonical-listing selection, and the
coarse anti-junk investability floor.

The yfinance screener exposes the same company under many venues (``NVDA`` /
``NVD.DE`` / ``1NVDA.MI`` / ``NVDA.SW``) with a byte-identical ``longName`` and no
ISIN. This module collapses those cross-listings to ONE canonical listing per
entity — group by normalized name, keep the most liquid (max USD average-daily
dollar-volume) line — and applies a deliberately loose USD-normalized anti-junk
floor. Everything here is pure and FX-rate-injected (no network), so the selection
logic is unit-testable; the screener source wires it in separately.

Theory basis (optimizer-theory): the floor is an existence / anti-shell filter,
NOT a size or capacity gate — small size is an alpha source (Banz, Fama-French),
so there is no size cap and no size tilt; the real market-impact capacity cut
(doc 22) lives downstream in optimizer/universe. Multi-currency: thresholds are a
USD numeraire; prices in sub-units (GBp pence = 1/100 GBP) are divided to the
major unit before any FX conversion, and share/volume COUNTS are never scaled.

The optimizer/universe HysteresisConfig / apply_screen / build_fx_pair_ticker
patterns are re-implemented here (not imported) — ingestion must not import
optimizer (guarded by tests/unit/hygiene/test_no_optimizer_import.py).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from app.utils.currency import MINOR_TO_MAJOR

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

# Sub-unit codes are quoted in 1/100 of the major unit (pence, agorot, cents).
_SUBUNIT_DIVISOR = 100.0


def normalize_name(name: str | None) -> str:
    """Canonical dedup key from a listing's display name.

    NFKD → ASCII de-accents (Moët/Nestlé collapse); uppercase; strip ``.``/``'``
    with NO separator (``S.A.`` → ``SA`` so it doesn't split from ``SA``); every
    other run of non-alphanumerics → a single space; trim. Legal suffixes
    (Corp/Inc/Ltd/…) are intentionally NOT stripped — doing so merges ~0.1% more
    clusters while adding false-merge risk. Returns ``""`` for an empty name.
    """
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    collapsed = ascii_name.upper().replace(".", "").replace("'", "")
    return _NON_ALNUM.sub(" ", collapsed).strip()


def split_currency(code: str | None) -> tuple[str | None, float]:
    """``currency`` → ``(major_unit_code, divisor)``.

    ``GBX``/``GBp`` → ``("GBP", 100)``, ``ILA`` → ``("ILS", 100)``, ``ZAC`` →
    ``("ZAR", 100)``; any already-major or unknown code → ``(code, 1)`` (fail-open,
    caller warns on unknown sub-units). The divisor applies to PRICE-derived
    quantities only — never to share or volume counts.
    """
    if code is None:
        return None, 1.0
    if code in MINOR_TO_MAJOR:
        return MINOR_TO_MAJOR[code], _SUBUNIT_DIVISOR
    return code, 1.0


@dataclass(frozen=True)
class FloorBand:
    """Entry/exit hysteresis band (pattern mirrors optimizer/universe.HysteresisConfig).

    A new entrant must clear ``entry``; an existing member is retained down to
    ``exit_``. The dead-band is intentionally wide here because an ingestion drop
    is destructive (it forfeits the ticker's stored price history).
    """

    entry: float
    exit_: float

    def __post_init__(self) -> None:
        if self.exit_ > self.entry:
            msg = f"exit_ ({self.exit_}) must be <= entry ({self.entry})"
            raise ValueError(msg)

    def threshold(self, *, is_member: bool) -> float:
        return self.exit_ if is_member else self.entry


@dataclass(frozen=True)
class IngestionFloorConfig:
    """Coarse anti-junk floor — USD numeraire. Nests strictly below the loosest
    optimizer preset (for_small_cap: mcap exit 35M, addv_3m exit 100k) so the two
    layers never double-cut. No price floor and no size cap: ingestion drops only
    non-instruments; small caps are kept (alpha source), the capacity cut is
    downstream.
    """

    mcap_usd: FloorBand = field(
        default_factory=lambda: FloorBand(entry=25_000_000.0, exit_=10_000_000.0)
    )
    addv_usd: FloorBand = field(
        default_factory=lambda: FloorBand(entry=50_000.0, exit_=25_000.0)
    )
    # A same-exchange sibling in a name cluster is preserved as a distinct
    # dual-class line only if it clears this (high) ADDV bar — recovers GOOG next
    # to GOOGL while dropping the illiquid BRK-A next to BRK-B.
    dual_class_min_addv_usd: float = 10_000_000.0


@dataclass(frozen=True)
class Listing:
    """The screener-inline fields the dedup+floor need (no per-ticker call)."""

    symbol: str
    exchange: str | None
    long_name: str | None
    short_name: str | None
    currency: str | None
    financial_currency: str | None
    price: float | None  # regularMarketPrice, LOCAL unit (pence for .L)
    avg_volume: float | None  # averageDailyVolume3Month, SHARES (a count)
    market_cap: float | None  # LOCAL major-unit currency
    shares_outstanding: float | None


@dataclass(frozen=True)
class Metrics:
    """USD-normalized quantities derived from a :class:`Listing`."""

    major_ccy: str | None
    price_major: float | None
    mcap_usd: float | None
    addv_usd: float | None


def _positive(value: float | None) -> float | None:
    """A finite strictly-positive number, else ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0.0 and f != float("inf") else None


def derive_metrics(listing: Listing, usd_per_major: dict[str, float]) -> Metrics:
    """USD-normalize a listing. FX is INJECTED (``{major_ccy: USD_per_1_major}``).

    Missing FX or missing inputs → the affected metric is ``None`` (fail-open: the
    floor treats ``None`` as "unknown, keep"). ``mcap_usd`` is RECONSTRUCTED from
    ``shares × price_major`` (self-consistent with the sub-unit correction); the
    reported ``market_cap`` is only a fallback and is cross-checked, because Yahoo
    reports ``.L`` market cap already in GBP while price is in pence — feeding both
    through ÷100 would double-count.
    """
    major, div = split_currency(listing.currency)
    price_major = None
    price = _positive(listing.price)
    if price is not None:
        price_major = price / div

    usd = usd_per_major.get(major) if major is not None else None
    usd = _positive(usd)

    addv_usd = None
    shares_adv = _positive(listing.avg_volume)
    if usd is not None and price_major is not None and shares_adv is not None:
        addv_usd = shares_adv * price_major * usd

    mcap_usd = None
    shares = _positive(listing.shares_outstanding)
    if usd is not None:
        if shares is not None and price_major is not None:
            mcap_usd = shares * price_major * usd  # preferred: reconstructed
            reported = _positive(listing.market_cap)
            if reported is not None:
                ratio = (reported * usd) / mcap_usd if mcap_usd else 0.0
                if ratio > 10.0 or (0.0 < ratio < 0.1):
                    logger.warning(
                        "mcap denomination mismatch for %s (%s): reported/reconstructed=%.1fx"
                        " — using reconstruction",
                        listing.symbol,
                        listing.currency,
                        ratio,
                    )
        else:
            reported = _positive(listing.market_cap)
            if reported is not None:
                mcap_usd = reported * usd  # fallback: reported is already major-unit

    return Metrics(
        major_ccy=major, price_major=price_major, mcap_usd=mcap_usd, addv_usd=addv_usd
    )


def _rank_key(
    listing: Listing, metrics: Metrics, exchange_pref: tuple[str, ...]
) -> tuple[float, int, float, int, int]:
    """Sort key for canonical selection (higher = more canonical, sorted desc).

    Primary: USD ADDV (dollar liquidity — collapses by orders of magnitude off the
    primary listing and is robust to the GBp mis-scaling). Tiebreaks: home listing
    (currency == financial_currency), USD market cap, shorter symbol (the common
    beats warrants/units/rights, which carry longer symbols), then a deterministic
    exchange preference order.
    """
    addv = metrics.addv_usd if metrics.addv_usd is not None else -1.0
    home = int(
        listing.currency is not None and listing.currency == listing.financial_currency
    )
    mcap = metrics.mcap_usd if metrics.mcap_usd is not None else -1.0
    pref = (
        exchange_pref.index(listing.exchange)
        if listing.exchange in exchange_pref
        else len(exchange_pref) + 1
    )
    return (addv, home, mcap, -len(listing.symbol), -pref)


def dedup_canonical(
    listings: list[Listing],
    usd_per_major: dict[str, float],
    *,
    config: IngestionFloorConfig | None = None,
    exchange_pref: tuple[str, ...] = (),
    keep_dual_class: bool = True,
) -> list[Listing]:
    """Collapse cross-listings to one canonical listing per entity.

    Grouping key: ``(normalize_name(longName|shortName), financial_currency)`` — the
    ``financial_currency`` guard prevents merging two genuinely different companies
    that share a normalized name (0/348 real clusters violated it in validation).
    A name-less listing falls back to its own symbol so it is never merged away.
    Within a group the max-``_rank_key`` line is canonical; when ``keep_dual_class``
    is set, same-exchange siblings clearing ``dual_class_min_addv_usd`` are also kept
    (recovers GOOG beside GOOGL). Order of survivors is deterministic.
    """
    cfg = config or IngestionFloorConfig()
    enriched = [(lst, derive_metrics(lst, usd_per_major)) for lst in listings]

    groups: dict[tuple[str, str | None], list[tuple[Listing, Metrics]]] = defaultdict(
        list
    )
    for lst, metrics in enriched:
        name_key = normalize_name(lst.long_name or lst.short_name)
        if not name_key:
            name_key = f"\x00sym\x00{lst.symbol}"  # never merge an unnamed listing
        groups[(name_key, lst.financial_currency)].append((lst, metrics))

    survivors: list[Listing] = []
    for members in groups.values():
        members.sort(
            key=lambda lm: _rank_key(lm[0], lm[1], exchange_pref), reverse=True
        )
        winner, _ = members[0]
        survivors.append(winner)
        if keep_dual_class:
            for sibling, sib_metrics in members[1:]:
                if (
                    sibling.exchange == winner.exchange
                    and sib_metrics.addv_usd is not None
                    and sib_metrics.addv_usd >= cfg.dual_class_min_addv_usd
                ):
                    survivors.append(sibling)
    # Stable output independent of input ordering (reproducible builds).
    survivors.sort(key=lambda lst: lst.symbol)
    return survivors


def passes_floor(
    metrics: Metrics, config: IngestionFloorConfig, *, is_member: bool = False
) -> bool:
    """Coarse anti-junk gate on the (already canonical) listing. Fail-open: an
    unknown (``None``) metric never rejects — only a KNOWN value below the band
    does. ``is_member`` selects the (lower) exit thresholds for existing members
    so a name doesn't churn out on a noisy month.
    """
    if metrics.price_major is None:  # unpriced / non-existent line
        return False
    mcap_below = (
        metrics.mcap_usd is not None
        and metrics.mcap_usd < config.mcap_usd.threshold(is_member=is_member)
    )
    addv_below = (
        metrics.addv_usd is not None
        and metrics.addv_usd < config.addv_usd.threshold(is_member=is_member)
    )
    return not (mcap_below or addv_below)
