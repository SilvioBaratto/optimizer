"""ETF-specific screen: AUM + liquidity filters, and share-class/cross-listing
dedup.

The equity screens (market cap, P/E, ROE …) do not apply to funds, so ETFs run
a dedicated pipeline: minimum AUM, minimum average daily dollar volume, and the
shared HistoricalDataFilter for the ≥750-trading-day history bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.universe.trading212.config import UniverseBuilderConfig


@dataclass
class AUMFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "AUMFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
        if not data:
            return False, "No data available"
        aum = data.get("totalAssets") or data.get("netAssets")
        if aum is None:
            return False, "No AUM (totalAssets) data"
        if aum < self.config.etf_min_aum:
            return (
                False,
                f"AUM ${aum / 1e6:.0f}M < ${self.config.etf_min_aum / 1e6:.0f}M",
            )
        return True, f"AUM ${aum / 1e6:.0f}M"


@dataclass
class ETFLiquidityFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "ETFLiquidityFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
        if not data:
            return False, "No data available"
        volume = data.get("averageVolume") or data.get("averageVolume10days")
        price = (
            data.get("regularMarketPrice")
            or data.get("currentPrice")
            or data.get("previousClose")
        )
        if not volume or not price:
            return False, "No volume/price for liquidity"
        addv = volume * price
        if addv < self.config.etf_min_addv:
            return (
                False,
                f"ADV ${addv / 1e6:.1f}M < ${self.config.etf_min_addv / 1e6:.1f}M",
            )
        return True, f"ADV ${addv / 1e6:.1f}M"


def dedup_etfs_by_isin(
    candidates: list[tuple[str, dict[str, Any]]],
    preference: tuple[str, ...],
) -> list[tuple[str, dict[str, Any]]]:
    """Collapse cross-listings of the same fund (identical ISIN) to a single
    listing on the most-preferred exchange. Candidates without an ISIN are kept
    as-is (cannot be deduped).

    Args:
        candidates: ``(exchange_name, t212_instrument)`` pairs.
        preference: exchange names, most-preferred first.

    Returns:
        The deduped candidate list, order-stable for kept items.
    """
    rank = {name: i for i, name in enumerate(preference)}
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    passthrough: list[tuple[str, dict[str, Any]]] = []

    for exchange_name, inst in candidates:
        isin = inst.get("isin")
        if not isin:
            passthrough.append((exchange_name, inst))
            continue
        current = best.get(isin)
        if current is None:
            best[isin] = (exchange_name, inst)
            continue
        # Lower rank index = more preferred. Unknown exchanges sort last.
        if rank.get(exchange_name, len(rank)) < rank.get(current[0], len(rank)):
            best[isin] = (exchange_name, inst)

    return list(best.values()) + passthrough
