"""ETF-specific screen helpers: share-class / cross-listing dedup.

The equity screens (market cap, P/E, ROE …) do not apply to funds, so ETFs run a
dedicated pipeline whose only gate is the shared HistoricalDataFilter (the
≥750-trading-day history bar); classification (leveraged/inverse/unclassifiable
rejected) happens upstream in the builder. Investability — liquidity, ADDV,
price — is deliberately NOT screened here: it is a downstream fund-layer concern
(optimizer/universe), computed on the price panel at portfolio construction.
Trading 212 exposes no AUM and Yahoo omits it for most UCITS listings, so an
ingestion-side size/liquidity gate can't be applied reliably anyway.
"""

from __future__ import annotations

from typing import Any


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
