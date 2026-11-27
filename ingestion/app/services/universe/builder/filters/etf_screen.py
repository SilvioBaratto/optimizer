"""ETF-specific helper: share-class / cross-listing dedup.

Ingestion applies no investability filtering. ETFs are admitted on classification
(leveraged/inverse/unclassifiable rejected upstream in the builder) alone; the one
transform kept here is ISIN dedup, which collapses cross-listings of the same fund
to a single listing (noise-reduction, not investability). Investability —
liquidity, ADDV, price, history — is a downstream fund-layer concern
(optimizer/universe), computed on the price panel at portfolio construction.
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
