"""Repository for market summaries (SPEC B3) — idempotent upserts.

Uses ``index_elements`` upserts (SQLite-testable); one row per
(market, symbol, as_of).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from portopt_db.models.market_data.market_summary import MarketSummary
from portopt_db.repository import RepositoryBase


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _str(v: Any, n: int) -> str | None:
    return str(v)[:n] if v is not None else None


class MarketSummaryRepository(RepositoryBase):
    """Persists and refreshes per-market equity/ETF summary snapshots."""

    def upsert_summaries(
        self,
        market: str,
        as_of: dt.date,
        rows: list[dict[str, Any]],
    ) -> int:
        """Write or refresh summary rows for one market snapshot.

        Deduplicates by symbol within the batch before upserting, so callers
        need not pre-deduplicate.  Rows missing a ``symbol`` key are silently
        skipped.

        Args:
            market: Market or exchange identifier stored on every row.
            as_of: Trading date these summaries represent.
            rows: Raw dicts; recognised keys are ``symbol``, ``short_name``,
                ``price``, ``change``, ``change_percent``, ``previous_close``,
                and ``market_state``.

        Returns:
            Number of rows written after deduplication and symbol filtering.
        """
        by_symbol: dict[str, dict[str, Any]] = {}
        for r in rows:
            symbol = r.get("symbol")
            if not symbol:
                continue
            by_symbol[str(symbol)] = {
                "id": uuid.uuid4(),
                "market": market,
                "symbol": str(symbol)[:40],
                "as_of": as_of,
                "short_name": _str(r.get("short_name"), 255),
                "price": _num(r.get("price")),
                "change": _num(r.get("change")),
                "change_percent": _num(r.get("change_percent")),
                "previous_close": _num(r.get("previous_close")),
                "market_state": _str(r.get("market_state"), 20),
            }
        prepared = list(by_symbol.values())
        if not prepared:
            return 0
        self._upsert(
            MarketSummary,
            prepared,
            index_elements=["market", "symbol", "as_of"],
            update_columns=[
                "short_name",
                "price",
                "change",
                "change_percent",
                "previous_close",
                "market_state",
                "updated_at",
            ],
        )
        return len(prepared)
