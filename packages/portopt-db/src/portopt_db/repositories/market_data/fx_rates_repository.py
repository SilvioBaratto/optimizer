"""Repository for the daily FX-rate feed — idempotent upserts + range reads.

The ingestion FX step writes rows here; the fund performance tracker reads them
and pivots into the ``optimizer.fx`` rates frame. Deduplicates on the natural
key ``(base_currency, quote_currency, date)`` so an at-least-once refetch
converges to one row. Caller owns the commit.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from portopt_db.models.market_data.fx_rates import FxRate
from portopt_db.repository import RepositoryBase


class FxRatesRepository(RepositoryBase):
    """Persist and read daily FX rates via idempotent upserts."""

    def upsert_rates(self, rows: list[dict[str, Any]]) -> int:
        """Upsert FX-rate rows, returning the count written.

        Rows are deduplicated on ``(base_currency, quote_currency, date)``
        before writing (last value wins within a single payload), so a payload
        may be replayed without accumulating duplicates. Rows missing any of
        base currency, quote currency, date, or rate are skipped.

        Args:
            rows: Mappings with ``base_currency``, ``quote_currency``, ``date``,
                and ``rate`` keys.

        Returns:
            Number of distinct rows passed to the upsert (including no-ops where
            the natural key already existed).
        """
        out: dict[tuple[str, str, dt.date], dict[str, Any]] = {}
        for r in rows:
            base = r.get("base_currency")
            quote = r.get("quote_currency")
            date = r.get("date")
            rate = r.get("rate")
            if not base or not quote or date is None or rate is None:
                continue
            out[(base, quote, date)] = {
                "id": uuid.uuid4(),
                "base_currency": base,
                "quote_currency": quote,
                "date": date,
                "rate": rate,
            }
        if not out:
            return 0
        self._upsert(
            FxRate,
            list(out.values()),
            constraint_name="uq_fx_rates_pair_date",
            update_columns=["rate", "updated_at"],
        )
        return len(out)

    def get_rates(
        self,
        base: str,
        quotes: Sequence[str] | None = None,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[FxRate]:
        """Read FX rates for one base currency, ordered by quote then date.

        Args:
            base: Base currency the returned rates are expressed in.
            quotes: Restrict to these quote currencies; ``None`` returns all.
            start: Inclusive lower date bound; ``None`` leaves it open.
            end: Inclusive upper date bound; ``None`` leaves it open.

        Returns:
            Matching :class:`FxRate` rows ordered by ``(quote_currency, date)``;
            empty when nothing matches. Suitable for pivoting into a rates frame.
        """
        stmt = select(FxRate).where(FxRate.base_currency == base)
        if quotes is not None:
            stmt = stmt.where(FxRate.quote_currency.in_(list(quotes)))
        if start is not None:
            stmt = stmt.where(FxRate.date >= start)
        if end is not None:
            stmt = stmt.where(FxRate.date <= end)
        stmt = stmt.order_by(FxRate.quote_currency, FxRate.date)
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["FxRatesRepository"]
