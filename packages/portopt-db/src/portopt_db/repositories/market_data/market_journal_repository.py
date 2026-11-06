"""Repository for global daily macro/market/news digest rows (``market_journal``).

Uses ``index_elements`` upserts (SQLite-testable); one row per ``as_of`` date.
The caller owns the session and controls commit/rollback.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import select

from portopt_db.models.market_data.market_journal import MarketJournal
from portopt_db.repository import RepositoryBase

_UPDATE_COLUMNS = [
    "macro_deltas",
    "market_moves",
    "news_themes",
    "narrative",
    "source_counts",
    "updated_at",
]


class MarketJournalRepository(RepositoryBase):
    """Persists and retrieves global daily digest rows.

    All writes go through the idempotent ``_upsert`` path keyed on ``as_of``.
    Reads never issue commits; that responsibility belongs to the caller.
    """

    def upsert_journal(
        self,
        as_of: date,
        *,
        macro_deltas: dict[str, Any],
        market_moves: dict[str, Any],
        news_themes: dict[str, Any],
        narrative: str,
        source_counts: dict[str, Any],
        region: str = "US",
    ) -> int:
        """Write or refresh the daily digest for one trading date.

        Idempotent: re-running for the same ``as_of`` overwrites all data
        columns and advances ``updated_at`` but preserves ``created_at``. The
        unique key is ``as_of`` alone, so a date holds one digest whatever its
        region; a write for a different region than the stored one is refused
        rather than replacing it.

        Args:
            as_of: Trading date for the digest row.
            macro_deltas: Key FRED series latest-vs-prior-observation deltas.
            market_moves: Index and sector moves from ``market_summaries``.
            news_themes: Top macro and ticker news themes aggregated by volume.
            narrative: Deterministic template string for agent injection.
            source_counts: Row-count provenance for each section.
            region: Market region identifier; defaults to ``"US"``.

        Returns:
            Number of rows processed (always 1 when a row is supplied).

        Raises:
            ValueError: A digest for *as_of* is already stored under another
                region.
        """
        stored_region = self.session.execute(
            select(MarketJournal.region).where(MarketJournal.as_of == as_of)
        ).scalar_one_or_none()
        if stored_region is not None and stored_region != region:
            raise ValueError(
                f"market_journal already holds a {stored_region!r} digest for "
                f"{as_of}; one digest per date is supported, so a {region!r} "
                "digest would replace it"
            )
        row = {
            "id": uuid.uuid4(),
            "as_of": as_of,
            "region": region,
            "macro_deltas": macro_deltas,
            "market_moves": market_moves,
            "news_themes": news_themes,
            "narrative": narrative,
            "source_counts": source_counts,
        }
        return self._upsert(
            MarketJournal,
            [row],
            index_elements=["as_of"],
            update_columns=_UPDATE_COLUMNS,
        )

    def get_for_date(self, as_of: date) -> MarketJournal | None:
        """Return the digest row for the given trading date, or ``None``.

        Args:
            as_of: Trading date to look up.

        Returns:
            The ``MarketJournal`` instance, or ``None`` if no row exists.
        """
        stmt = select(MarketJournal).where(MarketJournal.as_of == as_of)
        return self.session.execute(stmt).scalars().first()

    def get_range(self, start: date, end: date) -> Sequence[MarketJournal]:
        """Return digest rows whose ``as_of`` falls in ``[start, end]``.

        Results are ordered by ``as_of`` ascending so callers receive a
        chronological timeline without extra sorting.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Sequence of matching rows, possibly empty, oldest first.
        """
        stmt = (
            select(MarketJournal)
            .where(MarketJournal.as_of >= start, MarketJournal.as_of <= end)
            .order_by(MarketJournal.as_of)
        )
        return self.session.execute(stmt).scalars().all()
