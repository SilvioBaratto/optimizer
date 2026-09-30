"""SQLAlchemy model for the global daily macro/market/news digest (``market_journal``).

One row per trading day; written by the ingestion ``daily_events`` step from
already-ingested FRED, market-summary, and news rows.  The narrative is a
deterministic template — no LLM is involved — so every row is reproducible for
audit.  Fund agents and the cockpit read this table via ``MarketJournalRepository``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import JSON, Date, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# JSONB on PostgreSQL, plain JSON for SQLite in tests.
_JSON = JSON().with_variant(JSONB, "postgresql")


class MarketJournal(BaseModel):
    """One global macro/market/news digest row per trading day.

    Written by the ingestion ``daily_events`` step; read by the fund history
    tools and the cockpit.  Deterministic — the narrative is templated, never
    LLM-generated, so the row is reproducible for audit.

    Attributes:
        as_of: Trading date this digest represents; unique per region.
        region: Market region identifier, reserved for multi-region expansion.
        macro_deltas: Key FRED series latest-vs-prior-observation deltas.
        market_moves: Index and sector moves from ``market_summaries``.
        news_themes: Top ``macro_news_themes`` plus top-N ``ticker_news``
            headlines aggregated by volume.
        narrative: Deterministic template string assembled from the JSON
            sections; suitable for direct injection into an agent prompt.
        source_counts: Row counts feeding each section — cheap provenance
            metadata for audit and debugging.
    """

    __tablename__ = "market_journal"
    __table_args__ = (
        UniqueConstraint("as_of", name="uq_market_journal_as_of"),
        Index("ix_market_journal_as_of", "as_of"),
    )

    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    region: Mapped[str] = mapped_column(String(8), nullable=False, default="US")
    macro_deltas: Mapped[dict[str, Any]] = mapped_column(
        _JSON, nullable=False, default=dict
    )
    market_moves: Mapped[dict[str, Any]] = mapped_column(
        _JSON, nullable=False, default=dict
    )
    news_themes: Mapped[dict[str, Any]] = mapped_column(
        _JSON, nullable=False, default=dict
    )
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_counts: Mapped[dict[str, Any]] = mapped_column(
        _JSON, nullable=False, default=dict
    )


__all__ = ["MarketJournal"]
