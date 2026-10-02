"""Pydantic v2 schema for the global daily digest builder (``market_journal``)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# The FRED series catalog lives in ``fred_scraper.FRED_SERIES`` and must not be
# re-typed here (the ingestion import-hygiene guard allowlists certain OAS series
# ids only in that one file). An empty ``fred_series`` therefore means "use the
# service's default", which the service derives from that catalog.
DEFAULT_MARKETS: tuple[str, ...] = ("US",)


class MarketJournalBuildRequest(BaseModel):
    """Request body for building one global daily digest row.

    Attributes:
        as_of: Trading date to build the digest for; ``None`` means today.
        region: Market region identifier stored on the digest row.
        fred_series: FRED series folded into ``macro_deltas``; empty means the
            service's default catalog.
        markets: ``market_summaries`` markets folded into ``market_moves``.
        news_limit: Maximum recent macro-news articles folded into
            ``news_themes``.
    """

    as_of: date | None = Field(
        default=None,
        description="Trading date to build the digest for. None means today.",
    )
    region: str = Field(
        default="US",
        description="Market region identifier stored on the digest row.",
    )
    fred_series: tuple[str, ...] = Field(
        default=(),
        description="FRED series summarized into macro_deltas; empty = default.",
    )
    markets: tuple[str, ...] = Field(
        default=DEFAULT_MARKETS,
        description="market_summaries markets summarized into market_moves.",
    )
    news_limit: int = Field(
        default=20,
        description="Maximum recent macro-news articles folded into news_themes.",
    )
