"""Pydantic v2 schemas for macro regime data endpoints."""

from pydantic import BaseModel, Field

from app.schemas._shared import AsyncJobProgress


class FredFetchRequest(BaseModel):
    """Request body for FRED time-series fetch."""

    series_ids: list[str] | None = Field(
        default=None,
        description="FRED series IDs to fetch. None means all configured series.",
    )
    incremental: bool = Field(
        default=True,
        description=(
            "When True, re-fetch a trailing window behind the last stored date "
            "(to pick up restated values) instead of the full history."
        ),
    )


class MacroFetchProgress(AsyncJobProgress):
    """Progress info for a macro fetch background job."""

    current_country: str = ""


class MacroNewsFetchRequest(BaseModel):
    """Request body for macro news fetch."""

    fetch_full_content: bool = Field(
        default=True,
        description="Whether to scrape full article content.",
    )
    max_articles: int = Field(
        default=100,
        description="Maximum number of articles to fetch.",
    )
