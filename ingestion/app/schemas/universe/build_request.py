"""Pydantic v2 schemas for Trading212 universe endpoints."""

from pydantic import BaseModel, Field


class UniverseBuildRequest(BaseModel):
    """Request to trigger a universe build."""

    exchanges: list[str] | None = Field(
        default=None,
        description="Restrict to specific exchanges (e.g. ['NYSE', 'NASDAQ']). None = all configured.",
    )
    max_workers: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of concurrent workers for ticker discovery.",
    )
