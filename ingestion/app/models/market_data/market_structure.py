"""SQLAlchemy models for market-structure rollups (SPEC B1).

Sector / industry taxonomy + region-scoped leaders from ``yf.Sector``. Stored
as point-in-time (``as_of``) snapshots keyed by (sector_key, region) so history
is retained; ``region`` is the ISO-3166 alpha-2 code the rollup was fetched for.
No reverse relationship onto Instrument — these are market-wide, not per-ticker.
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger,
    Date,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models._shared import BaseModel


class SectorSnapshot(BaseModel):
    """Sector overview (market cap / weight / counts) per region, per snapshot."""

    __tablename__ = "sector_snapshots"
    __table_args__ = (
        UniqueConstraint("sector_key", "region", "as_of", name="uq_sector_snapshot"),
        Index("ix_sector_snapshots_key_region", "sector_key", "region"),
    )

    sector_key: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str] = mapped_column(String(8), nullable=False)
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(30), nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Numeric(28, 2), nullable=True)
    market_weight: Mapped[float | None] = mapped_column(Numeric(12, 8), nullable=True)
    companies_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    industries_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    employee_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class SectorIndustry(BaseModel):
    """Industry belonging to a sector (region-scoped taxonomy row)."""

    __tablename__ = "sector_industries"
    __table_args__ = (
        UniqueConstraint(
            "sector_key",
            "region",
            "as_of",
            "industry_key",
            name="uq_sector_industry",
        ),
        Index("ix_sector_industries_key_region", "sector_key", "region"),
    )

    sector_key: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str] = mapped_column(String(8), nullable=False)
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    industry_key: Mapped[str] = mapped_column(String(80), nullable=False)
    industry_name: Mapped[str | None] = mapped_column(String(150), nullable=True)


class SectorTopCompany(BaseModel):
    """Top constituent company of a sector (region-scoped)."""

    __tablename__ = "sector_top_companies"
    __table_args__ = (
        UniqueConstraint(
            "sector_key", "region", "as_of", "symbol", name="uq_sector_top_company"
        ),
        Index("ix_sector_top_companies_key_region", "sector_key", "region"),
    )

    sector_key: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str] = mapped_column(String(8), nullable=False)
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    weight: Mapped[float | None] = mapped_column(Numeric(12, 8), nullable=True)
    rating: Mapped[str | None] = mapped_column(String(50), nullable=True)


__all__ = ["SectorIndustry", "SectorSnapshot", "SectorTopCompany"]
