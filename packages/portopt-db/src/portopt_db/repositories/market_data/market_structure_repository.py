"""Repository for market-structure rollups (SPEC B1) — idempotent upserts.

Uses ``RepositoryBase._upsert`` with ``index_elements`` so it compiles on both
PostgreSQL and the SQLite test engine; an at-least-once re-run converges to one
row per natural key.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import select

from portopt_db.models.market_data.market_structure import (
    SectorIndustry,
    SectorSnapshot,
    SectorTopCompany,
)
from portopt_db.repository import RepositoryBase


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return int(f) if f is not None else None


class MarketStructureRepository(RepositoryBase):
    def upsert_sector_snapshot(
        self,
        sector_key: str,
        region: str,
        as_of: dt.date,
        *,
        name: str | None,
        symbol: str | None,
        market_cap: float | None,
        market_weight: float | None,
        companies_count: int | None,
        industries_count: int | None,
        employee_count: int | None,
    ) -> None:
        self._upsert(
            SectorSnapshot,
            [
                {
                    "id": uuid.uuid4(),
                    "sector_key": sector_key,
                    "region": region,
                    "as_of": as_of,
                    "name": name,
                    "symbol": symbol,
                    "market_cap": market_cap,
                    "market_weight": market_weight,
                    "companies_count": companies_count,
                    "industries_count": industries_count,
                    "employee_count": employee_count,
                }
            ],
            index_elements=["sector_key", "region", "as_of"],
            update_columns=[
                "name",
                "symbol",
                "market_cap",
                "market_weight",
                "companies_count",
                "industries_count",
                "employee_count",
                "updated_at",
            ],
        )

    def upsert_industries(
        self,
        sector_key: str,
        region: str,
        as_of: dt.date,
        industries: list[dict[str, Any]],
    ) -> int:
        by_key: dict[str, dict[str, Any]] = {}
        for ind in industries:
            key = ind.get("key")
            if not key:
                continue
            by_key[str(key)] = {
                "id": uuid.uuid4(),
                "sector_key": sector_key,
                "region": region,
                "as_of": as_of,
                "industry_key": str(key)[:80],
                "industry_name": (ind.get("name") or None),
            }
        rows = list(by_key.values())
        if not rows:
            return 0
        self._upsert(
            SectorIndustry,
            rows,
            index_elements=["sector_key", "region", "as_of", "industry_key"],
            update_columns=["industry_name", "updated_at"],
        )
        return len(rows)

    def upsert_top_companies(
        self,
        sector_key: str,
        region: str,
        as_of: dt.date,
        companies: list[dict[str, Any]],
    ) -> int:
        by_symbol: dict[str, dict[str, Any]] = {}
        for c in companies:
            symbol = c.get("symbol")
            if not symbol:
                continue
            by_symbol[str(symbol)] = {
                "id": uuid.uuid4(),
                "sector_key": sector_key,
                "region": region,
                "as_of": as_of,
                "symbol": str(symbol)[:30],
                "name": c.get("name") or None,
                "weight": _num(c.get("weight")),
                "rating": (str(c["rating"])[:50] if c.get("rating") else None),
            }
        rows = list(by_symbol.values())
        if not rows:
            return 0
        self._upsert(
            SectorTopCompany,
            rows,
            index_elements=["sector_key", "region", "as_of", "symbol"],
            update_columns=["name", "weight", "rating", "updated_at"],
        )
        return len(rows)

    def get_latest_sector_as_of(self, sector_key: str, region: str) -> dt.date | None:
        stmt = (
            select(SectorSnapshot.as_of)
            .where(
                SectorSnapshot.sector_key == sector_key,
                SectorSnapshot.region == region,
            )
            .order_by(SectorSnapshot.as_of.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_sector_snapshot(
        self, sector_key: str, region: str
    ) -> SectorSnapshot | None:
        stmt = (
            select(SectorSnapshot)
            .where(
                SectorSnapshot.sector_key == sector_key,
                SectorSnapshot.region == region,
            )
            .order_by(SectorSnapshot.as_of.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()
