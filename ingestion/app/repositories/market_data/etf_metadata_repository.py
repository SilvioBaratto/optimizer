"""Repository for ETF fund metadata — idempotent upserts + reads.

Every write is an ``INSERT ... ON CONFLICT DO UPDATE`` on the table's natural
key (via ``RepositoryBase._upsert`` with ``index_elements``, which compiles on
both PostgreSQL and the SQLite test engine), so an at-least-once re-run
converges to one row.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import select

from app.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFHolding,
    ETFMetadata,
    ETFSectorWeight,
)
from app.repositories._shared.base import RepositoryBase


class ETFMetadataRepository(RepositoryBase):
    def upsert_metadata(
        self,
        instrument_id: uuid.UUID,
        *,
        aum: float | None,
        nav: float | None,
        fund_family: str | None,
        legal_type: str | None,
        expense_ratio: float | None,
        base_currency: str | None,
        as_of: dt.date | None,
    ) -> None:
        self._upsert(
            ETFMetadata,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "aum": aum,
                    "nav": nav,
                    "fund_family": fund_family,
                    "legal_type": legal_type,
                    "expense_ratio": expense_ratio,
                    "base_currency": base_currency,
                    "as_of": as_of,
                }
            ],
            index_elements=["instrument_id"],
            update_columns=[
                "aum",
                "nav",
                "fund_family",
                "legal_type",
                "expense_ratio",
                "base_currency",
                "as_of",
                "updated_at",
            ],
        )

    def upsert_asset_classes(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        *,
        stock_pct: float | None,
        bond_pct: float | None,
        cash_pct: float | None,
        other_pct: float | None,
    ) -> None:
        self._upsert(
            ETFAssetClass,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "stock_pct": stock_pct,
                    "bond_pct": bond_pct,
                    "cash_pct": cash_pct,
                    "other_pct": other_pct,
                }
            ],
            index_elements=["instrument_id", "as_of"],
            update_columns=[
                "stock_pct",
                "bond_pct",
                "cash_pct",
                "other_pct",
                "updated_at",
            ],
        )

    def upsert_holdings(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        holdings: list[dict[str, Any]],
    ) -> int:
        rows = [
            {
                "id": uuid.uuid4(),
                "instrument_id": instrument_id,
                "as_of": as_of,
                "holding_symbol": h["symbol"],
                "holding_name": h.get("name"),
                "weight": h.get("weight"),
            }
            for h in holdings
            if h.get("symbol")
        ]
        if not rows:
            return 0
        self._upsert(
            ETFHolding,
            rows,
            index_elements=["instrument_id", "as_of", "holding_symbol"],
            update_columns=["holding_name", "weight", "updated_at"],
        )
        return len(rows)

    def upsert_sector_weights(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        weights: dict[str, float],
    ) -> int:
        rows = [
            {
                "id": uuid.uuid4(),
                "instrument_id": instrument_id,
                "as_of": as_of,
                "sector": sector,
                "weight": weight,
            }
            for sector, weight in weights.items()
        ]
        if not rows:
            return 0
        self._upsert(
            ETFSectorWeight,
            rows,
            index_elements=["instrument_id", "as_of", "sector"],
            update_columns=["weight", "updated_at"],
        )
        return len(rows)

    # ------------------------------------------------------------------ reads

    def get_metadata(self, instrument_id: uuid.UUID) -> ETFMetadata | None:
        return self.session.execute(
            select(ETFMetadata).where(ETFMetadata.instrument_id == instrument_id)
        ).scalar_one_or_none()

    def get_asset_classes(self, instrument_id: uuid.UUID) -> ETFAssetClass | None:
        return self.session.execute(
            select(ETFAssetClass)
            .where(ETFAssetClass.instrument_id == instrument_id)
            .order_by(ETFAssetClass.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()
