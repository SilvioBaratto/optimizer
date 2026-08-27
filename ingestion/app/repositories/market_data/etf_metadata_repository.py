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

from portopt_db.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFBondHoldings,
    ETFBondRating,
    ETFEquityHoldings,
    ETFFundOperations,
    ETFHolding,
    ETFMetadata,
    ETFSectorWeight,
)
from sqlalchemy import select

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
        category: str | None = None,
        description: str | None = None,
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
                    "category": category,
                    "description": description,
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
                "category",
                "description",
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
        # Dedup by holding_symbol within the batch: yfinance can repeat a symbol
        # (e.g. two share classes), and a multi-row ON CONFLICT that touches the
        # same natural key twice raises a PostgreSQL cardinality violation. Last
        # occurrence wins.
        by_symbol: dict[str, dict[str, Any]] = {}
        for h in holdings:
            symbol = h.get("symbol")
            if not symbol:
                continue
            by_symbol[symbol] = {
                "id": uuid.uuid4(),
                "instrument_id": instrument_id,
                "as_of": as_of,
                "holding_symbol": symbol,
                "holding_name": h.get("name"),
                "weight": h.get("weight"),
            }
        rows = list(by_symbol.values())
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

    def upsert_equity_holdings(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        metrics: dict[str, float],
    ) -> int:
        if not metrics:
            return 0
        self._upsert(
            ETFEquityHoldings,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "price_to_earnings": metrics.get("priceToEarnings"),
                    "price_to_book": metrics.get("priceToBook"),
                    "price_to_sales": (
                        metrics.get("priceToSales")
                        or metrics.get("priceToSalesTrailing12Months")
                    ),
                    "price_to_cashflow": metrics.get("priceToCashflow"),
                    "median_market_cap": metrics.get("medianMarketCap"),
                    "three_year_earnings_growth": metrics.get(
                        "threeYearEarningsGrowth"
                    ),
                }
            ],
            index_elements=["instrument_id", "as_of"],
            update_columns=[
                "price_to_earnings",
                "price_to_book",
                "price_to_sales",
                "price_to_cashflow",
                "median_market_cap",
                "three_year_earnings_growth",
                "updated_at",
            ],
        )
        return 1

    def upsert_bond_holdings(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        metrics: dict[str, float],
    ) -> int:
        if not metrics:
            return 0
        self._upsert(
            ETFBondHoldings,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "duration": metrics.get("duration"),
                    "maturity": metrics.get("maturity"),
                    "credit_quality": metrics.get("creditQuality"),
                }
            ],
            index_elements=["instrument_id", "as_of"],
            update_columns=["duration", "maturity", "credit_quality", "updated_at"],
        )
        return 1

    def upsert_bond_ratings(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        ratings: dict[str, float],
    ) -> int:
        rows = [
            {
                "id": uuid.uuid4(),
                "instrument_id": instrument_id,
                "as_of": as_of,
                "rating": rating,
                "weight": weight,
            }
            for rating, weight in ratings.items()
        ]
        if not rows:
            return 0
        self._upsert(
            ETFBondRating,
            rows,
            index_elements=["instrument_id", "as_of", "rating"],
            update_columns=["weight", "updated_at"],
        )
        return len(rows)

    def upsert_fund_operations(
        self,
        instrument_id: uuid.UUID,
        as_of: dt.date,
        metrics: dict[str, float],
    ) -> int:
        if not metrics:
            return 0
        self._upsert(
            ETFFundOperations,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "annual_report_expense_ratio": metrics.get(
                        "annualReportExpenseRatio"
                    ),
                    "annual_holdings_turnover": metrics.get("annualHoldingsTurnover"),
                    "total_net_assets": metrics.get("totalNetAssets"),
                }
            ],
            index_elements=["instrument_id", "as_of"],
            update_columns=[
                "annual_report_expense_ratio",
                "annual_holdings_turnover",
                "total_net_assets",
                "updated_at",
            ],
        )
        return 1

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
