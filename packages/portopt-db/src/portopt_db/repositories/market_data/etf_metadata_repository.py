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
from portopt_db.repository import RepositoryBase


def _pick(metrics: dict[str, float], *keys: str) -> float | None:
    """Return the first present metric across candidate keys.

    ``funds_data`` depth DataFrames are indexed by human **display labels**
    (e.g. ``"Price/Earnings"``, ``"Duration"``, ``"Annual Report Expense
    Ratio"``) — exactly what ``_first_col_dict`` emits as dict keys. Earlier code
    looked up the raw camelCase field names, which never matched the label form,
    so every metric landed NULL. The label spelling is tried first; the camelCase
    spelling is kept as a forward-compat fallback.
    """
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


class ETFMetadataRepository(RepositoryBase):
    """Repository for ETF fund metadata tables.

    Exposes idempotent upsert methods for each sub-table (metadata, asset
    classes, holdings, sector weights, equity/bond characteristics, fund
    operations) and read helpers that return the latest snapshot.
    """

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
        """Upsert core ETF metadata for an instrument.

        Natural key is ``instrument_id``; a re-run updates all columns except
        the primary key. Pass ``None`` for fields not available in the source
        data.

        Args:
            instrument_id: Identifies the instrument row this metadata belongs to.
            aum: Total assets under management in the fund's base currency.
            nav: Net asset value per share.
            fund_family: Name of the fund issuer or asset manager.
            legal_type: Legal structure (e.g. ``"ETF"``, ``"Open-End Fund"``).
            expense_ratio: Annual total expense ratio as a decimal fraction.
            base_currency: ISO 4217 currency code for fund NAV and AUM.
            as_of: Snapshot date of the source data.
            category: Morningstar or provider-assigned category label.
            description: Free-text fund description.
        """
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
        """Upsert the asset-class allocation breakdown for an instrument snapshot.

        Natural key is ``(instrument_id, as_of)``; percentages are decimal
        fractions (0–1).

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for this allocation.
            stock_pct: Fraction of NAV allocated to equities.
            bond_pct: Fraction of NAV allocated to fixed income.
            cash_pct: Fraction of NAV held in cash or equivalents.
            other_pct: Fraction of NAV in all other asset classes.
        """
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
        """Upsert the top-holdings list for an instrument snapshot.

        Each element of ``holdings`` must contain ``"symbol"``; rows without
        it are silently dropped. Within the batch, later occurrences of the
        same symbol overwrite earlier ones before the upsert fires.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for this holdings list.
            holdings: Raw holding dicts with keys ``symbol``, ``name``,
                ``weight``.

        Returns:
            Number of rows actually written after dedup; 0 when all rows lack
            a symbol.
        """
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
        """Upsert sector-weight allocations for an instrument snapshot.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for this allocation.
            weights: Maps sector label to weight fraction (0–1).

        Returns:
            Number of sector rows written; 0 when ``weights`` is empty.
        """
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
        """Upsert equity characteristic metrics for an instrument snapshot.

        Metric keys may be display labels (e.g. ``"Price/Earnings"``) or
        camelCase field names; ``_pick`` resolves both spellings.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for these metrics.
            metrics: Dict of metric name to numeric value from the source data.

        Returns:
            1 if a row was written; 0 when ``metrics`` is empty.
        """
        if not metrics:
            return 0
        self._upsert(
            ETFEquityHoldings,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "price_to_earnings": _pick(
                        metrics, "Price/Earnings", "priceToEarnings"
                    ),
                    "price_to_book": _pick(metrics, "Price/Book", "priceToBook"),
                    "price_to_sales": _pick(
                        metrics,
                        "Price/Sales",
                        "priceToSales",
                        "priceToSalesTrailing12Months",
                    ),
                    "price_to_cashflow": _pick(
                        metrics, "Price/Cashflow", "priceToCashflow"
                    ),
                    "median_market_cap": _pick(
                        metrics, "Median Market Cap", "medianMarketCap"
                    ),
                    "three_year_earnings_growth": _pick(
                        metrics, "3 Year Earnings Growth", "threeYearEarningsGrowth"
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
        """Upsert bond characteristic metrics for an instrument snapshot.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for these metrics.
            metrics: Dict of metric name (display label or camelCase) to value.

        Returns:
            1 if a row was written; 0 when ``metrics`` is empty.
        """
        if not metrics:
            return 0
        self._upsert(
            ETFBondHoldings,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "duration": _pick(metrics, "Duration", "duration"),
                    "maturity": _pick(metrics, "Maturity", "maturity"),
                    "credit_quality": _pick(
                        metrics, "Credit Quality", "creditQuality", "credit_quality"
                    ),
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
        """Upsert bond credit-rating weight breakdown for an instrument snapshot.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for this distribution.
            ratings: Maps rating label (e.g. ``"AAA"``) to weight fraction (0–1).

        Returns:
            Number of rating rows written; 0 when ``ratings`` is empty.
        """
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
        """Upsert operational metrics for an instrument snapshot.

        Args:
            instrument_id: Identifies the ETF instrument.
            as_of: Snapshot date for these metrics.
            metrics: Dict of metric name (display label or camelCase) to value.

        Returns:
            1 if a row was written; 0 when ``metrics`` is empty.
        """
        if not metrics:
            return 0
        self._upsert(
            ETFFundOperations,
            [
                {
                    "id": uuid.uuid4(),
                    "instrument_id": instrument_id,
                    "as_of": as_of,
                    "annual_report_expense_ratio": _pick(
                        metrics,
                        "Annual Report Expense Ratio",
                        "annualReportExpenseRatio",
                    ),
                    "annual_holdings_turnover": _pick(
                        metrics, "Annual Holdings Turnover", "annualHoldingsTurnover"
                    ),
                    "total_net_assets": _pick(
                        metrics, "Total Net Assets", "totalNetAssets"
                    ),
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
        """Return the ETF metadata row for an instrument.

        Args:
            instrument_id: Identifies the ETF instrument.

        Returns:
            The metadata row, or ``None`` if none has been ingested yet.
        """
        return self.session.execute(
            select(ETFMetadata).where(ETFMetadata.instrument_id == instrument_id)
        ).scalar_one_or_none()

    def get_asset_classes(self, instrument_id: uuid.UUID) -> ETFAssetClass | None:
        """Return the most-recent asset-class allocation for an instrument.

        Args:
            instrument_id: Identifies the ETF instrument.

        Returns:
            The latest ``ETFAssetClass`` row ordered by ``as_of`` descending,
            or ``None`` if none has been ingested yet.
        """
        return self.session.execute(
            select(ETFAssetClass)
            .where(ETFAssetClass.instrument_id == instrument_id)
            .order_by(ETFAssetClass.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()
