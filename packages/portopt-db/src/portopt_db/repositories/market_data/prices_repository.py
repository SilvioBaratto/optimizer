"""Repository for an instrument's price bars, dividends and stock splits."""

import logging
from collections.abc import Sequence
from datetime import date
from uuid import UUID

import pandas as pd
from sqlalchemy import func, select

from portopt_db.models.market_data.yfinance_data import (
    Dividend,
    PriceHistory,
    StockSplit,
)
from portopt_db.repositories.market_data._mapping import (
    _last_wins,
    _safe_date,
    _safe_float,
    _safe_int,
    _safe_str,
)
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)


# Trading 212 lists sub-unit currencies as GBX/ZAC while Yahoo tags them GBp/ZAc.
_SUB_UNIT_ALIASES = {"GBX": "GBp", "ZAC": "ZAc"}


def _same_currency_unit(a: str, b: str) -> bool:
    """Return True when two currency codes denote the exact same unit.

    The comparison is case-sensitive on purpose: ``GBP`` and ``GBp`` differ by a
    factor of 100.
    """
    return _SUB_UNIT_ALIASES.get(a, a) == _SUB_UNIT_ALIASES.get(b, b)


class PricesRepository(RepositoryBase):
    """Price bars, dividends and stock splits of an instrument."""

    def upsert_price_history(
        self,
        instrument_id: UUID,
        history_df: pd.DataFrame,
        price_unit: str | None = None,
    ) -> int:
        """Upsert daily OHLCV rows from a yfinance history DataFrame.

        ``price_unit`` records the listing currency the prices are quoted in
        (e.g. ``"GBX"``); values are stored as-is without currency conversion.
        """
        rows = []
        for idx, row_data in history_df.iterrows():
            # _safe_date coerces pd.NaT (a datetime subclass) to None; a NaT
            # index row would otherwise write date=NaT into a NOT NULL column
            # and abort the whole ticker's price INSERT.
            dt = _safe_date(idx)
            if dt is None:
                continue

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "date": dt,
                    "open": _safe_float(row_data.get("Open")),
                    "high": _safe_float(row_data.get("High")),
                    "low": _safe_float(row_data.get("Low")),
                    "close": _safe_float(row_data.get("Close")),
                    "volume": _safe_int(row_data.get("Volume")),
                    "dividends": _safe_float(row_data.get("Dividends")),
                    "stock_splits": _safe_float(row_data.get("Stock Splits")),
                    "capital_gains": _safe_float(row_data.get("Capital Gains")),
                    "price_unit": price_unit,
                }
            )

        rows = _last_wins(rows, "date")
        return self._upsert(
            PriceHistory,
            rows,
            constraint_name="uq_price_history_instrument_date",
        )

    def get_price_history(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> Sequence[PriceHistory]:
        """Return price bars for an instrument, optionally capped at ``limit``.

        All bars in the date window are returned unless ``limit`` is given: a
        default cap would silently drop the newest bars of a long backfill for
        ascending callers (stale marks) and the oldest for descending ones.

        Ordering is DESC by default (``limit`` keeps the most-recent bars).
        Pass ``ascending=True`` to order ASC instead — the ``limit`` then keeps
        the *earliest* bars, so a ``start_date`` window returns the first bars
        after it rather than the last (opt-in; the default is unchanged).
        """
        stmt = select(PriceHistory).where(PriceHistory.instrument_id == instrument_id)
        if start_date:
            stmt = stmt.where(PriceHistory.date >= start_date)
        if end_date:
            stmt = stmt.where(PriceHistory.date <= end_date)
        order_col = PriceHistory.date.asc() if ascending else PriceHistory.date.desc()
        stmt = stmt.order_by(order_col)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.execute(stmt).scalars().all()

    def upsert_dividends(
        self,
        instrument_id: UUID,
        dividends: pd.Series | pd.DataFrame,
        price_currency: str | None = None,
    ) -> int:
        """Upsert dividend data from yfinance.

        Limitation: because the table has no currency column, an instrument
        whose dividends are all paid in another currency (e.g. a London line
        whose payments are all tagged ``USD``) stores no rows. Its dividends
        staleness stamp then stays NULL, so the category is fetched again, and
        the skip warning logged again, on every incremental run.

        Args:
            instrument_id: The instrument these rows belong to.
            dividends: A Series (index=date, value=amount) or, when Yahoo tags
                payments with a currency, a DataFrame with ``Dividends`` and
                ``currency`` columns.
            price_currency: Currency the instrument's prices are quoted in
                (``"GBp"`` and ``"GBX"`` are equivalent). The table has no
                currency column, so amounts are stored in the price currency
                and DataFrame rows paid in any other currency, or in an unknown
                one when this is ``None``, are skipped instead of being stored
                in the wrong unit. Blank currency tags count as the price
                currency, and a Series is always taken as such.

        Returns:
            Number of rows written.
        """
        paid_in: pd.Series | None = None
        if isinstance(dividends, pd.DataFrame):
            if "Dividends" not in dividends.columns:
                return 0
            if "currency" in dividends.columns:
                paid_in = dividends["currency"]
            dividends = dividends["Dividends"]

        rows = []
        skipped_foreign = 0
        for pos, (idx, amount) in enumerate(dividends.items()):
            dt = _safe_date(idx)
            amt = _safe_float(amount)
            if dt is None or amt is None:
                continue
            if paid_in is not None:
                code = _safe_str(paid_in.iloc[pos])
                if code and not (
                    price_currency and _same_currency_unit(code, price_currency)
                ):
                    skipped_foreign += 1
                    continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "date": dt,
                    "amount": amt,
                }
            )

        if skipped_foreign:
            logger.warning(
                "Skipped %d dividend(s) for instrument %s paid in a currency "
                "other than the price currency %r",
                skipped_foreign,
                instrument_id,
                price_currency,
            )

        return self._upsert(
            Dividend,
            _last_wins(rows, "date"),
            constraint_name="uq_dividend_instrument_date",
        )

    def get_dividends(self, instrument_id: UUID) -> Sequence[Dividend]:
        """Return dividend rows for an instrument, ordered newest-first."""
        stmt = (
            select(Dividend)
            .where(Dividend.instrument_id == instrument_id)
            .order_by(Dividend.date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def upsert_splits(self, instrument_id: UUID, splits: pd.Series) -> int:
        """Upsert stock split data from yfinance Series (index=date, value=ratio)."""
        rows = []
        for idx, ratio in splits.items():
            dt = _safe_date(idx)
            r = _safe_float(ratio)
            if dt is None or r is None:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "date": dt,
                    "ratio": r,
                }
            )

        return self._upsert(
            StockSplit,
            _last_wins(rows, "date"),
            constraint_name="uq_stock_split_instrument_date",
        )

    def get_splits(self, instrument_id: UUID) -> Sequence[StockSplit]:
        """Return stock split rows for an instrument, ordered newest-first."""
        stmt = (
            select(StockSplit)
            .where(StockSplit.instrument_id == instrument_id)
            .order_by(StockSplit.date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def get_oldest_price_date(self, instrument_id: UUID) -> date | None:
        """Return the date of the instrument's earliest stored price bar.

        Lets a full refetch reach back over the whole stored series, so every
        bar is rewritten on the provider's current split/dividend adjustment
        basis. ``None`` when the instrument has no stored prices.
        """
        return self.session.execute(
            select(func.min(PriceHistory.date)).where(
                PriceHistory.instrument_id == instrument_id
            )
        ).scalar_one_or_none()
