"""Repository for dashboard data queries."""

from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.macro_regime import BondYield, FredObservation
from app.models.universe import Instrument
from app.models.yfinance_data import PriceHistory
from app.repositories.base import RepositoryBase


class DashboardRepository(RepositoryBase):
    """Read-only repository for dashboard-specific data access."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Market snapshot helpers
    # ------------------------------------------------------------------

    def get_latest_fred_observations(
        self,
        series_ids: list[str],
    ) -> dict[str, tuple[float, float]]:
        """Return the two most recent non-null values per FRED series.

        Returns:
            {series_id: (latest_value, previous_value)}.
            Series with fewer than 2 observations are omitted.
        """
        if not series_ids:
            return {}

        stmt = (
            select(
                FredObservation.series_id,
                FredObservation.date,
                FredObservation.value,
            )
            .where(FredObservation.series_id.in_(series_ids))
            .where(FredObservation.value.is_not(None))
            .order_by(FredObservation.series_id, FredObservation.date.desc())
        )
        rows = self.session.execute(stmt).all()

        # Group in Python for SQLite compat (no window functions needed)
        grouped: dict[str, list[float]] = {}
        for series_id, _, value in rows:
            if value is None:
                continue
            vals = grouped.setdefault(series_id, [])
            if len(vals) < 2:
                vals.append(float(value))

        return {
            sid: (vals[0], vals[1])
            for sid, vals in grouped.items()
            if len(vals) >= 2
        }

    def get_latest_fred_observation_dates(
        self,
        series_ids: list[str],
    ) -> date | None:
        """Return the most recent observation date across the given FRED series."""
        if not series_ids:
            return None

        stmt = (
            select(FredObservation.date)
            .where(FredObservation.series_id.in_(series_ids))
            .where(FredObservation.value.is_not(None))
            .order_by(FredObservation.date.desc())
            .limit(1)
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        return row

    def get_spy_prices(self, n: int = 2) -> list[float]:
        """Return the last *n* SPY close prices ordered oldest-first.

        Returns:
            [oldest, ..., newest]. Empty list if SPY not found.
        """
        stmt = (
            select(PriceHistory.close)
            .join(Instrument, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker == "SPY")
            .order_by(PriceHistory.date.desc())
            .limit(n)
        )
        rows = self.session.execute(stmt).scalars().all()
        # Reverse so index 0 = oldest
        return [float(v) for v in reversed(rows) if v is not None]

    def get_spy_latest_date(self) -> date | None:
        """Return the most recent SPY price date."""
        stmt = (
            select(PriceHistory.date)
            .join(Instrument, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker == "SPY")
            .order_by(PriceHistory.date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_ten_year_yield_usa(self) -> tuple[float, float] | None:
        """Return (yield_value, day_change) for US 10Y bond.

        Returns None if no row exists.
        """
        stmt = (
            select(BondYield)
            .where(BondYield.country.in_(("United States", "USA")))
            .where(BondYield.maturity == "10Y")
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        if row is None or row.yield_value is None:
            return None
        return (float(row.yield_value), float(row.day_change or 0.0))

    def get_ten_year_yield_reference_date(self) -> date | None:
        """Return the reference_date for US 10Y bond yield."""
        stmt = (
            select(BondYield.reference_date)
            .where(BondYield.country.in_(("United States", "USA")))
            .where(BondYield.maturity == "10Y")
        )
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_multi_ticker_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch close prices for multiple tickers as a pivoted DataFrame.

        Returns:
            DataFrame with index=date, columns=yfinance_ticker, values=close price.
            Empty DataFrame if no data found.
        """
        if not tickers:
            return pd.DataFrame()

        stmt = (
            select(
                Instrument.yfinance_ticker,
                PriceHistory.date,
                PriceHistory.close,
            )
            .join(PriceHistory, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .where(PriceHistory.date >= start_date)
            .where(PriceHistory.date <= end_date)
            .order_by(PriceHistory.date)
        )
        rows = self.session.execute(stmt).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
        df["close"] = df["close"].astype(float)
        return df.pivot(index="date", columns="ticker", values="close").sort_index()
