"""Repository for yfinance data access with PostgreSQL upsert support."""

import contextlib
import logging
import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, cast
from uuid import UUID

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from portopt_db.models.market_data.yfinance_data import (
    AnalystAction,
    AnalystPriceTarget,
    AnalystRecommendation,
    CapitalGain,
    Dividend,
    EarningsDate,
    EarningsEstimate,
    EarningsHistory,
    EsgScore,
    FinancialStatement,
    GrowthEstimate,
    InsiderPurchaseSummary,
    InsiderRosterHolder,
    InsiderTransaction,
    InstitutionalHolder,
    MajorHolders,
    MutualFundHolder,
    OptionContract,
    PriceHistory,
    RevenueEstimate,
    SecFiling,
    SharesOutstanding,
    StockSplit,
    TickerNews,
    TickerProfile,
    TickerProfileExtra,
)
from portopt_db.models.universe.universe import Instrument
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)

_SENTINEL_DATE = date(1970, 1, 1)


def _safe_val(v: Any) -> Any:
    """Convert pandas/numpy types to Python natives, NaN/NaT to None."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    item_fn = getattr(v, "item", None)  # numpy scalar
    if callable(item_fn):
        return item_fn()
    return v


def _safe_int(v: Any) -> int | None:
    v = _safe_val(v)
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v: Any) -> float | None:
    v = _safe_val(v)
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_str(v: Any, max_len: int | None = None) -> str | None:
    v = _safe_val(v)
    if v is None:
        return None
    s = str(v)
    if max_len:
        s = s[:max_len]
    return s


def _safe_date(v: Any) -> date | None:
    """Convert various date-like values to date."""
    v = _safe_val(v)
    if v is None:
        return None
    # pd.NaT / pd.NA are missing values but NaT subclasses datetime, so the
    # branch below would call NaT.date() -> NaT and leak a bogus date into a
    # NOT NULL column. Treat any pandas NA scalar as missing.
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v).date()
        except (ValueError, TypeError):
            return None
    if isinstance(v, int | float):
        try:
            return datetime.fromtimestamp(v).date()
        except (ValueError, TypeError, OSError):
            return None
    return None


class YFinanceRepository(RepositoryBase):
    """Sync repository for yfinance data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Ticker Profile
    # ------------------------------------------------------------------

    def upsert_profile(self, instrument_id: UUID, info: dict[str, Any]) -> int:
        """Upsert a ticker profile from yf.Ticker.info dict."""
        # Map yfinance info keys to model columns
        ex_div = info.get("exDividendDate")
        if isinstance(ex_div, int | float):
            ex_div = _safe_date(ex_div)
        else:
            ex_div = _safe_date(ex_div)

        row = {
            "instrument_id": instrument_id,
            "symbol": _safe_str(info.get("symbol"), 50),
            "short_name": _safe_str(info.get("shortName"), 500),
            "long_name": _safe_str(info.get("longName"), 500),
            "isin": _safe_str(info.get("isin"), 20),
            "exchange": _safe_str(info.get("exchange"), 50),
            "quote_type": _safe_str(info.get("quoteType"), 50),
            "currency": _safe_str(info.get("currency"), 10),
            "sector": _safe_str(info.get("sector"), 200),
            "industry": _safe_str(info.get("industry"), 200),
            "country": _safe_str(info.get("country"), 100),
            "website": _safe_str(info.get("website"), 500),
            "long_business_summary": _safe_str(info.get("longBusinessSummary")),
            "market_cap": _safe_int(info.get("marketCap")),
            "enterprise_value": _safe_int(info.get("enterpriseValue")),
            "shares_outstanding": _safe_int(info.get("sharesOutstanding")),
            "float_shares": _safe_int(info.get("floatShares")),
            "implied_shares_outstanding": _safe_int(
                info.get("impliedSharesOutstanding")
            ),
            "current_price": _safe_float(info.get("currentPrice")),
            "previous_close": _safe_float(info.get("previousClose")),
            "open_price": _safe_float(info.get("open")),
            "day_low": _safe_float(info.get("dayLow")),
            "day_high": _safe_float(info.get("dayHigh")),
            "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "fifty_day_average": _safe_float(info.get("fiftyDayAverage")),
            "two_hundred_day_average": _safe_float(info.get("twoHundredDayAverage")),
            "average_volume": _safe_int(info.get("averageVolume")),
            "average_volume_10days": _safe_int(info.get("averageVolume10days")),
            "regular_market_volume": _safe_int(info.get("regularMarketVolume")),
            "bid": _safe_float(info.get("bid")),
            "ask": _safe_float(info.get("ask")),
            "bid_size": _safe_int(info.get("bidSize")),
            "ask_size": _safe_int(info.get("askSize")),
            "beta": _safe_float(info.get("beta")),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "trailing_eps": _safe_float(info.get("trailingEps")),
            "forward_eps": _safe_float(info.get("forwardEps")),
            "price_to_sales_trailing_12months": _safe_float(
                info.get("priceToSalesTrailing12Months")
            ),
            "price_to_book": _safe_float(info.get("priceToBook")),
            "enterprise_to_revenue": _safe_float(info.get("enterpriseToRevenue")),
            "enterprise_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "book_value": _safe_float(info.get("bookValue")),
            "profit_margins": _safe_float(info.get("profitMargins")),
            "operating_margins": _safe_float(info.get("operatingMargins")),
            "gross_margins": _safe_float(info.get("grossMargins")),
            "ebitda_margins": _safe_float(info.get("ebitdaMargins")),
            "return_on_assets": _safe_float(info.get("returnOnAssets")),
            "return_on_equity": _safe_float(info.get("returnOnEquity")),
            "total_revenue": _safe_int(info.get("totalRevenue")),
            "revenue_per_share": _safe_float(info.get("revenuePerShare")),
            "revenue_growth": _safe_float(info.get("revenueGrowth")),
            "earnings_growth": _safe_float(info.get("earningsGrowth")),
            "earnings_quarterly_growth": _safe_float(
                info.get("earningsQuarterlyGrowth")
            ),
            "ebitda": _safe_int(info.get("ebitda")),
            "gross_profits": _safe_int(info.get("grossProfits")),
            "free_cashflow": _safe_int(info.get("freeCashflow")),
            "operating_cashflow": _safe_int(info.get("operatingCashflow")),
            "total_cash": _safe_int(info.get("totalCash")),
            "total_cash_per_share": _safe_float(info.get("totalCashPerShare")),
            "total_debt": _safe_int(info.get("totalDebt")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            "dividend_rate": _safe_float(info.get("dividendRate")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "ex_dividend_date": ex_div,
            "payout_ratio": _safe_float(info.get("payoutRatio")),
            "five_year_avg_dividend_yield": _safe_float(
                info.get("fiveYearAvgDividendYield")
            ),
            "trailing_annual_dividend_rate": _safe_float(
                info.get("trailingAnnualDividendRate")
            ),
            "trailing_annual_dividend_yield": _safe_float(
                info.get("trailingAnnualDividendYield")
            ),
            "last_dividend_value": _safe_float(info.get("lastDividendValue")),
            "target_high_price": _safe_float(info.get("targetHighPrice")),
            "target_low_price": _safe_float(info.get("targetLowPrice")),
            "target_mean_price": _safe_float(info.get("targetMeanPrice")),
            "target_median_price": _safe_float(info.get("targetMedianPrice")),
            "number_of_analyst_opinions": _safe_int(
                info.get("numberOfAnalystOpinions")
            ),
            "recommendation_key": _safe_str(info.get("recommendationKey"), 50),
            "recommendation_mean": _safe_float(info.get("recommendationMean")),
            "full_time_employees": _safe_int(info.get("fullTimeEmployees")),
        }

        return self._upsert(
            TickerProfile,
            [row],
            constraint_name="uq_ticker_profile_instrument",
        )

    def get_profile(self, instrument_id: UUID) -> TickerProfile | None:
        stmt = select(TickerProfile).where(TickerProfile.instrument_id == instrument_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def upsert_profile_extras(self, instrument_id: UUID, info: dict[str, Any]) -> int:
        """Upsert the 1:1 extra-info row (short interest, momentum, governance
        risk) mapped from the same yf.Ticker.info dict as ``upsert_profile``."""
        row = {
            "instrument_id": instrument_id,
            "shares_short": _safe_int(info.get("sharesShort")),
            "shares_short_prior_month": _safe_int(info.get("sharesShortPriorMonth")),
            "short_ratio": _safe_float(info.get("shortRatio")),
            "short_percent_of_float": _safe_float(info.get("shortPercentOfFloat")),
            "shares_percent_shares_out": _safe_float(
                info.get("sharesPercentSharesOut")
            ),
            "held_percent_insiders": _safe_float(info.get("heldPercentInsiders")),
            "held_percent_institutions": _safe_float(
                info.get("heldPercentInstitutions")
            ),
            "fifty_two_week_change": _safe_float(info.get("52WeekChange")),
            "sandp_52_week_change": _safe_float(info.get("SandP52WeekChange")),
            "sector_key": _safe_str(info.get("sectorKey"), 100),
            "industry_key": _safe_str(info.get("industryKey"), 150),
            "audit_risk": _safe_int(info.get("auditRisk")),
            "board_risk": _safe_int(info.get("boardRisk")),
            "compensation_risk": _safe_int(info.get("compensationRisk")),
            "shareholder_rights_risk": _safe_int(info.get("shareHolderRightsRisk")),
            "overall_risk": _safe_int(info.get("overallRisk")),
        }
        return self._upsert(
            TickerProfileExtra,
            [row],
            constraint_name="uq_ticker_profile_extras_instrument",
        )

    # ------------------------------------------------------------------
    # Options chain (SPEC A10 — high-volume, own scheduler step)
    # ------------------------------------------------------------------

    def get_options_as_of(self, instrument_id: UUID) -> date | None:
        """Most recent options snapshot date for the instrument, or None."""
        stmt = select(func.max(OptionContract.as_of)).where(
            OptionContract.instrument_id == instrument_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_options_as_of_bulk(
        self, instrument_ids: Sequence[UUID]
    ) -> dict[UUID, date]:
        """Latest options snapshot date per instrument in one grouped query.

        Avoids a MAX(as_of) round-trip per instrument across the full sweep.
        """
        ids = list(instrument_ids)
        if not ids:
            return {}
        stmt = (
            select(OptionContract.instrument_id, func.max(OptionContract.as_of))
            .where(OptionContract.instrument_id.in_(ids))
            .group_by(OptionContract.instrument_id)
        )
        return {
            row[0]: row[1]
            for row in self.session.execute(stmt).all()
            if row[1] is not None
        }

    def upsert_option_chain(
        self, instrument_id: UUID, rows: list[dict[str, Any]]
    ) -> int:
        """Upsert option-contract snapshot rows (already flattened + typed).

        Deduped in-batch on (as_of, contract_symbol): a single snapshot must not
        touch the same natural key twice (PostgreSQL ON CONFLICT cardinality).
        """
        deduped: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            symbol = r.get("contract_symbol")
            as_of = r.get("as_of")
            if not symbol or as_of is None:
                continue
            deduped[(as_of, symbol)] = {"instrument_id": instrument_id, **r}
        prepared = list(deduped.values())
        if not prepared:
            return 0
        return self._upsert(
            OptionContract,
            prepared,
            constraint_name="uq_option_contract",
        )

    def get_sectors_by_yfinance_ticker(self, tickers: Sequence[str]) -> dict[str, str]:
        """Return ``{yfinance_ticker: sector}`` for the given tickers.

        Joins ``TickerProfile`` to ``Instrument`` on ``instrument_id`` and keys the
        result by ``Instrument.yfinance_ticker``. Rows whose sector is NULL/empty are
        omitted, so a caller can treat "absent" uniformly as unknown.

        Lives here rather than in the calling service: the query spans two ORM models,
        and a service building it would have to import those models directly — a core
        file reaching past the repository into the persistence layer.
        """
        if not tickers:
            return {}
        stmt = (
            select(Instrument.yfinance_ticker, TickerProfile.sector)
            .join(TickerProfile, TickerProfile.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker.in_(list(tickers)))
        )
        return {
            row.yfinance_ticker: row.sector
            for row in self.session.execute(stmt).all()
            if row.sector
        }

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    def upsert_price_history(
        self,
        instrument_id: UUID,
        history_df: pd.DataFrame,
        price_unit: str | None = None,
    ) -> int:
        """Upsert daily OHLCV rows from a yfinance history DataFrame.

        ``price_unit`` records the listing currency the prices are quoted in
        (e.g. "GBX"); the values are stored as-is (SPEC OQ2).
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

        return self._upsert(
            PriceHistory,
            rows,
            constraint_name="uq_price_history_instrument_date",
        )

    def upsert_earnings_estimate(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert forward-period earnings estimates (index = period labels)."""
        rows = [
            {
                "instrument_id": instrument_id,
                "period": _safe_str(period, 10),
                "num_analysts": _safe_int(row.get("numberOfAnalysts")),
                "avg": _safe_float(row.get("avg")),
                "low": _safe_float(row.get("low")),
                "high": _safe_float(row.get("high")),
                "year_ago_eps": _safe_float(row.get("yearAgoEps")),
                "growth": _safe_float(row.get("growth")),
            }
            for period, row in df.iterrows()
        ]
        return self._upsert(
            EarningsEstimate,
            rows,
            constraint_name="uq_earnings_estimate_instrument_period",
        )

    def upsert_revenue_estimate(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert forward-period revenue estimates (index = period labels)."""
        rows = [
            {
                "instrument_id": instrument_id,
                "period": _safe_str(period, 10),
                "num_analysts": _safe_int(row.get("numberOfAnalysts")),
                "avg": _safe_float(row.get("avg")),
                "low": _safe_float(row.get("low")),
                "high": _safe_float(row.get("high")),
                "year_ago_revenue": _safe_float(row.get("yearAgoRevenue")),
                "growth": _safe_float(row.get("growth")),
            }
            for period, row in df.iterrows()
        ]
        return self._upsert(
            RevenueEstimate,
            rows,
            constraint_name="uq_revenue_estimate_instrument_period",
        )

    def upsert_growth_estimates(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert per-period growth estimates (stock/industry/sector/index trend).

        Column names vary across yfinance versions; read defensively.
        """

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        rows = [
            {
                "instrument_id": instrument_id,
                "period": _safe_str(period, 10),
                "stock_trend": _safe_float(_col(row, "stockTrend", "stock")),
                "industry_trend": _safe_float(_col(row, "industryTrend", "industry")),
                "sector_trend": _safe_float(_col(row, "sectorTrend", "sector")),
                "index_trend": _safe_float(_col(row, "indexTrend", "index")),
            }
            for period, row in df.iterrows()
        ]
        return self._upsert(
            GrowthEstimate,
            rows,
            constraint_name="uq_growth_estimate_instrument_period",
        )

    def upsert_earnings_history(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert historical EPS surprise (index = past-quarter dates).

        Deduped in-batch on ``period_date`` (last-wins): two source rows on the
        same calendar day would otherwise collide on ``uq_earnings_history_...``
        within one ON CONFLICT and abort the whole write.
        """
        by_date: dict[date, dict[str, Any]] = {}
        for idx, row in df.iterrows():
            period_date = _safe_date(idx)
            if period_date is None:
                continue
            by_date[period_date] = {
                "instrument_id": instrument_id,
                "period_date": period_date,
                "eps_estimate": _safe_float(row.get("epsEstimate")),
                "eps_actual": _safe_float(row.get("epsActual")),
                "eps_difference": _safe_float(row.get("epsDifference")),
                "surprise_percent": _safe_float(row.get("surprisePercent")),
            }
        rows = list(by_date.values())
        return self._upsert(
            EarningsHistory,
            rows,
            constraint_name="uq_earnings_history_instrument_period",
        )

    def upsert_earnings_dates(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert past/upcoming earnings dates (index = tz-aware datetimes).

        Column labels vary across yfinance versions; read defensively.
        """

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        by_date: dict[date, dict[str, Any]] = {}
        for idx, row in df.iterrows():
            earnings_date = _safe_date(idx)
            if earnings_date is None:
                continue
            by_date[earnings_date] = {
                "instrument_id": instrument_id,
                "earnings_date": earnings_date,
                "eps_estimate": _safe_float(_col(row, "EPS Estimate", "epsEstimate")),
                "eps_actual": _safe_float(_col(row, "Reported EPS", "epsActual")),
                "surprise_percent": _safe_float(
                    _col(row, "Surprise(%)", "surprisePercent")
                ),
            }
        rows = list(by_date.values())
        return self._upsert(
            EarningsDate,
            rows,
            constraint_name="uq_earnings_date_instrument_date",
        )

    def upsert_analyst_actions(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert analyst upgrade/downgrade actions (index = grade dates)."""

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        rows: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for idx, row in df.iterrows():
            action_date = _safe_date(idx)
            if action_date is None:
                continue
            firm = _safe_str(_col(row, "Firm", "firm"), 200) or ""
            to_grade = _safe_str(_col(row, "ToGrade", "toGrade"), 100) or ""
            key = (action_date, firm, to_grade)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "action_date": action_date,
                    "firm": firm,
                    "from_grade": _safe_str(_col(row, "FromGrade", "fromGrade"), 100),
                    "to_grade": to_grade,
                    "action": _safe_str(_col(row, "Action", "action"), 50),
                }
            )
        return self._upsert(AnalystAction, rows, constraint_name="uq_analyst_action")

    def upsert_esg_scores(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the latest ESG snapshot from yf.Ticker.sustainability.

        sustainability is a single-column frame indexed by metric name.
        """
        series = df.iloc[:, 0] if df.shape[1] >= 1 else df.squeeze()

        def _metric(name: str) -> float | None:
            try:
                return _safe_float(series.get(name))
            except Exception:  # defensive: shape varies across versions
                return None

        row = {
            "instrument_id": instrument_id,
            "total_esg": _metric("totalEsg"),
            "environment_score": _metric("environmentScore"),
            "social_score": _metric("socialScore"),
            "governance_score": _metric("governanceScore"),
            "highest_controversy": _metric("highestControversy"),
        }
        return self._upsert(EsgScore, [row], constraint_name="uq_esg_score_instrument")

    def upsert_sec_filings(
        self, instrument_id: UUID, filings: list[dict[str, Any]]
    ) -> int:
        """Upsert SEC filings from yf.Ticker.sec_filings (list of dicts)."""

        def _col(d: dict[str, Any], *names: str) -> Any:
            for name in names:
                if name in d:
                    return d.get(name)
            return None

        rows: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for filing in filings:
            filing_date = _safe_date(_col(filing, "date", "filingDate", "epochDate"))
            if filing_date is None:
                continue
            form_type = _safe_str(_col(filing, "type", "form", "formType"), 50) or ""
            title = _safe_str(_col(filing, "title", "description"), 500) or ""
            key = (filing_date, form_type, title)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "filing_date": filing_date,
                    "form_type": form_type,
                    "title": title,
                    "url": _safe_str(_col(filing, "edgarUrl", "url", "link"), 1000),
                }
            )
        return self._upsert(SecFiling, rows, constraint_name="uq_sec_filing")

    def get_price_history(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 5000,
    ) -> Sequence[PriceHistory]:
        stmt = select(PriceHistory).where(PriceHistory.instrument_id == instrument_id)
        if start_date:
            stmt = stmt.where(PriceHistory.date >= start_date)
        if end_date:
            stmt = stmt.where(PriceHistory.date <= end_date)
        stmt = stmt.order_by(PriceHistory.date.desc()).limit(limit)
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Financial Statements
    # ------------------------------------------------------------------

    def upsert_financial_statements(
        self,
        instrument_id: UUID,
        df: pd.DataFrame,
        statement_type: str,
        period_type: str,
        currency_code: str | None = None,
    ) -> int:
        """Upsert financial statement rows (EAV format).

        yfinance returns DataFrames where:
         - columns are period dates
         - index rows are line item names

        Parameters
        ----------
        currency_code : str or None
            Reporting currency for these rows (e.g. ``"GBP"`` for UK
            companies).  Should be the **major-unit** code — convert
            listing currencies like ``"GBX"`` via
            :func:`app.utils.currency.to_major_currency` before passing.
        """
        rows = []
        for col in df.columns:
            period_date = _safe_date(col)
            if period_date is None:
                continue
            for line_item in df.index:
                val = _safe_float(df.at[line_item, col])
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "statement_type": statement_type,
                        "period_type": period_type,
                        "period_date": period_date,
                        "line_item": _safe_str(line_item, 200),
                        "value": val,
                        "currency_code": currency_code,
                    }
                )

        return self._upsert(
            FinancialStatement,
            rows,
            constraint_name="uq_financial_statement_row",
        )

    def get_financial_statements(
        self,
        instrument_id: UUID,
        statement_type: str | None = None,
        period_type: str | None = None,
    ) -> Sequence[FinancialStatement]:
        stmt = select(FinancialStatement).where(
            FinancialStatement.instrument_id == instrument_id
        )
        if statement_type:
            stmt = stmt.where(FinancialStatement.statement_type == statement_type)
        if period_type:
            stmt = stmt.where(FinancialStatement.period_type == period_type)
        stmt = stmt.order_by(
            FinancialStatement.statement_type,
            FinancialStatement.period_date.desc(),
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

    def upsert_dividends(self, instrument_id: UUID, dividends: pd.Series) -> int:
        """Upsert dividend data from yfinance Series (index=date, value=amount)."""
        rows = []
        for idx, amount in dividends.items():
            dt = _safe_date(idx)
            amt = _safe_float(amount)
            if dt is None or amt is None:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "date": dt,
                    "amount": amt,
                }
            )

        return self._upsert(
            Dividend,
            rows,
            constraint_name="uq_dividend_instrument_date",
        )

    def get_dividends(self, instrument_id: UUID) -> Sequence[Dividend]:
        stmt = (
            select(Dividend)
            .where(Dividend.instrument_id == instrument_id)
            .order_by(Dividend.date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Stock Splits
    # ------------------------------------------------------------------

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
            rows,
            constraint_name="uq_stock_split_instrument_date",
        )

    def get_splits(self, instrument_id: UUID) -> Sequence[StockSplit]:
        stmt = (
            select(StockSplit)
            .where(StockSplit.instrument_id == instrument_id)
            .order_by(StockSplit.date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Shares Outstanding / Capital Gains (corporate-action extras)
    # ------------------------------------------------------------------

    def upsert_shares_outstanding(self, instrument_id: UUID, shares: Any) -> int:
        """Upsert point-in-time share counts from ``get_shares_full``.

        yfinance returns a Series (index=timestamp, value=shares); a
        single-column DataFrame is squeezed. Same date can repeat, so the
        first row per date wins the natural-key dedup.
        """
        series = shares
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0] if series.shape[1] else series.squeeze()

        rows: list[dict[str, Any]] = []
        seen: set[date] = set()
        for idx, val in series.items():
            dt = _safe_date(idx)
            n = _safe_int(val)
            if dt is None or n is None or dt in seen:
                continue
            seen.add(dt)
            rows.append({"instrument_id": instrument_id, "date": dt, "shares": n})

        return self._upsert(
            SharesOutstanding,
            rows,
            constraint_name="uq_shares_outstanding_instrument_date",
        )

    def upsert_capital_gains(self, instrument_id: UUID, gains: pd.Series) -> int:
        """Upsert capital-gain distributions (index=date, value=amount)."""
        rows = []
        for idx, amount in gains.items():
            dt = _safe_date(idx)
            amt = _safe_float(amount)
            if dt is None or amt is None:
                continue
            rows.append({"instrument_id": instrument_id, "date": dt, "amount": amt})

        return self._upsert(
            CapitalGain,
            rows,
            constraint_name="uq_capital_gain_instrument_date",
        )

    # ------------------------------------------------------------------
    # Analyst Recommendations
    # ------------------------------------------------------------------

    def upsert_recommendations(self, instrument_id: UUID, rec_df: pd.DataFrame) -> int:
        """Upsert analyst recommendations from recommendations_summary DataFrame."""
        rows = []
        for _, row_data in rec_df.iterrows():
            period = _safe_str(row_data.get("period"), 50)
            if not period:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "period": period,
                    "strong_buy": _safe_int(row_data.get("strongBuy")),
                    "buy": _safe_int(row_data.get("buy")),
                    "hold": _safe_int(row_data.get("hold")),
                    "sell": _safe_int(row_data.get("sell")),
                    "strong_sell": _safe_int(row_data.get("strongSell")),
                }
            )

        return self._upsert(
            AnalystRecommendation,
            rows,
            constraint_name="uq_analyst_rec_instrument_period",
        )

    def get_recommendations(
        self, instrument_id: UUID
    ) -> Sequence[AnalystRecommendation]:
        stmt = (
            select(AnalystRecommendation)
            .where(AnalystRecommendation.instrument_id == instrument_id)
            .order_by(AnalystRecommendation.period)
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Analyst Price Targets
    # ------------------------------------------------------------------

    def upsert_price_targets(self, instrument_id: UUID, targets: dict[str, Any]) -> int:
        """Upsert analyst price targets from dict."""
        row = {
            "instrument_id": instrument_id,
            "current": _safe_float(targets.get("current")),
            "low": _safe_float(targets.get("low")),
            "high": _safe_float(targets.get("high")),
            "mean": _safe_float(targets.get("mean")),
            "median": _safe_float(targets.get("median")),
        }

        return self._upsert(
            AnalystPriceTarget,
            [row],
            constraint_name="uq_analyst_pt_instrument",
        )

    def get_price_targets(self, instrument_id: UUID) -> AnalystPriceTarget | None:
        stmt = select(AnalystPriceTarget).where(
            AnalystPriceTarget.instrument_id == instrument_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Institutional Holders
    # ------------------------------------------------------------------

    def upsert_institutional_holders(
        self, instrument_id: UUID, holders_df: pd.DataFrame
    ) -> int:
        """Upsert institutional holders from DataFrame."""
        rows = []
        for _, row_data in holders_df.iterrows():
            name = _safe_str(row_data.get("Holder"), 500)
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "holder_name": name,
                    "date_reported": _safe_date(row_data.get("Date Reported")),
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "pct_held": _safe_float(row_data.get("pctHeld")),
                }
            )

        return self._upsert(
            InstitutionalHolder,
            rows,
            constraint_name="uq_inst_holder_instrument_name",
        )

    def get_institutional_holders(
        self, instrument_id: UUID
    ) -> Sequence[InstitutionalHolder]:
        stmt = (
            select(InstitutionalHolder)
            .where(InstitutionalHolder.instrument_id == instrument_id)
            .order_by(InstitutionalHolder.holder_name)
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Mutual Fund Holders
    # ------------------------------------------------------------------

    def upsert_mutualfund_holders(
        self, instrument_id: UUID, holders_df: pd.DataFrame
    ) -> int:
        """Upsert mutual fund holders from DataFrame."""
        rows = []
        for _, row_data in holders_df.iterrows():
            name = _safe_str(row_data.get("Holder"), 500)
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "holder_name": name,
                    "date_reported": _safe_date(row_data.get("Date Reported")),
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "pct_held": _safe_float(row_data.get("pctHeld")),
                }
            )

        return self._upsert(
            MutualFundHolder,
            rows,
            constraint_name="uq_mutual_fund_holder_instrument_name",
        )

    def get_mutualfund_holders(self, instrument_id: UUID) -> Sequence[MutualFundHolder]:
        stmt = (
            select(MutualFundHolder)
            .where(MutualFundHolder.instrument_id == instrument_id)
            .order_by(MutualFundHolder.holder_name)
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Insider Transactions
    # ------------------------------------------------------------------

    def upsert_insider_transactions(
        self, instrument_id: UUID, insiders_df: pd.DataFrame
    ) -> int:
        """Upsert insider transactions from DataFrame."""
        rows = []
        for _, row_data in insiders_df.iterrows():
            name = _safe_str(row_data.get("Insider"), 500)
            tx_type = _safe_str(row_data.get("Transaction"), 200)
            if not tx_type:
                # yfinance 1.3.0 stopped populating the "Transaction" column
                # (now always ""); the human-readable action moved to "Text"
                # (e.g. "Sale at price 290.00 per share."). Without this
                # fallback every row was skipped and insider_transactions
                # silently stayed empty with no error logged.
                tx_type = _safe_str(row_data.get("Text"), 200)
            if not name or not tx_type:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "insider_name": name,
                    "position": _safe_str(row_data.get("Position"), 500),
                    "transaction_type": tx_type,
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "start_date": _safe_date(row_data.get("Start Date"))
                    or _SENTINEL_DATE,
                    "ownership": _safe_str(row_data.get("Ownership"), 50),
                }
            )

        # Collapse rows that collide on the uq_insider_tx_row constraint
        # (instrument_id, insider_name, start_date, transaction_type). yfinance
        # returns several real transactions sharing all four (e.g. same grant,
        # same day, different share counts); a single INSERT ... ON CONFLICT
        # cannot touch the same conflict key twice ("ON CONFLICT DO UPDATE
        # command cannot affect row a second time"). Keep the last occurrence,
        # matching the DB upsert's last-write-wins semantics.
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for r in rows:
            key = (
                r["instrument_id"],
                r["insider_name"],
                r["start_date"],
                r["transaction_type"],
            )
            deduped[key] = r

        return self._upsert(
            InsiderTransaction,
            list(deduped.values()),
            constraint_name="uq_insider_tx_row",
        )

    def get_insider_transactions(
        self, instrument_id: UUID
    ) -> Sequence[InsiderTransaction]:
        stmt = (
            select(InsiderTransaction)
            .where(InsiderTransaction.instrument_id == instrument_id)
            .order_by(InsiderTransaction.start_date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Holders extras (major holders + insider summary + insider roster)
    # ------------------------------------------------------------------

    def upsert_major_holders(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the 1:1 ownership breakdown from ``major_holders``.

        yfinance returns a label-indexed single-value DataFrame
        (``insidersPercentHeld`` etc.); read the first column as a lookup.
        """
        lookup = df.iloc[:, 0] if df.shape[1] else df.squeeze()

        def _v(key: str) -> Any:
            try:
                return lookup.get(key)
            except (AttributeError, TypeError):
                return None

        row = {
            "instrument_id": instrument_id,
            "insiders_percent_held": _safe_float(_v("insidersPercentHeld")),
            "institutions_percent_held": _safe_float(_v("institutionsPercentHeld")),
            "institutions_float_percent_held": _safe_float(
                _v("institutionsFloatPercentHeld")
            ),
            "institutions_count": _safe_int(_v("institutionsCount")),
        }
        return self._upsert(
            MajorHolders, [row], constraint_name="uq_major_holders_instrument"
        )

    def upsert_insider_purchases(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the 1:1 6-month insider buy/sell summary.

        The label lives in the first column ("Insider Purchases Last 6m") with
        the figure in "Shares"; read known labels defensively.
        """
        label_col = df.columns[0]
        shares_by_label: dict[str, Any] = {}
        for _, r in df.iterrows():
            label = _safe_str(r.get(label_col), 100)
            if label:
                shares_by_label[label] = r.get("Shares")

        def _shares(*labels: str) -> int | None:
            for label in labels:
                if label in shares_by_label:
                    return _safe_int(shares_by_label[label])
            return None

        row = {
            "instrument_id": instrument_id,
            "purchase_shares": _shares("Purchases"),
            "sale_shares": _shares("Sales"),
            "net_shares": _shares(
                "Net Shares Purchased (Sold)", "Net Shares Purchased"
            ),
            "total_insider_shares": _shares("Total Insider Shares Held"),
        }
        return self._upsert(
            InsiderPurchaseSummary,
            [row],
            constraint_name="uq_insider_purchases_instrument",
        )

    def upsert_insider_roster(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert individual insiders from ``insider_roster_holders``."""

        def _col(r: Any, *names: str) -> Any:
            for name in names:
                if name in r:
                    return r.get(name)
            return None

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, r in df.iterrows():
            name = _safe_str(_col(r, "Name", "name"), 500)
            if not name or name in seen:
                continue
            seen.add(name)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "insider_name": name,
                    "position": _safe_str(_col(r, "Position", "position"), 500),
                    "most_recent_transaction": _safe_str(
                        _col(r, "Most Recent Transaction"), 200
                    ),
                    "latest_transaction_date": _safe_date(
                        _col(r, "Latest Transaction Date")
                    ),
                    "shares_owned_directly": _safe_int(
                        _col(r, "Shares Owned Directly")
                    ),
                    "shares_owned_indirectly": _safe_int(
                        _col(r, "Shares Owned Indirectly")
                    ),
                }
            )
        return self._upsert(
            InsiderRosterHolder,
            rows,
            constraint_name="uq_insider_roster_instrument_name",
        )

    # ------------------------------------------------------------------
    # Ticker News
    # ------------------------------------------------------------------

    def upsert_news(self, instrument_id: UUID, articles: list[dict[str, Any]]) -> int:
        """Upsert news articles from yf.Ticker.news list of dicts.

        Handles both old yfinance format (flat dict with ``uuid``, ``title``,
        ``providerPublishTime``) and new format (nested ``content`` dict with
        ``pubDate`` ISO 8601 string).
        """
        # Look up instrument name once for all articles
        ticker_name = self.session.execute(
            select(Instrument.name).where(Instrument.id == instrument_id)
        ).scalar_one_or_none()

        rows = []
        for article in articles:
            # New format nests data under "content"; old format is flat
            content = article.get("content", article)

            # UUID: new format uses top-level "id", old uses "uuid"
            news_uuid = _safe_str(article.get("id") or article.get("uuid"), 200)
            if not news_uuid:
                continue

            # Title
            title = _safe_str(content.get("title") or article.get("title"))

            # Publisher: new format nests under provider dict
            provider = content.get("provider")
            if isinstance(provider, dict):
                publisher = _safe_str(provider.get("displayName", ""), 500)
            else:
                publisher = _safe_str(article.get("publisher"), 500)

            # Link: new format uses canonicalUrl or previewUrl
            canonical = content.get("canonicalUrl")
            if isinstance(canonical, dict):
                link = _safe_str(canonical.get("url", ""))
            else:
                link = _safe_str(content.get("previewUrl") or article.get("link"))

            # Publish time: new format is ISO 8601, old is epoch int
            publish_time = None
            pub_date_str = content.get("pubDate")
            if pub_date_str and isinstance(pub_date_str, str):
                try:
                    from dateutil import parser as dateutil_parser

                    publish_time = dateutil_parser.isoparse(pub_date_str)
                    # Strip tz for consistency with TickerNews column type
                    publish_time = publish_time.replace(tzinfo=None)
                except (ValueError, TypeError):
                    pass
            if publish_time is None:
                pt = article.get("providerPublishTime")
                if pt is not None:
                    with contextlib.suppress(ValueError, TypeError, OSError):
                        publish_time = datetime.fromtimestamp(int(pt))

            # News type
            news_type = _safe_str(
                content.get("contentType") or article.get("type"), 100
            )

            # Full content (scraped upstream by NewsClient if available)
            full_content = article.get("full_content")

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "news_uuid": news_uuid,
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "publish_time": publish_time,
                    "news_type": news_type,
                    "ticker_name": _safe_str(ticker_name, 500),
                    "full_content": full_content,
                }
            )

        return self._upsert(
            TickerNews,
            rows,
            constraint_name="uq_ticker_news_instrument_uuid",
        )

    def get_news(self, instrument_id: UUID) -> Sequence[TickerNews]:
        stmt = (
            select(TickerNews)
            .where(TickerNews.instrument_id == instrument_id)
            .order_by(TickerNews.publish_time.desc().nullslast())
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Staleness info (for incremental fetch)
    # ------------------------------------------------------------------

    def get_staleness_info(self, instrument_id: UUID) -> dict[str, Any]:
        """Return staleness metadata for incremental fetch decisions.

        Returns a dict with:
          - price_max_date: MAX(date) from price_history (date or None)
          - {category}_updated_at: MAX(updated_at) from each non-price table (datetime or None)
        """
        result: dict[str, Any] = {}

        # Price: get the latest date
        price_row = self.session.execute(
            select(func.max(PriceHistory.date)).where(
                PriceHistory.instrument_id == instrument_id
            )
        ).scalar_one_or_none()
        result["price_max_date"] = price_row

        # For each other category, get MAX(updated_at)
        category_models = [
            ("profile", TickerProfile),
            ("financials", FinancialStatement),
            ("dividends", Dividend),
            ("splits", StockSplit),
            ("recommendations", AnalystRecommendation),
            ("price_targets", AnalystPriceTarget),
            ("institutional_holders", InstitutionalHolder),
            ("mutualfund_holders", MutualFundHolder),
            ("insider_transactions", InsiderTransaction),
            ("news", TickerNews),
        ]

        for category, model in category_models:
            val = self.session.execute(
                select(func.max(cast(Any, model).updated_at)).where(
                    cast(Any, model).instrument_id == instrument_id
                )
            ).scalar_one_or_none()
            result[f"{category}_updated_at"] = val

        return result

    def get_staleness_info_bulk(
        self, instrument_ids: Sequence[UUID]
    ) -> dict[UUID, dict[str, Any]]:
        """Batched :meth:`get_staleness_info` for a whole sweep.

        Returns one full staleness dict per requested id (all-None for an
        instrument with no rows yet), matching the single-query shape exactly —
        but via 11 grouped queries total instead of 11 per instrument.
        """
        ids = list(instrument_ids)
        category_models = [
            ("profile", TickerProfile),
            ("financials", FinancialStatement),
            ("dividends", Dividend),
            ("splits", StockSplit),
            ("recommendations", AnalystRecommendation),
            ("price_targets", AnalystPriceTarget),
            ("institutional_holders", InstitutionalHolder),
            ("mutualfund_holders", MutualFundHolder),
            ("insider_transactions", InsiderTransaction),
            ("news", TickerNews),
        ]
        result: dict[UUID, dict[str, Any]] = {
            iid: {
                "price_max_date": None,
                **{f"{cat}_updated_at": None for cat, _ in category_models},
            }
            for iid in ids
        }
        if not ids:
            return result

        for iid, mx in self.session.execute(
            select(PriceHistory.instrument_id, func.max(PriceHistory.date))
            .where(PriceHistory.instrument_id.in_(ids))
            .group_by(PriceHistory.instrument_id)
        ).all():
            result[iid]["price_max_date"] = mx

        for category, model in category_models:
            m = cast(Any, model)
            for iid, mx in self.session.execute(
                select(m.instrument_id, func.max(m.updated_at))
                .where(m.instrument_id.in_(ids))
                .group_by(m.instrument_id)
            ).all():
                result[iid][f"{category}_updated_at"] = mx

        return result

    # ------------------------------------------------------------------
    # Instrument queries (used by yfinance_data endpoint)
    # ------------------------------------------------------------------

    def get_instruments_with_yfinance_ticker(self) -> Sequence[Instrument]:
        """Return all instruments that have a non-empty yfinance_ticker, with exchange eager-loaded."""
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker.isnot(None))
                .where(Instrument.yfinance_ticker != "")
            )
            .scalars()
            .unique()
            .all()
        )

    def get_instrument_by_yfinance_ticker(
        self, yfinance_ticker: str
    ) -> Instrument | None:
        """Return an instrument by its yfinance_ticker, with exchange eager-loaded."""
        return self.session.execute(
            select(Instrument)
            .options(joinedload(Instrument.exchange))
            .where(Instrument.yfinance_ticker == yfinance_ticker)
        ).scalar_one_or_none()

    def get_benchmark_coverage(
        self,
        tickers: list[str],
    ) -> dict[str, tuple[int, date | None]]:
        """Return ``{ticker: (price_row_count, latest_price_date)}``.

        Tickers without an instrument row (or with zero prices) come back as
        ``(0, None)``. Drives the startup bootstrap and the scheduler's
        reference-index refresh, which re-seed anything missing or stale.
        """
        if not tickers:
            return {}

        stmt = (
            select(
                Instrument.yfinance_ticker,
                func.count(PriceHistory.id).label("price_rows"),
                func.max(PriceHistory.date).label("latest"),
            )
            .outerjoin(PriceHistory, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .group_by(Instrument.yfinance_ticker)
        )
        rows = self.session.execute(stmt).all()
        coverage: dict[str, tuple[int, date | None]] = dict.fromkeys(tickers, (0, None))
        for ticker, count, latest in rows:
            coverage[ticker] = (int(count or 0), latest)
        return coverage
