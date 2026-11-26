"""Repository for an instrument's profile, statements, share counts and filings."""

import logging
from collections.abc import Sequence
from datetime import date
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select

from portopt_db.models.market_data.yfinance_data import (
    FinancialStatement,
    SecFiling,
    SharesOutstanding,
    TickerProfile,
    TickerProfileExtra,
)
from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.market_data._mapping import (
    _last_wins,
    _refreshable_columns,
    _safe_date,
    _safe_float,
    _safe_int,
    _safe_percent_as_fraction,
    _safe_str,
)
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)


# Description and classification columns. When quoteSummary fails, yfinance
# serves the bare v7 quote payload, which has none of them although the security
# has not changed, so an absent value must not erase what a full fetch stored.
# Market and fundamental columns are deliberately not listed: Yahoo drops those
# keys for real (a suspended dividend, lost analyst coverage) and the row should
# follow.
_PROFILE_STICKY_COLUMNS = frozenset(
    {
        "isin",
        "exchange",
        "quote_type",
        "currency",
        "short_name",
        "long_name",
        "sector",
        "industry",
        "country",
        "website",
        "long_business_summary",
        "full_time_employees",
    }
)
_PROFILE_EXTRAS_STICKY_COLUMNS = frozenset({"sector_key", "industry_key"})


class FundamentalsRepository(RepositoryBase):
    """Profile, financial statements, share counts and SEC filings of an instrument."""

    def upsert_profile(self, instrument_id: UUID, info: dict[str, Any]) -> int:
        """Upsert a ticker profile from a ``yf.Ticker.info`` dict.

        Args:
            instrument_id: The instrument this profile belongs to.
            info: The raw dict from ``yf.Ticker.info``. ``isin`` is not part of
                that payload; the caller injects it from a separate lookup. An
                ``info`` without a usable ``isin`` keeps the ISIN already stored
                for the instrument (a first load stores NULL) instead of
                overwriting it with NULL; every other column is refreshed.

        Returns:
            Number of rows written (0 or 1).
        """
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
            "debt_to_equity": _safe_percent_as_fraction(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            "dividend_rate": _safe_float(info.get("dividendRate")),
            "dividend_yield": _safe_percent_as_fraction(info.get("dividendYield")),
            "ex_dividend_date": ex_div,
            "payout_ratio": _safe_float(info.get("payoutRatio")),
            "five_year_avg_dividend_yield": _safe_percent_as_fraction(
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
            update_columns=_refreshable_columns(row, _PROFILE_STICKY_COLUMNS),
        )

    def get_profile(self, instrument_id: UUID) -> TickerProfile | None:
        """Return the ticker profile for an instrument, or None if not yet fetched."""
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
            update_columns=_refreshable_columns(row, _PROFILE_EXTRAS_STICKY_COLUMNS),
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
        seen: set[tuple[Any, ...]] = set()
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

    def upsert_financial_statements(
        self,
        instrument_id: UUID,
        df: pd.DataFrame,
        statement_type: str,
        period_type: str,
        currency_code: str | None = None,
        as_of: date | None = None,
    ) -> int:
        """Upsert financial statement rows (EAV format).

        yfinance returns DataFrames where columns are period dates and index
        rows are line-item names (e.g. ``"TotalRevenue"``). Column labels may be
        dates, ISO strings or ``"M/D/YYYY"`` strings.

        Args:
            instrument_id: The instrument these rows belong to.
            df: The yfinance financial statement DataFrame.
            statement_type: Category label, e.g. ``"income_stmt"`` or ``"balance_sheet"``.
            period_type: Frequency label, e.g. ``"annual"`` or ``"quarterly"``.
            currency_code: Reporting currency in major-unit form (e.g. ``"GBP"``).
                Convert listing currencies like ``"GBX"`` before passing.
                ``None`` means the currency is unknown: new rows get a NULL
                label, and rows that already exist keep the label they have.
            as_of: Date to stamp on a ``"Current"`` column (the trailing value in
                valuation-measure frames, which has no date of its own). Without
                it that column is skipped.

        Returns:
            Number of rows written.
        """
        rows = []
        periods_parsed = 0
        for col_pos, col in enumerate(df.columns):
            period_date: date | None
            if (
                as_of is not None
                and isinstance(col, str)
                and col.strip().lower() == "current"
            ):
                period_date = as_of
            else:
                period_date = _safe_date(col)
            if period_date is None:
                continue
            periods_parsed += 1
            # Positional access: label lookup returns an array, not a scalar,
            # when a line-item label repeats.
            for row_pos, line_item in enumerate(df.index):
                val = _safe_float(df.iat[row_pos, col_pos])
                # The period that just left Yahoo's window comes back almost
                # all NaN. Writing those cells would overwrite the figures
                # stored while it was complete, and Yahoo cannot serve them
                # again.
                if val is None:
                    continue
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

        # A failed info lookup yields no currency; on conflict that must not turn
        # an already-correct label into NULL.
        update_columns = None if currency_code is not None else ["value", "updated_at"]

        if not periods_parsed and len(df.index) and len(df.columns):
            logger.warning(
                "No column of the %s/%s frame parsed as a period date "
                "(labels: %s); nothing stored",
                statement_type,
                period_type,
                list(df.columns[:5]),
            )

        return self._upsert(
            FinancialStatement,
            _last_wins(rows, "period_date", "line_item"),
            constraint_name="uq_financial_statement_row",
            update_columns=update_columns,
        )

    def get_financial_statements(
        self,
        instrument_id: UUID,
        statement_type: str | None = None,
        period_type: str | None = None,
    ) -> Sequence[FinancialStatement]:
        """Return financial statement rows for an instrument.

        Args:
            instrument_id: The instrument to query.
            statement_type: Optional filter, e.g. ``"income_stmt"`` or ``"balance_sheet"``.
            period_type: Optional filter, e.g. ``"annual"`` or ``"quarterly"``.

        Returns:
            Rows ordered by statement type then period date descending.
        """
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
