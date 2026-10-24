"""Repository for yfinance data access with PostgreSQL upsert support."""

import contextlib
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any, cast
from uuid import UUID

import pandas as pd
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session, aliased, joinedload

from portopt_db.models.market_data.yfinance_data import (
    AnalystAction,
    AnalystPriceTarget,
    AnalystRecommendation,
    Dividend,
    EarningsDate,
    EarningsEstimate,
    EarningsHistory,
    EpsRevisions,
    EpsTrend,
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

# Trading 212 lists sub-unit currencies as GBX/ZAC while Yahoo tags them GBp/ZAc.
_SUB_UNIT_ALIASES = {"GBX": "GBp", "ZAC": "ZAc"}


def _same_currency_unit(a: str, b: str) -> bool:
    """Return True when two currency codes denote the exact same unit.

    The comparison is case-sensitive on purpose: ``GBP`` and ``GBp`` differ by a
    factor of 100.
    """
    return _SUB_UNIT_ALIASES.get(a, a) == _SUB_UNIT_ALIASES.get(b, b)


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


def _insider_lot_order(row: dict[str, Any]) -> tuple[Any, ...]:
    """Sort key giving same-key insider lots a source-order-independent order."""
    shares, value = row["shares"], row["value"]
    return (
        shares is None,
        shares or 0,
        value is None,
        value or 0,
        row["ownership"] or "",
        row["position"] or "",
    )


def _safe_percent_as_fraction(v: Any) -> float | None:
    """Convert a figure Yahoo quotes in percent (``0.44`` = 0.44%) to a fraction."""
    f = _safe_float(v)
    return None if f is None else f / 100.0


def _safe_str(v: Any, max_len: int | None = None) -> str | None:
    v = _safe_val(v)
    if v is None:
        return None
    s = str(v)
    if max_len:
        s = s[:max_len]
    return s


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


def _refreshable_columns(row: dict[str, Any], sticky: frozenset[str]) -> list[str]:
    """Columns an ON CONFLICT update may overwrite for a single-row snapshot.

    Every column except the conflict key, minus the ``sticky`` ones this row
    leaves empty. ``updated_at`` stays in so the row is still stamped.
    """
    columns = [
        name
        for name, value in row.items()
        if name != "instrument_id" and not (name in sticky and value is None)
    ]
    return [*columns, "updated_at"]


def _estimate_update_columns(rows: list[dict[str, Any]]) -> list[str] | None:
    """Conflict-update columns for a forward-estimate batch.

    A fetch that carries no currency must not turn a label stored by an earlier
    one into NULL, so ``currency`` is left out of the update in that case.

    Returns:
        ``None`` (update every column) when any row has a currency, otherwise
        every column except ``currency``.
    """
    if not rows or any(r["currency"] is not None for r in rows):
        return None
    return _refreshable_columns(rows[0], frozenset({"currency"}))


def _last_wins(rows: list[dict[str, Any]], *key_fields: str) -> list[dict[str, Any]]:
    """Collapse rows that share a conflict key, keeping the last occurrence.

    A single ``INSERT ... ON CONFLICT DO UPDATE`` cannot affect the same key
    twice ("cannot affect row a second time"), and the error aborts the whole
    statement, so a duplicate in one feed would drop that category for the
    ticker on every run. ``instrument_id`` is constant within a batch and is
    left out of ``key_fields``.
    """
    return list({tuple(r[k] for k in key_fields): r for r in rows}.values())


def _latest_reported_per_holder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per holder name, preferring the most recent report date."""
    ordered = sorted(rows, key=lambda r: r["date_reported"] or _SENTINEL_DATE)
    return _last_wins(ordered, "holder_name")


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
            pass
        # yfinance valuation-measure columns are labelled "M/D/YYYY" without
        # zero padding.
        try:
            return datetime.strptime(v, "%m/%d/%Y").date()
        except (ValueError, TypeError):
            return None
    if isinstance(v, int | float):
        # Yahoo epoch dates sit at 00:00 UTC, so reading them in the host's
        # timezone would shift the calendar day on any host west of UTC.
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).date()
        except (ValueError, TypeError, OSError, OverflowError):
            return None
    return None


# (statement_type, period_type) pairs written by the six statement fetches.
FINANCIAL_STATEMENT_GROUPS: tuple[tuple[str, str], ...] = (
    ("income_statement", "annual"),
    ("income_statement", "quarterly"),
    ("balance_sheet", "annual"),
    ("balance_sheet", "quarterly"),
    ("cashflow", "annual"),
    ("cashflow", "quarterly"),
)

# valuation_measures rows share financial_statements with the statements but
# are refreshed independently, so they get their own marker.
VALUATION_MEASURES_STATEMENT_TYPE = "valuation_measures"
VALUATION_MEASURES_PERIOD_TYPE = "point_in_time"

_FUNDAMENTALS_CATEGORY_MODELS: tuple[tuple[str, type], ...] = (
    ("eps_trend", EpsTrend),
    ("eps_revisions", EpsRevisions),
    ("earnings_estimate", EarningsEstimate),
    ("revenue_estimate", RevenueEstimate),
    ("growth_estimates", GrowthEstimate),
    ("earnings_history", EarningsHistory),
    ("earnings_dates", EarningsDate),
    ("sec_filings", SecFiling),
    ("shares_outstanding", SharesOutstanding),
)

# Categories refreshed on the fundamentals window. Each one reports its own
# ``{category}_updated_at`` so a failed or empty fetch of one category is not
# hidden by the others being fresh.
FUNDAMENTAL_CATEGORIES: tuple[str, ...] = (
    "valuation_measures",
    *(category for category, _ in _FUNDAMENTALS_CATEGORY_MODELS),
)

_STALENESS_CATEGORY_MODELS: tuple[tuple[str, type], ...] = (
    ("profile", TickerProfile),
    ("dividends", Dividend),
    ("splits", StockSplit),
    ("recommendations", AnalystRecommendation),
    ("price_targets", AnalystPriceTarget),
    ("institutional_holders", InstitutionalHolder),
    ("mutualfund_holders", MutualFundHolder),
    ("insider_transactions", InsiderTransaction),
    ("news", TickerNews),
    *_FUNDAMENTALS_CATEGORY_MODELS,
)


def statement_staleness_key(statement_type: str, period_type: str) -> str:
    """Return the staleness-dict key for one financial statement group.

    Args:
        statement_type: Statement label, e.g. ``"balance_sheet"``.
        period_type: ``"annual"`` or ``"quarterly"``.

    Returns:
        The key under which :meth:`YFinanceRepository.get_staleness_info`
        reports that group's last write, e.g. ``"balance_sheet_annual_updated_at"``.
    """
    return f"{statement_type}_{period_type}_updated_at"


def _statement_staleness(
    latest: Mapping[tuple[str, str], datetime | None],
) -> dict[str, Any]:
    """Build the financial_statements entries of a staleness dict.

    Args:
        latest: ``MAX(updated_at)`` per ``(statement_type, period_type)`` group.

    Returns:
        One marker per statement group, ``financials_updated_at`` and
        ``valuation_measures_updated_at``. ``financials_updated_at`` is the
        *oldest* of the six statement groups, and ``None`` while any group has
        never been stored: a statement that failed during a rate-limit burst
        must not be hidden by one that succeeded.
    """
    entries: dict[str, Any] = {
        statement_staleness_key(stmt, period): latest.get((stmt, period))
        for stmt, period in FINANCIAL_STATEMENT_GROUPS
    }
    stamps = list(entries.values())
    entries["financials_updated_at"] = (
        None if any(stamp is None for stamp in stamps) else min(stamps)
    )
    entries["valuation_measures_updated_at"] = latest.get(
        (VALUATION_MEASURES_STATEMENT_TYPE, VALUATION_MEASURES_PERIOD_TYPE)
    )
    return entries


class YFinanceRepository(RepositoryBase):
    """Sync repository for yfinance data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Ticker Profile
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Options chain (high-volume, own scheduler step)
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
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
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
                "currency": _safe_str(row.get("currency"), 10),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert(
            EarningsEstimate,
            rows,
            constraint_name="uq_earnings_estimate_instrument_period",
            update_columns=_estimate_update_columns(rows),
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
                "currency": _safe_str(row.get("currency"), 10),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert(
            RevenueEstimate,
            rows,
            constraint_name="uq_revenue_estimate_instrument_period",
            update_columns=_estimate_update_columns(rows),
        )

    def upsert_growth_estimates(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert per-period growth estimates (stock + index trend).

        yfinance 1.6.0 emits only ``stockTrend`` / ``indexTrend`` (industry and
        sector trend were dropped upstream). Column names vary across versions, so
        read defensively.
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
                "index_trend": _safe_float(_col(row, "indexTrend", "index")),
            }
            for period, row in df.iterrows()
        ]
        return self._upsert(
            GrowthEstimate,
            _last_wins(rows, "period"),
            constraint_name="uq_growth_estimate_instrument_period",
        )

    def upsert_eps_trend(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the EPS estimate trend (index = period labels "0q"/"+1q"/...).

        Columns are snapshot ages (``current``/``7daysAgo``/``30daysAgo``/...) plus
        a per-period ``currency`` column (yfinance 1.2.2+).
        """
        rows = [
            {
                "instrument_id": instrument_id,
                "period": _safe_str(period, 10),
                "current_estimate": _safe_float(row.get("current")),
                "days_ago_7": _safe_float(row.get("7daysAgo")),
                "days_ago_30": _safe_float(row.get("30daysAgo")),
                "days_ago_60": _safe_float(row.get("60daysAgo")),
                "days_ago_90": _safe_float(row.get("90daysAgo")),
                "currency": _safe_str(row.get("currency"), 10),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert(
            EpsTrend,
            rows,
            constraint_name="uq_eps_trend_instrument_period",
            update_columns=_estimate_update_columns(rows),
        )

    def upsert_eps_revisions(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert EPS revision counts (index = period labels "0q"/"+1q"/...).

        Columns are up/down revision counts over the trailing 7/30-day windows.
        Yahoo emits ``downLast7Days`` (capital D); the lowercase spelling is
        accepted too.
        """
        rows = [
            {
                "instrument_id": instrument_id,
                "period": _safe_str(period, 10),
                "up_last_7days": _safe_int(row.get("upLast7days")),
                "up_last_30days": _safe_int(row.get("upLast30days")),
                # Yahoo capitalises the D only on this column.
                "down_last_7days": _safe_int(
                    row.get("downLast7Days", row.get("downLast7days"))
                ),
                "down_last_30days": _safe_int(row.get("downLast30days")),
                "currency": _safe_str(row.get("currency"), 10),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert(
            EpsRevisions,
            rows,
            constraint_name="uq_eps_revisions_instrument_period",
            update_columns=_estimate_update_columns(rows),
        )

    def upsert_earnings_history(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert historical EPS surprise (index = past-quarter dates).

        Deduped in-batch on ``period_date`` (last-wins): two source rows on the
        same calendar day would otherwise collide on ``uq_earnings_history_...``
        within one ON CONFLICT and abort the whole write.

        ``surprisePercent`` arrives as a fraction (0.04 = 4%) and is stored in
        percent points, the unit ``earnings_dates`` and the earnings calendar
        already use.
        """
        by_date: dict[date, dict[str, Any]] = {}
        for idx, row in df.iterrows():
            period_date = _safe_date(idx)
            if period_date is None:
                continue
            surprise = _safe_float(row.get("surprisePercent"))
            by_date[period_date] = {
                "instrument_id": instrument_id,
                "period_date": period_date,
                "eps_estimate": _safe_float(row.get("epsEstimate")),
                "eps_actual": _safe_float(row.get("epsActual")),
                "eps_difference": _safe_float(row.get("epsDifference")),
                "surprise_percent": None if surprise is None else surprise * 100.0,
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
        seen: set[tuple[Any, ...]] = set()
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

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

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
            _last_wins(rows, "period"),
            constraint_name="uq_analyst_rec_instrument_period",
        )

    def get_recommendations(
        self, instrument_id: UUID
    ) -> Sequence[AnalystRecommendation]:
        """Return analyst recommendation summary rows for an instrument."""
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
        """Return the analyst price target row for an instrument, or None."""
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
            _latest_reported_per_holder(rows),
            constraint_name="uq_inst_holder_instrument_name",
        )

    def get_institutional_holders(
        self, instrument_id: UUID
    ) -> Sequence[InstitutionalHolder]:
        """Return institutional holder rows for an instrument, ordered by holder name."""
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
            _latest_reported_per_holder(rows),
            constraint_name="uq_mutual_fund_holder_instrument_name",
        )

    def get_mutualfund_holders(self, instrument_id: UUID) -> Sequence[MutualFundHolder]:
        """Return mutual fund holder rows for an instrument, ordered by holder name."""
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
        """Upsert insider transactions from a ``Ticker.insider_transactions`` frame.

        Rows sharing (insider, start date, transaction type) are merged by
        summing shares and value, because the unique key cannot tell separate
        same-day tranches apart. A row whose Transaction and Text are both
        blank is kept under an empty transaction type; only rows without an
        insider name are dropped.

        Args:
            instrument_id: Instrument the transactions belong to.
            insiders_df: Frame as returned by yfinance.

        Returns:
            Number of distinct rows upserted.
        """
        rows: list[dict[str, Any]] = []
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
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "insider_name": name,
                    "position": _safe_str(row_data.get("Position"), 500),
                    # Real grants/vestings arrive with Transaction and Text both
                    # blank; the column is NOT NULL, so keep them under "".
                    "transaction_type": tx_type or "",
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "start_date": _safe_date(row_data.get("Start Date"))
                    or _SENTINEL_DATE,
                    "ownership": _safe_str(row_data.get("Ownership"), 50),
                }
            )

        # Yahoo returns several real transactions sharing (instrument, insider,
        # start_date, transaction_type), e.g. two grants of different size on
        # one day. A single INSERT ... ON CONFLICT cannot touch the same
        # conflict key twice, and keeping only one of them would drop the other
        # lots' shares/value for good, so they are numbered with ``lot_index``.
        # Sorting each group first pins every lot to the same slot regardless of
        # the order Yahoo lists them in, keeping refetches idempotent.
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for r in rows:
            key = (
                r["instrument_id"],
                r["insider_name"],
                r["start_date"],
                r["transaction_type"],
            )
            groups.setdefault(key, []).append(r)

        numbered: list[dict[str, Any]] = []
        for group in groups.values():
            group.sort(key=_insider_lot_order)
            for lot_index, r in enumerate(group):
                r["lot_index"] = lot_index
                numbered.append(r)

        return self._upsert(
            InsiderTransaction,
            numbered,
            constraint_name="uq_insider_tx_row",
        )

    def get_insider_transactions(
        self, instrument_id: UUID
    ) -> Sequence[InsiderTransaction]:
        """Return insider transaction rows for an instrument, ordered newest-first."""
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
                    # The column is timestamptz: keep UTC rather than a naive
                    # value the database would read in its session timezone.
                    if publish_time.tzinfo is None:
                        publish_time = publish_time.replace(tzinfo=timezone.utc)
                    else:
                        publish_time = publish_time.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    pass
            if publish_time is None:
                pt = article.get("providerPublishTime")
                if pt is not None:
                    with contextlib.suppress(
                        ValueError, TypeError, OSError, OverflowError
                    ):
                        publish_time = datetime.fromtimestamp(int(pt), tz=timezone.utc)

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

        # The same articles reappear in ticker.news for days and each run
        # re-scrapes them. A scrape that fails (timeout, paywall, consent wall)
        # yields no body, and rewriting every column would NULL a body stored by
        # an earlier run, so those rows leave full_content out of the conflict
        # update.
        rows = _last_wins(rows, "news_uuid")
        with_body = [r for r in rows if r["full_content"]]
        without_body = [r for r in rows if not r["full_content"]]
        keep_body_columns = [
            col.name
            for col in TickerNews.__table__.columns
            if col.name not in {"id", "created_at", "full_content"}
        ]

        written = 0
        if with_body:
            written += self._upsert(
                TickerNews,
                with_body,
                constraint_name="uq_ticker_news_instrument_uuid",
            )
        if without_body:
            written += self._upsert(
                TickerNews,
                without_body,
                constraint_name="uq_ticker_news_instrument_uuid",
                update_columns=keep_body_columns,
            )
        return written

    def get_news(self, instrument_id: UUID) -> Sequence[TickerNews]:
        """Return news articles for an instrument, ordered by publish time descending."""
        stmt = (
            select(TickerNews)
            .where(TickerNews.instrument_id == instrument_id)
            .order_by(TickerNews.publish_time.desc().nullslast())
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Staleness info (for incremental fetch)
    # ------------------------------------------------------------------

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

    def get_staleness_info(self, instrument_id: UUID) -> dict[str, Any]:
        """Return staleness metadata for incremental fetch decisions.

        Returns a dict with:
          - price_max_date: MAX(date) from price_history (date or None)
          - price_min_date: MIN(date) from price_history (date or None)
          - {category}_updated_at: MAX(updated_at) from each non-price table (datetime or None)
          - {statement_type}_{period_type}_updated_at: MAX(updated_at) of each of
            the six financial statement groups
          - financials_updated_at: the oldest of those six, ``None`` while any
            group is missing
          - valuation_measures_updated_at: MAX(updated_at) of the valuation rows
            stored alongside the statements
        """
        result: dict[str, Any] = {}

        price_max, price_min = self.session.execute(
            select(func.max(PriceHistory.date), func.min(PriceHistory.date)).where(
                PriceHistory.instrument_id == instrument_id
            )
        ).one()
        result["price_max_date"] = price_max
        result["price_min_date"] = price_min

        for category, model in _STALENESS_CATEGORY_MODELS:
            val = self.session.execute(
                select(func.max(cast(Any, model).updated_at)).where(
                    cast(Any, model).instrument_id == instrument_id
                )
            ).scalar_one_or_none()
            result[f"{category}_updated_at"] = val

        latest_by_group: dict[tuple[str, str], datetime | None] = {
            (stmt, period): stamp
            for stmt, period, stamp in self.session.execute(
                select(
                    FinancialStatement.statement_type,
                    FinancialStatement.period_type,
                    func.max(FinancialStatement.updated_at),
                )
                .where(FinancialStatement.instrument_id == instrument_id)
                .group_by(
                    FinancialStatement.statement_type,
                    FinancialStatement.period_type,
                )
            ).all()
        }
        result.update(_statement_staleness(latest_by_group))

        return result

    def get_staleness_info_bulk(
        self, instrument_ids: Sequence[UUID]
    ) -> dict[UUID, dict[str, Any]]:
        """Batched :meth:`get_staleness_info` for a whole sweep.

        Returns one full staleness dict per requested id (all-None for an
        instrument with no rows yet), matching the single-query shape exactly —
        but via one grouped query per category instead of one per instrument.
        """
        ids = list(instrument_ids)
        result: dict[UUID, dict[str, Any]] = {
            iid: {
                "price_max_date": None,
                "price_min_date": None,
                **{f"{cat}_updated_at": None for cat, _ in _STALENESS_CATEGORY_MODELS},
                **_statement_staleness({}),
            }
            for iid in ids
        }
        if not ids:
            return result

        for iid, mx, mn in self.session.execute(
            select(
                PriceHistory.instrument_id,
                func.max(PriceHistory.date),
                func.min(PriceHistory.date),
            )
            .where(PriceHistory.instrument_id.in_(ids))
            .group_by(PriceHistory.instrument_id)
        ).all():
            result[iid]["price_max_date"] = mx
            result[iid]["price_min_date"] = mn

        for category, model in _STALENESS_CATEGORY_MODELS:
            m = cast(Any, model)
            for iid, mx in self.session.execute(
                select(m.instrument_id, func.max(m.updated_at))
                .where(m.instrument_id.in_(ids))
                .group_by(m.instrument_id)
            ).all():
                result[iid][f"{category}_updated_at"] = mx

        latest_by_instrument: dict[UUID, dict[tuple[str, str], datetime | None]] = {}
        for iid, stmt, period, stamp in self.session.execute(
            select(
                FinancialStatement.instrument_id,
                FinancialStatement.statement_type,
                FinancialStatement.period_type,
                func.max(FinancialStatement.updated_at),
            )
            .where(FinancialStatement.instrument_id.in_(ids))
            .group_by(
                FinancialStatement.instrument_id,
                FinancialStatement.statement_type,
                FinancialStatement.period_type,
            )
        ).all():
            latest_by_instrument.setdefault(iid, {})[(stmt, period)] = stamp
        for iid, latest in latest_by_instrument.items():
            result[iid].update(_statement_staleness(latest))

        return result

    # ------------------------------------------------------------------
    # Instrument queries (used by yfinance_data endpoint)
    # ------------------------------------------------------------------

    def get_instruments_with_yfinance_ticker(self) -> Sequence[Instrument]:
        """Return the instruments to ingest: those with a non-empty yfinance_ticker.

        A delisted row whose symbol is now carried by an active row (the old side
        of a venue transfer) is left out, so one symbol is never fetched and
        stored under two instrument ids. Delisted rows with no active successor
        are kept. The exchange is eager-loaded.
        """
        successor = aliased(Instrument)
        superseded = and_(
            Instrument.delisted_at.is_not(None),
            exists().where(
                successor.yfinance_ticker == Instrument.yfinance_ticker,
                successor.delisted_at.is_(None),
            ),
        )
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker.isnot(None))
                .where(Instrument.yfinance_ticker != "")
                .where(~superseded)
            )
            .scalars()
            .unique()
            .all()
        )

    def get_instrument_by_yfinance_ticker(
        self, yfinance_ticker: str
    ) -> Instrument | None:
        """Return the instrument for a yfinance_ticker, with exchange eager-loaded.

        ``instruments`` is unique on ``(ticker, exchange)`` and ``yfinance_ticker``
        is indexed but not unique, so a symbol that moved venues leaves a delisted
        row beside its active one. The active listing wins; among several
        candidates the most recently delisted (or, for active rows, the most
        recently created) does, so the result is deterministic and a genuinely
        delisted holding still resolves.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.

        Returns:
            The chosen instrument, or ``None`` when no row has that symbol.
        """
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker == yfinance_ticker)
                .order_by(
                    Instrument.delisted_at.is_(None).desc(),
                    Instrument.delisted_at.desc(),
                    Instrument.created_at.desc(),
                    Instrument.ticker,
                    Instrument.id,
                )
                .limit(1)
            )
            .scalars()
            .first()
        )

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
