"""Repository for yfinance data access with PostgreSQL upsert support."""

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

_SENTINEL_DATE = date(1970, 1, 1)

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.yfinance_data import (
    AnalystPriceTarget,
    AnalystRecommendation,
    Dividend,
    FinancialStatement,
    InsiderTransaction,
    InstitutionalHolder,
    MutualFundHolder,
    PriceHistory,
    StockSplit,
    TickerNews,
    TickerProfile,
)

logger = logging.getLogger(__name__)


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


def _safe_int(v: Any) -> Optional[int]:
    v = _safe_val(v)
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v: Any) -> Optional[float]:
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


def _safe_str(v: Any, max_len: Optional[int] = None) -> Optional[str]:
    v = _safe_val(v)
    if v is None:
        return None
    s = str(v)
    if max_len:
        s = s[:max_len]
    return s


def _safe_date(v: Any) -> Optional[date]:
    """Convert various date-like values to date."""
    v = _safe_val(v)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v).date()
        except (ValueError, TypeError):
            return None
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(v).date()
        except (ValueError, TypeError, OSError):
            return None
    return None


class YFinanceRepository:
    """Sync repository for yfinance data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Generic upsert helper
    # ------------------------------------------------------------------

    def _upsert(
        self,
        model: type,
        rows: List[Dict[str, Any]],
        constraint_name: str,
        update_columns: Optional[List[str]] = None,
    ) -> int:
        """Insert rows with ON CONFLICT DO UPDATE. Returns count of rows processed."""
        if not rows:
            return 0

        stmt = pg_insert(model.__table__).values(rows)

        if update_columns:
            update_dict = {col: stmt.excluded[col] for col in update_columns}
        else:
            # Update all columns except the primary key and created_at
            exclude = {"id", "created_at"}
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in model.__table__.columns
                if col.name not in exclude
            }

        stmt = stmt.on_conflict_do_update(
            constraint=constraint_name,
            set_=update_dict,
        )

        self.session.execute(stmt)
        return len(rows)

    # ------------------------------------------------------------------
    # Ticker Profile
    # ------------------------------------------------------------------

    def upsert_profile(self, instrument_id: UUID, info: Dict[str, Any]) -> int:
        """Upsert a ticker profile from yf.Ticker.info dict."""
        # Map yfinance info keys to model columns
        ex_div = info.get("exDividendDate")
        if isinstance(ex_div, (int, float)):
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
            "implied_shares_outstanding": _safe_int(info.get("impliedSharesOutstanding")),
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
            "price_to_sales_trailing_12months": _safe_float(info.get("priceToSalesTrailing12Months")),
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
            "earnings_quarterly_growth": _safe_float(info.get("earningsQuarterlyGrowth")),
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
            "five_year_avg_dividend_yield": _safe_float(info.get("fiveYearAvgDividendYield")),
            "trailing_annual_dividend_rate": _safe_float(info.get("trailingAnnualDividendRate")),
            "trailing_annual_dividend_yield": _safe_float(info.get("trailingAnnualDividendYield")),
            "last_dividend_value": _safe_float(info.get("lastDividendValue")),
            "target_high_price": _safe_float(info.get("targetHighPrice")),
            "target_low_price": _safe_float(info.get("targetLowPrice")),
            "target_mean_price": _safe_float(info.get("targetMeanPrice")),
            "target_median_price": _safe_float(info.get("targetMedianPrice")),
            "number_of_analyst_opinions": _safe_int(info.get("numberOfAnalystOpinions")),
            "recommendation_key": _safe_str(info.get("recommendationKey"), 50),
            "recommendation_mean": _safe_float(info.get("recommendationMean")),
            "full_time_employees": _safe_int(info.get("fullTimeEmployees")),
        }

        return self._upsert(
            TickerProfile,
            [row],
            constraint_name="uq_ticker_profile_instrument",
        )

    def get_profile(self, instrument_id: UUID) -> Optional[TickerProfile]:
        stmt = select(TickerProfile).where(
            TickerProfile.instrument_id == instrument_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    def upsert_price_history(
        self, instrument_id: UUID, history_df: pd.DataFrame
    ) -> int:
        """Upsert daily OHLCV rows from a yfinance history DataFrame."""
        rows = []
        for idx, row_data in history_df.iterrows():
            dt = idx
            if isinstance(dt, pd.Timestamp):
                dt = dt.date()
            elif isinstance(dt, datetime):
                dt = dt.date()

            rows.append({
                "instrument_id": instrument_id,
                "date": dt,
                "open": _safe_float(row_data.get("Open")),
                "high": _safe_float(row_data.get("High")),
                "low": _safe_float(row_data.get("Low")),
                "close": _safe_float(row_data.get("Close")),
                "volume": _safe_int(row_data.get("Volume")),
                "dividends": _safe_float(row_data.get("Dividends")),
                "stock_splits": _safe_float(row_data.get("Stock Splits")),
            })

        return self._upsert(
            PriceHistory,
            rows,
            constraint_name="uq_price_history_instrument_date",
        )

    def get_price_history(
        self,
        instrument_id: UUID,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 5000,
    ) -> Sequence[PriceHistory]:
        stmt = (
            select(PriceHistory)
            .where(PriceHistory.instrument_id == instrument_id)
        )
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
    ) -> int:
        """Upsert financial statement rows (EAV format).

        yfinance returns DataFrames where:
         - columns are period dates
         - index rows are line item names
        """
        rows = []
        for col in df.columns:
            period_date = _safe_date(col)
            if period_date is None:
                continue
            for line_item in df.index:
                val = _safe_float(df.at[line_item, col])
                rows.append({
                    "instrument_id": instrument_id,
                    "statement_type": statement_type,
                    "period_type": period_type,
                    "period_date": period_date,
                    "line_item": _safe_str(line_item, 200),
                    "value": val,
                })

        return self._upsert(
            FinancialStatement,
            rows,
            constraint_name="uq_financial_statement_row",
        )

    def get_financial_statements(
        self,
        instrument_id: UUID,
        statement_type: Optional[str] = None,
        period_type: Optional[str] = None,
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

    def upsert_dividends(
        self, instrument_id: UUID, dividends: pd.Series
    ) -> int:
        """Upsert dividend data from yfinance Series (index=date, value=amount)."""
        rows = []
        for idx, amount in dividends.items():
            dt = _safe_date(idx)
            amt = _safe_float(amount)
            if dt is None or amt is None:
                continue
            rows.append({
                "instrument_id": instrument_id,
                "date": dt,
                "amount": amt,
            })

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
            rows.append({
                "instrument_id": instrument_id,
                "date": dt,
                "ratio": r,
            })

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
    # Analyst Recommendations
    # ------------------------------------------------------------------

    def upsert_recommendations(
        self, instrument_id: UUID, rec_df: pd.DataFrame
    ) -> int:
        """Upsert analyst recommendations from recommendations_summary DataFrame."""
        rows = []
        for _, row_data in rec_df.iterrows():
            period = _safe_str(row_data.get("period"), 50)
            if not period:
                continue
            rows.append({
                "instrument_id": instrument_id,
                "period": period,
                "strong_buy": _safe_int(row_data.get("strongBuy")),
                "buy": _safe_int(row_data.get("buy")),
                "hold": _safe_int(row_data.get("hold")),
                "sell": _safe_int(row_data.get("sell")),
                "strong_sell": _safe_int(row_data.get("strongSell")),
            })

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

    def upsert_price_targets(
        self, instrument_id: UUID, targets: Dict[str, Any]
    ) -> int:
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

    def get_price_targets(
        self, instrument_id: UUID
    ) -> Optional[AnalystPriceTarget]:
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
            rows.append({
                "instrument_id": instrument_id,
                "holder_name": name,
                "date_reported": _safe_date(row_data.get("Date Reported")),
                "shares": _safe_int(row_data.get("Shares")),
                "value": _safe_int(row_data.get("Value")),
                "pct_held": _safe_float(row_data.get("% Out")),
            })

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
            rows.append({
                "instrument_id": instrument_id,
                "holder_name": name,
                "date_reported": _safe_date(row_data.get("Date Reported")),
                "shares": _safe_int(row_data.get("Shares")),
                "value": _safe_int(row_data.get("Value")),
                "pct_held": _safe_float(row_data.get("% Out")),
            })

        return self._upsert(
            MutualFundHolder,
            rows,
            constraint_name="uq_mutual_fund_holder_instrument_name",
        )

    def get_mutualfund_holders(
        self, instrument_id: UUID
    ) -> Sequence[MutualFundHolder]:
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
            if not name or not tx_type:
                continue
            rows.append({
                "instrument_id": instrument_id,
                "insider_name": name,
                "position": _safe_str(row_data.get("Position"), 500),
                "transaction_type": tx_type,
                "shares": _safe_int(row_data.get("Shares")),
                "value": _safe_int(row_data.get("Value")),
                "start_date": _safe_date(row_data.get("Start Date")) or _SENTINEL_DATE,
                "ownership": _safe_str(row_data.get("Ownership"), 50),
            })

        return self._upsert(
            InsiderTransaction,
            rows,
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
    # Ticker News
    # ------------------------------------------------------------------

    def upsert_news(
        self, instrument_id: UUID, articles: List[Dict[str, Any]]
    ) -> int:
        """Upsert news articles from yf.Ticker.news list of dicts."""
        rows = []
        for article in articles:
            news_uuid = _safe_str(article.get("uuid"), 200)
            if not news_uuid:
                continue

            # Extract publish time - yfinance stores as epoch or nested
            publish_time = None
            pt = article.get("providerPublishTime")
            if pt is not None:
                try:
                    publish_time = datetime.fromtimestamp(int(pt))
                except (ValueError, TypeError, OSError):
                    pass

            # Extract related tickers
            related = article.get("relatedTickers")

            rows.append({
                "instrument_id": instrument_id,
                "news_uuid": news_uuid,
                "title": _safe_str(article.get("title")),
                "publisher": _safe_str(article.get("publisher"), 500),
                "link": _safe_str(article.get("link")),
                "publish_time": publish_time,
                "news_type": _safe_str(article.get("type"), 100),
                "related_tickers": ",".join(related) if isinstance(related, list) else None,
            })

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

    def get_staleness_info(self, instrument_id: UUID) -> Dict[str, Any]:
        """Return staleness metadata for incremental fetch decisions.

        Returns a dict with:
          - price_max_date: MAX(date) from price_history (date or None)
          - {category}_updated_at: MAX(updated_at) from each non-price table (datetime or None)
        """
        result: Dict[str, Any] = {}

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
                select(func.max(model.updated_at)).where(
                    model.instrument_id == instrument_id
                )
            ).scalar_one_or_none()
            result[f"{category}_updated_at"] = val

        return result
