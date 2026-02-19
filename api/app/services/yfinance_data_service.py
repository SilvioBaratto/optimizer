"""Service layer orchestrating yfinance data fetching and storage."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.yfinance_repository import YFinanceRepository
from app.services.trading_calendar import has_sufficient_history
from app.services.yfinance import YFinanceClient

logger = logging.getLogger(__name__)


@dataclass
class StalenessThresholds:
    """Configurable freshness thresholds per data category."""

    profile_hours: int = 24
    financials_hours: int = 168  # 7 days
    dividends_hours: int = 24
    splits_hours: int = 24
    recommendations_hours: int = 24
    price_targets_hours: int = 24
    institutional_holders_hours: int = 24
    mutualfund_holders_hours: int = 24
    insider_transactions_hours: int = 24
    # news: always fetched (no threshold)
    # prices: always fetched (date-ranged)
    price_overlap_days: int = 5
    min_history_tolerance: float = 0.95


DEFAULT_THRESHOLDS = StalenessThresholds()


def _is_fresh(
    updated_at: Optional[datetime],
    threshold_hours: int,
    now: datetime,
) -> bool:
    """Return True if the data is still within the freshness window."""
    if updated_at is None:
        return False
    # Ensure both are timezone-aware for comparison
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - updated_at) < timedelta(hours=threshold_hours)


class YFinanceDataService:
    """Fetches all yfinance data categories for a ticker and stores via repository."""

    def __init__(self, session: Session, yf_client: YFinanceClient):
        self.session = session
        self.repo = YFinanceRepository(session)
        self.yf_client = yf_client

    def fetch_and_store(
        self,
        instrument_id: UUID,
        yfinance_ticker: str,
        period: str = "5y",
        mode: str = "incremental",
        thresholds: Optional[StalenessThresholds] = None,
        exchange_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch all data categories for a single ticker and store.

        Args:
            instrument_id: Database instrument UUID.
            yfinance_ticker: Yahoo Finance ticker symbol.
            period: Price history period for full mode (e.g. "5y").
            mode: "full" preserves original behaviour; "incremental" skips
                  fresh categories and date-ranges the price fetch.
            thresholds: Override default staleness thresholds.
            exchange_name: Exchange name for calendar-based history validation.

        Returns dict with counts per category, list of errors, and skipped categories.
        """
        counts: Dict[str, int] = {}
        errors: List[str] = []
        skipped: List[str] = []

        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS

        # In incremental mode, query staleness info once upfront
        staleness: Optional[Dict[str, Any]] = None
        if mode == "incremental":
            try:
                staleness = self.repo.get_staleness_info(instrument_id)
            except Exception as e:
                logger.warning(
                    "Staleness query failed for %s, falling back to full fetch: %s",
                    yfinance_ticker, e,
                )
                staleness = None

        now = datetime.now(timezone.utc)

        ticker = self.yf_client.get_ticker(yfinance_ticker)

        # 1. Profile (info)
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("profile_updated_at"), thresholds.profile_hours, now)
        ):
            skipped.append("profile")
        else:
            try:
                info = self.yf_client.fetch_info(yfinance_ticker)
                if info:
                    counts["profile"] = self.repo.upsert_profile(instrument_id, info)
                else:
                    counts["profile"] = 0
            except Exception as e:
                errors.append(f"profile: {e}")
                logger.warning("Failed to fetch profile for %s: %s", yfinance_ticker, e)

        # 2. Price history
        try:
            if (
                mode == "incremental"
                and staleness is not None
                and staleness.get("price_max_date") is not None
            ):
                # Incremental: fetch from max_date - overlap_days to today
                max_date = staleness["price_max_date"]
                start_date = max_date - timedelta(days=thresholds.price_overlap_days)
                history = self.yf_client.fetch_history(
                    yfinance_ticker,
                    start=start_date.isoformat(),
                    end=date.today().isoformat(),
                    min_rows=0,
                )
            else:
                # Full mode or no existing data: use period
                history = self.yf_client.fetch_history(
                    yfinance_ticker, period=period, min_rows=1
                )

            if history is not None and not history.empty:
                # Validate history length for full-period fetches only
                is_full_period = not (
                    mode == "incremental"
                    and staleness is not None
                    and staleness.get("price_max_date") is not None
                )
                if is_full_period:
                    sufficient, expected, minimum = has_sufficient_history(
                        len(history),
                        exchange_name,
                        period,
                        thresholds.min_history_tolerance,
                    )
                    if not sufficient:
                        logger.info(
                            "Skipping price storage for %s: got %d rows, "
                            "expected ~%d (min %d) for %s on %s",
                            yfinance_ticker, len(history),
                            expected, minimum, period, exchange_name,
                        )
                        counts["prices"] = 0
                        skipped.append("prices:insufficient_history")
                        history = None

            if history is not None and not history.empty:
                counts["prices"] = self.repo.upsert_price_history(
                    instrument_id, history
                )
            else:
                counts["prices"] = counts.get("prices", 0)
        except Exception as e:
            errors.append(f"prices: {e}")
            logger.warning("Failed to fetch prices for %s: %s", yfinance_ticker, e)

        # 3. Financial statements (income, balance, cashflow - annual + quarterly)
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("financials_updated_at"), thresholds.financials_hours, now)
        ):
            skipped.append("financials")
        else:
            try:
                fs_count = 0
                for stmt_type, fetch_method, quarterly_flag in [
                    ("income_statement", self.yf_client.financials.fetch_income_stmt, False),
                    ("income_statement", self.yf_client.financials.fetch_income_stmt, True),
                    ("balance_sheet", self.yf_client.financials.fetch_balance_sheet, False),
                    ("balance_sheet", self.yf_client.financials.fetch_balance_sheet, True),
                    ("cashflow", self.yf_client.financials.fetch_cashflow, False),
                    ("cashflow", self.yf_client.financials.fetch_cashflow, True),
                ]:
                    period_type = "quarterly" if quarterly_flag else "annual"
                    try:
                        df = fetch_method(yfinance_ticker, quarterly=quarterly_flag)
                        if df is not None and not df.empty:
                            fs_count += self.repo.upsert_financial_statements(
                                instrument_id, df, stmt_type, period_type
                            )
                    except Exception as e:
                        errors.append(f"financials.{stmt_type}.{period_type}: {e}")
                        logger.warning(
                            "Failed %s %s for %s: %s",
                            stmt_type, period_type, yfinance_ticker, e,
                        )

                # Earnings (annual + quarterly)
                for quarterly_flag in [False, True]:
                    period_type = "quarterly" if quarterly_flag else "annual"
                    try:
                        df = self.yf_client.financials.fetch_earnings(
                            yfinance_ticker, quarterly=quarterly_flag
                        )
                        if df is not None and not df.empty:
                            fs_count += self.repo.upsert_financial_statements(
                                instrument_id, df, "earnings", period_type
                            )
                    except Exception as e:
                        errors.append(f"financials.earnings.{period_type}: {e}")
                        logger.warning(
                            "Failed earnings %s for %s: %s",
                            period_type, yfinance_ticker, e,
                        )

                counts["financials"] = fs_count
            except Exception as e:
                errors.append(f"financials: {e}")
                logger.warning("Failed financials for %s: %s", yfinance_ticker, e)

        # 4. Dividends
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("dividends_updated_at"), thresholds.dividends_hours, now)
        ):
            skipped.append("dividends")
        else:
            try:
                dividends = self.yf_client.corporate_actions.fetch_dividends(yfinance_ticker)
                if dividends is not None and not dividends.empty:
                    counts["dividends"] = self.repo.upsert_dividends(
                        instrument_id, dividends
                    )
                else:
                    counts["dividends"] = 0
            except Exception as e:
                errors.append(f"dividends: {e}")
                logger.warning("Failed dividends for %s: %s", yfinance_ticker, e)

        # 5. Stock splits
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("splits_updated_at"), thresholds.splits_hours, now)
        ):
            skipped.append("splits")
        else:
            try:
                splits = self.yf_client.corporate_actions.fetch_splits(yfinance_ticker)
                if splits is not None and not splits.empty:
                    counts["splits"] = self.repo.upsert_splits(instrument_id, splits)
                else:
                    counts["splits"] = 0
            except Exception as e:
                errors.append(f"splits: {e}")
                logger.warning("Failed splits for %s: %s", yfinance_ticker, e)

        # 6. Analyst recommendations
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("recommendations_updated_at"), thresholds.recommendations_hours, now)
        ):
            skipped.append("recommendations")
        else:
            try:
                rec_df = self.yf_client.analysis.fetch_recommendations_summary(yfinance_ticker)
                if rec_df is not None and not rec_df.empty:
                    counts["recommendations"] = self.repo.upsert_recommendations(
                        instrument_id, rec_df
                    )
                else:
                    counts["recommendations"] = 0
            except Exception as e:
                errors.append(f"recommendations: {e}")
                logger.warning("Failed recommendations for %s: %s", yfinance_ticker, e)

        # 7. Analyst price targets
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("price_targets_updated_at"), thresholds.price_targets_hours, now)
        ):
            skipped.append("price_targets")
        else:
            try:
                targets = self.yf_client.analysis.fetch_analyst_price_targets(yfinance_ticker)
                if targets:
                    counts["price_targets"] = self.repo.upsert_price_targets(
                        instrument_id, targets
                    )
                else:
                    counts["price_targets"] = 0
            except Exception as e:
                errors.append(f"price_targets: {e}")
                logger.warning("Failed price targets for %s: %s", yfinance_ticker, e)

        # 8. Institutional holders
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("institutional_holders_updated_at"), thresholds.institutional_holders_hours, now)
        ):
            skipped.append("institutional_holders")
        else:
            try:
                inst_df = self.yf_client.holders.fetch_institutional_holders(yfinance_ticker)
                if inst_df is not None and not inst_df.empty:
                    counts["institutional_holders"] = self.repo.upsert_institutional_holders(
                        instrument_id, inst_df
                    )
                else:
                    counts["institutional_holders"] = 0
            except Exception as e:
                errors.append(f"institutional_holders: {e}")
                logger.warning(
                    "Failed institutional holders for %s: %s", yfinance_ticker, e
                )

        # 9. Mutual fund holders
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("mutualfund_holders_updated_at"), thresholds.mutualfund_holders_hours, now)
        ):
            skipped.append("mutualfund_holders")
        else:
            try:
                mf_df = self.yf_client.holders.fetch_mutualfund_holders(yfinance_ticker)
                if mf_df is not None and not mf_df.empty:
                    counts["mutualfund_holders"] = self.repo.upsert_mutualfund_holders(
                        instrument_id, mf_df
                    )
                else:
                    counts["mutualfund_holders"] = 0
            except Exception as e:
                errors.append(f"mutualfund_holders: {e}")
                logger.warning("Failed mutual fund holders for %s: %s", yfinance_ticker, e)

        # 10. Insider transactions
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(staleness.get("insider_transactions_updated_at"), thresholds.insider_transactions_hours, now)
        ):
            skipped.append("insider_transactions")
        else:
            try:
                insiders_df = self.yf_client.holders.fetch_insider_transactions(yfinance_ticker)
                if insiders_df is not None and not insiders_df.empty:
                    counts["insider_transactions"] = self.repo.upsert_insider_transactions(
                        instrument_id, insiders_df
                    )
                else:
                    counts["insider_transactions"] = 0
            except Exception as e:
                errors.append(f"insider_transactions: {e}")
                logger.warning(
                    "Failed insider transactions for %s: %s", yfinance_ticker, e
                )

        # 11. News (always fetched — small payload, yfinance returns recent items only)
        try:
            news_list = ticker.news
            if news_list:
                counts["news"] = self.repo.upsert_news(instrument_id, news_list)
            else:
                counts["news"] = 0
        except Exception as e:
            errors.append(f"news: {e}")
            logger.warning("Failed news for %s: %s", yfinance_ticker, e)

        return {"counts": counts, "errors": errors, "skipped": skipped}
