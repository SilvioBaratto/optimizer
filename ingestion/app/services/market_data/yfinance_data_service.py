"""Service layer orchestrating yfinance data fetching and storage.

Bulk-ingestion error policy (issue #850): each per-field fetch (profile,
prices, financials, dividends, holders, news, …) runs under its own
``except Exception`` that logs at WARNING and appends the failure to the
per-ticker ``errors`` list before continuing. One bad field never aborts the
ticker, and one bad ticker never aborts the batch (``all_errors``). Those
lists are surfaced to the caller in the fetch result, so the drops are
intentional per-item resilience, not silent swallows.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pandas as pd

from app.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)
from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services._shared import ProgressCallback, _noop, has_sufficient_history
from app.services.market_data.yfinance import YFinanceClient
from app.utils.currency import to_major_currency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout watchdog
# ---------------------------------------------------------------------------
# yfinance 1.3 does not expose a timeout parameter on the `.info` property
# or financial-statement attributes (income_stmt, balance_sheet, etc.).
# Wrapping the blocking call in a daemon thread and joining with a wall-clock
# deadline is the only approach that enforces a true timeout without using
# signals (signals may only be set from the main thread, which is illegal
# inside the thread pool workers used by the parallel fetch path).
#
# `call_with_timeout` submits the callable to a fresh one-shot thread pool,
# waits at most `timeout` seconds for a result, and raises `TimeoutError` on
# expiry.  The underlying thread is a daemon so it does not prevent process
# shutdown even if the network call never returns.
#
# `fetch_history` already accepts a `timeout` kwarg in yfinance 1.3 (passed
# directly to `requests`) — for that call we still use the kwarg because it
# avoids spawning an extra thread, but we additionally wrap it here so that
# the retry loop in `_facade.py` is also bounded.


def call_with_timeout(fn: "Any", timeout: float) -> "Any":
    """Execute *fn* in a daemon thread; raise ``TimeoutError`` after *timeout* seconds.

    Args:
        fn: Zero-argument callable to execute.
        timeout: Maximum wall-clock seconds to wait.

    Returns:
        The return value of *fn*.

    Raises:
        TimeoutError: If *fn* does not complete within *timeout* seconds.
        Exception: Any exception raised by *fn* is re-raised in the caller.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="yf-timeout") as _pool:
        future: Future[Any] = _pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as err:
            # The worker thread is a daemon — it will be abandoned but won't
            # block process shutdown.
            raise TimeoutError(
                f"yfinance call did not complete within {timeout:.0f}s"
            ) from err


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
    etf_metadata_hours: int = 168  # 7 days
    # news: always fetched (no threshold)
    # prices: always fetched (date-ranged)
    price_overlap_days: int = 5
    min_history_tolerance: float = 0.95


DEFAULT_THRESHOLDS = StalenessThresholds()


def _is_fresh(
    updated_at: datetime | None,
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

    def __init__(
        self,
        repo: YFinanceRepository,
        yf_client: YFinanceClient,
        request_timeout: float | None = None,
    ) -> None:
        self.repo = repo
        self.yf_client = yf_client
        # Wall-clock timeout for every yfinance network call.  None disables
        # the watchdog (used in tests that mock fetch methods synchronously).
        self._request_timeout = request_timeout

    def _timed(self, fn: "Any") -> "Any":
        """Execute *fn*, enforcing ``self._request_timeout`` if set."""
        if self._request_timeout is None:
            return fn()
        return call_with_timeout(fn, self._request_timeout)

    def fetch_and_store(
        self,
        instrument_id: UUID,
        yfinance_ticker: str,
        period: str = "5y",
        mode: str = "incremental",
        thresholds: StalenessThresholds | None = None,
        exchange_name: str | None = None,
        currency_code: str | None = None,
        asset_class: str | None = None,
    ) -> dict[str, Any]:
        """Fetch all data categories for a single ticker and store.

        Args:
            instrument_id: Database instrument UUID.
            yfinance_ticker: Yahoo Finance ticker symbol.
            period: Price history period for full mode (e.g. "5y").
            mode: "full" preserves original behaviour; "incremental" skips
                  fresh categories and date-ranges the price fetch.
            thresholds: Override default staleness thresholds.
            exchange_name: Exchange name for calendar-based history validation.
            currency_code: Instrument listing currency (e.g. "GBX").
                Converted to major-unit (e.g. "GBP") before storing on
                financial statement rows.
            asset_class: Instrument asset class. When "fixed_income" or
                "multi_asset" (i.e. an ETF), the ETF fund-metadata branch runs.

        Returns dict with counts per category, list of errors, and skipped categories.
        """
        counts: dict[str, int] = {}
        errors: list[str] = []
        skipped: list[str] = []

        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS

        # In incremental mode, query staleness info once upfront
        staleness: dict[str, Any] | None = None
        if mode == "incremental":
            try:
                staleness = self.repo.get_staleness_info(instrument_id)
            except Exception as e:
                logger.warning(
                    "Staleness query failed for %s, falling back to full fetch: %s",
                    yfinance_ticker,
                    e,
                )
                staleness = None

        now = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # Incremental short-circuit: skip whole ticker if ALL categories fresh
        # ------------------------------------------------------------------
        # In incremental mode we know staleness upfront.  If every category
        # passes its freshness check there is nothing to fetch — avoid
        # constructing a yf.Ticker at all (which acquires the rate-limiter
        # even on a cache hit, adding ~100ms×N delay over 2907 tickers).
        if mode == "incremental" and staleness is not None:
            all_fresh = (
                _is_fresh(
                    staleness.get("profile_updated_at"), thresholds.profile_hours, now
                )
                and staleness.get("price_max_date") is not None  # prices have data
                and _is_fresh(
                    staleness.get("financials_updated_at"),
                    thresholds.financials_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("dividends_updated_at"),
                    thresholds.dividends_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("splits_updated_at"), thresholds.splits_hours, now
                )
                and _is_fresh(
                    staleness.get("recommendations_updated_at"),
                    thresholds.recommendations_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("price_targets_updated_at"),
                    thresholds.price_targets_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("institutional_holders_updated_at"),
                    thresholds.institutional_holders_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("mutualfund_holders_updated_at"),
                    thresholds.mutualfund_holders_hours,
                    now,
                )
                and _is_fresh(
                    staleness.get("insider_transactions_updated_at"),
                    thresholds.insider_transactions_hours,
                    now,
                )
            )
            if all_fresh:
                logger.debug(
                    "All categories fresh for %s — skipping network fetch",
                    yfinance_ticker,
                )
                skipped.append("all:fresh")
                return {"counts": counts, "errors": errors, "skipped": skipped}

        # Lazy ticker construction: only materialise the yf.Ticker (and
        # acquire the rate-limiter) when at least one category will be fetched.
        # The ticker object is used only by the news fetch (section 11); all
        # other categories call through the sub-clients which call _get_ticker
        # internally (and share the same cache).
        _ticker_holder: list[Any] = []  # mutable cell for lazy init

        def _get_lazy_ticker() -> Any:
            if not _ticker_holder:
                _ticker_holder.append(self.yf_client.get_ticker(yfinance_ticker))
            return _ticker_holder[0]

        # 1. Profile (info)
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("profile_updated_at"), thresholds.profile_hours, now
            )
        ):
            skipped.append("profile")
        else:
            try:
                info = self._timed(lambda: self.yf_client.fetch_info(yfinance_ticker))
                if info:
                    counts["profile"] = self.repo.upsert_profile(instrument_id, info)
                else:
                    counts["profile"] = 0
            except Exception as e:
                errors.append(f"profile: {e}")
                logger.warning("Failed to fetch profile for %s: %s", yfinance_ticker, e)

        # 2. Price history
        # fetch_history accepts a native `timeout` kwarg (passed to requests),
        # so we pass it directly without an extra watchdog thread.
        try:
            if (
                mode == "incremental"
                and staleness is not None
                and staleness.get("price_max_date") is not None
            ):
                # Incremental: fetch from max_date - overlap_days to today
                max_date = staleness["price_max_date"]
                start_date = max_date - timedelta(days=thresholds.price_overlap_days)
                # auto_adjust=True (default) — split-adjusted Close required by prices_to_returns() downstream
                history = self.yf_client.fetch_history(
                    yfinance_ticker,
                    start=start_date.isoformat(),
                    end=date.today().isoformat(),
                    min_rows=0,
                    timeout=self._request_timeout,
                )
            else:
                # Full mode or no existing data: use period
                # auto_adjust=True (default) — split-adjusted Close required by prices_to_returns() downstream
                history = self.yf_client.fetch_history(
                    yfinance_ticker,
                    period=period,
                    min_rows=1,
                    timeout=self._request_timeout,
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
                        # The insufficient-history guard exists to stop a
                        # truncated/glitchy re-fetch from clobbering an
                        # established price series. But for an instrument with
                        # NO stored prices yet, a short history is simply a
                        # young listing's complete, legitimate history (recent
                        # IPOs/SPACs). Discarding it permanently excluded ~102
                        # instruments from price_history with no error logged.
                        # Only discard when prior price data exists.
                        if staleness is not None:
                            price_max_date = staleness.get("price_max_date")
                        else:
                            price_max_date = self.repo.get_staleness_info(
                                instrument_id
                            ).get("price_max_date")
                        has_existing_prices = price_max_date is not None
                        if has_existing_prices:
                            logger.info(
                                "Skipping price storage for %s: got %d rows, "
                                "expected ~%d (min %d) for %s on %s "
                                "(protecting existing series)",
                                yfinance_ticker,
                                len(history),
                                expected,
                                minimum,
                                period,
                                exchange_name,
                            )
                            counts["prices"] = 0
                            skipped.append("prices:insufficient_history")
                            history = None
                        else:
                            logger.info(
                                "Storing short history for %s: %d rows "
                                "(below expected ~%d for %s) — young listing "
                                "with no prior price data",
                                yfinance_ticker,
                                len(history),
                                expected,
                                period,
                            )

            if history is not None and not history.empty:
                # Store the listing currency as the price unit (raw, unconverted) so
                # sub-unit series (GBX pence) are unambiguous downstream (SPEC OQ2).
                counts["prices"] = self.repo.upsert_price_history(
                    instrument_id, history, price_unit=currency_code
                )
            else:
                counts["prices"] = counts.get("prices", 0)
        except Exception as e:
            errors.append(f"prices: {e}")
            logger.warning("Failed to fetch prices for %s: %s", yfinance_ticker, e)

        # 3. Financial statements (income, balance, cashflow - annual + quarterly)
        # Financial statements from yfinance are in the company's reporting
        # currency (e.g. GBP for UK companies), not the listing quote
        # currency (e.g. GBX).  Store the major-unit code for auditability.
        reporting_currency = to_major_currency(currency_code)

        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("financials_updated_at"), thresholds.financials_hours, now
            )
        ):
            skipped.append("financials")
        else:
            try:
                fs_count = 0
                for stmt_type, fetch_method, quarterly_flag in [
                    (
                        "income_statement",
                        self.yf_client.financials.fetch_income_stmt,
                        False,
                    ),
                    (
                        "income_statement",
                        self.yf_client.financials.fetch_income_stmt,
                        True,
                    ),
                    (
                        "balance_sheet",
                        self.yf_client.financials.fetch_balance_sheet,
                        False,
                    ),
                    (
                        "balance_sheet",
                        self.yf_client.financials.fetch_balance_sheet,
                        True,
                    ),
                    ("cashflow", self.yf_client.financials.fetch_cashflow, False),
                    ("cashflow", self.yf_client.financials.fetch_cashflow, True),
                ]:
                    period_type = "quarterly" if quarterly_flag else "annual"
                    _fm, _qf = fetch_method, quarterly_flag
                    try:
                        df = self._timed(
                            lambda m=_fm, q=_qf: m(yfinance_ticker, quarterly=q)
                        )
                        if df is not None and not df.empty:
                            fs_count += self.repo.upsert_financial_statements(
                                instrument_id,
                                df,
                                stmt_type,
                                period_type,
                                currency_code=reporting_currency,
                            )
                    except Exception as e:
                        errors.append(f"financials.{stmt_type}.{period_type}: {e}")
                        logger.warning(
                            "Failed %s %s for %s: %s",
                            stmt_type,
                            period_type,
                            yfinance_ticker,
                            e,
                        )

                counts["financials"] = fs_count
            except Exception as e:
                errors.append(f"financials: {e}")
                logger.warning("Failed financials for %s: %s", yfinance_ticker, e)

        # 3a. Valuation measures (Ticker.valuation, 9-metric panel).
        # Open-EAV reuse of financial_statements: no migration, no new model.
        # Staleness-gated on the shared financial_statements table.
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("financials_updated_at"), thresholds.financials_hours, now
            )
        ):
            skipped.append("valuation_measures")
        else:
            try:
                val_df = self._timed(
                    lambda: self.yf_client.metadata.fetch_valuation_measures(
                        yfinance_ticker
                    )
                )
                if val_df is not None and not val_df.empty:
                    counts["valuation_measures"] = (
                        self.repo.upsert_financial_statements(
                            instrument_id,
                            val_df,
                            "valuation_measures",
                            "point_in_time",
                            currency_code=reporting_currency,
                        )
                    )
                else:
                    counts["valuation_measures"] = 0
            except Exception as e:
                errors.append(f"valuation_measures: {e}")
                logger.warning(
                    "Failed valuation_measures for %s: %s", yfinance_ticker, e
                )

        # 3b. EPS trend. yfinance returns string period labels ("0q", "+1q",
        # "0y", "+1y"); coerce columns to datetimes and drop NaT so the
        # repository's _safe_date does not silently drop rows.
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("financials_updated_at"), thresholds.financials_hours, now
            )
        ):
            skipped.append("eps_trend")
        else:
            try:
                eps_trend_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_eps_trend(yfinance_ticker)
                )
                if eps_trend_df is not None and not eps_trend_df.empty:
                    eps_trend_df = eps_trend_df.copy()
                    eps_trend_df.columns = pd.to_datetime(
                        eps_trend_df.columns, errors="coerce"
                    )
                    eps_trend_df = eps_trend_df.loc[:, eps_trend_df.columns.notna()]
                if eps_trend_df is not None and not eps_trend_df.empty:
                    counts["eps_trend"] = self.repo.upsert_financial_statements(
                        instrument_id,
                        eps_trend_df,
                        "eps_trend",
                        "estimate",
                        currency_code=None,
                    )
                else:
                    counts["eps_trend"] = 0
            except Exception as e:
                errors.append(f"eps_trend: {e}")
                logger.warning("Failed eps_trend for %s: %s", yfinance_ticker, e)

        # 3c. EPS revisions. Same period-label coercion as eps_trend.
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("financials_updated_at"), thresholds.financials_hours, now
            )
        ):
            skipped.append("eps_revisions")
        else:
            try:
                eps_rev_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_eps_revisions(yfinance_ticker)
                )
                if eps_rev_df is not None and not eps_rev_df.empty:
                    eps_rev_df = eps_rev_df.copy()
                    eps_rev_df.columns = pd.to_datetime(
                        eps_rev_df.columns, errors="coerce"
                    )
                    eps_rev_df = eps_rev_df.loc[:, eps_rev_df.columns.notna()]
                if eps_rev_df is not None and not eps_rev_df.empty:
                    counts["eps_revisions"] = self.repo.upsert_financial_statements(
                        instrument_id,
                        eps_rev_df,
                        "eps_revisions",
                        "estimate",
                        currency_code=None,
                    )
                else:
                    counts["eps_revisions"] = 0
            except Exception as e:
                errors.append(f"eps_revisions: {e}")
                logger.warning("Failed eps_revisions for %s: %s", yfinance_ticker, e)

        # 3d/3e/3f. Analyst estimates (earnings / revenue / growth) — dedicated
        # typed tables, staleness-gated on the shared fundamentals window.
        estimates_fresh = (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("financials_updated_at"), thresholds.financials_hours, now
            )
        )
        if estimates_fresh:
            skipped.append("earnings_estimate")
        else:
            try:
                ee_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_earnings_estimate(
                        yfinance_ticker
                    )
                )
                counts["earnings_estimate"] = (
                    self.repo.upsert_earnings_estimate(instrument_id, ee_df)
                    if ee_df is not None and not ee_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"earnings_estimate: {e}")
                logger.warning(
                    "Failed earnings_estimate for %s: %s", yfinance_ticker, e
                )

        if estimates_fresh:
            skipped.append("revenue_estimate")
        else:
            try:
                re_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_revenue_estimate(
                        yfinance_ticker
                    )
                )
                counts["revenue_estimate"] = (
                    self.repo.upsert_revenue_estimate(instrument_id, re_df)
                    if re_df is not None and not re_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"revenue_estimate: {e}")
                logger.warning("Failed revenue_estimate for %s: %s", yfinance_ticker, e)

        if estimates_fresh:
            skipped.append("growth_estimates")
        else:
            try:
                ge_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_growth_estimates(
                        yfinance_ticker
                    )
                )
                counts["growth_estimates"] = (
                    self.repo.upsert_growth_estimates(instrument_id, ge_df)
                    if ge_df is not None and not ge_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"growth_estimates: {e}")
                logger.warning("Failed growth_estimates for %s: %s", yfinance_ticker, e)

        # 3g. Earnings history (past-quarter EPS surprise).
        if estimates_fresh:
            skipped.append("earnings_history")
        else:
            try:
                eh_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_earnings_history(
                        yfinance_ticker
                    )
                )
                counts["earnings_history"] = (
                    self.repo.upsert_earnings_history(instrument_id, eh_df)
                    if eh_df is not None and not eh_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"earnings_history: {e}")
                logger.warning("Failed earnings_history for %s: %s", yfinance_ticker, e)

        # 3h. Earnings dates (past + upcoming).
        if estimates_fresh:
            skipped.append("earnings_dates")
        else:
            try:
                ed_df = self._timed(
                    lambda: self.yf_client.metadata.fetch_earnings_dates(
                        yfinance_ticker
                    )
                )
                counts["earnings_dates"] = (
                    self.repo.upsert_earnings_dates(instrument_id, ed_df)
                    if ed_df is not None and not ed_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"earnings_dates: {e}")
                logger.warning("Failed earnings_dates for %s: %s", yfinance_ticker, e)

        # 3i. Analyst upgrade/downgrade actions (analyst-rating cadence).
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("recommendations_updated_at"),
                thresholds.recommendations_hours,
                now,
            )
        ):
            skipped.append("analyst_actions")
        else:
            try:
                ad_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_upgrades_downgrades(
                        yfinance_ticker
                    )
                )
                counts["analyst_actions"] = (
                    self.repo.upsert_analyst_actions(instrument_id, ad_df)
                    if ad_df is not None and not ad_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"analyst_actions: {e}")
                logger.warning("Failed analyst_actions for %s: %s", yfinance_ticker, e)

        # 3j. ESG / sustainability (rarely changes; fundamentals window).
        if estimates_fresh:
            skipped.append("esg_scores")
        else:
            try:
                esg_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_sustainability(
                        yfinance_ticker
                    )
                )
                counts["esg_scores"] = (
                    self.repo.upsert_esg_scores(instrument_id, esg_df)
                    if esg_df is not None and not esg_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"esg_scores: {e}")
                logger.warning("Failed esg_scores for %s: %s", yfinance_ticker, e)

        # 4. Dividends
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("dividends_updated_at"), thresholds.dividends_hours, now
            )
        ):
            skipped.append("dividends")
        else:
            try:
                dividends = self._timed(
                    lambda: self.yf_client.corporate_actions.fetch_dividends(
                        yfinance_ticker
                    )
                )
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
            and _is_fresh(
                staleness.get("splits_updated_at"), thresholds.splits_hours, now
            )
        ):
            skipped.append("splits")
        else:
            try:
                splits = self._timed(
                    lambda: self.yf_client.corporate_actions.fetch_splits(
                        yfinance_ticker
                    )
                )
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
            and _is_fresh(
                staleness.get("recommendations_updated_at"),
                thresholds.recommendations_hours,
                now,
            )
        ):
            skipped.append("recommendations")
        else:
            try:
                rec_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_recommendations_summary(
                        yfinance_ticker
                    )
                )
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
            and _is_fresh(
                staleness.get("price_targets_updated_at"),
                thresholds.price_targets_hours,
                now,
            )
        ):
            skipped.append("price_targets")
        else:
            try:
                targets = self._timed(
                    lambda: self.yf_client.analysis.fetch_analyst_price_targets(
                        yfinance_ticker
                    )
                )
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
            and _is_fresh(
                staleness.get("institutional_holders_updated_at"),
                thresholds.institutional_holders_hours,
                now,
            )
        ):
            skipped.append("institutional_holders")
        else:
            try:
                inst_df = self._timed(
                    lambda: self.yf_client.holders.fetch_institutional_holders(
                        yfinance_ticker
                    )
                )
                if inst_df is not None and not inst_df.empty:
                    counts["institutional_holders"] = (
                        self.repo.upsert_institutional_holders(instrument_id, inst_df)
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
            and _is_fresh(
                staleness.get("mutualfund_holders_updated_at"),
                thresholds.mutualfund_holders_hours,
                now,
            )
        ):
            skipped.append("mutualfund_holders")
        else:
            try:
                mf_df = self._timed(
                    lambda: self.yf_client.holders.fetch_mutualfund_holders(
                        yfinance_ticker
                    )
                )
                if mf_df is not None and not mf_df.empty:
                    counts["mutualfund_holders"] = self.repo.upsert_mutualfund_holders(
                        instrument_id, mf_df
                    )
                else:
                    counts["mutualfund_holders"] = 0
            except Exception as e:
                errors.append(f"mutualfund_holders: {e}")
                logger.warning(
                    "Failed mutual fund holders for %s: %s", yfinance_ticker, e
                )

        # 10. Insider transactions
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("insider_transactions_updated_at"),
                thresholds.insider_transactions_hours,
                now,
            )
        ):
            skipped.append("insider_transactions")
        else:
            try:
                insiders_df = self._timed(
                    lambda: self.yf_client.holders.fetch_insider_transactions(
                        yfinance_ticker
                    )
                )
                if insiders_df is not None and not insiders_df.empty:
                    counts["insider_transactions"] = (
                        self.repo.upsert_insider_transactions(
                            instrument_id, insiders_df
                        )
                    )
                else:
                    counts["insider_transactions"] = 0
            except Exception as e:
                errors.append(f"insider_transactions: {e}")
                logger.warning(
                    "Failed insider transactions for %s: %s", yfinance_ticker, e
                )

        # 11. News (always fetched with full article content scraped)
        try:
            news_list = self._timed(lambda: _get_lazy_ticker().news)
            if news_list:
                from app.services.market_data.yfinance.news.scraper import (
                    ArticleScraper,
                )

                scraper = ArticleScraper(delay=0.5)
                for article in news_list:
                    content = article.get("content", article)
                    canonical = content.get("canonicalUrl")
                    link = (
                        canonical.get("url", "")
                        if isinstance(canonical, dict)
                        else content.get("previewUrl") or article.get("link")
                    )
                    if link:
                        result = scraper.fetch(link)
                        if result["success"] and result["content"]:
                            article["full_content"] = result["content"]
                counts["news"] = self.repo.upsert_news(instrument_id, news_list)
            else:
                counts["news"] = 0
        except Exception as e:
            errors.append(f"news: {e}")
            logger.warning("Failed news for %s: %s", yfinance_ticker, e)

        # --- ETF fund metadata (fixed_income / multi_asset instruments only) ---
        if asset_class in ("fixed_income", "multi_asset"):
            self._fetch_etf_metadata(
                instrument_id,
                yfinance_ticker,
                mode,
                thresholds,
                now,
                counts,
                errors,
                skipped,
            )

        return {"counts": counts, "errors": errors, "skipped": skipped}

    def _fetch_etf_metadata(
        self,
        instrument_id: UUID,
        yfinance_ticker: str,
        mode: str,
        thresholds: StalenessThresholds,
        now: datetime,
        counts: dict[str, int],
        errors: list[str],
        skipped: list[str],
    ) -> None:
        """Fetch + store ETF fund metadata (profile, asset-class split, holdings,
        sector weights) via the funds sub-client. Staleness-gated in incremental
        mode; best-effort (a non-fund simply yields nothing)."""
        etf_repo = ETFMetadataRepository(self.repo.session)
        if mode == "incremental":
            existing = etf_repo.get_metadata(instrument_id)
            if existing is not None and _is_fresh(
                existing.updated_at, thresholds.etf_metadata_hours, now
            ):
                skipped.append("etf_metadata")
                return

        try:
            as_of = now.date()
            profile = self._timed(
                lambda: self.yf_client.funds.fetch_fund_profile(yfinance_ticker)
            )
            if profile:
                etf_repo.upsert_metadata(
                    instrument_id,
                    aum=profile.get("aum"),
                    nav=profile.get("nav"),
                    fund_family=profile.get("fund_family"),
                    legal_type=profile.get("legal_type"),
                    expense_ratio=profile.get("expense_ratio"),
                    base_currency=profile.get("base_currency"),
                    as_of=as_of,
                )
                counts["etf_metadata"] = 1

            fdata = self._timed(
                lambda: self.yf_client.funds.fetch_funds_data(yfinance_ticker)
            )
            if fdata:
                ac = fdata.get("asset_classes") or {}
                etf_repo.upsert_asset_classes(
                    instrument_id,
                    as_of,
                    stock_pct=ac.get("stockPosition"),
                    bond_pct=ac.get("bondPosition"),
                    cash_pct=ac.get("cashPosition"),
                    other_pct=ac.get("otherPosition"),
                )
                counts["etf_asset_classes"] = 1
                holdings = fdata.get("top_holdings") or []
                if holdings:
                    counts["etf_holdings"] = etf_repo.upsert_holdings(
                        instrument_id, as_of, holdings
                    )
                sectors = fdata.get("sector_weightings") or {}
                if sectors:
                    counts["etf_sector_weights"] = etf_repo.upsert_sector_weights(
                        instrument_id, as_of, sectors
                    )
        except Exception as e:
            errors.append(f"etf_metadata: {e}")
            logger.warning("Failed ETF metadata for %s: %s", yfinance_ticker, e)


# ---------------------------------------------------------------------------
# Standalone bulk fetch (callable from routes and scheduler)
# ---------------------------------------------------------------------------


def run_bulk_yfinance_fetch(
    request: Any,
    yf_client: YFinanceClient,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute bulk yfinance fetch for all instruments.

    Dispatches to a thread-pool implementation when ``request.workers > 1``;
    otherwise uses the in-process serial path with a single shared session.

    Args:
        request: ``YFinanceFetchRequest`` with ``period``, ``mode``, ``workers``.
        yf_client: Configured yfinance client instance.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``tickers_processed``, ``counts``, ``error_count``,
        and ``skipped_category_count``.
    """
    if request.workers > 1:
        return _run_bulk_yfinance_fetch_parallel(
            request, yf_client, on_progress=on_progress, workers=request.workers
        )

    from app.database import database_manager

    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        instruments = repo.get_instruments_with_yfinance_ticker()

        total = len(instruments)
        on_progress(total=total)

        all_errors: list[str] = []
        total_counts: dict[str, int] = {}
        total_skipped: int = 0

        from app.config import settings as _settings

        service = YFinanceDataService(
            repo,
            yf_client,
            request_timeout=float(_settings.yfinance_request_timeout_seconds),
        )

        for idx, instrument in enumerate(instruments, 1):
            ticker = instrument.yfinance_ticker
            on_progress(current=idx, current_ticker=ticker)

            if not ticker:
                all_errors.append(f"{instrument.id}: missing yfinance_ticker, skipped")
                continue

            try:
                result = service.fetch_and_store(
                    instrument_id=instrument.id,
                    yfinance_ticker=ticker,
                    period=request.period,
                    mode=request.mode,
                    exchange_name=instrument.exchange_name,
                    currency_code=instrument.currency_code,
                    asset_class=getattr(instrument, "asset_class", None),
                )

                for k, v in result["counts"].items():
                    total_counts[k] = total_counts.get(k, 0) + v
                total_skipped += len(result.get("skipped", []))
                for err in result["errors"]:
                    all_errors.append(f"{ticker}: {err}")

                session.commit()

            except Exception as e:
                logger.error("Failed to process %s: %s", ticker, e)
                all_errors.append(f"{ticker}: {e}")
                session.rollback()

        result_dict = {
            "tickers_processed": total,
            "counts": total_counts,
            "error_count": len(all_errors),
            "skipped_category_count": total_skipped,
        }
        on_progress(
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            errors=all_errors,
            result=result_dict,
        )
        logger.info("Bulk yfinance fetch completed: %d tickers", total)

    return result_dict


# ---------------------------------------------------------------------------
# Parallel bulk fetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InstrumentSpec:
    """Detached instrument fields used by worker threads."""

    instrument_id: UUID
    yfinance_ticker: str
    exchange_name: str | None
    currency_code: str | None
    asset_class: str | None = None


def _load_instrument_specs(database_manager: Any) -> list[_InstrumentSpec]:
    """Load instruments inside a short-lived main-thread session."""
    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        return [
            _InstrumentSpec(
                instrument_id=inst.id,
                yfinance_ticker=inst.yfinance_ticker or "",
                exchange_name=inst.exchange_name,
                currency_code=inst.currency_code,
                asset_class=getattr(inst, "asset_class", None),
            )
            for inst in repo.get_instruments_with_yfinance_ticker()
        ]


def _fetch_one_ticker(
    spec: _InstrumentSpec,
    request: Any,
    yf_client: YFinanceClient,
    database_manager: Any,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch a single ticker inside an isolated per-thread session."""
    if not spec.yfinance_ticker:
        raise ValueError("missing yfinance_ticker")
    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        service = YFinanceDataService(repo, yf_client, request_timeout=request_timeout)
        try:
            result = service.fetch_and_store(
                instrument_id=spec.instrument_id,
                yfinance_ticker=spec.yfinance_ticker,
                period=request.period,
                mode=request.mode,
                exchange_name=spec.exchange_name,
                currency_code=spec.currency_code,
                asset_class=spec.asset_class,
            )
            session.commit()
            return result
        except Exception:
            session.rollback()
            raise


def _aggregate_outcome(
    spec: _InstrumentSpec,
    outcome: dict[str, Any] | Exception,
    counts: dict[str, int],
    errors: list[str],
) -> int:
    """Fold a single worker outcome into the running aggregates.

    Returns the number of skipped sub-categories for this ticker.
    """
    if isinstance(outcome, Exception):
        errors.append(f"{spec.yfinance_ticker}: {outcome}")
        return 0
    for key, value in outcome["counts"].items():
        counts[key] = counts.get(key, 0) + value
    for err in outcome["errors"]:
        errors.append(f"{spec.yfinance_ticker}: {err}")
    return len(outcome.get("skipped", []))


def _run_bulk_yfinance_fetch_parallel(
    request: Any,
    yf_client: YFinanceClient,
    *,
    on_progress: ProgressCallback,
    workers: int,
) -> dict[str, Any]:
    """Execute bulk yfinance fetch across a thread pool.

    Each worker opens its own session via ``database_manager.get_session()``;
    sessions never cross thread boundaries. A ``threading.Lock``-guarded
    counter drives ``on_progress``. Worker exceptions are isolated and
    aggregated into the final ``error_count``.
    """
    from app.config import settings as _settings
    from app.database import database_manager

    request_timeout = float(_settings.yfinance_request_timeout_seconds)

    specs = _load_instrument_specs(database_manager)
    total = len(specs)
    on_progress(total=total)

    counter = [0]
    counter_lock = threading.Lock()
    aggregate_lock = threading.Lock()
    counts: dict[str, int] = {}
    errors: list[str] = []
    skipped = 0

    def _worker(
        spec: _InstrumentSpec,
    ) -> tuple[_InstrumentSpec, dict[str, Any] | Exception]:
        try:
            return spec, _fetch_one_ticker(
                spec, request, yf_client, database_manager, request_timeout
            )
        except Exception as exc:
            return spec, exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, spec) for spec in specs]
        for future in as_completed(futures):
            spec, outcome = future.result()
            with counter_lock:
                counter[0] += 1
                idx = counter[0]
            on_progress(current=idx, current_ticker=spec.yfinance_ticker)
            with aggregate_lock:
                skipped += _aggregate_outcome(spec, outcome, counts, errors)

    result_dict = {
        "tickers_processed": total,
        "counts": counts,
        "error_count": len(errors),
        "skipped_category_count": skipped,
    }
    on_progress(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        errors=errors,
        result=result_dict,
    )
    logger.info(
        "Parallel yfinance fetch completed: %d tickers (%d workers)", total, workers
    )
    return result_dict
