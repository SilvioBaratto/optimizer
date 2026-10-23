"""Service layer orchestrating yfinance data fetching and storage.

Bulk-ingestion error policy: each per-field fetch (profile,
prices, financials, dividends, holders, news, …) runs under its own
``except Exception`` that logs at WARNING and appends the failure to the
per-ticker ``errors`` list before continuing. One bad field never aborts the
ticker, and one bad ticker never aborts the batch (``all_errors``). Those
lists are surfaced to the caller in the fetch result, so the drops are
intentional per-item resilience, not silent swallows. Every repository write
runs in its own SAVEPOINT (``YFinanceDataService._write``) so a failed category
cannot abort the shared transaction and take the ticker's other writes with it.

A request that Yahoo never answered (unreachable, rate limited, timed out) is
not "no data": ``fetch_and_store`` reports it as ``transport_failures`` and a
``transport:`` error, and a sweep in which most requests went unanswered ends
as a failed job (``YahooUnreachableError``) instead of a quiet "completed".
"""

import contextvars
import logging
import math
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pandas as pd
from portopt_db.repositories.market_data.etf_metadata_repository import (
    ETFMetadataRepository,
)
from portopt_db.repositories.market_data.yfinance_repository import (
    FINANCIAL_STATEMENT_GROUPS,
    FUNDAMENTAL_CATEGORIES,
    VALUATION_MEASURES_PERIOD_TYPE,
    VALUATION_MEASURES_STATEMENT_TYPE,
    YFinanceRepository,
    statement_staleness_key,
)

from app.services._shared import (
    ProgressCallback,
    _noop,
    has_sufficient_history,
    parse_period_years,
)
from app.services._shared.trading_calendar import last_completed_session
from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance import (
    YFinanceClient,
    record_failure,
    track_request_outcomes,
)
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
# `call_with_timeout` runs the callable on a fresh daemon thread, waits at most
# `timeout` seconds for it, and raises `TimeoutError` on expiry.  The caller is
# released at the deadline and the thread is abandoned rather than joined: a
# `ThreadPoolExecutor` (non-daemon workers, joined on shutdown) would hold the
# caller until the call finished and could keep the interpreter alive at exit.
# An abandoned call runs on in the background and its result is dropped.
#
# `fetch_history` already accepts a `timeout` kwarg in yfinance 1.3 (passed
# directly to `requests`) — for that call we still use the kwarg because it
# avoids spawning an extra thread, but we additionally wrap it here so that
# the retry loop in `_facade.py` is also bounded.


def call_with_timeout(fn: "Any", timeout: float) -> "Any":
    """Execute *fn* in a daemon thread; raise ``TimeoutError`` after *timeout* seconds.

    On expiry the caller is released immediately; *fn* keeps running in the
    abandoned thread and whatever it returns or raises afterwards is dropped.

    Args:
        fn: Zero-argument callable to execute.
        timeout: Maximum wall-clock seconds to wait.

    Returns:
        The return value of *fn*.

    Raises:
        TimeoutError: If *fn* does not complete within *timeout* seconds.
        Exception: Any exception raised by *fn* is re-raised in the caller.
    """
    outcome: list[tuple[bool, Any]] = []
    finished = threading.Event()
    # A copy of the caller's context carries its request-outcome scope into the
    # worker thread, which would otherwise start with an empty one.
    context = contextvars.copy_context()

    def _run() -> None:
        try:
            outcome.append((True, context.run(fn)))
        except BaseException as exc:
            # BaseException so an exit signal raised in the worker still
            # reaches the caller instead of leaving it with no outcome.
            outcome.append((False, exc))
        finally:
            finished.set()

    # A daemon thread that is abandoned at the deadline: joining it (as a
    # ThreadPoolExecutor does on exit) would block the caller until the hung call
    # ends, and a non-daemon worker could keep the interpreter alive.
    threading.Thread(target=_run, name="yf-timeout", daemon=True).start()
    if not finished.wait(timeout):
        message = f"yfinance call did not complete within {timeout:.0f}s"
        record_failure(f"TimeoutError: {message}")
        raise TimeoutError(message)

    succeeded, value = outcome[0]
    if succeeded:
        return value
    raise value


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
    # How long a clean empty answer for a fundamentals category or statement
    # settles it. Shorter than financials_hours because the client cannot tell
    # "no data" from "retries exhausted": a fetch that merely failed is retried
    # within days, not after a week.
    no_data_hours: int = 72
    # news: always fetched (no threshold)
    # prices: always fetched (date-ranged)
    price_overlap_days: int = 5
    min_history_tolerance: float = 0.95


DEFAULT_THRESHOLDS = StalenessThresholds()

# One memory for every bulk sweep in the process: services are built per run (or
# per ticker in the parallel path), so a memory owned by the service would be
# forgotten before the next daily run could use it.
SHARED_EMPTY_FETCHES = EmptyFetchMemory()

# Sentinel: distinguishes "caller did not supply staleness" (query it) from
# "caller supplied None" (no data yet → treat as stale) on fetch_and_store.
_UNSET: Any = object()


def _prices_are_current(
    price_max_date: date | None,
    exchange_name: str | None,
    today: date,
) -> bool:
    """Return True if an incremental price fetch would add no new bar.

    The incremental fetch ends at *today* (exclusive), so the newest bar it can
    return is the last session before *today*. Prices need no fetch only when
    the stored series already reaches that session; fresh fundamentals say
    nothing about it.
    """
    if price_max_date is None:
        return False
    if isinstance(price_max_date, datetime):
        price_max_date = price_max_date.date()
    return price_max_date >= last_completed_session(exchange_name, today)


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


def _financial_currency(info: dict[str, Any] | None) -> str | None:
    """Return the major-unit currency Yahoo reports a company's statements in.

    Args:
        info: The ``yf.Ticker.info`` dict, or ``None`` when unavailable.

    Returns:
        ISO-4217 major code from ``financialCurrency`` (sub-units such as
        ``GBp`` collapsed), or ``None`` when Yahoo does not say. Never falls back
        to the listing currency: ADRs and cross-listings routinely report in a
        different currency, and a guessed label is worse than an absent one.
    """
    if not isinstance(info, dict):
        return None
    raw = info.get("financialCurrency")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return to_major_currency(raw.strip())


def _valuation_column_date(label: Any, as_of: date) -> date | None:
    """Resolve one ``Ticker.valuation`` column label to the date it describes."""
    if isinstance(label, datetime):
        return None if pd.isna(label) else label.date()
    if isinstance(label, date):
        return label
    if not isinstance(label, str):
        return None
    text = label.strip()
    if text.lower() == "current":
        return as_of
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _date_valuation_columns(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Relabel a ``Ticker.valuation`` frame so every column is a timestamp.

    yfinance labels the columns ``"Current"`` plus ``"M/D/YYYY"`` strings, which
    the date-keyed statement repository cannot parse and silently skips, so the
    labels must be resolved to dates first. ``"Current"`` is the latest trailing
    value and is stamped with *as_of*; columns that resolve to an already-seen
    date or to no date are dropped, since a duplicate would collide on the
    table's unique key.

    Args:
        df: Measures as rows, one column per period plus ``"Current"``.
        as_of: Date to stamp the ``"Current"`` column with (the fetch date).

    Returns:
        A frame with datetime column labels; empty when none resolved.
    """
    keep: list[int] = []
    resolved_dates: list[date] = []
    seen: set[date] = set()
    for position, label in enumerate(df.columns):
        resolved = _valuation_column_date(label, as_of)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        keep.append(position)
        resolved_dates.append(resolved)
    dated = df.iloc[:, keep].copy()
    dated.columns = pd.DatetimeIndex(resolved_dates)
    return dated


# Re-fetched closes inside this relative band of the stored ones share one
# adjustment basis: NUMERIC(20,6) rounding stays orders of magnitude below it,
# while a dividend of one basis point of price (the smallest that moves a
# return) is above it.
_BASIS_REL_TOL = 1e-4
_BASIS_ABS_TOL = 1e-6

# A re-based history may open a few sessions after the first stored bar
# (holiday, first-day gap) and still cover the whole stored series.
_REBASE_START_TOLERANCE = timedelta(days=7)

# Yahoo resolves a period such as "5y" on its own clock, a day or two off ours: a
# stored bar this close to the period's edge could fall just outside the fetch.
_PERIOD_EDGE_MARGIN = timedelta(days=7)


def _today() -> date:
    """Return the current date; the one clock read the price-fetch logic uses."""
    return date.today()


def _years_before(day: date, years: int) -> date:
    """Return *day* moved back by whole *years*, clamping 29 Feb to 28 Feb."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _bar_date(stamp: Any) -> date | None:
    """Return the calendar date of a history index value, or None if missing."""
    try:
        ts = pd.Timestamp(stamp)
    except (TypeError, ValueError):
        return None
    return ts.date() if isinstance(ts, pd.Timestamp) else None


def _adjustment_basis_changed(
    stored_bars: Iterable[Any], fresh: pd.DataFrame | None
) -> bool:
    """Report whether re-fetched closes disagree with the stored closes.

    yfinance returns closes adjusted for splits and dividends as of the fetch
    date, so a split or ex-dividend date after the last fetch rescales every
    earlier bar. Stored bars on the old basis and re-fetched bars on the new one
    therefore differ by a constant factor on any date they share.

    Args:
        stored_bars: Stored price rows exposing ``date`` and ``close``.
        fresh: Re-fetched history with a ``Close`` column and a date index.

    Returns:
        True when any date present on both sides differs beyond rounding noise.
    """
    if fresh is None or fresh.empty or "Close" not in fresh:
        return False
    fresh_close: dict[date, float] = {}
    for stamp, close in fresh["Close"].items():
        day = _bar_date(stamp)
        if day is not None and pd.notna(close):
            fresh_close[day] = float(close)
    for bar in stored_bars:
        new = fresh_close.get(bar.date)
        if bar.close is None or new is None:
            continue
        if not math.isclose(
            float(bar.close), new, rel_tol=_BASIS_REL_TOL, abs_tol=_BASIS_ABS_TOL
        ):
            return True
    return False


class YFinanceDataService:
    """Fetches all yfinance data categories for a ticker and stores via repository."""

    def __init__(
        self,
        repo: YFinanceRepository,
        yf_client: YFinanceClient,
        request_timeout: float | None = None,
        empty_fetches: EmptyFetchMemory | None = None,
    ) -> None:
        self.repo = repo
        self.yf_client = yf_client
        # Private by default so unrelated callers (and tests) never see each
        # other's empty answers; bulk sweeps inject SHARED_EMPTY_FETCHES.
        self._empty_fetches = (
            empty_fetches if empty_fetches is not None else EmptyFetchMemory()
        )
        # Wall-clock timeout for every yfinance network call.  None disables
        # the watchdog (used in tests that mock fetch methods synchronously).
        self._request_timeout = request_timeout

    def _timed(self, fn: "Any") -> "Any":
        """Execute *fn*, enforcing ``self._request_timeout`` if set."""
        if self._request_timeout is None:
            return fn()
        return call_with_timeout(fn, self._request_timeout)

    def _write(self, write: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run one category's repository write inside its own SAVEPOINT.

        PostgreSQL aborts the whole transaction after any failed statement, and
        COMMIT of an aborted transaction silently becomes ROLLBACK. With one
        shared session, a single bad category would therefore fail every later
        category and discard the ticker's earlier writes, while ``counts`` still
        reported them. The savepoint rolls back only the category that raised;
        the exception still propagates to that category's own handler.

        Args:
            write: Repository upsert to run.
            *args: Positional arguments for *write*.
            **kwargs: Keyword arguments for *write*.

        Returns:
            Whatever *write* returns (the number of rows it stored).
        """
        with self.repo.session.begin_nested():
            return write(*args, **kwargs)

    def _overlap_was_readjusted(
        self,
        instrument_id: UUID,
        fresh: pd.DataFrame | None,
        start_date: date,
        max_date: date,
    ) -> bool:
        """Compare the stored overlap bars with the freshly fetched window.

        The newest stored bar is left out: an exchange still open at fetch time
        stores an intraday snapshot there, and its final close differing is not
        a re-adjustment. Every older bar is final, and any split or dividend
        since the last fetch rescales all of them.
        """
        stored = self.repo.get_price_history(
            instrument_id,
            start_date=start_date,
            end_date=max_date - timedelta(days=1),
        )
        return _adjustment_basis_changed(stored, fresh)

    def _covers_stored_series(self, instrument_id: UUID, history: pd.DataFrame) -> bool:
        """Return True when *history* opens no later than the first stored bar.

        Such a history rewrites every stored bar on one basis, so it is safe to
        store even when it is shorter than the exchange calendar predicts (a
        young listing, or a thin name Yahoo has no-trade gaps for). A history
        that opens later would re-base only its own span and leave the older bars
        on the previous basis. Every stored bar is within a full fetch's reach:
        it starts at the oldest stored bar whenever the period would stop short.
        """
        first = next(
            iter(self.repo.get_price_history(instrument_id, limit=1, ascending=True)),
            None,
        )
        if first is None:
            return False
        opens = _bar_date(history.index.min())
        return opens is not None and opens <= first.date + _REBASE_START_TOLERANCE

    def _full_fetch_start(self, instrument_id: UUID, period: str) -> date | None:
        """Return where a full-history fetch must start to span the stored series.

        Nothing prunes ``price_history``, so a series outgrows a rolling period
        within a week of its first load. A fetch that reaches back only ``period``
        would re-base the newest bars and leave every older one on the previous
        adjustment basis, a seam that ``prices_to_returns`` turns into a fake
        return. Periods that are not whole years ("max", "6mo") are left to
        yfinance.

        Args:
            instrument_id: Instrument whose stored bars set the start.
            period: yfinance period string the caller asked for.

        Returns:
            The oldest stored bar's date when *period* would stop short of it,
            else None to fetch by *period* (which also backfills unstored bars).
        """
        years = parse_period_years(period)
        if years is None:
            return None
        try:
            oldest = next(
                iter(
                    self.repo.get_price_history(instrument_id, limit=1, ascending=True)
                ),
                None,
            )
        except Exception as e:
            # The widening only improves the basis of old bars; a failed lookup must
            # not cost the ticker its price refresh.
            logger.warning(
                "Oldest stored price lookup failed for %s, refetching %s only: %s",
                instrument_id,
                period,
                e,
            )
            return None
        if oldest is None:
            return None
        if oldest.date > _years_before(_today(), years) + _PERIOD_EDGE_MARGIN:
            return None
        return oldest.date

    def _fetch_full_price_history(
        self,
        instrument_id: UUID,
        yfinance_ticker: str,
        period: str,
        *,
        may_have_stored_bars: bool = True,
    ) -> pd.DataFrame | None:
        # A first load has nothing stored to reach back to, so skip the lookup.
        start = (
            self._full_fetch_start(instrument_id, period)
            if may_have_stored_bars
            else None
        )
        window: dict[str, Any] = (
            {"period": period} if start is None else {"start": start.isoformat()}
        )
        # fetch_history accepts a native `timeout` kwarg (passed to requests),
        # so we pass it directly without an extra watchdog thread. Closes are
        # adjusted as of the fetch date (yfinance default), which is why every
        # stored bar has to come from one fetch to share a basis.
        return self.yf_client.fetch_history(
            yfinance_ticker,
            **window,
            min_rows=1,
            # actions=True adds Dividends / Stock Splits columns the
            # price-history upsert persists (Capital Gains no longer
            # emitted by Yahoo → stays NULL).
            actions=True,
            timeout=self._request_timeout,
        )

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
        staleness: Any = _UNSET,
    ) -> dict[str, Any]:
        """Fetch all data categories for a single ticker and store.

        Args:
            instrument_id: Database instrument UUID.
            yfinance_ticker: Yahoo Finance ticker symbol.
            period: Price history period for full mode (e.g. "5y"). A stored
                series older than a whole-year period is fetched from its oldest
                bar instead, so one fetch re-bases all of it.
            mode: "full" re-downloads every category and the whole stored price
                  series; "incremental" skips fresh categories and date-ranges
                  the price fetch.
            thresholds: Override default staleness thresholds.
            exchange_name: Exchange name for calendar-based history validation.
            currency_code: Instrument listing currency (e.g. "GBX"). Stored raw
                as the price unit; its major unit (e.g. "GBP") labels the
                market-cap based valuation rows. Financial statements are
                labelled with Yahoo's ``financialCurrency`` instead.
            asset_class: Instrument asset class. When "fixed_income" or
                "multi_asset" (i.e. an ETF), the ETF fund-metadata branch runs.

        Returns:
            Dict with ``counts`` per category, ``errors``, ``skipped``
            categories, ``transport_failures`` (requests Yahoo never answered,
            each also summarised as one ``transport:`` error) and
            ``requests_answered`` (requests Yahoo answered, with or without
            data), so an unreachable Yahoo cannot pass for an empty answer.
        """
        with track_request_outcomes() as outcomes:
            result = self._fetch_and_store(
                instrument_id,
                yfinance_ticker,
                period=period,
                mode=mode,
                thresholds=thresholds,
                exchange_name=exchange_name,
                currency_code=currency_code,
                asset_class=asset_class,
                staleness=staleness,
            )
        result["transport_failures"] = len(outcomes.failures)
        result["requests_answered"] = outcomes.answered
        if outcomes.failures:
            result["errors"].append(
                f"transport: {len(outcomes.failures)} request(s) got no answer "
                f"from Yahoo (first: {outcomes.failures[0]})"
            )
        return result

    def _fetch_and_store(
        self,
        instrument_id: UUID,
        yfinance_ticker: str,
        period: str,
        mode: str,
        thresholds: StalenessThresholds | None,
        exchange_name: str | None,
        currency_code: str | None,
        asset_class: str | None,
        staleness: Any,
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        errors: list[str] = []
        skipped: list[str] = []

        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS

        # In incremental mode, query staleness info once upfront — unless the
        # caller (a bulk sweep) pre-loaded it in one grouped query and injected
        # it here, avoiding a per-instrument round-trip.
        if mode != "incremental":
            staleness = None
        elif staleness is _UNSET:
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

        # A fundamentals gate is settled when its marker is inside the
        # fundamentals window, or when the last fetch came back empty recently:
        # an empty answer writes no row, so without the second clause a category
        # Yahoo has nothing for would be requested on every run.
        def _settled(marker_key: str) -> bool:
            return (
                mode == "incremental"
                and staleness is not None
                and (
                    _is_fresh(
                        staleness.get(marker_key), thresholds.financials_hours, now
                    )
                    or self._empty_fetches.is_recent(
                        instrument_id, marker_key, thresholds.no_data_hours, now
                    )
                )
            )

        # Gated per category on the marker of the category's own table: a
        # category whose fetch failed while the statements were stored must not
        # be hidden by them for the whole window.
        def _fundamentals_fresh(category: str) -> bool:
            return _settled(f"{category}_updated_at")

        # Gated per statement: one statement stored during a rate-limit burst
        # must not hide the ones that failed until the window expires.
        def _statement_fresh(stmt_type: str, period_type: str) -> bool:
            return _settled(statement_staleness_key(stmt_type, period_type))

        def _record_fetch(marker_key: str, stored: int) -> None:
            if stored:
                return
            # While the breaker is open an empty result is more likely a lost
            # request than Yahoo's answer; leaving it unrecorded keeps the next
            # run retrying instead of hiding the category.
            if self.yf_client.circuit_breaker.is_active:
                return
            self._empty_fetches.remember(instrument_id, marker_key, now)

        # ------------------------------------------------------------------
        # Incremental short-circuit: skip whole ticker if ALL categories fresh
        # ------------------------------------------------------------------
        # In incremental mode we know staleness upfront.  If every category
        # passes its freshness check and the stored prices already reach the
        # last completed session there is nothing to fetch — avoid
        # constructing a yf.Ticker at all (which acquires the rate-limiter
        # even on a cache hit, adding ~100ms×N delay over 2907 tickers).
        if mode == "incremental" and staleness is not None:
            all_fresh = (
                _is_fresh(
                    staleness.get("profile_updated_at"), thresholds.profile_hours, now
                )
                and all(
                    _statement_fresh(st, pt) for st, pt in FINANCIAL_STATEMENT_GROUPS
                )
                and all(
                    _fundamentals_fresh(category) for category in FUNDAMENTAL_CATEGORIES
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
                and _prices_are_current(
                    staleness.get("price_max_date"), exchange_name, now.date()
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
        # The ticker object is needed only by the news fetch; all other
        # categories call through sub-clients that invoke _get_ticker
        # internally and share the same cache.
        _ticker_holder: list[Any] = []  # mutable cell for lazy init
        profile_info: dict[str, Any] | None = None

        def _get_lazy_ticker() -> Any:
            if not _ticker_holder:
                _ticker_holder.append(self.yf_client.get_ticker(yfinance_ticker))
            return _ticker_holder[0]

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
                    profile_info = info
                    # yfinance's .info does not carry ISIN — fetch it separately
                    # (Ticker.isin) and inject so ticker_profiles.isin populates.
                    # The lookup is a separate third-party call: its failure must
                    # not cost the profile that was just fetched.
                    isin = None
                    try:
                        isin = self._timed(
                            lambda: self.yf_client.metadata.fetch_isin(yfinance_ticker)
                        )
                    except Exception as e:
                        errors.append(f"isin: {e}")
                        logger.warning(
                            "Failed to fetch ISIN for %s: %s", yfinance_ticker, e
                        )
                    if isin:
                        info["isin"] = isin
                    counts["profile"] = self._write(
                        self.repo.upsert_profile, instrument_id, info
                    )
                    # Extra info fields (short interest, momentum, governance risk)
                    # reuse the already-fetched dict — no separate network call.
                    counts["profile_extras"] = self._write(
                        self.repo.upsert_profile_extras, instrument_id, info
                    )
                else:
                    counts["profile"] = 0
            except Exception as e:
                errors.append(f"profile: {e}")
                logger.warning("Failed to fetch profile for %s: %s", yfinance_ticker, e)

        try:
            incremental_window = (
                mode == "incremental"
                and staleness is not None
                and staleness.get("price_max_date") is not None
            )
            # Set only when a re-base fetch replaced the incremental window, so
            # the window can still be stored if the full history is unusable.
            window_history: pd.DataFrame | None = None
            if incremental_window:
                max_date = staleness["price_max_date"]
                start_date = max_date - timedelta(days=thresholds.price_overlap_days)
                history = self.yf_client.fetch_history(
                    yfinance_ticker,
                    start=start_date.isoformat(),
                    end=date.today().isoformat(),
                    min_rows=0,
                    # actions=True adds Dividends / Stock Splits columns the
                    # price-history upsert persists (Capital Gains no longer
                    # emitted by Yahoo → stays NULL).
                    actions=True,
                    timeout=self._request_timeout,
                )
                # A split or dividend since the last fetch rescales all earlier
                # bars, but the window rewrites only its own: re-fetch the whole
                # period so the stored series stays on a single basis.
                if self._overlap_was_readjusted(
                    instrument_id, history, start_date, max_date
                ):
                    try:
                        rebased = self._fetch_full_price_history(
                            instrument_id, yfinance_ticker, period
                        )
                    except Exception as e:
                        logger.warning(
                            "Re-base fetch failed for %s, storing the window only: %s",
                            yfinance_ticker,
                            e,
                        )
                        rebased = None
                    if rebased is not None and not rebased.empty:
                        window_history, history = history, rebased
            else:
                history = self._fetch_full_price_history(
                    instrument_id,
                    yfinance_ticker,
                    period,
                    may_have_stored_bars=not (
                        isinstance(staleness, dict)
                        and staleness.get("price_max_date") is None
                    ),
                )

            if history is not None and not history.empty:
                is_full_period = not incremental_window or window_history is not None
                if is_full_period:
                    sufficient, expected, minimum = has_sufficient_history(
                        len(history),
                        exchange_name,
                        period,
                        thresholds.min_history_tolerance,
                    )
                    if not sufficient:
                        # The insufficient-history guard exists to stop a
                        # truncated/glitchy re-fetch from re-basing only part of
                        # an established price series. For an instrument with
                        # NO stored prices yet, a short history is simply a
                        # young listing's complete, legitimate history (recent
                        # IPOs/SPACs). Discarding it permanently excluded ~102
                        # instruments from price_history with no error logged.
                        stored = (
                            staleness
                            if staleness is not None
                            else self.repo.get_staleness_info(instrument_id)
                        )
                        has_existing_prices = stored.get("price_max_date") is not None
                        price_min_date = stored.get("price_min_date")
                        if has_existing_prices and price_min_date is not None:
                            # A young listing with stored prices has no
                            # sessions before its earliest stored bar, so judge
                            # the refetch over that lifetime. The anchor must
                            # come from storage, not the fetched frame: a
                            # head-truncated refetch is complete when measured
                            # from its own first bar.
                            sufficient, expected, minimum = has_sufficient_history(
                                len(history),
                                exchange_name,
                                period,
                                thresholds.min_history_tolerance,
                                since=price_min_date,
                            )
                        # A history that reaches back to the first stored bar
                        # rewrites every stored price on one adjustment basis;
                        # refusing it would pin a young listing to a stale basis
                        # for good. One that opens later would re-base only its
                        # own span, so it is still refused.
                        if not sufficient and not (
                            has_existing_prices
                            and self._covers_stored_series(instrument_id, history)
                        ):
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
                                history = window_history
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
                        elif not sufficient:
                            logger.info(
                                "Storing short history for %s: %d rows "
                                "(below expected ~%d for %s) — young listing, "
                                "history spans every stored price",
                                yfinance_ticker,
                                len(history),
                                expected,
                                period,
                            )

            if history is not None and not history.empty:
                # Store the listing currency as the price unit (raw, unconverted) so
                # sub-unit series (GBX pence) are unambiguous downstream.
                counts["prices"] = self._write(
                    self.repo.upsert_price_history,
                    instrument_id,
                    history,
                    price_unit=currency_code,
                )
            else:
                counts["prices"] = counts.get("prices", 0)
                if incremental_window:
                    # yfinance swallows network faults inside history() and
                    # returns an empty frame. The overlap window always spans
                    # stored bars, so an empty answer for a ticker that has a
                    # series is the only trace of a Yahoo that did not answer.
                    skipped.append("prices:unavailable")
        except Exception as e:
            errors.append(f"prices: {e}")
            logger.warning("Failed to fetch prices for %s: %s", yfinance_ticker, e)

        # Statements are denominated in the company's reporting currency, which
        # Yahoo exposes as info["financialCurrency"] and which differs from the
        # listing currency for ADRs and cross-listings (TSM lists in USD but
        # reports in TWD; SHEL.L lists in GBp but reports in USD).
        listing_major_currency = to_major_currency(currency_code)
        statement_currency_cell: list[str | None] = []

        def _statement_currency() -> str | None:
            # Resolved on first use so a ticker with nothing to store never pays
            # for an info request; the profile step usually already supplied it.
            if not statement_currency_cell:
                info_for_currency = profile_info
                if not info_for_currency:
                    try:
                        info_for_currency = self._timed(
                            lambda: self.yf_client.fetch_info(yfinance_ticker)
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not resolve financial currency for %s: %s",
                            yfinance_ticker,
                            e,
                        )
                statement_currency_cell.append(_financial_currency(info_for_currency))
            return statement_currency_cell[0]

        if all(_statement_fresh(st, pt) for st, pt in FINANCIAL_STATEMENT_GROUPS):
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
                    if _statement_fresh(stmt_type, period_type):
                        continue
                    _fm, _qf = fetch_method, quarterly_flag
                    try:
                        df = self._timed(
                            lambda m=_fm, q=_qf: m(yfinance_ticker, quarterly=q)
                        )
                        group_rows = 0
                        if df is not None and not df.empty:
                            group_rows = self._write(
                                self.repo.upsert_financial_statements,
                                instrument_id,
                                df,
                                stmt_type,
                                period_type,
                                currency_code=_statement_currency(),
                            )
                            fs_count += group_rows
                        _record_fetch(
                            statement_staleness_key(stmt_type, period_type), group_rows
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
        # Market cap and EV are quote-based, so the rows carry the listing
        # currency's major.
        if _fundamentals_fresh("valuation_measures"):
            skipped.append("valuation_measures")
        else:
            try:
                val_df = self._timed(
                    lambda: self.yf_client.metadata.fetch_valuation_measures(
                        yfinance_ticker
                    )
                )
                if val_df is not None and not val_df.empty:
                    dated_val_df = _date_valuation_columns(val_df, now.date())
                    if dated_val_df.empty:
                        logger.warning(
                            "valuation_measures for %s: none of the columns %s "
                            "resolved to a date; nothing stored",
                            yfinance_ticker,
                            list(val_df.columns)[:5],
                        )
                        counts["valuation_measures"] = 0
                    else:
                        counts["valuation_measures"] = self._write(
                            self.repo.upsert_financial_statements,
                            instrument_id,
                            dated_val_df,
                            VALUATION_MEASURES_STATEMENT_TYPE,
                            VALUATION_MEASURES_PERIOD_TYPE,
                            currency_code=listing_major_currency,
                        )
                else:
                    counts["valuation_measures"] = 0
            except Exception as e:
                errors.append(f"valuation_measures: {e}")
                logger.warning(
                    "Failed valuation_measures for %s: %s", yfinance_ticker, e
                )

        # 3b. EPS trend — the consensus EPS estimate for each forward period as it
        # stood now vs 7/30/60/90 days ago. The panel is indexed by period label
        # ("0q", "+1q", "0y", "+1y") with snapshot-age columns; neither axis is a
        # date, so it lands in the typed eps_trend table (not financial_statements).
        if _fundamentals_fresh("eps_trend"):
            skipped.append("eps_trend")
        else:
            try:
                eps_trend_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_eps_trend(yfinance_ticker)
                )
                counts["eps_trend"] = (
                    self._write(self.repo.upsert_eps_trend, instrument_id, eps_trend_df)
                    if eps_trend_df is not None and not eps_trend_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"eps_trend: {e}")
                logger.warning("Failed eps_trend for %s: %s", yfinance_ticker, e)

        # 3c. EPS revisions — up/down revision counts over the trailing 7/30-day
        # windows, per forward-period label. Same period-label index as eps_trend;
        # stored in the typed eps_revisions table.
        if _fundamentals_fresh("eps_revisions"):
            skipped.append("eps_revisions")
        else:
            try:
                eps_rev_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_eps_revisions(yfinance_ticker)
                )
                counts["eps_revisions"] = (
                    self._write(
                        self.repo.upsert_eps_revisions, instrument_id, eps_rev_df
                    )
                    if eps_rev_df is not None and not eps_rev_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"eps_revisions: {e}")
                logger.warning("Failed eps_revisions for %s: %s", yfinance_ticker, e)

        # 3d/3e/3f. Analyst estimates (earnings / revenue / growth) — dedicated
        # typed tables, each gated on its own fundamentals-window marker.
        if _fundamentals_fresh("earnings_estimate"):
            skipped.append("earnings_estimate")
        else:
            try:
                ee_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_earnings_estimate(
                        yfinance_ticker
                    )
                )
                counts["earnings_estimate"] = (
                    self._write(
                        self.repo.upsert_earnings_estimate, instrument_id, ee_df
                    )
                    if ee_df is not None and not ee_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"earnings_estimate: {e}")
                logger.warning(
                    "Failed earnings_estimate for %s: %s", yfinance_ticker, e
                )

        if _fundamentals_fresh("revenue_estimate"):
            skipped.append("revenue_estimate")
        else:
            try:
                re_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_revenue_estimate(
                        yfinance_ticker
                    )
                )
                counts["revenue_estimate"] = (
                    self._write(self.repo.upsert_revenue_estimate, instrument_id, re_df)
                    if re_df is not None and not re_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"revenue_estimate: {e}")
                logger.warning("Failed revenue_estimate for %s: %s", yfinance_ticker, e)

        if _fundamentals_fresh("growth_estimates"):
            skipped.append("growth_estimates")
        else:
            try:
                ge_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_growth_estimates(
                        yfinance_ticker
                    )
                )
                counts["growth_estimates"] = (
                    self._write(self.repo.upsert_growth_estimates, instrument_id, ge_df)
                    if ge_df is not None and not ge_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"growth_estimates: {e}")
                logger.warning("Failed growth_estimates for %s: %s", yfinance_ticker, e)

        # 3g. Earnings history (past-quarter EPS surprise).
        if _fundamentals_fresh("earnings_history"):
            skipped.append("earnings_history")
        else:
            try:
                eh_df = self._timed(
                    lambda: self.yf_client.analysis.fetch_earnings_history(
                        yfinance_ticker
                    )
                )
                counts["earnings_history"] = (
                    self._write(self.repo.upsert_earnings_history, instrument_id, eh_df)
                    if eh_df is not None and not eh_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"earnings_history: {e}")
                logger.warning("Failed earnings_history for %s: %s", yfinance_ticker, e)

        # 3h. Earnings dates (past + upcoming).
        if _fundamentals_fresh("earnings_dates"):
            skipped.append("earnings_dates")
        else:
            try:
                ed_df = self._timed(
                    lambda: self.yf_client.metadata.fetch_earnings_dates(
                        yfinance_ticker
                    )
                )
                counts["earnings_dates"] = (
                    self._write(self.repo.upsert_earnings_dates, instrument_id, ed_df)
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
                    self._write(self.repo.upsert_analyst_actions, instrument_id, ad_df)
                    if ad_df is not None and not ad_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"analyst_actions: {e}")
                logger.warning("Failed analyst_actions for %s: %s", yfinance_ticker, e)

        # 3k. SEC filings (list of dicts; fundamentals window).
        if _fundamentals_fresh("sec_filings"):
            skipped.append("sec_filings")
        else:
            try:
                filings = self._timed(
                    lambda: self.yf_client.financials.fetch_sec_filings(yfinance_ticker)
                )
                counts["sec_filings"] = (
                    self._write(self.repo.upsert_sec_filings, instrument_id, filings)
                    if filings
                    else 0
                )
            except Exception as e:
                errors.append(f"sec_filings: {e}")
                logger.warning("Failed sec_filings for %s: %s", yfinance_ticker, e)

        # 3l. Shares outstanding (fundamentals window; point-in-time float).
        if _fundamentals_fresh("shares_outstanding"):
            skipped.append("shares_outstanding")
        else:
            try:
                shares = self._timed(
                    lambda: self.yf_client.corporate_actions.fetch_shares_full(
                        yfinance_ticker
                    )
                )
                counts["shares_outstanding"] = (
                    self._write(
                        self.repo.upsert_shares_outstanding, instrument_id, shares
                    )
                    if shares is not None and not getattr(shares, "empty", True)
                    else 0
                )
            except Exception as e:
                errors.append(f"shares_outstanding: {e}")
                logger.warning(
                    "Failed shares_outstanding for %s: %s", yfinance_ticker, e
                )

        # A category present in counts was fetched without raising; its count says
        # whether Yahoo had anything.
        for category in FUNDAMENTAL_CATEGORIES:
            if category in counts:
                _record_fetch(f"{category}_updated_at", counts[category])

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
                    counts["dividends"] = self._write(
                        self.repo.upsert_dividends, instrument_id, dividends
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
                    counts["splits"] = self._write(
                        self.repo.upsert_splits, instrument_id, splits
                    )
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
                    counts["recommendations"] = self._write(
                        self.repo.upsert_recommendations, instrument_id, rec_df
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
                    counts["price_targets"] = self._write(
                        self.repo.upsert_price_targets, instrument_id, targets
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
                    counts["institutional_holders"] = self._write(
                        self.repo.upsert_institutional_holders, instrument_id, inst_df
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
                    counts["mutualfund_holders"] = self._write(
                        self.repo.upsert_mutualfund_holders, instrument_id, mf_df
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
                    counts["insider_transactions"] = self._write(
                        self.repo.upsert_insider_transactions,
                        instrument_id,
                        insiders_df,
                    )
                else:
                    counts["insider_transactions"] = 0
            except Exception as e:
                errors.append(f"insider_transactions: {e}")
                logger.warning(
                    "Failed insider transactions for %s: %s", yfinance_ticker, e
                )

        # 10b. Major holders (ownership breakdown; institutional window).
        if (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("institutional_holders_updated_at"),
                thresholds.institutional_holders_hours,
                now,
            )
        ):
            skipped.append("major_holders")
        else:
            try:
                mh_df = self._timed(
                    lambda: self.yf_client.holders.fetch_major_holders(yfinance_ticker)
                )
                counts["major_holders"] = (
                    self._write(self.repo.upsert_major_holders, instrument_id, mh_df)
                    if mh_df is not None and not mh_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"major_holders: {e}")
                logger.warning("Failed major_holders for %s: %s", yfinance_ticker, e)

        # 10c/10d. Insider purchases + roster (insider window).
        insider_fresh = (
            mode == "incremental"
            and staleness is not None
            and _is_fresh(
                staleness.get("insider_transactions_updated_at"),
                thresholds.insider_transactions_hours,
                now,
            )
        )
        if insider_fresh:
            skipped.append("insider_purchases")
        else:
            try:
                ip_df = self._timed(
                    lambda: self.yf_client.holders.fetch_insider_purchases(
                        yfinance_ticker
                    )
                )
                counts["insider_purchases"] = (
                    self._write(
                        self.repo.upsert_insider_purchases, instrument_id, ip_df
                    )
                    if ip_df is not None and not ip_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"insider_purchases: {e}")
                logger.warning(
                    "Failed insider_purchases for %s: %s", yfinance_ticker, e
                )

        if insider_fresh:
            skipped.append("insider_roster")
        else:
            try:
                ir_df = self._timed(
                    lambda: self.yf_client.holders.fetch_insider_roster_holders(
                        yfinance_ticker
                    )
                )
                counts["insider_roster"] = (
                    self._write(self.repo.upsert_insider_roster, instrument_id, ir_df)
                    if ir_df is not None and not ir_df.empty
                    else 0
                )
            except Exception as e:
                errors.append(f"insider_roster: {e}")
                logger.warning("Failed insider_roster for %s: %s", yfinance_ticker, e)

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
                counts["news"] = self._write(
                    self.repo.upsert_news, instrument_id, news_list
                )
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
            fdata = self._timed(
                lambda: self.yf_client.funds.fetch_funds_data(yfinance_ticker)
            )
            overview = (fdata or {}).get("fund_overview") or {}
            description = (fdata or {}).get("description")
            profile = profile or {}
            if profile or overview or description:
                self._write(
                    etf_repo.upsert_metadata,
                    instrument_id,
                    aum=profile.get("aum"),
                    nav=profile.get("nav"),
                    fund_family=profile.get("fund_family"),
                    legal_type=profile.get("legal_type"),
                    expense_ratio=profile.get("expense_ratio"),
                    # Yahoo reports the listing currency, a sub-unit ('GBp') for
                    # LSE listings; a sub-unit is never a fund's base currency.
                    base_currency=to_major_currency(profile.get("base_currency")),
                    category=overview.get("categoryName") or overview.get("category"),
                    description=description,
                    as_of=as_of,
                )
                counts["etf_metadata"] = 1

            if fdata:
                ac = fdata.get("asset_classes") or {}
                buckets = {
                    "stock_pct": ac.get("stockPosition"),
                    "bond_pct": ac.get("bondPosition"),
                    "cash_pct": ac.get("cashPosition"),
                    "other_pct": ac.get("otherPosition"),
                }
                # An all-NULL row would become the newest as_of snapshot and
                # shadow the last real allocation.
                if any(v is not None for v in buckets.values()):
                    self._write(
                        etf_repo.upsert_asset_classes, instrument_id, as_of, **buckets
                    )
                    counts["etf_asset_classes"] = 1
                holdings = fdata.get("top_holdings") or []
                if holdings:
                    counts["etf_holdings"] = self._write(
                        etf_repo.upsert_holdings, instrument_id, as_of, holdings
                    )
                sectors = fdata.get("sector_weightings") or {}
                if sectors:
                    counts["etf_sector_weights"] = self._write(
                        etf_repo.upsert_sector_weights, instrument_id, as_of, sectors
                    )
                # Depth (SPEC A8): equity/bond metrics, ratings, operations.
                equity = fdata.get("equity_holdings") or {}
                if equity:
                    counts["etf_equity_holdings"] = self._write(
                        etf_repo.upsert_equity_holdings, instrument_id, as_of, equity
                    )
                bonds = fdata.get("bond_holdings") or {}
                if bonds:
                    counts["etf_bond_holdings"] = self._write(
                        etf_repo.upsert_bond_holdings, instrument_id, as_of, bonds
                    )
                ratings = fdata.get("bond_ratings") or {}
                if ratings:
                    counts["etf_bond_ratings"] = self._write(
                        etf_repo.upsert_bond_ratings, instrument_id, as_of, ratings
                    )
                operations = fdata.get("fund_operations") or {}
                if operations:
                    counts["etf_fund_operations"] = self._write(
                        etf_repo.upsert_fund_operations,
                        instrument_id,
                        as_of,
                        operations,
                    )
        except Exception as e:
            errors.append(f"etf_metadata: {e}")
            logger.warning("Failed ETF metadata for %s: %s", yfinance_ticker, e)


# ---------------------------------------------------------------------------
# Standalone bulk fetch (callable from routes and scheduler)
# ---------------------------------------------------------------------------


def _backfill_instrument_isin_from_profiles() -> None:
    """Propagate ISIN from ``ticker_profiles`` onto ``instruments`` after a sweep.

    The Screener universe build stores no ISIN; the per-instrument profile fetch
    does (``ticker_profiles.isin``). Copy it onto the canonical ``instruments.isin``
    for rows still missing one. Idempotent and cheap (a single UPDATE); best-effort
    so an enrichment failure never sinks the fetch result.
    """
    from portopt_db.repositories.universe.universe_repository import UniverseRepository

    from app.database import database_manager

    try:
        with database_manager.get_session() as session:
            filled = UniverseRepository(session).backfill_isin_from_profiles()
            session.commit()
        if filled:
            logger.info("Backfilled ISIN onto %d instruments from profiles", filled)
    except Exception:
        logger.exception("ISIN backfill from ticker_profiles failed (non-fatal)")


# Share of a sweep's yfinance requests that may go unanswered before the sweep
# counts as an outage. Below it the affected categories simply refill on the
# next run; at or above it the run says nothing about prices being current, so
# it must not report success or let the pipeline carry on as if they were.
_OUTAGE_FAILURE_RATIO = 0.5


class YahooUnreachableError(RuntimeError):
    """A yfinance sweep got no answer from Yahoo for most of its requests."""


@dataclass
class _SweepRequests:
    """Yahoo's responsiveness across a sweep, summed from each ticker's result."""

    failed: int = 0
    answered: int = 0

    def add(self, result: dict[str, Any]) -> None:
        self.failed += result.get("transport_failures", 0)
        self.answered += result.get("requests_answered", 0)

    @property
    def total(self) -> int:
        return self.failed + self.answered

    @property
    def is_outage(self) -> bool:
        return self.failed > 0 and self.failed >= _OUTAGE_FAILURE_RATIO * self.total


def _conclude_sweep(
    result_dict: dict[str, Any],
    errors: list[str],
    requests: _SweepRequests,
    on_progress: ProgressCallback,
) -> None:
    """Report the sweep as completed, or raise when Yahoo was unreachable.

    The outage path leaves the job's terminal status to the caller that catches
    the exception: writing "failed" here as well would count the transition
    (metrics, webhook) twice.

    Args:
        result_dict: Sweep summary; gains the request tallies.
        errors: Per-ticker error lines to store on the job.
        requests: Request outcomes summed over the sweep.
        on_progress: Job progress callback.

    Raises:
        YahooUnreachableError: When the share of unanswered requests reaches
            ``_OUTAGE_FAILURE_RATIO``.
    """
    result_dict["transport_failures"] = requests.failed
    result_dict["requests_answered"] = requests.answered
    if requests.is_outage:
        on_progress(errors=errors, result=result_dict)
        raise YahooUnreachableError(
            f"{requests.failed} of {requests.total} yfinance requests got no "
            "answer from Yahoo"
        )
    on_progress(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        errors=errors,
        result=result_dict,
    )


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
        ``skipped_category_count``, ``transport_failures`` and
        ``requests_answered``.

    Raises:
        YahooUnreachableError: When most requests got no answer from Yahoo. The
            job row keeps the result and errors; the ISIN backfill is skipped
            and the next sweep fills it.
    """
    if request.workers > 1:
        result_dict = _run_bulk_yfinance_fetch_parallel(
            request, yf_client, on_progress=on_progress, workers=request.workers
        )
        _backfill_instrument_isin_from_profiles()
        return result_dict

    from app.database import database_manager

    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        instruments = repo.get_instruments_with_yfinance_ticker()

        total = len(instruments)
        on_progress(total=total)

        all_errors: list[str] = []
        total_counts: dict[str, int] = {}
        total_skipped: int = 0
        sweep_requests = _SweepRequests()

        from app.config import settings as _settings

        service = YFinanceDataService(
            repo,
            yf_client,
            request_timeout=float(_settings.yfinance_request_timeout_seconds),
            empty_fetches=SHARED_EMPTY_FETCHES,
        )

        # Incremental: one grouped staleness query for the whole sweep instead
        # of 11 MAX() round-trips per instrument inside fetch_and_store.
        staleness_by_id: dict[UUID, dict[str, Any]] = {}
        if request.mode == "incremental":
            staleness_by_id = repo.get_staleness_info_bulk([i.id for i in instruments])

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
                    staleness=(
                        staleness_by_id.get(instrument.id)
                        if request.mode == "incremental"
                        else _UNSET
                    ),
                )

                for k, v in result["counts"].items():
                    total_counts[k] = total_counts.get(k, 0) + v
                total_skipped += len(result.get("skipped", []))
                sweep_requests.add(result)
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
        _conclude_sweep(result_dict, all_errors, sweep_requests, on_progress)
        logger.info("Bulk yfinance fetch completed: %d tickers", total)

    _backfill_instrument_isin_from_profiles()
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
    staleness: Any = _UNSET,
) -> dict[str, Any]:
    """Fetch a single ticker inside an isolated per-thread session."""
    if not spec.yfinance_ticker:
        raise ValueError("missing yfinance_ticker")
    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        service = YFinanceDataService(
            repo,
            yf_client,
            request_timeout=request_timeout,
            empty_fetches=SHARED_EMPTY_FETCHES,
        )
        try:
            result = service.fetch_and_store(
                instrument_id=spec.instrument_id,
                yfinance_ticker=spec.yfinance_ticker,
                period=request.period,
                mode=request.mode,
                exchange_name=spec.exchange_name,
                currency_code=spec.currency_code,
                asset_class=spec.asset_class,
                staleness=staleness,
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

    # Incremental: pre-load staleness for the whole sweep in one main-thread
    # session (grouped queries) instead of 11 MAX() round-trips per worker.
    staleness_by_id: dict[UUID, dict[str, Any]] = {}
    if request.mode == "incremental":
        with database_manager.get_session() as _s:
            staleness_by_id = YFinanceRepository(_s).get_staleness_info_bulk(
                [s.instrument_id for s in specs]
            )

    counter = [0]
    counter_lock = threading.Lock()
    aggregate_lock = threading.Lock()
    counts: dict[str, int] = {}
    errors: list[str] = []
    skipped = 0
    sweep_requests = _SweepRequests()

    def _worker(
        spec: _InstrumentSpec,
    ) -> tuple[_InstrumentSpec, dict[str, Any] | Exception]:
        try:
            return spec, _fetch_one_ticker(
                spec,
                request,
                yf_client,
                database_manager,
                request_timeout,
                staleness=(
                    staleness_by_id.get(spec.instrument_id)
                    if request.mode == "incremental"
                    else _UNSET
                ),
            )
        except Exception as exc:
            return spec, exc

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(_worker, spec) for spec in specs]
        for future in as_completed(futures):
            spec, outcome = future.result()
            with counter_lock:
                counter[0] += 1
                idx = counter[0]
            on_progress(current=idx, current_ticker=spec.yfinance_ticker)
            with aggregate_lock:
                skipped += _aggregate_outcome(spec, outcome, counts, errors)
                if not isinstance(outcome, Exception):
                    sweep_requests.add(outcome)
    finally:
        # on_progress raises CancelledError once the job's row was reaped. A
        # plain ``with`` block would still drain every queued ticker on exit,
        # leaving the sweep running as a zombie beside a freshly started run.
        executor.shutdown(wait=True, cancel_futures=True)

    result_dict = {
        "tickers_processed": total,
        "counts": counts,
        "error_count": len(errors),
        "skipped_category_count": skipped,
    }
    _conclude_sweep(result_dict, errors, sweep_requests, on_progress)
    logger.info(
        "Parallel yfinance fetch completed: %d tickers (%d workers)", total, workers
    )
    return result_dict
