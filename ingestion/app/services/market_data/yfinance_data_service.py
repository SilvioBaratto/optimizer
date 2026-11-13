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
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, replace
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
    ArticleScraperProtocol,
    CircuitBreakerProtocol,
    RequestOutcomes,
    YFinanceClient,
    fetch_ticker_news,
    record_failure,
    track_request_outcomes,
)
from app.services.market_data.yfinance.categories import (
    CategorySpec,
    FetchResult,
    is_empty_collection,
    is_settled,
    run_category,
    write_in_savepoint,
)
from app.services.market_data.yfinance.fetch_plan import (
    FetchContext,
    FetchMode,
    FetchPlan,
    StalenessThresholds,
)

# Kept under its private name: this module calls it by that name, and its
# tests import it from here.
from app.services.market_data.yfinance.fetch_plan import is_fresh as _is_fresh

# Imported as a module so the scraper class is looked up when a sweep builds
# one, which is where tests replace it.
from app.services.market_data.yfinance.news import scraper as news_scraper
from app.services.market_data.yfinance.news._links import extract_article_link
from app.services.market_data.yfinance.price_policy import (
    PriceBasis,
    PriceStorageDecision,
    PriceStorageReason,
    decide_price_storage,
    spans_stored_series,
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


def call_with_timeout(
    fn: "Any",
    timeout: float,
    circuit_breaker: CircuitBreakerProtocol | None = None,
) -> "Any":
    """Execute *fn* in a daemon thread; raise ``TimeoutError`` after *timeout* seconds.

    On expiry the caller is released immediately; *fn* keeps running in the
    abandoned thread and whatever it returns or raises afterwards is dropped.

    Args:
        fn: Zero-argument callable to execute.
        timeout: Maximum wall-clock seconds to wait.
        circuit_breaker: Breaker guarding the requests *fn* makes. An open
            window is waited out before the deadline starts. ``check()`` sleeps
            the window out, so inside *fn* that wait would be charged to
            *timeout*: the call would be recorded as a hang, and the abandoned
            thread would sleep on and fire its request when the window closes,
            one per timed-out call. When the breaker has hit its attempt limit
            ``check()`` raises instead of waiting; that is left to the guarded
            request inside *fn*, which decides what to do with it.

    Returns:
        The return value of *fn*.

    Raises:
        TimeoutError: If *fn* does not complete within *timeout* seconds.
        Exception: Any exception raised by *fn* is re-raised in the caller.
    """
    if circuit_breaker is not None:
        with suppress(RuntimeError):
            circuit_breaker.check()

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


DEFAULT_THRESHOLDS = StalenessThresholds()

# One memory for every bulk sweep in the process: services are built per run (or
# per ticker in the parallel path), so a memory owned by the service would be
# forgotten before the next daily run could use it.
SHARED_EMPTY_FETCHES = EmptyFetchMemory()

# Staleness marker -> the categories gated on it. An empty answer settles a
# marker only when every one of its categories came back empty.
_EVENT_MARKER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "dividends_updated_at": ("dividends",),
    "splits_updated_at": ("splits",),
    "recommendations_updated_at": ("recommendations", "analyst_actions"),
    "price_targets_updated_at": ("price_targets",),
    "institutional_holders_updated_at": ("institutional_holders", "major_holders"),
    "mutualfund_holders_updated_at": ("mutualfund_holders",),
    "insider_transactions_updated_at": (
        "insider_transactions",
        "insider_purchases",
        "insider_roster",
    ),
}

# Every marker the incremental short circuit checks besides the profile, in the
# order the sweep fetches their categories: statements, fundamentals, events.
_MARKERS_BEFORE_REQUESTS: tuple[str, ...] = (
    *(statement_staleness_key(*group) for group in FINANCIAL_STATEMENT_GROUPS),
    *(f"{category}_updated_at" for category in FUNDAMENTAL_CATEGORIES),
    *_EVENT_MARKER_CATEGORIES,
)

# Period type of the statement groups requested with ``quarterly=True``.
_QUARTERLY = "quarterly"

# Sentinel: distinguishes "caller did not supply staleness" (query it) from
# "caller supplied None" (no data yet → treat as stale) on fetch_and_store.
_UNSET: Any = object()


def _is_etf_fund(instrument_type: str | None, asset_class: str | None) -> bool:
    """Whether the instrument has ETF fund data (sectors, holdings) to fetch.

    The instrument type decides when it is known, so equity and commodity ETFs
    qualify. ``asset_class == "equity"`` cannot stand in for it: every STOCK
    carries that value too. Callers that do not know the type keep the legacy
    rule, under which only bond and multi-asset funds qualified.
    """
    if instrument_type is not None:
        return instrument_type.upper() == "ETF"
    return asset_class in ("fixed_income", "multi_asset")


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


def _stored_price_max_date(plan: FetchPlan) -> Any:
    """Return the newest stored bar's date when the plan may rely on it, else None.

    Only an incremental plan with a staleness snapshot knows what is stored; any
    other plan fetches the whole period.
    """
    staleness = plan.staleness
    if not plan.uses_staleness or staleness is None:
        return None
    return staleness.get("price_max_date")


def _fetch_mode(mode: str) -> FetchMode:
    """Return the plan mode for a caller's mode string."""
    # Any mode other than "incremental" has always fetched everything, where
    # FetchMode(mode) would reject it.
    return FetchMode.INCREMENTAL if mode == FetchMode.INCREMENTAL else FetchMode.FULL


def _nothing_due(
    ctx: FetchContext, plan: FetchPlan, recently_empty: Callable[[str], bool]
) -> bool:
    """Report whether an incremental sweep can leave the ticker without a request.

    True when the profile is fresh, every other category is settled and the
    stored prices already reach the last completed session. The sweep then
    avoids building a ``yf.Ticker`` at all, which takes the rate limiter even on
    a cache hit and so adds ~100 ms per ticker across the universe.

    Args:
        ctx: Ticker and settings of the sweep.
        plan: Freshness decision of the sweep.
        recently_empty: Memory of empty answers; see ``is_settled``.

    Returns:
        False for any sweep that does not use staleness.
    """
    if not plan.uses_staleness or plan.is_due("profile_updated_at"):
        return False
    settled = (
        is_settled(plan, marker, recently_empty) for marker in _MARKERS_BEFORE_REQUESTS
    )
    return all(settled) and _prices_are_current(
        _stored_price_max_date(plan), ctx.exchange_name, ctx.now.date()
    )


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


def _dated_valuation_panel(panel: pd.DataFrame, ctx: FetchContext) -> pd.DataFrame:
    """Date the columns of a valuation panel, warning when none of them resolves.

    Args:
        panel: Yahoo's valuation panel, one column per period plus ``"Current"``.
        ctx: The sweep's context; its time stamps ``"Current"`` and its ticker
            names the panel in the warning.

    Returns:
        The panel with datetime column labels; empty when no column resolved,
        which leaves the category with nothing to store.
    """
    dated = _date_valuation_columns(panel, ctx.now.date())
    if dated.empty:
        logger.warning(
            "valuation_measures for %s: none of the columns %s "
            "resolved to a date; nothing stored",
            ctx.yfinance_ticker,
            list(panel.columns)[:5],
        )
    return dated


def _article_id(article: dict[str, Any]) -> Any:
    """Return a feed article's id, which older feeds call ``uuid``."""
    return article.get("id") or article.get("uuid")


def _preview_or_feed_link(
    article: Mapping[str, Any], content: Mapping[str, Any]
) -> str | None:
    """Fall back from a missing canonical URL to the preview URL, then ``link``.

    Only the article's own ``link`` counts, not one inside ``content``. The sweep
    only tests the answer for truth, so a falsy one means nothing to scrape.
    """
    return content.get("previewUrl") or article.get("link")


def _has_no_share_counts(shares: Any) -> bool:
    """Report whether a share-count answer holds nothing to store.

    An answer without a frame's ``empty`` flag counts as nothing to store
    rather than failing the category.
    """
    return shares is None or bool(getattr(shares, "empty", True))


# Re-fetched closes inside this relative band of the stored ones share one
# adjustment basis: NUMERIC(20,6) rounding stays orders of magnitude below it,
# while a dividend of one basis point of price (the smallest that moves a
# return) is above it.
_BASIS_REL_TOL = 1e-4
_BASIS_ABS_TOL = 1e-6

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


@dataclass(frozen=True)
class _FetchedPrices:
    """One price fetch of the sweep, with the window a re-base replaced.

    Attributes:
        history: Bars to store: the incremental window, or a full-period history.
        window: The incremental window a re-base fetch replaced, kept so it can
            still be stored when the full history is refused; ``None`` otherwise.
        incremental: Whether the fetch started from the newest stored bar.
    """

    history: pd.DataFrame | None
    window: pd.DataFrame | None = None
    incremental: bool = False

    @property
    def is_full_period(self) -> bool:
        """Whether the history covers the whole period rather than a window."""
        return not self.incremental or self.window is not None


# The rendered reason of a short full-period history that is stored anyway.
_SHORT_HISTORY_STORED: dict[PriceStorageReason, str] = {
    PriceStorageReason.YOUNG_LISTING: "young listing with no prior price data",
    PriceStorageReason.SPANS_STORED_SERIES: (
        "young listing, history spans every stored price"
    ),
}


def _log_price_decision(
    ctx: FetchContext,
    history_len: int,
    basis: PriceBasis,
    decision: PriceStorageDecision,
) -> None:
    """Log a short full-period history the policy refused or stored anyway."""
    if decision.reason is PriceStorageReason.PROTECTS_STORED_SERIES:
        logger.info(
            "Skipping price storage for %s: got %d rows, expected ~%d (min %d) "
            "for %s on %s (protecting existing series)",
            ctx.yfinance_ticker,
            history_len,
            basis.expected_sessions,
            basis.minimum_sessions,
            ctx.period,
            ctx.exchange_name,
        )
        return
    why = _SHORT_HISTORY_STORED.get(decision.reason)
    if why is not None:
        logger.info(
            "Storing short history for %s: %d rows (below expected ~%d for %s) — %s",
            ctx.yfinance_ticker,
            history_len,
            basis.expected_sessions,
            ctx.period,
            why,
        )


@dataclass(frozen=True)
class _EmptyAnswers:
    """One ticker's access to the memory of categories Yahoo answered empty.

    An empty answer stores no row, so without this memory a category Yahoo has
    nothing for would be requested again on every run.

    Attributes:
        memory: Empty answers remembered across the sweeps of the process.
        ctx: Ticker and reference time of the sweep.
        yf_client: Client whose breaker tells a lost request from an answer.
        outcomes: Requests of the ticker that Yahoo never answered.
    """

    memory: EmptyFetchMemory
    ctx: FetchContext
    yf_client: YFinanceClient
    outcomes: RequestOutcomes

    def recently_empty(self, marker: str) -> bool:
        """Report whether Yahoo answered the category behind *marker* empty lately.

        Read on every call, because the sweep adds to the memory while it runs.
        """
        return self.memory.is_recent(
            self.ctx.instrument_id,
            marker,
            self.ctx.thresholds.no_data_hours,
            self.ctx.now,
        )

    def remember(self, marker: str, stored: int) -> None:
        """Remember the category behind *marker* as empty when it stored nothing.

        Args:
            marker: Staleness key of the category or statement group.
            stored: Rows the category stored this sweep.
        """
        if stored:
            return
        # While the breaker is open an empty result is more likely a lost request
        # than Yahoo's answer; leaving it unrecorded keeps the next run retrying
        # instead of hiding the category. Only rate limits open the breaker, so a
        # 5xx or an unreachable host is told apart through the tally. It carries
        # no category, so one lost request keeps every empty answer of the ticker
        # retryable.
        if self.yf_client.circuit_breaker.is_active or self.outcomes.failures:
            return
        self.memory.remember(self.ctx.instrument_id, marker, self.ctx.now)


class YFinanceDataService:
    """Fetches all yfinance data categories for a ticker and stores via repository."""

    def __init__(
        self,
        repo: YFinanceRepository,
        yf_client: YFinanceClient,
        request_timeout: float | None = None,
        empty_fetches: EmptyFetchMemory | None = None,
        news_scrape_delay: float | None = None,
    ) -> None:
        """Wire the service to its repository, client and sweep settings.

        Args:
            repo: Repository every category is stored through.
            yf_client: Client every Yahoo request goes through.
            request_timeout: Wall-clock seconds allowed per yfinance call;
                ``None`` disables the watchdog.
            empty_fetches: Memory of categories Yahoo answered empty; ``None``
                gives the service a private one.
            news_scrape_delay: Seconds to wait before each article page scraped
                for a ticker's news; ``None`` takes
                ``YFINANCE_NEWS_SCRAPE_DELAY_SECONDS`` from the daemon settings.
        """
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
        if news_scrape_delay is None:
            from app.config import settings

            news_scrape_delay = settings.yfinance_news_scrape_delay_seconds
        self._news_scrape_delay = news_scrape_delay

    def _timed(self, fn: "Any") -> "Any":
        """Execute *fn*, enforcing ``self._request_timeout`` if set.

        The circuit-breaker window is waited out before the deadline starts
        (see ``call_with_timeout``).
        """
        if self._request_timeout is None:
            return fn()
        return call_with_timeout(
            fn, self._request_timeout, circuit_breaker=self.yf_client.circuit_breaker
        )

    def _write(self, write: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run one category's repository write inside its own SAVEPOINT.

        The same savepoint ``run_category`` stores through; see
        ``write_in_savepoint`` for why every category needs one.

        Args:
            write: Repository upsert to run.
            *args: Positional arguments for *write*.
            **kwargs: Keyword arguments for *write*.

        Returns:
            Whatever *write* returns (the number of rows it stored).

        Raises:
            RuntimeError: When the database rejected the statement. The message
                is the driver's cause, without the SQL text; the original error
                stays on ``__cause__``.
        """
        return write_in_savepoint(self.repo.session, write, *args, **kwargs)

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
        return spans_stored_series(_bar_date(history.index.min()), first.date)

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
        # Today's bar is still forming while the exchange is open, and the
        # freshness gate takes the newest stored date as final; the incremental
        # fetch stops before today for the same reason. yfinance cannot combine
        # "ytd" with an end date, so that window is spelled out as a start.
        today = _today()
        if start is None and period == "ytd":
            start = date(today.year, 1, 1)
        window: dict[str, Any] = (
            {"period": period} if start is None else {"start": start.isoformat()}
        )
        window["end"] = today.isoformat()
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
        instrument_type: str | None = None,
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
            asset_class: Instrument asset class. Only consulted when
                ``instrument_type`` is unknown: "fixed_income" or "multi_asset"
                then mark an ETF.
            staleness: Pre-fetched staleness info for incremental mode.
            instrument_type: Instrument type ("ETF", "STOCK", ...). When
                "ETF", the ETF fund-metadata branch runs whatever the asset
                class, so equity and commodity ETFs get sector weights too.
                ``None`` keeps the legacy ``asset_class`` rule.

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
                instrument_type=instrument_type,
                outcomes=outcomes,
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
        instrument_type: str | None,
        outcomes: RequestOutcomes,
    ) -> dict[str, Any]:
        """Run one ticker's sweep inside ``fetch_and_store``'s request tally.

        Builds the sweep's context and freshness plan once, then runs the steps;
        *outcomes* lets an empty answer tell Yahoo's silence from a lost request.
        """
        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS
        staleness = self._staleness_snapshot(
            instrument_id, yfinance_ticker, mode, staleness
        )
        now = datetime.now(timezone.utc)
        plan = FetchPlan(
            mode=_fetch_mode(mode), staleness=staleness, thresholds=thresholds, now=now
        )
        ctx = FetchContext(
            instrument_id=instrument_id,
            yfinance_ticker=yfinance_ticker,
            period=period,
            mode=plan.mode,
            exchange_name=exchange_name,
            currency_code=currency_code,
            asset_class=asset_class,
            instrument_type=instrument_type,
            now=now,
            thresholds=thresholds,
        )
        result = FetchResult()
        empty_answers = _EmptyAnswers(
            self._empty_fetches, ctx, self.yf_client, outcomes
        )
        self._sweep(ctx, plan, result, empty_answers)
        return result.to_response()

    def _staleness_snapshot(
        self, instrument_id: UUID, yfinance_ticker: str, mode: str, staleness: Any
    ) -> Any:
        """Return the staleness snapshot an incremental sweep decides on.

        Args:
            instrument_id: Instrument whose stored categories are looked up.
            yfinance_ticker: Ticker named in the warning of a failed lookup.
            mode: The sweep mode; only an incremental sweep uses a snapshot.
            staleness: The caller's snapshot, or ``_UNSET`` to query it here. A
                bulk sweep loads every instrument's in one grouped query and
                injects it, avoiding a round-trip per instrument.

        Returns:
            The snapshot, or ``None`` for a full sweep and when the lookup failed,
            which makes every category due.
        """
        if mode != FetchMode.INCREMENTAL:
            return None
        if staleness is not _UNSET:
            return staleness
        try:
            return self.repo.get_staleness_info(instrument_id)
        except Exception as e:
            logger.warning(
                "Staleness query failed for %s, falling back to full fetch: %s",
                yfinance_ticker,
                e,
            )
            return None

    def _sweep(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        empty_answers: _EmptyAnswers,
    ) -> None:
        """Run every step of the ticker's sweep, or none when nothing is due."""
        if _nothing_due(ctx, plan, empty_answers.recently_empty):
            logger.debug(
                "All categories fresh for %s — skipping network fetch",
                ctx.yfinance_ticker,
            )
            result.skipped.append("all:fresh")
            return
        profile_info = self._fetch_profile(ctx, plan, result)
        self._fetch_prices(ctx, plan, result)
        self._fetch_statements(
            ctx, plan, result, profile_info=profile_info, empty_answers=empty_answers
        )
        self._fetch_fundamentals(ctx, plan, result, empty_answers)
        self._fetch_events(ctx, plan, result, empty_answers)
        self._fetch_news(ctx, plan, result, self._article_scraper())
        if _is_etf_fund(ctx.instrument_type, ctx.asset_class):
            self._fetch_etf_metadata(
                ctx.instrument_id,
                ctx.yfinance_ticker,
                ctx.mode,
                ctx.thresholds,
                ctx.now,
                result.counts,
                result.errors,
                result.skipped,
            )

    def _fetch_profile(
        self, ctx: FetchContext, plan: FetchPlan, result: FetchResult
    ) -> dict[str, Any] | None:
        """Refresh the ticker's profile and its extra fields from Yahoo's ``info``.

        Args:
            ctx: Ticker and settings of the sweep.
            plan: Decides whether the profile is due.
            result: Receives the profile counts, the ``profile`` skip or the
                ``profile:`` and ``isin:`` errors.

        Returns:
            The ``info`` Yahoo answered, kept even when storing it failed, so the
            statements can read the currency they are reported in without a
            second request. ``None`` when the profile was not due, Yahoo sent
            nothing or the request failed.
        """
        if not plan.is_due("profile_updated_at"):
            result.skipped.append("profile")
            return None
        profile_info: dict[str, Any] | None = None
        try:
            info = self._timed(lambda: self.yf_client.fetch_info(ctx.yfinance_ticker))
            if not info:
                result.counts["profile"] = 0
                return None
            profile_info = info
            self._store_profile(ctx, info, result)
        except Exception as e:
            result.errors.append(f"profile: {e}")
            logger.warning("Failed to fetch profile for %s: %s", ctx.yfinance_ticker, e)
        return profile_info

    def _store_profile(
        self, ctx: FetchContext, info: dict[str, Any], result: FetchResult
    ) -> None:
        """Store the profile and its extra fields, with the ISIN added to *info*."""
        # yfinance's .info does not carry the ISIN, so Ticker.isin is asked
        # separately and injected for ticker_profiles.isin.
        isin = self._fetch_isin(ctx, result)
        if isin:
            info["isin"] = isin
        result.counts["profile"] = self._write(
            self.repo.upsert_profile, ctx.instrument_id, info
        )
        # Extra info fields (short interest, momentum, governance risk) reuse the
        # already-fetched dict, with no separate network call.
        result.counts["profile_extras"] = self._write(
            self.repo.upsert_profile_extras, ctx.instrument_id, info
        )

    def _fetch_isin(self, ctx: FetchContext, result: FetchResult) -> Any:
        """Return the ticker's ISIN, or ``None`` with an ``isin:`` error on failure."""
        # A separate third-party call: its failure must not cost the profile
        # that was just fetched.
        try:
            return self._timed(
                lambda: self.yf_client.metadata.fetch_isin(ctx.yfinance_ticker)
            )
        except Exception as e:
            result.errors.append(f"isin: {e}")
            logger.warning("Failed to fetch ISIN for %s: %s", ctx.yfinance_ticker, e)
            return None

    def _fetch_prices(
        self, ctx: FetchContext, plan: FetchPlan, result: FetchResult
    ) -> None:
        """Refresh the ticker's price history and report it as ``prices``.

        An incremental plan that knows the newest stored bar fetches the window
        since it, and the whole period instead when a split or dividend re-based
        the overlap. Any other plan fetches the whole period. A full-period history
        is stored only when ``decide_price_storage`` allows it.

        Args:
            ctx: Ticker and settings of the sweep.
            plan: Supplies the newest stored bar of an incremental sweep.
            result: Receives the ``prices`` count, the
                ``prices:insufficient_history`` or ``prices:unavailable`` skip, or
                the ``prices:`` error.
        """
        try:
            fetched = self._fetch_price_history(ctx, plan)
            history = self._history_to_store(ctx, plan, fetched, result)
            self._store_prices(ctx, history, fetched.incremental, result)
        except Exception as e:
            result.errors.append(f"prices: {e}")
            logger.warning("Failed to fetch prices for %s: %s", ctx.yfinance_ticker, e)

    def _fetch_price_history(
        self, ctx: FetchContext, plan: FetchPlan
    ) -> _FetchedPrices:
        """Fetch the incremental window, or the whole period when there is none."""
        max_date = _stored_price_max_date(plan)
        if max_date is not None:
            return self._fetch_price_window(ctx, max_date)
        staleness = plan.staleness
        first_load = (
            isinstance(staleness, dict) and staleness.get("price_max_date") is None
        )
        history = self._fetch_full_price_history(
            ctx.instrument_id,
            ctx.yfinance_ticker,
            ctx.period,
            may_have_stored_bars=not first_load,
        )
        return _FetchedPrices(history=history)

    def _fetch_price_window(self, ctx: FetchContext, max_date: date) -> _FetchedPrices:
        """Fetch the bars since *max_date*, re-based when the overlap moved."""
        start_date = max_date - timedelta(days=ctx.thresholds.price_overlap_days)
        window = self.yf_client.fetch_history(
            ctx.yfinance_ticker,
            start=start_date.isoformat(),
            end=date.today().isoformat(),
            min_rows=0,
            # The actions flag adds the Dividends and Stock Splits columns the
            # price-history upsert persists; Yahoo no longer sends Capital Gains,
            # which stays NULL.
            actions=True,
            timeout=self._request_timeout,
        )
        # A split or dividend since the last fetch rescales all earlier bars, but
        # the window rewrites only its own: re-fetch the whole period so the
        # stored series stays on a single basis.
        if self._overlap_was_readjusted(
            ctx.instrument_id, window, start_date, max_date
        ):
            rebased = self._fetch_rebased_history(ctx)
            if rebased is not None and not rebased.empty:
                return _FetchedPrices(history=rebased, window=window, incremental=True)
        return _FetchedPrices(history=window, incremental=True)

    def _fetch_rebased_history(self, ctx: FetchContext) -> pd.DataFrame | None:
        """Fetch the whole period for a re-base; ``None`` when that fails."""
        try:
            return self._fetch_full_price_history(
                ctx.instrument_id, ctx.yfinance_ticker, ctx.period
            )
        except Exception as e:
            logger.warning(
                "Re-base fetch failed for %s, storing the window only: %s",
                ctx.yfinance_ticker,
                e,
            )
            return None

    def _history_to_store(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        fetched: _FetchedPrices,
        result: FetchResult,
    ) -> pd.DataFrame | None:
        """Return the bars to store: the fetched history, or the window it replaced.

        A full-period history the policy refuses is reported as skipped and leaves
        the stored series alone; the incremental window a re-base replaced is
        stored instead, when there was one.
        """
        history = fetched.history
        if history is None or history.empty or not fetched.is_full_period:
            return history
        young_listing, basis = self._judge_full_history(ctx, plan, history)
        decision = decide_price_storage(len(history), young_listing, basis)
        _log_price_decision(ctx, len(history), basis, decision)
        if decision.should_store:
            return history
        result.counts["prices"] = 0
        result.skipped.append("prices:insufficient_history")
        return fetched.window

    def _judge_full_history(
        self, ctx: FetchContext, plan: FetchPlan, history: pd.DataFrame
    ) -> tuple[bool, PriceBasis]:
        """Return whether nothing is stored yet, and how *history* measures up.

        The stored series is looked up only for a history the period finds short,
        and its first bar only when the series' lifetime finds it short too, so a
        sufficient history costs no query.
        """
        basis = self._sessions_basis(ctx, len(history))
        if basis.sufficient:
            return False, basis
        stored = (
            plan.staleness
            if plan.staleness is not None
            else self.repo.get_staleness_info(ctx.instrument_id)
        )
        if stored.get("price_max_date") is None:
            return True, basis
        since = stored.get("price_min_date")
        if since is not None:
            # A young listing with stored prices has no sessions before its first
            # stored bar. The anchor comes from storage, not the fetched frame: a
            # head-truncated refetch is complete when measured from its own start.
            basis = self._sessions_basis(ctx, len(history), since=since)
        if basis.sufficient:
            return False, basis
        spans = self._covers_stored_series(ctx.instrument_id, history)
        return False, replace(basis, spans_stored_series=spans)

    def _sessions_basis(
        self, ctx: FetchContext, history_len: int, since: date | None = None
    ) -> PriceBasis:
        """Judge *history_len* rows against the exchange calendar from *since* on."""
        sufficient, expected, minimum = has_sufficient_history(
            history_len,
            ctx.exchange_name,
            ctx.period,
            ctx.thresholds.min_history_tolerance,
            since=since,
        )
        return PriceBasis(
            sufficient=sufficient,
            expected_sessions=expected,
            minimum_sessions=minimum,
        )

    def _store_prices(
        self,
        ctx: FetchContext,
        history: pd.DataFrame | None,
        incremental: bool,
        result: FetchResult,
    ) -> None:
        """Store *history* as the ``prices`` count, or report that none came."""
        if history is None or history.empty:
            result.counts["prices"] = result.counts.get("prices", 0)
            if incremental:
                # yfinance swallows network faults inside history() and returns
                # an empty frame. The overlap window always spans stored bars, so
                # an empty answer for a ticker that has a series is the only trace
                # of a Yahoo that did not answer.
                result.skipped.append("prices:unavailable")
            return
        # The listing currency is stored raw as the price unit, so sub-unit series
        # (GBX pence) are unambiguous downstream.
        result.counts["prices"] = self._write(
            self.repo.upsert_price_history,
            ctx.instrument_id,
            history,
            price_unit=ctx.currency_code,
        )

    def _fetch_statements(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        *,
        profile_info: dict[str, Any] | None,
        empty_answers: _EmptyAnswers,
    ) -> None:
        """Refresh the six financial-statement groups as the one ``financials``.

        Each group is gated on its own marker, so one statement stored during a
        rate-limit burst does not hide the ones that failed until the window
        expires. The step is skipped as ``financials`` only when every group is
        settled.

        Args:
            ctx: Ticker and settings of the sweep.
            plan: Decides which groups are due.
            result: Receives the summed ``financials`` count, the ``financials``
                skip, and an error per failed group (or one ``financials:`` error
                when the client cannot serve statements at all).
            profile_info: The profile step's ``info``; it names the reporting
                currency, so a group with rows needs no extra request.
            empty_answers: Settles groups Yahoo recently answered empty and
                remembers the groups that stored nothing.
        """
        markers = (
            statement_staleness_key(*group) for group in FINANCIAL_STATEMENT_GROUPS
        )
        if all(is_settled(plan, m, empty_answers.recently_empty) for m in markers):
            result.skipped.append("financials")
            return
        try:
            # The groups report failures under their own labels but count, and are
            # skipped, as the one "financials" category: they share only the
            # sweep's errors.
            groups = FetchResult(errors=result.errors)
            currency = self._statement_currency(ctx, profile_info)
            for spec in self._statement_specs(currency):
                self._run_category(spec, ctx, plan, groups, empty_answers)
                # Each group has its own marker, so an empty group is remembered
                # whatever its siblings answered.
                if spec.name in groups.counts:
                    empty_answers.remember(spec.marker, groups.counts[spec.name])
            result.counts["financials"] = sum(groups.counts.values())
        except Exception as e:
            result.errors.append(f"financials: {e}")
            logger.warning("Failed financials for %s: %s", ctx.yfinance_ticker, e)

    def _statement_currency(
        self, ctx: FetchContext, profile_info: dict[str, Any] | None
    ) -> Callable[[], str | None]:
        """Return a lookup of the currency the company reports its statements in.

        Statements are denominated in the reporting currency, which Yahoo exposes
        as ``info["financialCurrency"]`` and which differs from the listing
        currency for ADRs and cross-listings (TSM lists in USD but reports in
        TWD; SHEL.L lists in GBp but reports in USD). The lookup resolves on first
        use, so a ticker with nothing to store never pays for an ``info`` request;
        the profile step usually already supplied it.
        """
        resolved: list[str | None] = []

        def statement_currency() -> str | None:
            if not resolved:
                info = profile_info or self._fetch_info_for_currency(ctx)
                resolved.append(_financial_currency(info))
            return resolved[0]

        return statement_currency

    def _fetch_info_for_currency(self, ctx: FetchContext) -> dict[str, Any] | None:
        """Request ``info`` for its reporting currency; ``None`` when that fails."""
        try:
            info: dict[str, Any] | None = self._timed(
                lambda: self.yf_client.fetch_info(ctx.yfinance_ticker)
            )
        except Exception as e:
            logger.warning(
                "Could not resolve financial currency for %s: %s",
                ctx.yfinance_ticker,
                e,
            )
            return None
        return info

    def _run_category(
        self,
        spec: CategorySpec[Any],
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        empty_answers: _EmptyAnswers,
    ) -> None:
        """Run one category through ``run_category`` on this service's session."""
        run_category(
            spec,
            ctx,
            plan,
            result,
            session=self.repo.session,
            recently_empty=empty_answers.recently_empty,
        )

    def _fetch_fundamentals(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        empty_answers: _EmptyAnswers,
    ) -> None:
        """Refresh the fundamentals categories, then remember the empty ones.

        Each category is gated on the marker of its own table: one whose fetch
        failed while the statements were stored must not be hidden by them for
        the whole window.
        """
        listing_currency = to_major_currency(ctx.currency_code)
        for spec in self._fundamental_specs(ctx, listing_currency):
            self._run_category(spec, ctx, plan, result, empty_answers)
        # A category present in counts was fetched without raising; its count
        # says whether Yahoo had anything.
        for category in FUNDAMENTAL_CATEGORIES:
            if category in result.counts:
                empty_answers.remember(
                    f"{category}_updated_at", result.counts[category]
                )

    def _fetch_events(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        empty_answers: _EmptyAnswers,
    ) -> None:
        """Refresh the corporate-action, analyst and holder categories.

        A marker whose categories were all fetched without raising and stored
        nothing is one Yahoo holds no rows for; it is remembered so the next run
        does not ask again.
        """
        for spec in self._event_specs():
            self._run_category(spec, ctx, plan, result, empty_answers)
        counts = result.counts
        for marker, categories in _EVENT_MARKER_CATEGORIES.items():
            if all(category in counts for category in categories):
                stored = int(any(counts[category] for category in categories))
                empty_answers.remember(marker, stored)

    def _article_scraper(self) -> ArticleScraperProtocol:
        """Return the scraper of the sweep's article pages, at the configured pace."""
        return news_scraper.ArticleScraper(delay=self._news_scrape_delay)

    def _fetch_news(
        self,
        ctx: FetchContext,
        plan: FetchPlan,
        result: FetchResult,
        scraper: ArticleScraperProtocol,
    ) -> None:
        """Refresh the ticker's news feed, with the body of every new article.

        News has no staleness marker: the feed is requested on every sweep, and
        only the bodies of articles not stored yet are scraped.

        Args:
            ctx: Ticker and settings of the sweep.
            plan: Not consulted: news has no freshness marker, so the feed is
                due on every sweep.
            result: Receives the ``news`` count or the ``news:`` error.
            scraper: Fetches the body of each new article's page.
        """
        try:
            news_list = self._timed(
                lambda: fetch_ticker_news(ctx.yfinance_ticker, self.yf_client)
            )
            if not news_list:
                result.counts["news"] = 0
                return
            self._scrape_new_articles(ctx, news_list, scraper)
            result.counts["news"] = self._write(
                self.repo.upsert_news, ctx.instrument_id, news_list
            )
        except Exception as e:
            result.errors.append(f"news: {e}")
            logger.warning("Failed news for %s: %s", ctx.yfinance_ticker, e)

    def _scrape_new_articles(
        self,
        ctx: FetchContext,
        news_list: list[dict[str, Any]],
        scraper: ArticleScraperProtocol,
    ) -> None:
        """Add ``full_content`` to each article whose body is not stored yet."""
        # The feed repeats the same articles for days. Re-scraping a body already
        # stored costs a throttled page request per article per sweep; the upsert
        # leaves a stored body alone when none is sent.
        feed_ids = [
            str(news_id)
            for article in news_list
            if (news_id := _article_id(article)) is not None
        ]
        scraped_ids = self.repo.get_scraped_news_uuids(ctx.instrument_id, feed_ids)
        for article in news_list:
            news_id = _article_id(article)
            if news_id is not None and str(news_id) in scraped_ids:
                continue
            link = extract_article_link(article, _preview_or_feed_link)
            if not link:
                continue
            page = scraper.fetch(link)
            if page["success"] and page["content"]:
                article["full_content"] = page["content"]

    def _statement_specs(
        self, statement_currency: Callable[[], str | None]
    ) -> tuple[CategorySpec[Any], ...]:
        """Return one spec per financial-statement group, in the sweep's order.

        The client's statement requests are looked up here, before any group
        runs, so a client that cannot serve statements at all fails them as one
        ``financials`` error instead of six.

        Args:
            statement_currency: Resolves the currency the company reports its
                statements in; called only when a group has rows to store.

        Returns:
            The specs of ``FINANCIAL_STATEMENT_GROUPS``, each named
            ``"financials.<statement>.<period>"`` and gated on its own marker.
        """
        financials = self.yf_client.financials
        requests: dict[str, Callable[..., pd.DataFrame | None]] = {
            "income_statement": financials.fetch_income_stmt,
            "balance_sheet": financials.fetch_balance_sheet,
            "cashflow": financials.fetch_cashflow,
        }
        return tuple(
            self._statement_spec(
                statement_type,
                period_type,
                requests[statement_type],
                statement_currency,
            )
            for statement_type, period_type in FINANCIAL_STATEMENT_GROUPS
        )

    def _statement_spec(
        self,
        statement_type: str,
        period_type: str,
        request: Callable[..., pd.DataFrame | None],
        statement_currency: Callable[[], str | None],
    ) -> CategorySpec[Any]:
        """Return the spec of one statement group; see ``_statement_specs``."""
        quarterly = period_type == _QUARTERLY

        def fetch(ctx: FetchContext) -> Any:
            return self._timed(
                lambda: request(ctx.yfinance_ticker, quarterly=quarterly)
            )

        return CategorySpec(
            name=f"financials.{statement_type}.{period_type}",
            marker=statement_staleness_key(statement_type, period_type),
            fetch=fetch,
            store=self._upsert(
                lambda repo: repo.upsert_financial_statements,
                statement_type,
                period_type,
                options=lambda _ctx: {"currency_code": statement_currency()},
            ),
        )

    def _fundamental_specs(
        self, ctx: FetchContext, listing_currency: str | None
    ) -> tuple[CategorySpec[Any], ...]:
        """Return the specs of the fundamentals step, in the order it fetches them.

        Each category is gated on its own ``<category>_updated_at`` marker on the
        fundamentals window, except analyst actions: they are gated on the
        recommendations window but fetched in this step, since fetching them
        elsewhere would reorder the sweep's requests and errors.

        Args:
            ctx: The sweep's context; its time dates the valuation panel.
            listing_currency: Major unit of the listing currency. It labels the
                valuation rows, because market cap and EV are quote-based.

        Returns:
            The specs of the sweep's fundamentals step.
        """
        return (
            self._valuation_spec(ctx, listing_currency),
            *self._eps_specs(),
            *self._estimate_specs(),
            *self._earnings_specs(),
            *self._filing_and_share_specs(),
        )

    def _valuation_spec(
        self, ctx: FetchContext, listing_currency: str | None
    ) -> CategorySpec[Any]:
        """Return the valuation-measures spec; see ``_fundamental_specs``."""
        return CategorySpec(
            name="valuation_measures",
            marker="valuation_measures_updated_at",
            fetch=self._request(
                lambda client: client.metadata.fetch_valuation_measures
            ),
            transform=lambda panel: _dated_valuation_panel(panel, ctx),
            # Reuses the open EAV statements table under its own statement
            # type, so the panel needs no table or migration of its own.
            store=self._upsert(
                lambda repo: repo.upsert_financial_statements,
                VALUATION_MEASURES_STATEMENT_TYPE,
                VALUATION_MEASURES_PERIOD_TYPE,
                options=lambda _ctx: {"currency_code": listing_currency},
            ),
        )

    def _eps_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the EPS trend and EPS revisions specs, in sweep order."""
        return (
            CategorySpec(
                name="eps_trend",
                marker="eps_trend_updated_at",
                fetch=self._request(lambda client: client.analysis.fetch_eps_trend),
                store=self._upsert(lambda repo: repo.upsert_eps_trend),
            ),
            CategorySpec(
                name="eps_revisions",
                marker="eps_revisions_updated_at",
                fetch=self._request(lambda client: client.analysis.fetch_eps_revisions),
                store=self._upsert(lambda repo: repo.upsert_eps_revisions),
            ),
        )

    def _estimate_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the earnings, revenue and growth estimate specs, in sweep order."""
        return (
            CategorySpec(
                name="earnings_estimate",
                marker="earnings_estimate_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_earnings_estimate
                ),
                store=self._upsert(lambda repo: repo.upsert_earnings_estimate),
            ),
            CategorySpec(
                name="revenue_estimate",
                marker="revenue_estimate_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_revenue_estimate
                ),
                store=self._upsert(lambda repo: repo.upsert_revenue_estimate),
            ),
            CategorySpec(
                name="growth_estimates",
                marker="growth_estimates_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_growth_estimates
                ),
                store=self._upsert(lambda repo: repo.upsert_growth_estimates),
            ),
        )

    def _earnings_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the earnings history and dates specs, then analyst actions."""
        return (
            CategorySpec(
                name="earnings_history",
                marker="earnings_history_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_earnings_history
                ),
                store=self._upsert(lambda repo: repo.upsert_earnings_history),
            ),
            CategorySpec(
                name="earnings_dates",
                marker="earnings_dates_updated_at",
                fetch=self._request(
                    lambda client: client.metadata.fetch_earnings_dates
                ),
                store=self._upsert(lambda repo: repo.upsert_earnings_dates),
            ),
            CategorySpec(
                name="analyst_actions",
                marker="recommendations_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_upgrades_downgrades
                ),
                store=self._upsert(lambda repo: repo.upsert_analyst_actions),
            ),
        )

    def _filing_and_share_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the SEC filings and shares outstanding specs, in sweep order."""
        return (
            CategorySpec(
                name="sec_filings",
                marker="sec_filings_updated_at",
                fetch=self._request(lambda client: client.financials.fetch_sec_filings),
                store=self._upsert(lambda repo: repo.upsert_sec_filings),
                is_empty=is_empty_collection,
            ),
            CategorySpec(
                name="shares_outstanding",
                marker="shares_outstanding_updated_at",
                fetch=self._request(
                    lambda client: client.corporate_actions.fetch_shares_full
                ),
                store=self._upsert(lambda repo: repo.upsert_shares_outstanding),
                is_empty=_has_no_share_counts,
            ),
        )

    def _event_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the corporate-action, analyst-rating and holder specs, in order.

        Each category is gated on the marker of its own event window. Major
        holders share the institutional holders' marker, and insider purchases
        and the insider roster share the insider transactions' marker, which is
        why ``_EVENT_MARKER_CATEGORIES`` remembers a marker as empty only when
        every category behind it answered empty.

        Returns:
            The specs of the sweep's event step, in the order it fetches them.
        """
        return (
            *self._corporate_action_specs(),
            *self._analyst_rating_specs(),
            *self._holder_specs(),
            *self._shared_marker_holder_specs(),
        )

    def _corporate_action_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the dividends and splits specs, in sweep order."""
        return (
            CategorySpec(
                name="dividends",
                marker="dividends_updated_at",
                fetch=self._request(
                    lambda client: client.corporate_actions.fetch_dividends
                ),
                store=self._upsert(
                    lambda repo: repo.upsert_dividends,
                    options=lambda ctx: {"price_currency": ctx.currency_code},
                ),
            ),
            CategorySpec(
                name="splits",
                marker="splits_updated_at",
                fetch=self._request(
                    lambda client: client.corporate_actions.fetch_splits
                ),
                store=self._upsert(lambda repo: repo.upsert_splits),
            ),
        )

    def _analyst_rating_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the analyst recommendation and price target specs, in sweep order."""
        return (
            CategorySpec(
                name="recommendations",
                marker="recommendations_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_recommendations_summary
                ),
                store=self._upsert(lambda repo: repo.upsert_recommendations),
            ),
            CategorySpec(
                name="price_targets",
                marker="price_targets_updated_at",
                fetch=self._request(
                    lambda client: client.analysis.fetch_analyst_price_targets
                ),
                store=self._upsert(lambda repo: repo.upsert_price_targets),
                is_empty=is_empty_collection,
            ),
        )

    def _holder_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the holder specs gated on a marker of their own, in sweep order."""
        return (
            CategorySpec(
                name="institutional_holders",
                marker="institutional_holders_updated_at",
                fetch=self._request(
                    lambda client: client.holders.fetch_institutional_holders
                ),
                store=self._upsert(lambda repo: repo.upsert_institutional_holders),
            ),
            CategorySpec(
                name="mutualfund_holders",
                marker="mutualfund_holders_updated_at",
                fetch=self._request(
                    lambda client: client.holders.fetch_mutualfund_holders
                ),
                store=self._upsert(lambda repo: repo.upsert_mutualfund_holders),
            ),
            CategorySpec(
                name="insider_transactions",
                marker="insider_transactions_updated_at",
                fetch=self._request(
                    lambda client: client.holders.fetch_insider_transactions
                ),
                store=self._upsert(lambda repo: repo.upsert_insider_transactions),
            ),
        )

    def _shared_marker_holder_specs(self) -> tuple[CategorySpec[Any], ...]:
        """Return the holder specs gated on another category's marker, in order."""
        return (
            CategorySpec(
                name="major_holders",
                marker="institutional_holders_updated_at",
                fetch=self._request(lambda client: client.holders.fetch_major_holders),
                store=self._upsert(lambda repo: repo.upsert_major_holders),
            ),
            CategorySpec(
                name="insider_purchases",
                marker="insider_transactions_updated_at",
                fetch=self._request(
                    lambda client: client.holders.fetch_insider_purchases
                ),
                store=self._upsert(lambda repo: repo.upsert_insider_purchases),
            ),
            CategorySpec(
                name="insider_roster",
                marker="insider_transactions_updated_at",
                fetch=self._request(
                    lambda client: client.holders.fetch_insider_roster_holders
                ),
                store=self._upsert(lambda repo: repo.upsert_insider_roster),
            ),
        )

    def _request(
        self, pick: Callable[[YFinanceClient], Callable[[str], Any]]
    ) -> Callable[[FetchContext], Any]:
        """Return a fetch that makes one client request for the sweep's ticker.

        The request is looked up when the fetch runs, not here, so a sub-client
        that cannot serve it fails that category alone.

        Args:
            pick: Selects the request, e.g.
                ``lambda client: client.corporate_actions.fetch_splits``.

        Returns:
            A fetch that calls the request with the context's ticker, under the
            service's request timeout.
        """

        def fetch(ctx: FetchContext) -> Any:
            return self._timed(lambda: pick(self.yf_client)(ctx.yfinance_ticker))

        return fetch

    def _upsert(
        self,
        pick: Callable[[YFinanceRepository], Callable[..., int]],
        *args: Any,
        options: Callable[[FetchContext], dict[str, Any]] | None = None,
    ) -> Callable[[Any, FetchContext], int]:
        """Return a store that hands a category's answer to a repository write.

        The write is only named here: ``run_category`` calls the store inside the
        category's savepoint, the one place besides ``_write`` a repository write
        runs.

        Args:
            pick: Selects the write, e.g. ``lambda repo: repo.upsert_splits``.
            *args: Arguments the write takes after the instrument id and the
                answer.
            options: Builds the write's keyword arguments from the sweep's
                context. Called when the store runs, so only for an answer with
                rows to store.

        Returns:
            A store that calls ``write(instrument_id, answer, *args, **options)``
            and returns the rows it stored.
        """

        def store(payload: Any, ctx: FetchContext) -> int:
            keywords = options(ctx) if options is not None else {}
            return pick(self.repo)(ctx.instrument_id, payload, *args, **keywords)

        return store

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
        if mode == FetchMode.INCREMENTAL:
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
            etf_counts: dict[str, int] = {}
            # One savepoint for the whole fund snapshot: a later failed write must
            # not leave metadata behind without its holdings, and the counts are
            # published only after it succeeds.
            with self.repo.session.begin_nested():
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
                        category=overview.get("categoryName")
                        or overview.get("category"),
                        description=description,
                        as_of=as_of,
                    )
                    etf_counts["etf_metadata"] = 1

                if fdata:
                    ac = fdata.get("asset_classes") or {}
                    buckets = {
                        "stock_pct": ac.get("stockPosition"),
                        "bond_pct": ac.get("bondPosition"),
                        "cash_pct": ac.get("cashPosition"),
                        "preferred_pct": ac.get("preferredPosition"),
                        "convertible_pct": ac.get("convertiblePosition"),
                        "other_pct": ac.get("otherPosition"),
                    }
                    # An all-NULL row would become the newest as_of snapshot and
                    # shadow the last real allocation.
                    if any(v is not None for v in buckets.values()):
                        self._write(
                            etf_repo.upsert_asset_classes,
                            instrument_id,
                            as_of,
                            **buckets,
                        )
                        etf_counts["etf_asset_classes"] = 1
                    holdings = fdata.get("top_holdings") or []
                    if holdings:
                        etf_counts["etf_holdings"] = self._write(
                            etf_repo.upsert_holdings, instrument_id, as_of, holdings
                        )
                    sectors = fdata.get("sector_weightings") or {}
                    if sectors:
                        etf_counts["etf_sector_weights"] = self._write(
                            etf_repo.upsert_sector_weights,
                            instrument_id,
                            as_of,
                            sectors,
                        )
                    # Depth (SPEC A8): equity/bond metrics, ratings, operations.
                    equity = fdata.get("equity_holdings") or {}
                    if equity:
                        etf_counts["etf_equity_holdings"] = self._write(
                            etf_repo.upsert_equity_holdings,
                            instrument_id,
                            as_of,
                            equity,
                        )
                    bonds = fdata.get("bond_holdings") or {}
                    if bonds:
                        etf_counts["etf_bond_holdings"] = self._write(
                            etf_repo.upsert_bond_holdings, instrument_id, as_of, bonds
                        )
                    ratings = fdata.get("bond_ratings") or {}
                    if ratings:
                        etf_counts["etf_bond_ratings"] = self._write(
                            etf_repo.upsert_bond_ratings, instrument_id, as_of, ratings
                        )
                    operations = fdata.get("fund_operations") or {}
                    if operations:
                        etf_counts["etf_fund_operations"] = self._write(
                            etf_repo.upsert_fund_operations,
                            instrument_id,
                            as_of,
                            operations,
                        )
            counts.update(etf_counts)
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
        if request.mode == FetchMode.INCREMENTAL:
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
                        if request.mode == FetchMode.INCREMENTAL
                        else _UNSET
                    ),
                    instrument_type=getattr(instrument, "instrument_type", None),
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

# Pooled connections the sweep needs outside the worker threads: the main
# thread's progress writes, the job heartbeat thread, and the APScheduler job
# store or a concurrent pipeline step.
_NON_WORKER_DB_CONNECTIONS = 3


def pool_safe_worker_limit(pool_size: int, max_overflow: int) -> int:
    """Return the largest worker count the connection pool can serve safely.

    Every worker pins one pooled connection from its first upsert until the
    ticker's commit, across all remaining network calls and news scrapes.
    Workers beyond the pool capacity queue on checkout and fail with a
    ``QueuePool`` timeout, losing that ticker's data, while the progress and
    heartbeat writes starve for the same connections.

    Args:
        pool_size: Persistent connections held by the engine's pool.
        max_overflow: Extra connections the pool may open under load.

    Returns:
        ``pool_size + max_overflow`` minus the connections reserved for the
        non-worker threads, never below 1.
    """
    return max(1, pool_size + max_overflow - _NON_WORKER_DB_CONNECTIONS)


@dataclass(frozen=True)
class _InstrumentSpec:
    """Detached instrument fields used by worker threads."""

    instrument_id: UUID
    yfinance_ticker: str
    exchange_name: str | None
    currency_code: str | None
    asset_class: str | None = None
    instrument_type: str | None = None


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
                instrument_type=getattr(inst, "instrument_type", None),
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
                instrument_type=spec.instrument_type,
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
    if request.mode == FetchMode.INCREMENTAL:
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
                    if request.mode == FetchMode.INCREMENTAL
                    else _UNSET
                ),
            )
        except Exception as exc:
            return spec, exc

    def _report_progress(idx: int, ticker: str) -> None:
        # Progress is observability only: one transient DB error among ~10k
        # writes must not fail a sweep whose data is committed per ticker.
        # Cancellation is the one error that has to stop the job.
        try:
            on_progress(current=idx, current_ticker=ticker)
        except CancelledError:
            raise
        except Exception:
            logger.warning(
                "yfinance progress write failed at ticker %d/%d; continuing",
                idx,
                total,
                exc_info=True,
            )

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(_worker, spec) for spec in specs]
        for future in as_completed(futures):
            spec, outcome = future.result()
            with counter_lock:
                counter[0] += 1
                idx = counter[0]
            _report_progress(idx, spec.yfinance_ticker)
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
