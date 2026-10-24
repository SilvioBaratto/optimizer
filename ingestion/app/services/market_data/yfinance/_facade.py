"""Singleton facade that composes all yfinance sub-clients behind one entry point."""

import logging
import threading
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ._base import TICKER_REQUEST_KEY, fetch_with_fault_retry
from ._config import reapply_yfinance_log_level
from .funds import FundsClient
from .infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    install_fault_witness,
    is_rate_limit_error,
    note_ticker,
    retry_with_backoff,
)
from .market import CalendarsClient, MarketClient, SearchClient, SectorsClient
from .protocols import CacheProtocol, CircuitBreakerProtocol, RateLimiterProtocol
from .screener import ScreenerClient
from .ticker import (
    AnalysisClient,
    CorporateActionsClient,
    FinancialsClient,
    HoldersClient,
    MetadataClient,
)

logger = logging.getLogger(__name__)

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# Every cached ``yf.Ticker`` memoises what it fetched (profile, statements,
# holders, scraped article bodies), so the cache size is resident memory. A
# sweep runs at most 16 workers (``YFINANCE_FETCH_WORKERS``), each on one symbol
# at a time, so a few hundred entries keep every in-flight symbol warm. Sizing
# it to the universe instead would pin on the order of a GiB until exit.
DEFAULT_TICKER_CACHE_SIZE = 256

# Outlasts the longest scheduled exchange closure (Lunar New Year, Golden Week)
# while another exchange keeps trading; a ticker silent for longer has stopped
# trading. Calendar time rather than a bar count, so hourly and daily panels
# tolerate the same closure.
_MAX_TRAILING_GAP = pd.Timedelta(days=10)
# Ten days is under two weekly bars and under one monthly bar, so coarse
# intervals are allowed this many bars of silence instead.
_MAX_TRAILING_GAP_BARS = 5


def _align_close_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align per-ticker Close series into one gap-free panel.

    Interior gaps are carried forward, as the exchanges' holiday calendars
    differ. A ticker whose last print trails the panel end by more than
    ``_MAX_TRAILING_GAP`` (or ``_MAX_TRAILING_GAP_BARS`` bars when that is
    longer) is dropped instead, as is one that never printed: carrying its
    last price to the end would present a delisted or halted ticker as a
    constant-price, zero-volatility asset. The panel is then trimmed to the
    dates every remaining ticker covers, so no leading gaps remain.

    Args:
        prices: Close prices, one column per ticker, indexed by bar time.

    Returns:
        The aligned panel, possibly empty when no date is shared.
    """
    prices = prices.dropna(how="all")
    if prices.empty:
        return prices
    index = prices.index
    # Bars since each ticker's last real print: the NaN run at the panel's end.
    trailing_bars = (prices.notna().iloc[::-1].cumsum() == 0).sum()
    max_gap = _MAX_TRAILING_GAP
    bar_span = index.to_series().diff().median()
    if pd.notna(bar_span):
        max_gap = max(max_gap, _MAX_TRAILING_GAP_BARS * bar_span)
    panel_end = index[-1]
    stopped = [
        ticker
        for ticker, bars in trailing_bars.items()
        if bars >= len(index) or panel_end - index[-1 - bars] > max_gap
    ]
    if stopped:
        logger.warning(
            "Dropping %d ticker(s) with no price for over %g days before the "
            "panel end: %s",
            len(stopped),
            max_gap / pd.Timedelta(days=1),
            ", ".join(map(str, stopped)),
        )
        prices = prices.drop(columns=stopped)
    # Trim to the date range where every ticker has data; portfolios need a
    # consistent shared history with no leading gaps.
    return prices.ffill().dropna()


class YFinanceClient:
    """Process-wide singleton facade over the yfinance API.

    Owns the shared cache, rate-limiter, and circuit-breaker; all sub-clients
    hold references to the same instances so their internal state is consistent.
    """

    _instance: "YFinanceClient | None" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        cache_size: int = DEFAULT_TICKER_CACHE_SIZE,
        cache_ttl: float = 3600.0,
        rate_limit_delay: float = 0.1,
        default_max_retries: int = 3,
    ) -> "YFinanceClient":
        """Return the singleton, constructing it on first call.

        Args:
            cache_size: Maximum number of ``yf.Ticker`` objects held in the LRU
                cache.
            cache_ttl: Seconds before a cached ticker entry expires.
            rate_limit_delay: Minimum seconds between successive ticker requests,
                shared by every symbol and worker.
            default_max_retries: Retry budget used when callers omit
                ``max_retries``.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache=LRUCache(
                            capacity=cache_size,
                            default_ttl=cache_ttl,
                        ),
                        rate_limiter=RateLimiter(delay=rate_limit_delay),
                        circuit_breaker=CircuitBreaker(),
                        default_max_retries=default_max_retries,
                    )
        return cls._instance

    # Exactly the registered @cached_property sub-client names — reset_instance
    # pops each from __dict__ to force re-creation against fresh infrastructure.
    _CACHED_SUBCLIENT_NAMES: tuple[str, ...] = (
        "financials",
        "analysis",
        "holders",
        "corporate_actions",
        "metadata",
        "search",
        "funds",
        "screener",
        "sectors",
        "calendars",
        "market",
    )

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton and evict all cached sub-clients.

        Intended for tests that need a fresh instance with different parameters.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cache.clear()
                for name in cls._CACHED_SUBCLIENT_NAMES:
                    cls._instance.__dict__.pop(name, None)
                cls._instance = None

    def __init__(
        self,
        cache: CacheProtocol,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        """Prefer ``get_instance`` for normal use; this is exposed for tests.

        Args:
            cache: Ticker object cache; all sub-clients share the same instance.
            rate_limiter: Spaces successive ticker requests across all symbols.
            circuit_breaker: Prevents cascading failures when Yahoo is degraded.
            default_max_retries: Retry budget used when callers omit
                ``max_retries``.
        """
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries
        install_fault_witness()

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Return a cached ``yf.Ticker``, creating one on cache miss.

        Fetch attempts call this just before they send their request, so the
        shared request delay is enforced here on cache hits and misses alike.

        Args:
            symbol: The ticker symbol to look up (e.g. ``"AAPL"``).
        """
        note_ticker(symbol)

        # Constructing a Ticker sends nothing, so pacing only the cache misses
        # would never space the requests that reuse a cached Ticker.
        self.rate_limiter.acquire(TICKER_REQUEST_KEY)
        ticker = self.cache.get(symbol)

        if ticker is not None:
            logger.debug("Cache hit for ticker '%s'", symbol)
            return ticker

        logger.debug("Cache miss for ticker '%s', fetching from yfinance", symbol)

        ticker = yf.Ticker(symbol)
        self.cache.put(symbol, ticker)

        return ticker

    def fetch_info(
        self,
        symbol: str,
        max_retries: int | None = None,
        min_fields: int = 10,
    ) -> dict[str, Any] | None:
        """Fetch the ``info`` dict for *symbol*, retrying on transient errors.

        Args:
            symbol: Ticker symbol.
            max_retries: Override the instance default retry budget.
            min_fields: Minimum number of non-empty keys required to accept the
                response as valid; guards against Yahoo returning a stub dict.

        Returns:
            The raw ``yf.Ticker.info`` dict, or ``None`` when Yahoo has no
            usable profile or every attempt failed.
        """
        max_retries = max_retries or self.default_max_retries
        logger.debug("Fetching info for '%s' (max_retries=%d)", symbol, max_retries)

        def _action() -> dict[str, Any] | None:
            return self.get_ticker(symbol).info

        result = fetch_with_fault_retry(
            _action,
            max_retries,
            cache=self.cache,
            circuit_breaker=self.circuit_breaker,
            rate_limit_predicate=self._is_rate_limit_error,
            is_valid=lambda info: info is not None and len(info) >= min_fields,
            on_attempt_failed=lambda: self.cache.delete(symbol),
        )

        if result is None:
            logger.debug("Failed to fetch info for '%s'", symbol)
            # yfinance keeps an empty answer, such as the {} it leaves when it
            # swallows an HTTP error, on the Ticker for the cache's lifetime.
            self.cache.delete(symbol)

        return result

    def fetch_history(
        self,
        symbol: str,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        max_retries: int | None = None,
        min_rows: int = 10,
        *,
        auto_adjust: bool = True,
        back_adjust: bool = False,
        repair: bool = True,
        actions: bool = False,
        prepost: bool = False,
        keepna: bool = False,
        rounding: bool = False,
        timeout: float | None = None,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV history for *symbol*, retrying on transient errors.

        Args:
            symbol: Ticker symbol.
            period: Lookback period string accepted by yfinance (e.g. ``"5y"``).
                Ignored when *start* is provided; with only *end*, it sets the
                length of the window that ends there.
            start: ISO date string for the start of the range (inclusive).
            end: ISO date string for the end of the range (exclusive).
            interval: Bar interval (e.g. ``"1d"``, ``"1h"``).
            max_retries: Override the instance default retry budget.
            min_rows: Minimum number of rows required to accept the response.
            auto_adjust: Adjust OHLC for splits/dividends via yfinance.
            back_adjust: Back-adjust prices (mutually exclusive with
                ``auto_adjust`` in practice).
            repair: Enable yfinance's bad-price reconstruction.  Requires
                scikit-learn at runtime; disabling it silently drops ~22% of
                tickers that trigger reconstruction.
            actions: Include dividend and split columns.
            prepost: Include pre/post-market data.
            keepna: Retain NaN rows rather than dropping them.
            rounding: Round prices to 2 decimal places.
            timeout: Per-request network timeout in seconds. ``None`` keeps
                yfinance's own finite default instead of disabling the timeout.

        Returns:
            DataFrame with a DatetimeIndex, or ``None`` on failure.
        """
        max_retries = max_retries or self.default_max_retries
        logger.debug("Fetching history for '%s' (max_retries=%d)", symbol, max_retries)

        history_kwargs: dict[str, Any] = {
            "interval": interval,
            "auto_adjust": auto_adjust,
            "back_adjust": back_adjust,
            "repair": repair,
            "actions": actions,
            "prepost": prepost,
            "keepna": keepna,
            "rounding": rounding,
        }
        # An explicit timeout=None would override yfinance's finite default and
        # leave a stalled connection waiting forever.
        if timeout is not None:
            history_kwargs["timeout"] = timeout

        def _action() -> pd.DataFrame | None:
            ticker = self.get_ticker(symbol)

            try:
                if start is not None:
                    return ticker.history(start=start, end=end, **history_kwargs)
                if end is not None:
                    return ticker.history(period=period, end=end, **history_kwargs)
                return ticker.history(period=period, **history_kwargs)
            finally:
                reapply_yfinance_log_level()

        result = fetch_with_fault_retry(
            _action,
            max_retries,
            cache=self.cache,
            circuit_breaker=self.circuit_breaker,
            rate_limit_predicate=self._is_rate_limit_error,
            is_valid=lambda hist: (
                hist is not None and not hist.empty and len(hist) >= min_rows
            ),
            on_attempt_failed=lambda: self.cache.delete(symbol),
        )

        if result is None:
            logger.debug("Failed to fetch history for '%s'", symbol)

        return result

    def fetch_price_and_benchmark(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "5y",
        max_retries: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
        """Fetch aligned price history and info for *symbol* and its benchmark.

        Args:
            symbol: Ticker symbol to fetch.
            benchmark: Symbol used as the comparison benchmark.
            period: Lookback period string accepted by yfinance (e.g. ``"5y"``).
            max_retries: Override the instance default retry budget.

        Returns:
            A three-tuple of (stock_history, benchmark_history, stock_info).
            Any element may be ``None`` when data is unavailable or the fetch
            failed.
        """
        max_retries = max_retries or self.default_max_retries

        stock_hist = self.fetch_history(symbol, period=period, max_retries=max_retries)
        stock_info = self.fetch_info(symbol, max_retries=max_retries)

        if stock_hist is None or stock_hist.empty:
            return None, None, None

        benchmark_hist = self.fetch_history(
            benchmark, period=period, max_retries=max_retries
        )

        if benchmark_hist is None or benchmark_hist.empty:
            return stock_hist, None, stock_info

        # yfinance may return tz-aware or tz-naive indices depending on the
        # exchange; normalising to date strings avoids spurious intersection
        # misses.
        try:
            stock_date_strs = [ts.strftime("%Y-%m-%d") for ts in stock_hist.index]
            bench_date_strs = [ts.strftime("%Y-%m-%d") for ts in benchmark_hist.index]

            common_date_strs = set(stock_date_strs) & set(bench_date_strs)

            if len(common_date_strs) > 0:
                stock_mask = pd.Series(stock_date_strs).isin(common_date_strs).values
                bench_mask = pd.Series(bench_date_strs).isin(common_date_strs).values

                stock_hist = cast("pd.DataFrame", stock_hist[stock_mask])
                benchmark_hist = cast("pd.DataFrame", benchmark_hist[bench_mask])
            else:
                benchmark_hist = None
        except Exception:
            # Benchmark alignment is optional: if it fails the stock history is
            # still returned with benchmark_hist=None and the caller proceeds
            # without a benchmark series, rather than failing the whole fetch.
            benchmark_hist = None

        return stock_hist, benchmark_hist, stock_info

    def bulk_download(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str = "5y",
        interval: str = "1d",
        threads: bool = True,
        group_by: str = "ticker",
        auto_adjust: bool = True,
        progress: bool = False,
        max_retries: int | None = None,
        *,
        repair: bool = True,
    ) -> pd.DataFrame | None:
        """Download price history for multiple symbols in a single yfinance call.

        Args:
            symbols: Ticker symbols to download.
            start: ISO date string for the start of the range (inclusive).
            end: ISO date string for the end of the range (exclusive).
            period: Lookback period when *start* is not provided; with only
                *end*, it sets the length of the window that ends there.
            interval: Bar interval (e.g. ``"1d"``).
            threads: Use threading inside yfinance for parallel downloads.
            group_by: Column grouping strategy in the returned DataFrame.
            auto_adjust: Adjust OHLC for splits/dividends.
            progress: Show a yfinance progress bar.
            max_retries: Override the instance default retry budget.
            repair: Enable yfinance's bad-price reconstruction, matching
                ``fetch_history`` so a bulk download and its per-ticker
                fallback return the same prices. yfinance itself defaults to
                off for downloads.

        Returns:
            A potentially MultiIndex DataFrame, or ``None`` on failure.
        """
        max_retries = max_retries or self.default_max_retries
        logger.info(
            "Bulk downloading %d symbols (max_retries=%d)",
            len(symbols),
            max_retries,
        )

        kwargs: dict[str, Any] = {
            "tickers": symbols,
            "interval": interval,
            "threads": threads,
            "group_by": group_by,
            "auto_adjust": auto_adjust,
            "progress": progress,
            "repair": repair,
        }
        if start is not None:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period
            if end is not None:
                kwargs["end"] = end

        def _action() -> pd.DataFrame | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("bulk_download")
            try:
                data = yf.download(**kwargs)
            finally:
                # Price repair lowers the shared yfinance log level without a
                # lock, in every download thread, just as it does for
                # fetch_history.
                reapply_yfinance_log_level()

            if data is None or (hasattr(data, "empty") and data.empty):
                return None
            return data

        result = retry_with_backoff(
            _action,
            max_retries,
            is_valid=lambda d: d is not None,
            is_rate_limit_error=self._is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

        if result is None:
            logger.warning("Bulk download failed for %d symbols", len(symbols))

        return result

    def fetch_prices_dataframe(
        self,
        symbols: list[str],
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Return an aligned Close-price DataFrame (DatetimeIndex rows, ticker columns).

        Suitable for feeding into ``skfolio.preprocessing.prices_to_returns()``.
        Tries bulk download first, falls back to individual fetches on failure.
        Tickers that stopped trading before the panel's end are omitted rather
        than carried forward at their last price.
        """
        logger.info(
            "Fetching prices DataFrame for %d symbols",
            len(symbols),
        )

        data = self.bulk_download(
            symbols=symbols,
            start=start,
            end=end,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
        )

        if data is not None:
            prices = self._extract_close_prices(data, symbols)
            if prices is not None:
                return prices

        logger.info(
            "Bulk download failed or incomplete, falling back to individual fetches"
        )
        return self._fetch_individual_prices(symbols, period, start, end, interval)

    def _extract_close_prices(
        self,
        data: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame | None:
        """Extract Close prices from a bulk-download MultiIndex DataFrame."""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                frames: dict[str, pd.Series] = {}
                for sym in symbols:
                    if sym in data.columns.get_level_values(0):
                        col = data[sym]["Close"]
                        if col is not None and not col.empty:
                            frames[sym] = cast("pd.Series", col)
                    elif ("Close", sym) in data.columns:
                        col = data["Close"][sym]
                        if col is not None and not col.empty:
                            frames[sym] = cast("pd.Series", col)
            else:
                # Single ticker — columns are just price fields
                if "Close" in data.columns:
                    frames = {symbols[0]: data["Close"]}
                else:
                    return None

            if not frames:
                return None

            prices = _align_close_prices(pd.DataFrame(frames))

            if prices.empty:
                logger.warning("Extracted prices DataFrame is empty after cleanup")
                return None

            return prices

        except Exception:
            # Bulk close-price extraction is best-effort: a malformed frame is
            # logged with traceback and degraded to None so the caller falls
            # back to per-ticker fetches rather than failing the batch.
            logger.warning(
                "Failed to extract close prices from bulk data", exc_info=True
            )
            return None

    def _fetch_individual_prices(
        self,
        symbols: list[str],
        period: str,
        start: str | None,
        end: str | None,
        interval: str,
    ) -> pd.DataFrame | None:
        """Fetch Close prices one ticker at a time and combine."""
        frames: dict[str, pd.Series] = {}

        for sym in symbols:
            hist = self.fetch_history(
                symbol=sym,
                period=period,
                start=start,
                end=end,
                interval=interval,
            )
            if hist is not None and not hist.empty and "Close" in hist.columns:
                frames[sym] = hist["Close"]
            else:
                logger.warning("No price data for '%s', skipping", sym)

        if not frames:
            logger.warning("No price data retrieved for any symbol")
            return None

        prices = _align_close_prices(pd.DataFrame(frames))

        if prices.empty:
            logger.warning("Individual prices DataFrame is empty after cleanup")
            return None

        logger.info(
            "Built prices DataFrame: %d rows x %d columns",
            len(prices),
            len(prices.columns),
        )
        return prices

    def get_cache_stats(self) -> dict[str, Any]:
        """Return a snapshot of cache and retry configuration for observability."""
        return {
            "size": self.cache.size(),
            "capacity": getattr(self.cache, "capacity", "unknown"),
            "default_max_retries": self.default_max_retries,
        }

    def clear_cache(self) -> None:
        self.cache.clear()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        return is_rate_limit_error(error)

    # ------------------------------------------------------------------
    # Sub-client facade properties (lazy-initialized)
    # ------------------------------------------------------------------

    def _ticker_sub_client_kwargs(self) -> dict[str, Any]:
        return {
            "cache": self.cache,
            "rate_limiter": self.rate_limiter,
            "circuit_breaker": self.circuit_breaker,
            "default_max_retries": self.default_max_retries,
        }

    def _module_sub_client_kwargs(self) -> dict[str, Any]:
        return {
            "rate_limiter": self.rate_limiter,
            "circuit_breaker": self.circuit_breaker,
            "default_max_retries": self.default_max_retries,
        }

    @cached_property
    def financials(self) -> FinancialsClient:
        return FinancialsClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def analysis(self) -> AnalysisClient:
        return AnalysisClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def holders(self) -> HoldersClient:
        return HoldersClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def corporate_actions(self) -> CorporateActionsClient:
        return CorporateActionsClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def metadata(self) -> MetadataClient:
        return MetadataClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def search(self) -> SearchClient:
        return SearchClient(**self._module_sub_client_kwargs())

    @cached_property
    def funds(self) -> FundsClient:
        return FundsClient(**self._ticker_sub_client_kwargs())

    @cached_property
    def screener(self) -> ScreenerClient:
        return ScreenerClient(**self._module_sub_client_kwargs())

    @cached_property
    def sectors(self) -> SectorsClient:
        return SectorsClient(**self._module_sub_client_kwargs())

    @cached_property
    def calendars(self) -> CalendarsClient:
        return CalendarsClient(**self._module_sub_client_kwargs())

    @cached_property
    def market(self) -> MarketClient:
        return MarketClient(**self._module_sub_client_kwargs())


def get_yfinance_client(
    cache_size: int = DEFAULT_TICKER_CACHE_SIZE,
    cache_ttl: float = 3600.0,
    rate_limit_delay: float = 0.1,
    default_max_retries: int = 3,
) -> YFinanceClient:
    """Return the process-wide singleton ``YFinanceClient``.

    Convenience wrapper over ``YFinanceClient.get_instance``.

    Args:
        cache_size: Maximum number of cached ticker objects.
        cache_ttl: Seconds before a cached ticker expires.
        rate_limit_delay: Minimum seconds between successive ticker requests,
            shared by every symbol and worker.
        default_max_retries: Retry budget for callers that omit ``max_retries``.
    """
    return YFinanceClient.get_instance(
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        rate_limit_delay=rate_limit_delay,
        default_max_retries=default_max_retries,
    )
