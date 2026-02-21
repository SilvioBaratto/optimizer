import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from .infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    is_rate_limit_error,
    retry_with_backoff,
)
from .market import (
    AsyncStreamingClient,
    CalendarsClient,
    MarketClient,
    ScreenerClient,
    SearchClient,
    SectorIndustryClient,
    StreamingClient,
)
from .protocols import CacheProtocol, CircuitBreakerProtocol, RateLimiterProtocol
from .ticker import (
    AnalysisClient,
    CorporateActionsClient,
    FinancialsClient,
    FundsClient,
    HoldersClient,
    MetadataClient,
)

logger = logging.getLogger(__name__)

# Load environment variables from project root .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class YFinanceClient:
    _instance: "YFinanceClient | None" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        cache_size: int = 3000,
        cache_ttl: float = 3600.0,
        rate_limit_delay: float = 0.1,
        default_max_retries: int = 3,
    ) -> "YFinanceClient":
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

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cache.clear()
                cls._instance = None

    def __init__(
        self,
        cache: CacheProtocol,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def get_ticker(self, symbol: str) -> yf.Ticker:
        ticker = self.cache.get(symbol)

        if ticker is not None:
            logger.debug("Cache hit for ticker '%s'", symbol)
            return ticker

        logger.debug("Cache miss for ticker '%s', fetching from yfinance", symbol)
        self.rate_limiter.acquire(symbol)

        ticker = yf.Ticker(symbol)
        self.cache.put(symbol, ticker)

        return ticker

    def fetch_info(
        self,
        symbol: str,
        max_retries: int | None = None,
        min_fields: int = 10,
    ) -> dict[str, Any] | None:
        max_retries = max_retries or self.default_max_retries
        logger.debug("Fetching info for '%s' (max_retries=%d)", symbol, max_retries)

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            ticker = self.get_ticker(symbol)
            return ticker.info

        result = retry_with_backoff(
            _action,
            max_retries,
            is_valid=lambda info: info is not None and len(info) >= min_fields,
            is_rate_limit_error=self._is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

        if result is None:
            logger.debug("Failed to fetch info for '%s'", symbol)

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
    ) -> pd.DataFrame | None:
        max_retries = max_retries or self.default_max_retries
        logger.debug("Fetching history for '%s' (max_retries=%d)", symbol, max_retries)

        def _action() -> pd.DataFrame | None:
            self.circuit_breaker.check()
            ticker = self.get_ticker(symbol)

            if start is not None:
                return ticker.history(start=start, end=end, interval=interval)
            return ticker.history(period=period, interval=interval)

        result = retry_with_backoff(
            _action,
            max_retries,
            is_valid=lambda hist: hist is not None
            and not hist.empty
            and len(hist) >= min_rows,
            is_rate_limit_error=self._is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
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

        # Align dates (timezone-agnostic matching)
        try:
            stock_date_strs = [ts.strftime("%Y-%m-%d") for ts in stock_hist.index]
            bench_date_strs = [ts.strftime("%Y-%m-%d") for ts in benchmark_hist.index]

            common_date_strs = set(stock_date_strs) & set(bench_date_strs)

            if len(common_date_strs) > 0:
                stock_mask = pd.Series(stock_date_strs).isin(common_date_strs).values
                bench_mask = pd.Series(bench_date_strs).isin(common_date_strs).values

                stock_hist = stock_hist[stock_mask]
                benchmark_hist = benchmark_hist[bench_mask]
            else:
                benchmark_hist = None
        except Exception:
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
        auto_adjust: bool = False,
        progress: bool = False,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
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
        }
        if start is not None:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period

        def _action() -> pd.DataFrame | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("bulk_download")
            data = yf.download(**kwargs)

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
                            frames[sym] = col
                    elif ("Close", sym) in data.columns:
                        col = data["Close"][sym]
                        if col is not None and not col.empty:
                            frames[sym] = col
            else:
                # Single ticker — columns are just price fields
                if "Close" in data.columns:
                    frames = {symbols[0]: data["Close"]}
                else:
                    return None

            if not frames:
                return None

            prices = pd.DataFrame(frames)
            prices = prices.ffill().dropna(how="all")
            # Drop leading rows where any column is NaN
            prices = prices.dropna()

            if prices.empty:
                logger.warning("Extracted prices DataFrame is empty after cleanup")
                return None

            return prices

        except Exception:
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

        prices = pd.DataFrame(frames)
        prices = prices.ffill().dropna(how="all").dropna()

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

    @property
    def financials(self) -> FinancialsClient:
        if not hasattr(self, "_financials"):
            self._financials = FinancialsClient(**self._ticker_sub_client_kwargs())
        return self._financials

    @property
    def analysis(self) -> AnalysisClient:
        if not hasattr(self, "_analysis"):
            self._analysis = AnalysisClient(**self._ticker_sub_client_kwargs())
        return self._analysis

    @property
    def holders(self) -> HoldersClient:
        if not hasattr(self, "_holders"):
            self._holders = HoldersClient(**self._ticker_sub_client_kwargs())
        return self._holders

    @property
    def corporate_actions(self) -> CorporateActionsClient:
        if not hasattr(self, "_corporate_actions"):
            self._corporate_actions = CorporateActionsClient(
                **self._ticker_sub_client_kwargs()
            )
        return self._corporate_actions

    @property
    def metadata(self) -> MetadataClient:
        if not hasattr(self, "_metadata"):
            self._metadata = MetadataClient(**self._ticker_sub_client_kwargs())
        return self._metadata

    @property
    def funds(self) -> FundsClient:
        if not hasattr(self, "_funds"):
            self._funds = FundsClient(**self._ticker_sub_client_kwargs())
        return self._funds

    @property
    def market(self) -> MarketClient:
        if not hasattr(self, "_market"):
            self._market = MarketClient(**self._module_sub_client_kwargs())
        return self._market

    @property
    def sectors(self) -> SectorIndustryClient:
        if not hasattr(self, "_sectors"):
            self._sectors = SectorIndustryClient(**self._module_sub_client_kwargs())
        return self._sectors

    @property
    def search(self) -> SearchClient:
        if not hasattr(self, "_search"):
            self._search = SearchClient(**self._module_sub_client_kwargs())
        return self._search

    @property
    def screener(self) -> ScreenerClient:
        if not hasattr(self, "_screener"):
            self._screener = ScreenerClient(**self._module_sub_client_kwargs())
        return self._screener

    @property
    def calendars(self) -> CalendarsClient:
        if not hasattr(self, "_calendars"):
            self._calendars = CalendarsClient(**self._module_sub_client_kwargs())
        return self._calendars

    @property
    def streaming(self) -> StreamingClient:
        if not hasattr(self, "_streaming"):
            self._streaming = StreamingClient()
        return self._streaming

    @property
    def async_streaming(self) -> AsyncStreamingClient:
        if not hasattr(self, "_async_streaming"):
            self._async_streaming = AsyncStreamingClient()
        return self._async_streaming


def get_yfinance_client(
    cache_size: int = 3000,
    cache_ttl: float = 3600.0,
    rate_limit_delay: float = 0.1,
    default_max_retries: int = 3,
) -> YFinanceClient:
    return YFinanceClient.get_instance(
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        rate_limit_delay=rate_limit_delay,
        default_max_retries=default_max_retries,
    )
