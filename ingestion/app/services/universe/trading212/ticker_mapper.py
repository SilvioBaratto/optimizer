import time
from dataclasses import dataclass, field

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.config import UniverseBuilderConfig


@dataclass
class YFinanceTickerMapper:
    config: UniverseBuilderConfig
    cache: TickerMappingCache | None = None
    _yf_client: YFinanceClient | None = field(default=None, repr=False)
    max_retries: int = 5

    def __post_init__(self):
        if self.cache is None:
            self.cache = TickerMappingCache()

    @property
    def yf_client(self) -> YFinanceClient:
        if self._yf_client is None:
            self._yf_client = YFinanceClient.get_instance()
        return self._yf_client

    def discover(self, symbol: str, exchange_name: str | None = None) -> str | None:
        try:
            # Check cache first
            if exchange_name and self.cache:
                cached = self.cache.get_mapping(symbol, exchange_name)
                if cached and self._verify_ticker(cached):
                    return cached

            # Yahoo Finance uses dashes instead of slashes for share classes
            clean_symbol = symbol.replace("/", "-")

            # Build list of tickers to try
            ticker_attempts = self._build_ticker_attempts(clean_symbol, exchange_name)

            # Try each ticker
            for attempt_ticker in ticker_attempts:
                if self._verify_ticker(attempt_ticker):
                    if exchange_name and self.cache:
                        self.cache.save_mapping(symbol, exchange_name, attempt_ticker)
                    return attempt_ticker

            return None

        except Exception:
            # Best-effort per-ticker discovery inside a bulk universe build:
            # _verify_ticker already classifies known yfinance errors, so an
            # unexpected failure degrades to "no mapping" and the row is
            # skipped (surfaced via BuildResult.errors), never fatal.
            return None

    def _build_ticker_attempts(
        self, clean_symbol: str, exchange_name: str | None
    ) -> list[str]:
        attempts = []

        if exchange_name:
            suffix = self.config.get_yahoo_suffix(exchange_name)
            if suffix is not None:
                preferred_ticker = clean_symbol + suffix
                attempts.append(preferred_ticker)

        if clean_symbol not in attempts:
            attempts.append(clean_symbol)

        return attempts

    def _verify_ticker(self, ticker: str) -> bool:
        for retry in range(self.max_retries):
            try:
                info = self.yf_client.fetch_info(ticker, max_retries=1, min_fields=5)

                return bool(
                    info
                    and len(info) > 5
                    and ("currentPrice" in info or "regularMarketPrice" in info)
                )

            except Exception as e:
                # Per-ticker verification in a bulk build: the error string is
                # classified (rate-limit → backoff/retry, not-found → reject) and
                # the ticker degrades to unverified. No raise — one bad ticker
                # must not abort the universe build.
                error_str = str(e).lower()

                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if retry < self.max_retries - 1:
                        wait_times = [60, 300, 900, 1800, 3600]
                        wait_time = (
                            wait_times[retry] if retry < len(wait_times) else 3600
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        return False

                if any(
                    x in error_str
                    for x in ["not found", "404", "invalid crumb", "unauthorized"]
                ):
                    return False

                return False

        return False
