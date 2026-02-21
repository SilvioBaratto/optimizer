import time
from dataclasses import dataclass, field

from app.services.trading212.cache.ticker_cache import TickerMappingCache
from app.services.trading212.config import UniverseBuilderConfig
from app.services.yfinance import YFinanceClient


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
                if cached:
                    if self._verify_ticker(cached):
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

                if info and len(info) > 5:
                    if "currentPrice" in info or "regularMarketPrice" in info:
                        return True

                return False

            except Exception as e:
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

    def fetch_basic_data(self, yf_ticker: str, max_retries: int = 3) -> dict | None:
        for attempt in range(max_retries):
            try:
                info = self.yf_client.fetch_info(
                    yf_ticker, max_retries=1, min_fields=10
                )

                if not info or len(info) < 10:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    return None

                return {
                    "marketCap": info.get("marketCap"),
                    "currentPrice": info.get("currentPrice"),
                    "regularMarketPrice": info.get("regularMarketPrice"),
                    "volume": info.get("volume"),
                    "averageVolume": info.get("averageVolume"),
                    "averageVolume10days": info.get("averageVolume10days"),
                    "sharesOutstanding": info.get("sharesOutstanding"),
                    "beta": info.get("beta"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "exchange": info.get("exchange"),
                    "trailingPE": info.get("trailingPE"),
                    "priceToBook": info.get("priceToBook"),
                    "priceToSalesTrailing12Months": info.get(
                        "priceToSalesTrailing12Months"
                    ),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "profitMargins": info.get("profitMargins"),
                    "operatingMargins": info.get("operatingMargins"),
                    "grossMargins": info.get("grossMargins"),
                    "debtToEquity": info.get("debtToEquity"),
                    "totalDebt": info.get("totalDebt"),
                    "totalAssets": info.get("totalAssets"),
                    "currentRatio": info.get("currentRatio"),
                    "dividendYield": info.get("dividendYield"),
                    "dividendRate": info.get("dividendRate"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                    "operatingCashflow": info.get("operatingCashflow"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsGrowth"),
                    "52WeekChange": info.get("52WeekChange"),
                }

            except Exception as e:
                error_str = str(e).lower()

                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 10
                        time.sleep(wait_time)
                        continue
                    return None
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return None

        return None
