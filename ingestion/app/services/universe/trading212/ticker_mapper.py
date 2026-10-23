import logging
from dataclasses import dataclass, field

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.config import UniverseBuilderConfig

logger = logging.getLogger(__name__)


@dataclass
class YFinanceTickerMapper:
    """Maps Trading 212 symbols to verified yfinance tickers.

    Resolves the exchange-suffixed form (e.g. ``AAPL.L``); only venues whose
    Yahoo suffix is empty, or that have no suffix mapping, use the bare
    symbol. Uses the in-memory ``TickerMappingCache`` to avoid redundant
    yfinance round-trips within a single universe build job.
    """

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
        """Return the verified yfinance ticker for a T212 symbol, or ``None``.

        Checks the cache before issuing any yfinance request. On a miss,
        verifies the ticker built from the symbol and the exchange's Yahoo
        suffix. A verified ticker is written back to the cache for the
        remainder of the build.

        Args:
            symbol: The T212 ``shortName`` (e.g. ``AAPL_US_EQ``).
            exchange_name: T212 exchange name used to look up the Yahoo suffix.

        Returns:
            A yfinance ticker string with live price data, or ``None`` when
            no matching ticker can be verified.
        """
        try:
            if exchange_name and self.cache:
                cached = self.cache.get_mapping(symbol, exchange_name)
                if cached and self._verify_ticker(cached):
                    return cached

            # Yahoo Finance uses dashes instead of slashes for share classes
            clean_symbol = symbol.replace("/", "-")

            ticker_attempts = self._build_ticker_attempts(clean_symbol, exchange_name)

            for attempt_ticker in ticker_attempts:
                if self._verify_ticker(attempt_ticker):
                    if exchange_name and self.cache:
                        self.cache.save_mapping(symbol, exchange_name, attempt_ticker)
                    return attempt_ticker

            return None

        except Exception:
            # Best-effort per-ticker discovery inside a bulk universe build:
            # an unexpected failure degrades to "no mapping" and the row is
            # skipped, never fatal.
            return None

    def _build_ticker_attempts(
        self, clean_symbol: str, exchange_name: str | None
    ) -> list[str]:
        suffix = self.config.get_yahoo_suffix(exchange_name) if exchange_name else None

        # Verification only checks that Yahoo returns a price, so a bare-symbol
        # fallback on a suffixed venue would accept an unrelated security that
        # shares the short symbol (BA.L is BAE Systems, bare BA is Boeing in USD).
        if suffix:
            return [clean_symbol + suffix]

        return [clean_symbol]

    def _verify_ticker(self, ticker: str) -> bool:
        """Report whether Yahoo returns a live price for *ticker*.

        ``fetch_info`` swallows every request error and returns ``None``, so a
        throttled call is told apart from a no-data answer by the shared
        circuit breaker it trips. While throttled the call is repeated; the
        breaker blocks the next ``fetch_info`` until its cooldown ends, so no
        sleep is needed here. A no-data answer is final.
        """
        for _ in range(self.max_retries):
            info = self.yf_client.fetch_info(ticker, max_retries=1, min_fields=5)

            if info is not None:
                return bool(
                    len(info) > 5
                    and ("currentPrice" in info or "regularMarketPrice" in info)
                )

            if not self.yf_client.circuit_breaker.is_active:
                return False

        logger.warning(
            "yfinance stayed throttled while verifying %s; treating it as unmapped",
            ticker,
        )
        return False
