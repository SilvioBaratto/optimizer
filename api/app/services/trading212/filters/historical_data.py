import time
from dataclasses import dataclass, field
from typing import Any

from app.services.trading212.config import UniverseBuilderConfig
from app.services.yfinance import YFinanceClient


@dataclass
class HistoricalDataFilter:
    config: UniverseBuilderConfig
    _yf_client: YFinanceClient | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "HistoricalDataFilter"

    @property
    def yf_client(self) -> YFinanceClient:
        if self._yf_client is None:
            self._yf_client = YFinanceClient.get_instance()
        return self._yf_client

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
        # data parameter unused but required by protocol
        _ = data

        has_coverage, days_available = self._check_historical_data(yf_ticker)

        if not has_coverage:
            years = days_available / 252 if days_available > 0 else 0
            return (
                False,
                f"Insufficient history: {days_available} days ({years:.1f}y) from period='5y'",
            )

        years = days_available / 252
        return True, f"Historical data: {days_available} days ({years:.1f}y)"

    def _check_historical_data(
        self, yf_ticker: str, max_retries: int = 3
    ) -> tuple[bool, int]:
        for attempt in range(max_retries):
            try:
                # Fetch 5 years of historical data
                hist = self.yf_client.fetch_history(
                    yf_ticker,
                    period="5y",
                    max_retries=1,  # We handle retries ourselves
                    min_rows=1,  # Accept any amount initially
                )

                if hist is None or hist.empty:
                    return False, 0

                days_available = len(hist)

                # Sanity check: Just ensure we got a reasonable amount of data
                # (at least ~3 years worth = 750 trading days)
                # If yfinance returned data for period="5y", trust it's approximately correct
                has_coverage = days_available >= self.config.min_trading_days

                return has_coverage, days_available

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting and timeouts
                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 10
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, 0
                else:
                    # Other errors
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        return False, 0

        return False, 0
