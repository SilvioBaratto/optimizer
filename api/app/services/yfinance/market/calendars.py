"""Sub-client for market calendars: earnings, IPO, splits, economic events."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol
from ..infrastructure import is_rate_limit_error, retry_with_backoff

logger = logging.getLogger(__name__)


class CalendarsClient:
    """Wraps ``yf.Calendars`` for market-wide calendar data."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def fetch_earnings_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching earnings calendar (start=%s, end=%s)", start, end)
        return self._fetch_calendar_attr("earnings_calendar", start, end, max_retries)

    def fetch_ipo_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching IPO calendar (start=%s, end=%s)", start, end)
        return self._fetch_calendar_attr("ipo_info_calendar", start, end, max_retries)

    def fetch_splits_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching splits calendar (start=%s, end=%s)", start, end)
        return self._fetch_calendar_attr("splits_calendar", start, end, max_retries)

    def fetch_economic_events_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching economic events calendar (start=%s, end=%s)", start, end)
        return self._fetch_calendar_attr("economic_events_calendar", start, end, max_retries)

    def _fetch_calendar_attr(
        self,
        attr: str,
        start: str | None,
        end: str | None,
        max_retries: int | None,
    ) -> pd.DataFrame | None:
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> Any:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"calendar_{attr}")
            cal = yf.Calendars(start=start, end=end)
            return getattr(cal, attr, None)

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
