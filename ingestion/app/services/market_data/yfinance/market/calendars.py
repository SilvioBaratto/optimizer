"""Module-level client for market-wide calendars via ``yf.Calendars``.

Wraps the four calendar rollups (earnings / IPO / splits / economic events).
Each fetch returns a list of record dicts (the DataFrame rows) or ``None`` so
the service/repo can map them defensively without importing yfinance.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)


def _records(value: Any) -> list[dict[str, Any]]:
    """A ``yf.Calendars`` property → list of record dicts (index kept as a col)."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.reset_index().to_dict("records")
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    return []


class CalendarsClient:
    """Wraps ``yf.Calendars`` (earnings / IPO / splits / economic events)."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def _fetch(self, attr: str, max_retries: int | None) -> list[dict[str, Any]] | None:
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"calendars:{attr}")
            cal = yf.Calendars()
            return _records(getattr(cal, attr, None))

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def fetch_earnings(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._fetch("earnings_calendar", max_retries)

    def fetch_ipos(self, max_retries: int | None = None) -> list[dict[str, Any]] | None:
        return self._fetch("ipo_info_calendar", max_retries)

    def fetch_splits(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._fetch("splits_calendar", max_retries)

    def fetch_economic_events(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._fetch("economic_events_calendar", max_retries)
