"""Sub-client for market status and summary via ``yf.Market``."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol
from ..infrastructure import is_rate_limit_error, retry_with_backoff

logger = logging.getLogger(__name__)


class MarketClient:
    """Wraps ``yf.Market`` for market-level queries."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def fetch_status(
        self,
        market: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching market status for '%s'", market)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"market_{market}")
            m = yf.Market(market)
            return m.status

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def fetch_summary(
        self,
        market: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching market summary for '%s'", market)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"market_{market}")
            m = yf.Market(market)
            return m.summary

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
