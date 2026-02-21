"""Base client providing shared ticker caching and resilience infrastructure.

All ticker-based sub-clients extend this class to share a single cache,
rate limiter, and circuit breaker instance with the main ``YFinanceClient``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import yfinance as yf

from .infrastructure import is_rate_limit_error, retry_with_backoff
from .protocols import CacheProtocol, CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseClient:
    """Shared infrastructure for ticker-based sub-clients."""

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

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Return a cached ``yf.Ticker``, creating one on cache miss."""
        ticker = self.cache.get(symbol)
        if ticker is not None:
            logger.debug("Cache hit for ticker '%s'", symbol)
            return ticker

        logger.debug("Cache miss for ticker '%s', fetching from yfinance", symbol)
        self.rate_limiter.acquire(symbol)
        ticker = yf.Ticker(symbol)
        self.cache.put(symbol, ticker)
        return ticker

    def _fetch_with_resilience(
        self,
        action: Callable[[], T],
        max_retries: int | None = None,
        *,
        is_valid: Callable[[T], bool] | None = None,
    ) -> T | None:
        """Execute *action* with retry, circuit-breaker, and rate-limit handling."""
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _guarded() -> T:
            self.circuit_breaker.check()
            return action()

        return retry_with_backoff(
            _guarded,
            retries,
            is_valid=is_valid,
            is_rate_limit_error=self._is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        return is_rate_limit_error(error)
