"""Base client providing shared ticker caching and resilience infrastructure.

All ticker-based sub-clients extend this class to share a single cache,
rate limiter, and circuit breaker instance with the main ``YFinanceClient``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import yfinance as yf

from .infrastructure import is_no_data_error, is_rate_limit_error, retry_with_backoff
from .protocols import CacheProtocol, CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


def fetch_with_resilience(
    action: Callable[[], T],
    retries: int,
    *,
    circuit_breaker: CircuitBreakerProtocol,
    rate_limit_predicate: Callable[[Exception], bool] = is_rate_limit_error,
    is_valid: Callable[[T], bool] | None = None,
) -> T | None:
    """Run one yfinance request, retrying transient failures only.

    Yahoo's final answers end the call at once: a no-data exception, or a
    response that *is_valid* rejects, returns ``None`` without another
    request. yfinance already turns a 404 for a data module the symbol lacks
    into an empty result, and network faults surface as exceptions that are
    retried, so an empty result is Yahoo's answer rather than a lost request.

    Args:
        action: Zero-argument callable performing one yfinance request.
        retries: Attempt budget, including the first attempt.
        circuit_breaker: Breaker checked before each attempt and tripped by
            rate limits.
        rate_limit_predicate: Recognises rate-limit errors that should trip
            the breaker.
        is_valid: Predicate a usable result must satisfy; a result it rejects
            (typically an empty frame) is returned as ``None``.

    Returns:
        The result of *action*, or ``None`` when Yahoo has no usable data or
        every attempt failed.
    """

    def _guarded() -> T:
        circuit_breaker.check()
        return action()

    try:
        result = retry_with_backoff(
            _guarded,
            retries,
            is_rate_limit_error=rate_limit_predicate,
            on_rate_limit=circuit_breaker.trigger,
            on_success=lambda _: circuit_breaker.reset(),
            is_permanent_error=is_no_data_error,
        )
    except Exception as exc:
        logger.debug("Yahoo has no data for this request: %s", exc)
        return None

    if result is not None and is_valid is not None and not is_valid(result):
        return None
    return result


class BaseClient:
    """Shared infrastructure for ticker-based sub-clients."""

    def __init__(
        self,
        cache: CacheProtocol,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        """Initialize the base client with shared resilience infrastructure.

        Args:
            cache: Ticker object cache shared across all sub-clients.
            rate_limiter: Throttles per-symbol request frequency.
            circuit_breaker: Prevents cascading failures when Yahoo is degraded.
            default_max_retries: Retry budget used when callers omit
                ``max_retries``.
        """
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
        """Execute *action* through :func:`fetch_with_resilience`.

        Args:
            action: Zero-argument callable performing one yfinance request.
            max_retries: Attempt budget; ``None`` uses the client default.
            is_valid: Predicate a usable result must satisfy; a result it
                rejects (typically an empty frame) is returned as ``None``.

        Returns:
            The result of *action*, or ``None`` when Yahoo has no usable data
            or every attempt failed.
        """
        return fetch_with_resilience(
            action,
            max_retries if max_retries is not None else self.default_max_retries,
            circuit_breaker=self.circuit_breaker,
            rate_limit_predicate=self._is_rate_limit_error,
            is_valid=is_valid,
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        return is_rate_limit_error(error)
