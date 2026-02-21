"""Sub-client for search and lookup via ``yf.Search`` and ``yf.Lookup``."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

_LOOKUP_ASSET_TYPES = {
    "stock": "get_stock",
    "etf": "get_etf",
    "mutualfund": "get_mutualfund",
    "index": "get_index",
    "future": "get_future",
    "currency": "get_currency",
    "cryptocurrency": "get_cryptocurrency",
}


class SearchClient:
    """Wraps ``yf.Search`` and ``yf.Lookup``."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def search(
        self,
        query: str,
        max_results: int = 8,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Searching for '%s' (max_results=%d)", query, max_results)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("search")
            s = yf.Search(query, max_results=max_results)
            return {
                "quotes": s.quotes,
                "news": s.news,
            }

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def lookup(
        self,
        query: str,
        asset_type: str = "stock",
        count: int = 25,
        max_retries: int | None = None,
    ) -> list[dict[str, Any]] | None:
        logger.debug("Lookup '%s' (type=%s, count=%d)", query, asset_type, count)
        retries = max_retries if max_retries is not None else self.default_max_retries

        method_name = _LOOKUP_ASSET_TYPES.get(asset_type)
        if method_name is None:
            logger.error(
                "Unknown asset_type '%s'. Valid: %s",
                asset_type,
                list(_LOOKUP_ASSET_TYPES),
            )
            return None

        def _action() -> list[dict[str, Any]] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("lookup")
            lk = yf.Lookup(query)
            method = getattr(lk, method_name)
            return method(count=count)

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
