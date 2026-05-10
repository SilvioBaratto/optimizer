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

_SEARCH_EXTRA_ATTRS: tuple[str, ...] = ("lists", "research", "nav")


def _collect_extras(search_obj: Any, flags: dict[str, bool]) -> dict[str, Any]:
    """Return enabled-extras keyed by attribute, ``None`` when absent."""
    return {
        attr: getattr(search_obj, attr, None)
        for attr in _SEARCH_EXTRA_ATTRS
        if flags[attr]
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
        news_count: int = 5,
        include_lists: bool = False,
        include_research: bool = False,
        include_nav: bool = False,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run a Yahoo search and return ``quotes``/``news`` plus opt-in extras.

        ``max_results`` controls the quote count; ``news_count`` controls the
        news count (forwarded to ``yf.Search``). ``include_lists`` /
        ``include_research`` / ``include_nav`` toggle inclusion of the
        corresponding key in the returned dict (value via ``getattr`` so a
        missing attribute resolves to ``None`` rather than raising).
        """
        logger.debug(
            "Searching for '%s' (max_results=%d, news_count=%d, "
            "include_lists=%s, include_research=%s, include_nav=%s)",
            query,
            max_results,
            news_count,
            include_lists,
            include_research,
            include_nav,
        )
        retries = max_retries if max_retries is not None else self.default_max_retries

        flags = {
            "lists": include_lists,
            "research": include_research,
            "nav": include_nav,
        }

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("search")
            s = yf.Search(query, max_results=max_results, news_count=news_count)
            return {"quotes": s.quotes, "news": s.news, **_collect_extras(s, flags)}

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
