"""Sub-client for equity/fund/ETF screening via yfinance's ``yf.screen``.

yfinance 1.6.0 exposes screening as the module function ``yf.screen(query, ...)``
plus the ``EquityQuery`` / ``FundQuery`` / ``ETFQuery`` builders and the
``PREDEFINED_SCREENER_QUERIES`` catalogue — there is no ``yf.Screener`` class.
Results are capped at 250 per call; paginate with ``offset``.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

MAX_SIZE = 250


class ScreenerClient:
    """Wraps ``yf.screen`` with the shared resilience infrastructure."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def screen(
        self,
        query: Any,
        *,
        size: int = 25,
        offset: int = 0,
        count: int | None = None,
        sort_field: str | None = None,
        sort_asc: bool = True,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run one screen. ``query`` is an ``EquityQuery``/``FundQuery``/``ETFQuery``
        or a predefined-screen name.

        yfinance sizes **custom** ``*Query`` objects via ``size=`` (default 100, max 250)
        and **predefined** name strings via ``count=`` (default 25, max 250). Pass
        ``count`` for a predefined name (see :meth:`screen_predefined`); otherwise ``size``
        is used. Page with ``offset``.
        """
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("screener")
            if count is not None:
                return yf.screen(
                    query,
                    offset=offset,
                    count=min(count, MAX_SIZE),
                    sortField=sort_field,
                    sortAsc=sort_asc,
                )
            return yf.screen(
                query,
                offset=offset,
                size=min(size, MAX_SIZE),
                sortField=sort_field,
                sortAsc=sort_asc,
            )

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def screen_predefined(
        self,
        name: str,
        *,
        count: int = 25,
        offset: int = 0,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run a predefined screen by name (see ``yf.PREDEFINED_SCREENER_QUERIES``).

        Predefined screens are sized via ``count`` (default 25, max 250), not ``size``.
        """
        if name not in yf.PREDEFINED_SCREENER_QUERIES:
            logger.error(
                "Unknown predefined screen '%s'. Valid: %s",
                name,
                list(yf.PREDEFINED_SCREENER_QUERIES),
            )
            return None
        return self.screen(name, count=count, offset=offset, max_retries=max_retries)
