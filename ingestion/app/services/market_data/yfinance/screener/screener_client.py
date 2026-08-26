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
        sort_field: str | None = None,
        sort_asc: bool = True,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run one screen. ``query`` is an ``EquityQuery``/``FundQuery``/``ETFQuery``
        or a predefined-screen name. ``size`` is capped at 250; page with ``offset``."""
        retries = max_retries if max_retries is not None else self.default_max_retries
        capped = min(size, MAX_SIZE)

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("screener")
            return yf.screen(
                query,
                offset=offset,
                size=capped,
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
        size: int = 25,
        offset: int = 0,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run a predefined screen by name (see ``yf.PREDEFINED_SCREENER_QUERIES``)."""
        if name not in yf.PREDEFINED_SCREENER_QUERIES:
            logger.error(
                "Unknown predefined screen '%s'. Valid: %s",
                name,
                list(yf.PREDEFINED_SCREENER_QUERIES),
            )
            return None
        return self.screen(name, size=size, offset=offset, max_retries=max_retries)
