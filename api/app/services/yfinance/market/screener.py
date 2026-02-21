"""Sub-client for screening via ``yf.screen``, ``EquityQuery``, ``FundQuery``."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)


class ScreenerClient:
    """Wraps ``yf.screen()`` and predefined screener queries."""

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
        offset: int = 0,
        size: int = 25,
        sort_field: str = "ticker",
        sort_asc: bool = True,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Running screener (offset=%d, size=%d)", offset, size)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> pd.DataFrame | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire("screener")
            result = yf.screen(
                query,
                offset=offset,
                size=size,
                sortField=sort_field,
                sortAsc=sort_asc,
            )
            return result

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def get_predefined_screeners(self) -> dict[str, Any]:
        return yf.PREDEFINED_SCREENER_QUERIES
