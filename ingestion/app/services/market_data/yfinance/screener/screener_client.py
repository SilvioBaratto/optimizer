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

from ..infrastructure import is_no_data_error, is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

MAX_SIZE = 250


class ScreenerError(RuntimeError):
    """A screen request failed for good, so its result must not be read as empty.

    Callers that page a ranked query treat an empty page as the end of the
    results; raising keeps a transient Yahoo failure from passing for that.
    """


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
        sort_asc: bool | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Run one screen with a custom or predefined query.

        yfinance sizes custom ``*Query`` objects via ``size`` (max 250) and predefined
        name strings via ``count`` (max 250). Passing ``count`` signals a predefined
        name; otherwise ``size`` applies. Page results with ``offset``.

        A predefined name's first page (``offset`` 0) goes to Yahoo's predefined
        endpoint, which honors ``count`` and the screen's own ranking. Later pages
        must use the offset endpoint, which sizes pages by ``size`` instead, so
        ``count`` is sent as ``size`` there.

        Args:
            query: An ``EquityQuery``, ``FundQuery``, ``ETFQuery``, or a predefined-
                screen name string.
            size: Result page size for custom queries (max 250).
            offset: Pagination offset.
            count: Result page size when ``query`` is a predefined name; overrides
                ``size``.
            sort_field: yfinance field name to sort results by; None keeps the
                predefined screen's own field (``ticker`` for custom queries).
            sort_asc: Sort ascending when True; None keeps the predefined screen's
                own direction (descending for custom queries).
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Screen result dict from yfinance.

        Raises:
            ScreenerError: When every attempt fails, or Yahoo gives a final no-data
                answer that a retry cannot change. The original error, when there
                is one, is chained as ``__cause__``.
        """
        retries = max_retries if max_retries is not None else self.default_max_retries
        last_error: Exception | None = None

        def _action() -> dict[str, Any] | None:
            nonlocal last_error
            self.circuit_breaker.check()
            self.rate_limiter.acquire("screener")
            page_size = min(size if count is None else count, MAX_SIZE)
            kwargs: dict[str, Any] = {}
            if isinstance(query, str) and offset == 0:
                # Any explicit offset diverts yfinance to the offset endpoint, which
                # ignores ``count``; leaving it out keeps the predefined endpoint.
                kwargs["size" if count is None else "count"] = page_size
            else:
                kwargs["offset"] = offset
                kwargs["size"] = page_size
            # Pinning a sort here would override a predefined screen's own ranking.
            if sort_field is not None:
                kwargs["sortField"] = sort_field
            if sort_asc is not None:
                kwargs["sortAsc"] = sort_asc
            try:
                return yf.screen(query, **kwargs)
            except Exception as exc:
                last_error = exc
                raise

        try:
            result = retry_with_backoff(
                _action,
                retries,
                is_valid=lambda v: v is not None,
                is_rate_limit_error=is_rate_limit_error,
                on_rate_limit=self.circuit_breaker.trigger,
                on_success=lambda _: self.circuit_breaker.reset(),
                is_permanent_error=is_no_data_error,
            )
        except Exception as exc:
            raise ScreenerError(f"screen failed with a final answer: {exc}") from exc
        if result is None:
            raise ScreenerError(
                f"screen failed after {retries} attempt(s): {last_error}"
            ) from last_error
        return result

    def screen_predefined(
        self,
        name: str,
        *,
        count: int = 25,
        offset: int = 0,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Run a predefined screen by name from ``yf.PREDEFINED_SCREENER_QUERIES``.

        Predefined screens are sized via ``count`` (max 250), not ``size``, and keep
        their own ranking (field and direction).

        Args:
            name: Predefined screen name; must be in ``yf.PREDEFINED_SCREENER_QUERIES``.
            count: Maximum results to return (max 250).
            offset: Pagination offset.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Screen result dict from yfinance, or None if the name is unrecognised.

        Raises:
            ScreenerError: When every attempt fails (see :meth:`screen`).
        """
        if name not in yf.PREDEFINED_SCREENER_QUERIES:
            logger.error(
                "Unknown predefined screen '%s'. Valid: %s",
                name,
                list(yf.PREDEFINED_SCREENER_QUERIES),
            )
            return None
        return self.screen(name, count=count, offset=offset, max_retries=max_retries)
