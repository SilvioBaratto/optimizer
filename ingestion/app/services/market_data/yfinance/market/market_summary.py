"""Module-level client for market summaries via ``yf.Market``.

``yf.Market(id).summary`` returns a dict of index/quote summaries for a regional
market. Only ``summary`` is regional (``status`` is US-only and ignored here).
Values may be ``{"raw": ..., "fmt": ...}`` wrappers, so they are unwrapped to
scalars before being returned as a flat list of record dicts.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

MARKET_IDENTIFIERS: tuple[str, ...] = (
    "US",
    "GB",
    "ASIA",
    "EUROPE",
    "RATES",
    "COMMODITIES",
    "CURRENCIES",
    "CRYPTOCURRENCIES",
)


def _raw(v: Any) -> Any:
    """Unwrap Yahoo ``{"raw": x, "fmt": ...}`` value wrappers to the scalar."""
    if isinstance(v, dict):
        return v.get("raw", v.get("fmt"))
    return v


def _summary_to_list(summary: Any) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    out: list[dict[str, Any]] = []
    for exchange, q in summary.items():
        if not isinstance(q, dict):
            continue
        out.append(
            {
                # yfinance keys summary by exchange code; the instrument symbol
                # (e.g. "^GSPC") lives in q["symbol"]. Fall back to the key.
                "symbol": str(q.get("symbol") or exchange),
                "short_name": q.get("shortName") or q.get("longName"),
                "price": _raw(q.get("regularMarketPrice")),
                "change": _raw(q.get("regularMarketChange")),
                "change_percent": _raw(q.get("regularMarketChangePercent")),
                "previous_close": _raw(q.get("regularMarketPreviousClose")),
                "market_state": q.get("marketState"),
            }
        )
    return out


class MarketClient:
    """Wraps ``yf.Market`` summary for the documented regional identifiers."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def fetch_summary(
        self, market: str, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        """Return the regional market summary as a flat list of record dicts."""
        logger.debug("Fetching market summary for '%s'", market)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"market:{market}")
            return _summary_to_list(yf.Market(market).summary)

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
