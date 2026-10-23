"""Module-level client for regional market summaries from Yahoo.

Reads the ``quote/marketSummary`` endpoint that backs ``yf.Market`` directly
rather than through ``yf.Market(id).summary``: yfinance keys that property by
exchange code, so futures, FX, rates and crypto quotes that share one code
overwrite each other and most of a market's rows disappear. The raw response is
a list, which is kept whole. Values may be ``{"raw": ..., "fmt": ...}``
wrappers, so they are unwrapped to scalars before being returned as a flat list
of record dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from yfinance.const import _QUERY1_URL_
from yfinance.data import YfData

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

_SUMMARY_URL = f"{_QUERY1_URL_}/v6/finance/quote/marketSummary"
_SUMMARY_FIELDS = (
    "shortName",
    "regularMarketPrice",
    "regularMarketChange",
    "regularMarketChangePercent",
)

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


def _summary_to_list(quotes: Any) -> list[dict[str, Any]]:
    if not isinstance(quotes, list):
        return []
    out: list[dict[str, Any]] = []
    for q in quotes:
        if not isinstance(q, dict):
            continue
        # Index/FX quotes occasionally omit "symbol"; the exchange code is then
        # the only identifier Yahoo gives.
        symbol = q.get("symbol") or q.get("exchange")
        if not symbol:
            continue
        out.append(
            {
                "symbol": str(symbol),
                "short_name": q.get("shortName") or q.get("longName"),
                "price": _raw(q.get("regularMarketPrice")),
                "change": _raw(q.get("regularMarketChange")),
                "change_percent": _raw(q.get("regularMarketChangePercent")),
                "previous_close": _raw(q.get("regularMarketPreviousClose")),
                "market_state": q.get("marketState"),
            }
        )
    return out


def _fetch_quotes(market: str) -> Any:
    """Return the raw ``marketSummaryResponse.result`` list for *market*.

    Uses ``get_raw_json`` so the response bypasses yfinance's process-wide
    ``cache_get``, which would replay a stale or failed answer to every later
    sweep and retry of the same market.
    """
    payload = YfData().get_raw_json(
        _SUMMARY_URL,
        params={
            "fields": ",".join(_SUMMARY_FIELDS),
            "formatted": False,
            "lang": "en-US",
            "market": market,
        },
    )
    return payload["marketSummaryResponse"]["result"]


class MarketClient:
    """Fetches Yahoo's market summary for the documented regional identifiers."""

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
        """Return the regional market summary as a flat list of record dicts.

        Args:
            market: Regional market identifier from ``MARKET_IDENTIFIERS``
                (e.g. ``"US"``, ``"EUROPE"``).
            max_retries: Override the instance default retry count.

        Returns:
            One record per quote Yahoo returns, or ``None`` on unrecoverable
            failure or an unknown *market*.
        """
        logger.debug("Fetching market summary for '%s'", market)
        if market not in MARKET_IDENTIFIERS:
            logger.error(
                "Unknown market '%s'. Valid: %s", market, list(MARKET_IDENTIFIERS)
            )
            return None
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"market:{market}")
            return _summary_to_list(_fetch_quotes(market))

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
