"""Per-day spot FX rate provider (issue #775).

Returns `float | None` for (from_ccy, as_of_date) → rate-to-base lookups.
Backed by `LRUCache` and wraps the yfinance call with `CircuitBreaker`,
`RateLimiter`, and `retry_with_backoff` from `app.services.infrastructure`.
Never raises on missing data — callers attach `PositionFlag.FX_MISSING`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.services.infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    retry_with_backoff,
)
from app.services.infrastructure.retry import is_transient_network_error
from optimizer.fx._rates import build_fx_pair_ticker

logger = logging.getLogger(__name__)

_PENCE_NORMALIZED: frozenset[str] = frozenset({"GBX", "GBP_PENCE", "GBP_P"})
_PENCE_CASE_SENSITIVE: frozenset[str] = frozenset({"GBp"})
_DAY_SECONDS: float = 86400.0
_DEFAULT_RETRIES: int = 3


def _default_cache() -> LRUCache:
    return LRUCache(capacity=500, default_ttl=_DAY_SECONDS)


def _default_breaker() -> CircuitBreaker:
    return CircuitBreaker(service_name="FX rates", max_attempts=5)


def _default_limiter() -> RateLimiter:
    return RateLimiter()


@dataclass
class FxRateProvider:
    """Spot-FX provider with per-day cache and infrastructure-grade resilience."""

    base_currency: str = "EUR"
    cache: LRUCache | None = None
    circuit_breaker: CircuitBreaker | None = None
    rate_limiter: RateLimiter | None = None
    _cache: LRUCache = field(init=False, repr=False)
    _breaker: CircuitBreaker = field(init=False, repr=False)
    _limiter: RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache = self.cache or _default_cache()
        self._breaker = self.circuit_breaker or _default_breaker()
        self._limiter = self.rate_limiter or _default_limiter()

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_rate_to_base(self, from_ccy: str, as_of_date: date) -> float | None:
        self._reject_pence(from_ccy)
        normalized = from_ccy.upper()
        base = self.base_currency.upper()

        if normalized == base:
            return 1.0

        if self._breaker.is_active:
            return None

        cache_key = f"{normalized}_{base}_{as_of_date.isoformat()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return float(cached)

        rate = self._fetch_rate(normalized, base, as_of_date)
        if rate is None:
            return None
        self._cache.put(cache_key, rate, ttl=_DAY_SECONDS)
        return rate

    @staticmethod
    def _reject_pence(raw: str) -> None:
        if raw in _PENCE_CASE_SENSITIVE or raw.upper() in _PENCE_NORMALIZED:
            raise ValueError(
                f"FxRateProvider rejects pence input '{raw}'. "
                "Normalize to major currency (e.g. 'GBP') first."
            )

    def _fetch_rate(
        self, from_ccy: str, base: str, as_of_date: date
    ) -> float | None:
        self._limiter.acquire(key=from_ccy)
        ticker = build_fx_pair_ticker(from_ccy, base, cross_via_usd=True)
        if ticker is None:
            return 1.0

        if isinstance(ticker, tuple):
            return self._fetch_cross_rate(ticker, as_of_date)
        return self._fetch_direct_rate(ticker, from_ccy, as_of_date)

    def _fetch_direct_rate(
        self, ticker: str, from_ccy: str, as_of_date: date
    ) -> float | None:
        close = self._download_close(ticker, as_of_date)
        if close is None:
            return None
        return close if ticker.startswith(from_ccy) else 1.0 / close

    def _fetch_cross_rate(
        self,
        tickers: tuple[str, str],
        as_of_date: date,
    ) -> float | None:
        from_usd, base_usd = tickers
        from_close = self._download_close(from_usd, as_of_date)
        base_close = self._download_close(base_usd, as_of_date)
        if from_close is None or base_close is None or base_close == 0:
            return None
        return from_close / base_close

    def _download_close(self, ticker: str, as_of_date: date) -> float | None:
        def action() -> pd.DataFrame | None:
            return yf.download(
                ticker,
                start=as_of_date,
                end=as_of_date + timedelta(days=1),
                auto_adjust=False,
                progress=False,
            )

        try:
            df = retry_with_backoff(
                action,
                max_retries=_DEFAULT_RETRIES,
                base_delay=1.0,
                max_delay=10.0,
                is_rate_limit_error=is_transient_network_error,
                on_rate_limit=self._breaker.trigger,
            )
        except Exception:
            # Per-rate isolation: an unexpected FX download failure is logged
            # with traceback and degraded to None so the caller skips this rate
            # (one currency pair never aborts the multi-rate fetch).
            logger.exception("FX download crashed for %s", ticker)
            return None

        return _extract_close(df)


def _extract_close(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    close_series = df["Close"].dropna()
    if close_series.empty:
        return None
    return float(close_series.iloc[-1])
