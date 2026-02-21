"""Sub-client for sector and industry data via ``yf.Sector`` and ``yf.Industry``."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)


class SectorIndustryClient:
    """Wraps ``yf.Sector`` and ``yf.Industry``."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    # ------------------------------------------------------------------
    # Sector
    # ------------------------------------------------------------------

    def fetch_sector_overview(
        self,
        sector_key: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching sector overview for '%s'", sector_key)
        return self._fetch_sector_attr(sector_key, "overview", max_retries)

    def fetch_sector_top_companies(
        self,
        sector_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching sector top companies for '%s'", sector_key)
        return self._fetch_sector_attr(sector_key, "top_companies", max_retries)

    def fetch_sector_top_etfs(
        self,
        sector_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching sector top ETFs for '%s'", sector_key)
        return self._fetch_sector_attr(sector_key, "top_etfs", max_retries)

    def fetch_sector_top_mutual_funds(
        self,
        sector_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching sector top mutual funds for '%s'", sector_key)
        return self._fetch_sector_attr(sector_key, "top_mutual_funds", max_retries)

    # ------------------------------------------------------------------
    # Industry
    # ------------------------------------------------------------------

    def fetch_industry_overview(
        self,
        industry_key: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching industry overview for '%s'", industry_key)
        return self._fetch_industry_attr(industry_key, "overview", max_retries)

    def fetch_industry_top_companies(
        self,
        industry_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching industry top companies for '%s'", industry_key)
        return self._fetch_industry_attr(industry_key, "top_companies", max_retries)

    def fetch_industry_top_etfs(
        self,
        industry_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching industry top ETFs for '%s'", industry_key)
        return self._fetch_industry_attr(industry_key, "top_etfs", max_retries)

    def fetch_industry_top_mutual_funds(
        self,
        industry_key: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching industry top mutual funds for '%s'", industry_key)
        return self._fetch_industry_attr(industry_key, "top_mutual_funds", max_retries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_sector_attr(
        self, sector_key: str, attr: str, max_retries: int | None
    ) -> Any | None:
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> Any | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"sector_{sector_key}")
            s = yf.Sector(sector_key)
            return getattr(s, attr, None)

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def _fetch_industry_attr(
        self, industry_key: str, attr: str, max_retries: int | None
    ) -> Any | None:
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> Any | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"industry_{industry_key}")
            ind = yf.Industry(industry_key)
            return getattr(ind, attr, None)

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
