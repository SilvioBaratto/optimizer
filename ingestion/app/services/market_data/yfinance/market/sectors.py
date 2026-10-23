"""Module-level client for sector / industry rollups via ``yf.Sector``.

``yf.Sector(key, region=)`` exposes a global taxonomy (11 sector keys) whose
list-style rollups (``top_companies``, ``industries``) are region-scoped. This
client returns a plain dict per (sector, region) so the service/repo can map it
without importing yfinance.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFDataException

from ..infrastructure import is_no_data_error, is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

SECTOR_KEYS: tuple[str, ...] = (
    "technology",
    "healthcare",
    "financial-services",
    "consumer-cyclical",
    "communication-services",
    "industrials",
    "consumer-defensive",
    "energy",
    "basic-materials",
    "real-estate",
    "utilities",
)


def _industries_to_list(value: Any) -> list[dict[str, Any]]:
    """``sector.industries`` → ``[{"key","name"}]`` (DataFrame or dict tolerant)."""
    out: list[dict[str, Any]] = []
    if isinstance(value, pd.DataFrame) and not value.empty:
        for key, row in value.iterrows():
            out.append({"key": str(key), "name": row.get("name") or row.get("Name")})
    elif isinstance(value, dict):
        for key, name in value.items():
            out.append({"key": str(key), "name": name})
    return out


def _present(value: Any) -> Any:
    """Return *value*, or ``None`` when pandas marks it missing (NaN, NaT, None).

    NaN is truthy, so ``value or fallback`` keeps it and a later ``str()``
    stores the text ``"nan"`` as if it were real data.
    """
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else value


def _top_companies_to_list(value: Any) -> list[dict[str, Any]]:
    """``sector.top_companies`` → ``[{"symbol","name","weight","rating"}]``.

    Fields yfinance leaves NaN (an unrated constituent, an unnamed row) come
    back as ``None``; a genuine ``0.0`` weight is kept.
    """
    if not isinstance(value, pd.DataFrame) or value.empty:
        return []
    out: list[dict[str, Any]] = []
    for symbol, row in value.iterrows():
        out.append(
            {
                "symbol": str(symbol),
                "name": _present(row.get("name")) or _present(row.get("Name")),
                "weight": _present(
                    row.get("market weight")
                    if "market weight" in row
                    else row.get("market_weight")
                ),
                "rating": _present(row.get("rating")) or _present(row.get("Rating")),
            }
        )
    return out


class SectorsClient:
    """Wraps ``yf.Sector`` (and its region-scoped rollups)."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def fetch_sector(
        self,
        key: str,
        region: str = "US",
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Return overview + industries + top companies for a sector/region.

        Args:
            key: Sector key from ``SECTOR_KEYS`` (e.g. ``"technology"``).
            region: Regional scope for list-style rollups (default ``"US"``).
            max_retries: Override the instance default retry count.

        Returns:
            Dict with ``key``, ``name``, ``symbol``, ``overview``,
            ``industries``, and ``top_companies``, or ``None`` on failure.

        Raises:
            YFDataException: Yahoo exposed no name and no overview for the pair.
                yfinance swallows the failed fetch (a throttle or a 404) and
                leaves every attribute empty, which would otherwise look like a
                valid, truthy result.  It is raised as a final no-data answer so
                it is not retried, and the caller can record the failure instead
                of persisting an all-NULL snapshot.
        """
        logger.debug("Fetching sector '%s' (region=%s)", key, region)
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> dict[str, Any] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"sector:{key}:{region}")
            sec = yf.Sector(key, region=region)
            overview = sec.overview
            if not isinstance(overview, dict):
                overview = {}
            if sec.name is None and all(v is None for v in overview.values()):
                raise YFDataException(f"no sector data for {key}/{region}")
            return {
                "key": sec.key or key,
                "name": sec.name,
                "symbol": sec.symbol,
                "overview": overview,
                "industries": _industries_to_list(sec.industries),
                "top_companies": _top_companies_to_list(sec.top_companies),
            }

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_permanent_error=is_no_data_error,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )
