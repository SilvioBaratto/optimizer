"""Module-level client for market-wide calendars via ``yf.Calendars``.

Wraps the four calendar rollups (earnings / IPO / splits / economic events).
Rather than the bare convenience properties (which return only ~12 rows over a
7-day window and, for earnings, most-active names only), each fetch drives the
``get_*_calendar`` methods over a forward window with offset pagination, so a
market-wide sweep captures the full set. Yahoo caps each page at 100 rows; we
page up to ``_MAX_ROWS`` and log if that cap is hit (no silent truncation).

Earnings additionally fetch a **backward** window: a forward-only calendar only
carries the consensus ``EPS Estimate`` (upcoming events have no result yet), so
``Reported EPS`` / ``Surprise(%)`` are always NULL. The past window backfills the
realized EPS and surprise — the raw material for the earnings-surprise (SUE /
PEAD) signal — for names that have already reported. The upsert is keyed on
``(ticker, event_date)``, so the estimate→actual→surprise progression is
idempotent across runs.

Each fetch returns a list of record dicts (the DataFrame rows) or ``None`` so
the service/repo can map them defensively without importing yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from ..infrastructure import is_rate_limit_error, retry_with_backoff
from ..protocols import CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

_PAGE = 100  # Yahoo caps calendar pages at 100 (calendars.py: min(limit, 100))
_MAX_ROWS = 2000  # safety cap across pages; logged if reached
_WINDOW_DAYS = 90  # forward window fetched each run
# Backward windows: fetched as a separate pass (own row budget) so forward rows
# never crowd out the realized values. Earnings spans a quarterly reporting cycle;
# economic events a monthly release cycle. Both backfill the "actual" columns that
# a forward-only calendar can never carry.
_EARNINGS_LOOKBACK_DAYS = 35
_ECON_LOOKBACK_DAYS = 30


def _records(value: Any) -> list[dict[str, Any]]:
    """A ``yf.Calendars`` property → list of record dicts (index kept as a col)."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.reset_index().to_dict("records")
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    return []


class CalendarsClient:
    """Wraps ``yf.Calendars`` (earnings / IPO / splits / economic events)."""

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def _paginate(
        self,
        method: str,
        extra: dict[str, Any],
        max_retries: int | None,
        *,
        lookback_days: int = 0,
        forward_days: int = _WINDOW_DAYS,
    ) -> list[dict[str, Any]] | None:
        retries = max_retries if max_retries is not None else self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            self.circuit_breaker.check()
            self.rate_limiter.acquire(f"calendars:{method}")
            today = datetime.now(timezone.utc).date()
            start = today - timedelta(days=lookback_days)
            end = today + timedelta(days=forward_days)
            fn = getattr(yf.Calendars(), method)

            collected: list[dict[str, Any]] = []
            offset = 0
            while len(collected) < _MAX_ROWS:
                recs = _records(
                    fn(start=start, end=end, limit=_PAGE, offset=offset, **extra)
                )
                if not recs:
                    break
                collected.extend(recs)
                if len(recs) < _PAGE:
                    break
                offset += _PAGE
            if len(collected) >= _MAX_ROWS:
                logger.warning(
                    "calendars %s hit the %d-row cap; more rows may exist",
                    method,
                    _MAX_ROWS,
                )
            return collected[:_MAX_ROWS]

        return retry_with_backoff(
            _action,
            retries,
            is_valid=lambda v: v is not None,
            is_rate_limit_error=is_rate_limit_error,
            on_rate_limit=self.circuit_breaker.trigger,
            on_success=lambda _: self.circuit_breaker.reset(),
        )

    def _fetch_windowed(
        self,
        method: str,
        extra: dict[str, Any],
        max_retries: int | None,
        *,
        lookback_days: int,
    ) -> list[dict[str, Any]] | None:
        """Two passes with independent row budgets: a forward window (upcoming rows,
        estimate/forecast + prior) and a backward window (past ``lookback_days``,
        which is the only place realized "actual" values exist). Merged forward-first
        so the repo upsert (keyed on the event's natural key) lets a realized actual
        win over the same event's earlier estimate.
        """
        forward = self._paginate(method, extra, max_retries)
        past = self._paginate(
            method, extra, max_retries, lookback_days=lookback_days, forward_days=0
        )
        if forward is None and past is None:
            return None
        return [*(forward or []), *(past or [])]

    def fetch_earnings(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        # Whole market, not just most-active names. Backward pass backfills realized
        # EPS + surprise for names that already reported.
        return self._fetch_windowed(
            "get_earnings_calendar",
            {"filter_most_active": False},
            max_retries,
            lookback_days=_EARNINGS_LOOKBACK_DAYS,
        )

    def fetch_ipos(self, max_retries: int | None = None) -> list[dict[str, Any]] | None:
        return self._paginate("get_ipo_info_calendar", {}, max_retries)

    def fetch_splits(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._paginate("get_splits_calendar", {}, max_retries)

    def fetch_economic_events(
        self, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None:
        # Backward pass backfills the realized "actual" prints for events that have
        # already occurred (a forward-only window only ever carries the prior value;
        # Yahoo does not populate the "Expected" forecast at all).
        return self._fetch_windowed(
            "get_economic_events_calendar",
            {},
            max_retries,
            lookback_days=_ECON_LOOKBACK_DAYS,
        )
