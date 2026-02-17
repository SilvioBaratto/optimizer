"""Sub-client for ticker metadata: ISIN, fast_info, calendar, options, earnings dates."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class MetadataClient(BaseClient):
    """Wraps ``yf.Ticker`` metadata-related attributes."""

    def fetch_isin(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> str | None:
        logger.debug("Fetching ISIN for '%s'", symbol)

        def _action() -> str | None:
            return self._get_ticker(symbol).isin

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and v != "" and v != "-",
        )

    def fetch_fast_info(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fast_info for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fi = self._get_ticker(symbol).fast_info
            if fi is None:
                return None
            return dict(fi)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_calendar(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching calendar for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            cal = self._get_ticker(symbol).calendar
            if cal is None:
                return None
            if isinstance(cal, pd.DataFrame):
                return cal.to_dict()
            return dict(cal) if not isinstance(cal, dict) else cal

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_options_expirations(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> tuple[str, ...] | None:
        logger.debug("Fetching options expirations for '%s'", symbol)

        def _action() -> tuple[str, ...] | None:
            return self._get_ticker(symbol).options

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_option_chain(
        self,
        symbol: str,
        date: str | None = None,
        max_retries: int | None = None,
    ) -> Any | None:
        logger.debug("Fetching option chain for '%s' (date=%s)", symbol, date)

        def _action() -> Any | None:
            ticker = self._get_ticker(symbol)
            if date is not None:
                return ticker.option_chain(date)
            return ticker.option_chain()

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_earnings_dates(
        self,
        symbol: str,
        limit: int = 12,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching earnings dates for '%s' (limit=%d)", symbol, limit)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).get_earnings_dates(limit=limit)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_history_metadata(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching history metadata for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            return self._get_ticker(symbol).history_metadata

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )
