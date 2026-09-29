"""Sub-client for dividends, splits, actions, capital gains, and shares."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class CorporateActionsClient(BaseClient):
    """Wraps ``yf.Ticker`` corporate-action attributes."""

    def fetch_dividends(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.Series | None:
        """Fetch dividend payment history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Dividend Series indexed by date, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching dividends for '%s'", symbol)

        def _action() -> pd.Series | None:
            return self._get_ticker(symbol).dividends

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda s: s is not None and not s.empty,
        )

    def fetch_splits(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.Series | None:
        """Fetch stock split history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Splits Series indexed by date, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching splits for '%s'", symbol)

        def _action() -> pd.Series | None:
            return self._get_ticker(symbol).splits

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda s: s is not None and not s.empty,
        )

    def fetch_actions(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch combined dividends and splits history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Actions DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching actions for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).actions

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_shares_full(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch full share count series for symbol over a date range.

        Args:
            symbol: Ticker symbol.
            start: Start date (ISO format); None fetches from earliest available.
            end: End date (ISO format); None fetches to latest available.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Share count DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching shares full for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return cast(
                pd.DataFrame,
                self._get_ticker(symbol).get_shares_full(start=start, end=end),
            )

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not getattr(df, "empty", True),
        )
