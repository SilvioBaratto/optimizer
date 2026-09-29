"""Sub-client for holder information (institutional, mutual fund, insider)."""

from __future__ import annotations

import logging

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class HoldersClient(BaseClient):
    """Wraps ``yf.Ticker`` holder attributes."""

    def fetch_major_holders(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch major holder breakdown for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Major holders DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching major holders for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).major_holders

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_institutional_holders(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch institutional holder positions for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Institutional holders DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching institutional holders for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).institutional_holders

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_mutualfund_holders(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch mutual fund holder positions for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Mutual fund holders DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching mutual fund holders for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).mutualfund_holders

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_insider_transactions(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch insider transaction history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Insider transactions DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching insider transactions for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).insider_transactions

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_insider_purchases(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch insider purchase history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Insider purchases DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching insider purchases for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).insider_purchases

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_insider_roster_holders(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch current insider roster and holdings for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Insider roster DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching insider roster holders for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).insider_roster_holders

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )
