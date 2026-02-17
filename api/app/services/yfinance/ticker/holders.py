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
        logger.debug("Fetching insider roster holders for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).insider_roster_holders

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )
