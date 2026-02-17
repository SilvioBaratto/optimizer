"""Sub-client for ETF/mutual fund data: holdings, sectors, bonds, performance."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class FundsClient(BaseClient):
    """Wraps ``yf.Ticker.funds_data`` attributes."""

    def _get_funds_data(self, symbol: str) -> Any:
        """Return the ``funds_data`` object for *symbol*."""
        return self._get_ticker(symbol).funds_data

    def fetch_fund_overview(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund overview for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.fund_overview

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_top_holdings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching fund top holdings for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.top_holdings

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_fund_sector_weightings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund sector weightings for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.sector_weightings

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_bond_holdings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund bond holdings for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.bond_holdings

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_bond_ratings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund bond ratings for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.bond_ratings

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_equity_holdings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund equity holdings for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.equity_holdings

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_operations(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund operations for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.fund_operations

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_asset_classes(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching fund asset classes for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.asset_classes

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_fund_description(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> str | None:
        logger.debug("Fetching fund description for '%s'", symbol)

        def _action() -> str | None:
            fd = self._get_funds_data(symbol)
            if fd is None:
                return None
            return fd.description

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and v != "",
        )
