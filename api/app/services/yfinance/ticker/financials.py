"""Sub-client for financial statements and SEC filings."""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class FinancialsClient(BaseClient):
    """Wraps ``yf.Ticker`` financial-statement attributes."""

    def fetch_income_stmt(
        self,
        symbol: str,
        quarterly: bool = False,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching income statement for '%s' (quarterly=%s)", symbol, quarterly)
        attr = "quarterly_income_stmt" if quarterly else "income_stmt"

        def _action() -> pd.DataFrame | None:
            return getattr(self._get_ticker(symbol), attr)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_balance_sheet(
        self,
        symbol: str,
        quarterly: bool = False,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching balance sheet for '%s' (quarterly=%s)", symbol, quarterly)
        attr = "quarterly_balance_sheet" if quarterly else "balance_sheet"

        def _action() -> pd.DataFrame | None:
            return getattr(self._get_ticker(symbol), attr)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_cashflow(
        self,
        symbol: str,
        quarterly: bool = False,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching cashflow for '%s' (quarterly=%s)", symbol, quarterly)
        attr = "quarterly_cashflow" if quarterly else "cashflow"

        def _action() -> pd.DataFrame | None:
            return getattr(self._get_ticker(symbol), attr)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_earnings(
        self,
        symbol: str,
        quarterly: bool = False,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching earnings for '%s' (quarterly=%s)", symbol, quarterly)
        attr = "quarterly_earnings" if quarterly else "earnings"

        def _action() -> pd.DataFrame | None:
            return getattr(self._get_ticker(symbol), attr)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_sec_filings(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> list[dict[str, Any]] | None:
        logger.debug("Fetching SEC filings for '%s'", symbol)

        def _action() -> list[dict[str, Any]] | None:
            return cast(list[dict[str, Any]], self._get_ticker(symbol).sec_filings)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )
