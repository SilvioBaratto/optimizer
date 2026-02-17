"""Sub-client for analyst recommendations, estimates, and sustainability."""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class AnalysisClient(BaseClient):
    """Wraps ``yf.Ticker`` analyst/research attributes."""

    def fetch_recommendations(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching recommendations for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return cast(pd.DataFrame, self._get_ticker(symbol).recommendations)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_recommendations_summary(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching recommendations summary for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return cast(pd.DataFrame, self._get_ticker(symbol).recommendations_summary)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_upgrades_downgrades(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching upgrades/downgrades for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return cast(pd.DataFrame, self._get_ticker(symbol).upgrades_downgrades)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_analyst_price_targets(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        logger.debug("Fetching analyst price targets for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            return self._get_ticker(symbol).analyst_price_targets

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_earnings_estimate(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching earnings estimate for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).earnings_estimate

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_revenue_estimate(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching revenue estimate for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).revenue_estimate

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_earnings_history(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching earnings history for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).earnings_history

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_growth_estimates(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching growth estimates for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).growth_estimates

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_sustainability(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        logger.debug("Fetching sustainability for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).sustainability

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )
