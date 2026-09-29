"""Sub-client for analyst recommendations and estimates."""

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
        """Fetch analyst buy/sell/hold recommendations for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Recommendations DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch aggregated recommendation counts by period for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Recommendations summary DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch analyst upgrades and downgrades history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Upgrades/downgrades DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch analyst consensus price targets for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Price targets dict, or None if unavailable or all retries fail.
        """
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
        """Fetch forward earnings estimates by period for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Earnings estimate DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch forward revenue estimates by period for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Revenue estimate DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch historical EPS actuals vs. estimates for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Earnings history DataFrame, or None if unavailable or all retries fail.
        """
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
        """Fetch analyst growth estimates for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Growth estimates DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching growth estimates for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).growth_estimates

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_eps_trend(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch EPS trend (current vs. revised estimates) for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            EPS trend DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching EPS trend for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).eps_trend

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_eps_revisions(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch EPS estimate revision history for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            EPS revisions DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching EPS revisions for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).eps_revisions

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )
