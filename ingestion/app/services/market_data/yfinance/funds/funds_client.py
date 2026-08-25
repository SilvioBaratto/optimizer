"""Sub-client for ETF / fund data (re-added after the strip).

Wraps ``yf.Ticker.funds_data`` (asset-class split, top holdings, sector weights)
and the fund fields on ``yf.Ticker.info`` (AUM / NAV / family / expense ratio).
Best-effort: for a non-fund (equity) the fund data is empty and these return
``None`` rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .._base import BaseClient

logger = logging.getLogger(__name__)


class FundsClient(BaseClient):
    """Wraps ``yf.Ticker`` fund attributes."""

    def fetch_funds_data(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None:
        """Asset-class split, top holdings and sector weights, or ``None`` when
        the ticker is not a fund / carries no fund data."""
        logger.debug("Fetching funds data for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            return self._parse_funds_data(self._get_ticker(symbol))

        # None is a legitimate "not a fund" answer — do not retry on it.
        return self._fetch_with_resilience(
            _action, max_retries, is_valid=lambda _: True
        )

    def fetch_fund_profile(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None:
        """Headline fund metadata from ``info`` (AUM/NAV/family/…)."""
        logger.debug("Fetching fund profile for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            info = self._get_ticker(symbol).info or {}
            profile = {
                "aum": info.get("totalAssets"),
                "nav": info.get("navPrice"),
                "fund_family": info.get("fundFamily"),
                "legal_type": info.get("legalType"),
                "expense_ratio": (
                    info.get("annualReportExpenseRatio") or info.get("expenseRatio")
                ),
                "base_currency": info.get("currency"),
            }
            if not any(v is not None for v in profile.values()):
                return None
            return profile

        return self._fetch_with_resilience(
            _action, max_retries, is_valid=lambda _: True
        )

    @staticmethod
    def _parse_funds_data(ticker: Any) -> dict[str, Any] | None:
        try:
            fd = ticker.funds_data
        except Exception:
            return None
        if fd is None:
            return None

        asset_classes = _as_dict(getattr(fd, "asset_classes", None))
        sector_weightings = _as_dict(getattr(fd, "sector_weightings", None))
        top_holdings = _parse_holdings(getattr(fd, "top_holdings", None))

        if not asset_classes and not sector_weightings and not top_holdings:
            return None
        return {
            "asset_classes": asset_classes,
            "sector_weightings": sector_weightings,
            "top_holdings": top_holdings,
        }


def _as_dict(value: Any) -> dict[str, Any]:
    try:
        return dict(value) if value else {}
    except Exception:
        return {}


def _parse_holdings(frame: Any) -> list[dict[str, Any]]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    holdings: list[dict[str, Any]] = []
    for symbol, row in frame.iterrows():
        pct = row.get("Holding Percent")
        holdings.append(
            {
                "symbol": str(symbol),
                "name": row.get("Name"),
                "weight": float(pct) if pd.notna(pct) else None,
            }
        )
    return holdings
