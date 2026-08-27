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
                # Yahoo omits totalAssets for many UCITS ETFs; netAssets is the
                # same figure under a second key when present.
                "aum": info.get("totalAssets") or info.get("netAssets"),
                "nav": info.get("navPrice"),
                "fund_family": info.get("fundFamily"),
                "legal_type": info.get("legalType"),
                # netExpenseRatio is the key Yahoo actually populates for ETFs
                # (US-listed carry it; UCITS listings mostly do not). The other
                # two are legacy mutual-fund keys kept as fallbacks.
                "expense_ratio": (
                    info.get("netExpenseRatio")
                    or info.get("annualReportExpenseRatio")
                    or info.get("expenseRatio")
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

        description = getattr(fd, "description", None)
        if not isinstance(description, str):
            description = None
        return {
            "asset_classes": asset_classes,
            "sector_weightings": sector_weightings,
            "top_holdings": top_holdings,
            # Depth (SPEC A8): metric-indexed DataFrames flattened to {metric:
            # value}; bond_ratings / fund_overview are already dict-shaped.
            "equity_holdings": _first_col_dict(getattr(fd, "equity_holdings", None)),
            "bond_holdings": _first_col_dict(getattr(fd, "bond_holdings", None)),
            "fund_operations": _first_col_dict(getattr(fd, "fund_operations", None)),
            "bond_ratings": _as_dict(getattr(fd, "bond_ratings", None)),
            "fund_overview": _as_dict(getattr(fd, "fund_overview", None)),
            "description": description,
        }


def _as_dict(value: Any) -> dict[str, Any]:
    try:
        return dict(value) if value else {}
    except Exception:
        return {}


def _first_col_dict(frame: Any) -> dict[str, float]:
    """Flatten a metric-indexed DataFrame (equity/bond holdings, operations) to
    ``{metric: value}`` reading the fund's own (first) column; the second column
    is usually the category average and is dropped."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    col = frame.iloc[:, 0]
    out: dict[str, float] = {}
    for key, val in col.items():
        if pd.notna(val):
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
    return out


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
