"""Sub-client for ETF / fund data.

Wraps ``yf.Ticker.funds_data`` (asset-class split, top holdings, sector weights)
and the fund fields on ``yf.Ticker.info`` (AUM / NAV / family / expense ratio).
Best-effort: for a non-fund (equity) the fund data is empty and these return
``None`` rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.utils.currency import to_major_currency

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
                "expense_ratio": _expense_ratio_fraction(info),
                # navPrice/totalAssets are in the fund's own currency, which for
                # a UCITS ETF differs from the quote currency in info["currency"]
                # (IUSA.L quotes in GBp but its NAV is USD). financialCurrency is
                # the fund's reporting currency, but Yahoo leaves it empty for
                # many funds (BND, TDEM, UCITS); leaving those NULL beats
                # mislabelling NAV/AUM with the listing code.
                "base_currency": to_major_currency(info.get("financialCurrency")),
            }
            if not any(v is not None for v in profile.values()):
                return None
            if profile["base_currency"] is None:
                # Applied only after the fund-data check so a bare equity quote
                # cannot pass as a fund profile on the strength of its currency.
                profile["base_currency"] = _us_listing_usd(info)
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
            # Metric-indexed DataFrames are flattened to {metric: value};
            # bond_ratings / fund_overview are already dict-shaped.
            "equity_holdings": _equity_holdings(getattr(fd, "equity_holdings", None)),
            "bond_holdings": _first_col_dict(getattr(fd, "bond_holdings", None)),
            "fund_operations": _first_col_dict(getattr(fd, "fund_operations", None)),
            "bond_ratings": _as_dict(getattr(fd, "bond_ratings", None)),
            "fund_overview": _as_dict(getattr(fd, "fund_overview", None)),
            "description": description,
        }


def _us_listing_usd(info: dict[str, Any]) -> str | None:
    """``"USD"`` for a USD-quoted fund on a US listing, else ``None``.

    US-listed funds report NAV and AUM in dollars, so the quote currency is a
    safe stand-in when ``financialCurrency`` is missing. Yahoo's ``market``
    field marks the listing venue (``us_market`` / ``gb_market``); ``region``
    cannot be used because it carries the fund's domicile and reads ``US`` for
    London-listed UCITS ETFs on US indices.
    """
    if info.get("market") == "us_market" and info.get("currency") == "USD":
        return "USD"
    return None


def _expense_ratio_fraction(info: dict[str, Any]) -> float | None:
    """Annual expense ratio as a decimal fraction (0.0009 = 0.09% p.a.).

    ``netExpenseRatio`` is the key Yahoo actually populates for ETFs, but it is
    quoted in percent points (SPY 0.0945 = 0.0945%), whereas the legacy
    mutual-fund keys ``annualReportExpenseRatio`` / ``expenseRatio`` are already
    fractions. Normalising here keeps the stored column in a single unit. Keys
    are tried in order on ``is not None`` so a genuine 0.0 (zero-fee fund) is
    kept rather than read as missing.
    """
    net = info.get("netExpenseRatio")
    if net is not None:
        return float(net) / 100.0
    for key in ("annualReportExpenseRatio", "expenseRatio"):
        value = info.get(key)
        if value is not None:
            return float(value)
    return None


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


# Yahoo serves these four fund valuation rows as reciprocals (earnings yield,
# book/price, ...), unlike the genuine multiples in ``info``.
_RECIPROCAL_VALUATION_KEYS = (
    "Price/Earnings",
    "Price/Book",
    "Price/Sales",
    "Price/Cashflow",
)


def _equity_holdings(frame: Any) -> dict[str, float]:
    """Flatten ``equity_holdings`` and turn Yahoo's valuation reciprocals into
    multiples (SPY P/E 0.04132 -> 24.2).

    A value of 0.0 is Yahoo's "no data" sentinel (bond ETFs, negative
    earnings), and a negative reciprocal has no meaningful multiple, so both
    are dropped rather than stored as a misleading 0 / inverted figure.
    """
    metrics = _first_col_dict(frame)
    for key in _RECIPROCAL_VALUATION_KEYS:
        value = metrics.pop(key, None)
        if value is not None and value > 0:
            metrics[key] = 1.0 / value
    return metrics


def _parse_holdings(frame: Any) -> list[dict[str, Any]]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    holdings: list[dict[str, Any]] = []
    for symbol, row in frame.iterrows():
        pct: Any = row.get("Holding Percent")
        holdings.append(
            {
                "symbol": str(symbol),
                "name": row.get("Name"),
                "weight": float(pct) if pd.notna(pct) else None,
            }
        )
    return holdings
