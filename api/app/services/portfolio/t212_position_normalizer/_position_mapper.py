"""T212 → NormalizedPosition mapper (issue #776).

Per-position pure logic — no DB session, no orchestration. The orchestrator
(issue #777) handles DB lookup, weight assignment, reconciliation, batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.schemas.portfolio import NormalizedPosition, PositionFlag
from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
    FxRateProvider,
)
from app.services.portfolio.t212_position_normalizer._suffix_resolver import (
    resolve_t212_suffix,
)

_PENCE_RAW: frozenset[str] = frozenset({"GBX", "GBP_PENCE", "GBP_P"})
_PENCE_CASE_SENSITIVE: frozenset[str] = frozenset({"GBp"})
_PENCE_DIVISOR: float = 100.0


@dataclass
class T212PositionMapper:
    """Convert one raw T212 position dict into a `NormalizedPosition`."""

    fx_provider: FxRateProvider
    base_currency: str = "EUR"

    def map_position(
        self,
        raw: dict[str, Any],
        *,
        yf_ticker: str | None,
        as_of_date: date,
    ) -> NormalizedPosition:
        t212_ticker = self._require_ticker(raw)
        qty = self._require_quantity(raw)
        currency, price_native = self._normalize_currency(raw)
        flags: list[PositionFlag] = []

        ticker = self._resolve_ticker(yf_ticker, t212_ticker, flags)
        self._stale_price_flag(raw.get("currentPrice"), flags)
        eur_value = self._compute_eur_value(
            qty, price_native, currency, as_of_date, flags
        )
        ppl_eur = self._compute_ppl_eur(raw)

        return NormalizedPosition(
            ticker=ticker,
            t212_ticker=t212_ticker,
            qty=qty,
            price_native=price_native,
            currency=currency,
            eur_value=eur_value,
            eur_weight=0.0,
            ppl_eur=ppl_eur,
            flags=flags,
        )

    @staticmethod
    def _require_ticker(raw: dict[str, Any]) -> str:
        ticker = raw.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("raw position is missing required 'ticker' field")
        return ticker

    @staticmethod
    def _require_quantity(raw: dict[str, Any]) -> float:
        if "quantity" not in raw:
            raise ValueError("raw position is missing required 'quantity' field")
        return float(raw["quantity"])

    def _normalize_currency(self, raw: dict[str, Any]) -> tuple[str, float]:
        raw_ccy = raw.get("currencyCode") or self.base_currency
        price = _safe_price(raw.get("currentPrice"))
        if raw_ccy in _PENCE_CASE_SENSITIVE or raw_ccy.upper() in _PENCE_RAW:
            return "GBP", price / _PENCE_DIVISOR
        return raw_ccy.upper(), price

    def _resolve_ticker(
        self, yf_ticker: str | None, t212_ticker: str, flags: list[PositionFlag]
    ) -> str:
        if yf_ticker:
            return yf_ticker
        flags.append(PositionFlag.UNMAPPED)
        resolved = resolve_t212_suffix(t212_ticker)
        return f"{resolved.base_symbol}{resolved.probable_yf_suffix}"

    @staticmethod
    def _stale_price_flag(
        raw_price: Any, flags: list[PositionFlag]
    ) -> None:
        if raw_price is None:
            flags.append(PositionFlag.STALE_PRICE)
            return
        try:
            if float(raw_price) <= 0.0:
                flags.append(PositionFlag.STALE_PRICE)
        except (TypeError, ValueError):
            flags.append(PositionFlag.STALE_PRICE)

    def _compute_eur_value(
        self,
        qty: float,
        price_native: float,
        currency: str,
        as_of_date: date,
        flags: list[PositionFlag],
    ) -> float:
        if price_native <= 0.0:
            return 0.0
        if currency == self.base_currency.upper():
            return qty * price_native
        rate = self.fx_provider.get_rate_to_base(currency, as_of_date)
        if rate is None:
            flags.append(PositionFlag.FX_MISSING)
            return 0.0
        return qty * price_native * rate

    def _compute_ppl_eur(self, raw: dict[str, Any]) -> float | None:
        if self.base_currency.upper() != "EUR":
            return None
        ppl = raw.get("ppl")
        if ppl is None:
            return None
        fx_ppl = raw.get("fxPpl") or 0.0
        return float(ppl) + float(fx_ppl)


def _safe_price(raw_price: Any) -> float:
    if raw_price is None:
        return 0.0
    try:
        return float(raw_price)
    except (TypeError, ValueError):
        return 0.0
