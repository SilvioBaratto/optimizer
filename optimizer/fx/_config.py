"""Configuration for multi-currency FX conversion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FxConversionMode(str, Enum):
    """How FX conversion is applied in the pipeline."""

    NONE = "none"
    TO_BASE = "to_base"
    DECOMPOSE = "decompose"


class BaseCurrency(str, Enum):
    """Supported base currencies."""

    EUR = "EUR"
    GBP = "GBP"
    USD = "USD"


class FxDataSource(str, Enum):
    """Source for FX rate data."""

    YFINANCE = "yfinance"
    FRED = "fred"


@dataclass(frozen=True)
class FxConfig:
    """Immutable configuration for multi-currency FX conversion.

    Parameters
    ----------
    base_currency : BaseCurrency
        Target currency for price conversion.
    mode : FxConversionMode
        ``NONE`` disables conversion (default, backward-compatible).
        ``TO_BASE`` converts all prices to the base currency.
        ``DECOMPOSE`` converts and also produces an FX return
        decomposition.
    data_source : FxDataSource
        Where to source FX rates.
    fill_limit : int
        Maximum number of trading days to forward-fill missing FX
        rates (weekends, holidays).
    require_full_coverage : bool
        If ``True``, raise ``DataError`` when any required FX pair
        has insufficient data.  If ``False``, log a warning and
        leave base-currency-denominated tickers unchanged.
    cross_via_usd : bool
        When ``True``, compute cross-rates via USD (e.g. GBP/EUR =
        GBP/USD / EUR/USD) rather than looking up direct pairs.
        yfinance direct cross pairs are often sparse.
    """

    base_currency: BaseCurrency = BaseCurrency.EUR
    mode: FxConversionMode = FxConversionMode.NONE
    data_source: FxDataSource = FxDataSource.YFINANCE
    fill_limit: int = 5
    require_full_coverage: bool = False
    cross_via_usd: bool = True

    @classmethod
    def for_eur_base(cls) -> FxConfig:
        """EUR base currency with conversion enabled."""
        return cls(base_currency=BaseCurrency.EUR, mode=FxConversionMode.TO_BASE)

    @classmethod
    def for_gbp_base(cls) -> FxConfig:
        """GBP base currency with conversion enabled."""
        return cls(base_currency=BaseCurrency.GBP, mode=FxConversionMode.TO_BASE)

    @classmethod
    def for_usd_base(cls) -> FxConfig:
        """USD base currency with conversion enabled."""
        return cls(base_currency=BaseCurrency.USD, mode=FxConversionMode.TO_BASE)

    @classmethod
    def for_decomposition(
        cls, base_currency: BaseCurrency = BaseCurrency.EUR
    ) -> FxConfig:
        """Conversion with full return decomposition."""
        return cls(base_currency=base_currency, mode=FxConversionMode.DECOMPOSE)
