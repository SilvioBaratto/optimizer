"""Configuration for multi-currency FX conversion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.exceptions import ConfigurationError


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

    Attributes:
        base_currency: Target currency for price conversion.
        mode: ``NONE`` disables conversion (default, backward-compatible).
            ``TO_BASE`` converts all prices to the base currency.
            ``DECOMPOSE`` converts and also produces an FX return decomposition.
        data_source: Where to source FX rates (read by the rate loader, not
            by ``build_fx_converter``).
        fill_limit: Maximum age, in business days, of an FX rate that may be
            carried forward over weekends and holidays.  ``0`` converts with
            same-day rates only.
        require_full_coverage: If ``True``, raise ``DataError`` when any
            required FX pair has insufficient data, or when a ticker has no
            known currency (absent from ``currency_map`` or mapped to
            ``None`` / NaN / a blank string). If ``False``, log a warning and
            leave the affected tickers unconverted: an FX pair with no data
            leaves its tickers in local currency, and a ticker with no known
            currency is valued as if quoted in the base major currency.
        cross_via_usd: When ``True``, compute cross-rates via USD
            (e.g. GBP/EUR = GBP/USD / EUR/USD) rather than looking up direct
            pairs. yfinance direct cross pairs are often sparse. Read by
            whoever loads the rates, not by ``build_fx_converter``, which
            receives them ready-made.
        strict: If ``True``, raise ``ConfigurationError`` when ``mode != NONE``
            but ``currency_map`` or ``fx_rates`` is ``None`` or empty, and
            treat missing FX coverage as an error (as if
            ``require_full_coverage`` were set). If ``False`` (default), log a
            ``WARNING`` and skip conversion for the affected tickers; empty
            ``fx_rates`` is accepted silently when every ticker is already
            quoted in the base currency.
    """

    base_currency: BaseCurrency = BaseCurrency.EUR
    mode: FxConversionMode = FxConversionMode.NONE
    data_source: FxDataSource = FxDataSource.YFINANCE
    fill_limit: int = 5
    require_full_coverage: bool = False
    cross_via_usd: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if self.fill_limit < 0:
            raise ConfigurationError(
                f"fill_limit must be non-negative, got {self.fill_limit}."
            )

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

    @classmethod
    def for_strict_conversion(
        cls, base_currency: BaseCurrency = BaseCurrency.EUR
    ) -> FxConfig:
        """Conversion enabled; raises ``ConfigurationError`` if inputs are missing."""
        return cls(
            base_currency=base_currency,
            mode=FxConversionMode.TO_BASE,
            strict=True,
        )
