"""Factory function for building FX converter from config."""

from __future__ import annotations

import logging

import pandas as pd

from optimizer.exceptions import ConfigurationError
from optimizer.fx._config import FxConfig, FxConversionMode
from optimizer.fx._converter import FxPriceConverter

logger = logging.getLogger(__name__)


def build_fx_converter(
    config: FxConfig,
    *,
    fx_rates: pd.DataFrame | None,
    currency_map: dict[str, str] | None,
) -> FxPriceConverter:
    """Build a ready-to-use FxPriceConverter from config.

    ``FxConversionMode.NONE`` yields a pass-through converter (prices are only
    cast to float), whatever the other arguments are.  For every other mode the
    inputs are checked: with ``config.strict`` a missing or empty ``fx_rates`` /
    ``currency_map`` raises, and any rate currency a ticker needs but lacks
    raises at ``fit()`` (strict implies ``require_full_coverage``), so a strict
    converter never returns mixed-currency prices.  Without ``strict`` a missing
    ``currency_map`` is logged as a warning (every ticker is then taken to be
    in the base currency), and a rate currency a ticker needs but lacks is
    logged by the converter's ``fit()``; the affected tickers are left
    unconverted.  Empty ``fx_rates`` alone is not warned about here: it is
    legitimate when every ticker is quoted in the base currency.

    Args:
        config: FX conversion settings (mode, base currency, fill limit,
            coverage policy).
        fx_rates: Pre-loaded rate DataFrame (dates × currency columns).  Each
            column holds units-of-base per one unit-of-foreign.
        currency_map: Ticker-to-ISO-currency-code mapping used to look up each
            asset's denomination.

    Returns:
        Configured converter ready for ``fit()`` / ``transform()``.

    Raises:
        ConfigurationError: If ``config.strict`` is set, ``config.mode`` is not
            ``NONE`` and ``fx_rates`` or ``currency_map`` is ``None`` or empty.
    """
    base_currency = config.base_currency.value

    if config.mode is FxConversionMode.NONE:
        return FxPriceConverter(
            base_currency=base_currency,
            currency_map={},
            fx_rates=None,
            fill_limit=config.fill_limit,
        )

    if config.strict:
        missing = [
            name
            for name, value in (("fx_rates", fx_rates), ("currency_map", currency_map))
            if value is None or len(value) == 0
        ]
        if missing:
            raise ConfigurationError(
                f"FX conversion mode {config.mode.value!r} needs "
                f"{' and '.join(missing)}, but got none or empty."
            )
    elif currency_map is None or len(currency_map) == 0:
        # fit() flags foreign tickers that lack a rate, but with no currency_map
        # it sees none, so this gap would otherwise be silent.
        logger.warning(
            "FX conversion mode %r needs currency_map, but got none or empty. "
            "Every ticker is treated as base-currency and left unconverted.",
            config.mode.value,
        )

    return FxPriceConverter(
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
        fill_limit=config.fill_limit,
        require_full_coverage=config.require_full_coverage or config.strict,
    )
