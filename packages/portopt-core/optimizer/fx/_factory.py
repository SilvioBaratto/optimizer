"""Factory function for building FX converter from config."""

from __future__ import annotations

import pandas as pd

from optimizer.fx._config import FxConfig
from optimizer.fx._converter import FxPriceConverter


def build_fx_converter(
    config: FxConfig,
    *,
    fx_rates: pd.DataFrame,
    currency_map: dict[str, str],
) -> FxPriceConverter:
    """Build a ready-to-use FxPriceConverter from config.

    Args:
        config: FX conversion settings (base currency, fill limit, coverage policy).
        fx_rates: Pre-loaded rate DataFrame (dates × currency columns).  Each
            column holds units-of-base per one unit-of-foreign.
        currency_map: Ticker-to-ISO-currency-code mapping used to look up each
            asset's denomination.

    Returns:
        Configured converter ready for ``fit()`` / ``transform()``.
    """
    return FxPriceConverter(
        base_currency=config.base_currency.value,
        currency_map=currency_map,
        fx_rates=fx_rates,
        fill_limit=config.fill_limit,
        require_full_coverage=config.require_full_coverage,
    )
