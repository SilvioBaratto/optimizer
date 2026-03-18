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
    """Build a ready-to-use :class:`FxPriceConverter` from config.

    Parameters
    ----------
    config : FxConfig
        FX conversion configuration.
    fx_rates : pd.DataFrame
        Pre-loaded FX rate DataFrame (dates x currencies).  Each
        column holds units-of-base per one unit-of-foreign.
    currency_map : dict[str, str]
        Ticker → ISO currency code mapping.

    Returns
    -------
    FxPriceConverter
        Configured converter ready for ``fit()`` / ``transform()``.
    """
    return FxPriceConverter(
        base_currency=config.base_currency.value,
        currency_map=currency_map,
        fx_rates=fx_rates,
        fill_limit=config.fill_limit,
        require_full_coverage=config.require_full_coverage,
    )
