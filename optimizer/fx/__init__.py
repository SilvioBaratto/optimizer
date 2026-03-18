"""Multi-currency FX conversion and return decomposition.

Provides tools for converting local-currency prices to a single base
currency before feeding into the optimization pipeline, and for
decomposing total returns into stock and FX components.
"""

from optimizer.fx._config import BaseCurrency, FxConfig, FxConversionMode, FxDataSource
from optimizer.fx._converter import FxPriceConverter
from optimizer.fx._decomposition import FxReturnDecomposition, decompose_fx_returns
from optimizer.fx._factory import build_fx_converter
from optimizer.fx._rates import align_fx_rates, build_fx_pair_ticker

__all__ = [
    "BaseCurrency",
    "FxConfig",
    "FxConversionMode",
    "FxDataSource",
    "FxPriceConverter",
    "FxReturnDecomposition",
    "align_fx_rates",
    "build_fx_converter",
    "build_fx_pair_ticker",
    "decompose_fx_returns",
]
