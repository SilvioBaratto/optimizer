"""Distance estimator selection.

All distance estimators are sklearn-compatible — they expose ``.fit(X)``
and store a square ``distance_`` matrix.

Input contract: ``X`` is a clean, aligned **linear**-return DataFrame
(``(n_observations, n_assets)``, tickers as columns). Every skfolio distance
estimator validates ``X`` and **rejects NaN / non-finite values**; the
correlation family additionally needs equal-length columns. Ingestion data
(5y ragged history ⇒ NaN gaps and unequal-length series, ``Numeric`` ⇒
``Decimal``, mixed ``price_unit`` currency/scale) must be float-cast,
scale/currency-normalised and NaN-aligned upstream (e.g. ``SelectComplete``
pre-selection or ``DataFrame.dropna``) before fitting — this package is a pure
builder and cleans nothing itself.
"""

from optimizer.distance._config import (
    DistanceConfig,
    DistanceEstimatorType,
    NBinsMethod,
)
from optimizer.distance._factory import build_distance

__all__ = [
    "DistanceConfig",
    "DistanceEstimatorType",
    "NBinsMethod",
    "build_distance",
]
