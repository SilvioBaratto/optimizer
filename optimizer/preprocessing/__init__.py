"""Custom sklearn-compatible preprocessing transformers.

Time-series transformers (``DataValidator``, ``OutlierTreater``,
``SectorImputer``, ``RegressionImputer``) operate per-asset across time
(axis=0). Cross-sectional transformers (``CS*``) operate per-period
across assets (axis=1). They are not interchangeable.
"""

from optimizer.preprocessing._cs_transformers import (
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)
from optimizer.preprocessing._delisting import apply_delisting_returns
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._regression_imputer import RegressionImputer
from optimizer.preprocessing._validation import DataValidator

__all__ = [
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSWinsorizer",
    "DataValidator",
    "OutlierTreater",
    "RegressionImputer",
    "SectorImputer",
    "apply_delisting_returns",
]
