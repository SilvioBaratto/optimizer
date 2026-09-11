"""Custom sklearn-compatible preprocessing transformers.

Time-series transformers (``DataValidator``, ``OutlierTreater``,
``SectorImputer``, ``RegressionImputer``) operate per-asset across time
(axis=0). Cross-sectional transformers (``CS*``) operate per-period
across assets (axis=1). They are not interchangeable.

``prices_to_returns`` / :func:`to_returns` convert a price panel to linear
returns and run *outside* the pipeline (they change data semantics).
:func:`make_cleaning_pipeline` assembles the axis=0 transformers into a
single ``sklearn.pipeline.Pipeline``; :func:`make_cs_transformer` builds a
cross-sectional transformer from a serialisable config.
"""

from skfolio.preprocessing import prices_to_returns

from optimizer.preprocessing._config import CleaningConfig, ImputerStrategy
from optimizer.preprocessing._cs_transformers import (
    BaseCSTransformer,
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSTransformerConfig,
    CSTransformerType,
    CSWinsorizer,
    make_cs_transformer,
)
from optimizer.preprocessing._delisting import apply_delisting_returns
from optimizer.preprocessing._factory import make_cleaning_pipeline
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._regression_imputer import RegressionImputer
from optimizer.preprocessing._returns import JoinMethod, ReturnsConfig, to_returns
from optimizer.preprocessing._validation import DataValidator

__all__ = [
    "BaseCSTransformer",
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSTransformerConfig",
    "CSTransformerType",
    "CSWinsorizer",
    "CleaningConfig",
    "DataValidator",
    "ImputerStrategy",
    "JoinMethod",
    "OutlierTreater",
    "RegressionImputer",
    "ReturnsConfig",
    "SectorImputer",
    "apply_delisting_returns",
    "make_cleaning_pipeline",
    "make_cs_transformer",
    "prices_to_returns",
    "to_returns",
]
