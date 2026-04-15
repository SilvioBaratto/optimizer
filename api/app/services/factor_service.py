"""Re-export shim — kept for backward compatibility.

New code should import directly from the specific sub-modules:
  - app.services._factor_helpers
  - app.services.factor_compute_service
  - app.services.factor_scoring_service
  - app.services.factor_analysis_service
"""

__all__ = [
    "FactorDataError",
    "build_exposure_constraints_for_tickers",
    "compute_factor_scores",
    "compute_quintile_spread_for_tickers",
    "compute_regime_tilt",
    "run_factor_compute",
    "select_stocks_for_tickers",
    "validate_factors",
    "_load_fundamentals",
    "_build_factor_scores_dict",
    "_build_multiindex_factor_df",
    "_build_standardized_scores_df",
]

from app.services._factor_helpers import FactorDataError  # noqa: F401
from app.services.factor_analysis_service import (  # noqa: F401
    build_exposure_constraints_for_tickers,
    compute_quintile_spread_for_tickers,
    compute_regime_tilt,
    select_stocks_for_tickers,
)
from app.services.factor_compute_service import (  # noqa: F401
    _load_fundamentals,
    run_factor_compute,
)
from app.services.factor_scoring_service import (  # noqa: F401
    _build_factor_scores_dict,
    _build_multiindex_factor_df,
    _build_standardized_scores_df,
    compute_factor_scores,
    validate_factors,
)
