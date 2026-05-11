"""Factor validation utilities."""

from __future__ import annotations

import logging

import pandas as pd
from numpy.linalg import LinAlgError

from optimizer.factors._validation import FactorValidationReport, compute_vif

logger = logging.getLogger(__name__)


def validate_factors(
    factor_scores_history: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
    standardized: pd.DataFrame,
) -> FactorValidationReport:
    """Validate factor scores and compute diagnostic statistics.

    Parameters
    ----------
    factor_scores_history : dict[str, pd.DataFrame]
        Mapping of factor name → (dates × tickers) history of scores.
    returns_history : pd.DataFrame
        (dates × tickers) forward returns aligned with ``factor_scores_history``.
    standardized : pd.DataFrame
        Cross-sectional standardized factor matrix (tickers × factors) used
        for VIF computation.

    Returns
    -------
    FactorValidationReport
        Validation report.  ``vif_scores`` is ``None`` when VIF computation
        fails due to collinearity (``LinAlgError``) or invalid input
        (``ValueError``).  ``TypeError`` is propagated to the caller.
    """
    vif_scores: pd.Series | None = None

    try:
        vif_scores = compute_vif(standardized)
    except (LinAlgError, ValueError) as exc:
        logger.warning(
            "VIF computation failed — skipping multicollinearity diagnostics: %s",
            exc,
        )
        vif_scores = None
    # TypeError is intentionally not caught — let it propagate

    return FactorValidationReport(vif_scores=vif_scores)
