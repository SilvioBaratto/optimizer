"""Pipeline composition: pre-selection + optimiser → sklearn Pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sklearn.pipeline import Pipeline

from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.pre_selection._pipeline import build_preselection_pipeline

if TYPE_CHECKING:
    import datetime as dt

logger = logging.getLogger(__name__)


def build_portfolio_pipeline(
    optimizer: Any,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    expiration_dates: dict[str, dt.datetime] | None = None,
) -> Pipeline:
    """Compose a full sklearn Pipeline: pre-selection → optimiser.

    The resulting pipeline is a single estimator for cross-validation
    and hyperparameter tuning.  Pre-selection is performed *within*
    each CV fold, preventing data leakage.

    Parameters
    ----------
    optimizer : BaseOptimization
        A skfolio optimiser (e.g. from ``build_mean_risk()``)
        used as the final pipeline estimator.
    pre_selection_config : PreSelectionConfig or None
        Pre-selection configuration.  ``None`` uses default settings.
    sector_mapping : dict[str, str] or None
        Ticker → sector mapping for :class:`SectorImputer`.
    expiration_dates : dict[str, datetime.datetime] or None
        Ticker → expiration date, forwarded to the pre-selection pipeline's
        ``SelectNonExpiring`` step (non-serialisable, hence a factory keyword
        rather than a config field).  Only takes effect when the pre-selection
        config sets ``use_non_expiring`` with a positive
        ``expiration_lookahead``; ``None`` (default) retains every asset.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-ready pipeline whose ``fit(X)`` cleans and filters
        returns then optimises, and whose ``predict(X)`` produces
        a skfolio ``Portfolio``.

    Examples
    --------
    >>> from optimizer.optimization import MeanRiskConfig, build_mean_risk
    >>> from optimizer.pipeline import build_portfolio_pipeline
    >>> optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
    >>> pipeline = build_portfolio_pipeline(optimizer)
    >>> pipeline.fit(X)            # X = returns DataFrame
    >>> portfolio = pipeline.predict(X)
    >>> print(portfolio.sharpe_ratio)
    """
    preselection = build_preselection_pipeline(
        config=pre_selection_config,
        sector_mapping=sector_mapping,
        expiration_dates=expiration_dates,
    )

    # Flatten pre-selection steps + final optimiser into one pipeline
    # so that get_params() exposes all nested parameters for tuning.
    steps = [*preselection.steps, ("optimizer", optimizer)]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
