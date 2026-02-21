"""Pipeline composition: pre-selection + optimiser → sklearn Pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.pre_selection._pipeline import build_preselection_pipeline


def build_portfolio_pipeline(
    optimizer: Any,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> Pipeline:
    """Compose a full sklearn Pipeline: pre-selection → optimiser.

    The resulting pipeline is a single estimator for cross-validation
    and hyperparameter tuning.  Pre-selection is performed *within*
    each CV fold, preventing data leakage.

    Parameters
    ----------
    optimizer : BaseOptimization
        A skfolio optimiser (e.g. from ``build_mean_risk()``,
        ``build_hrp()``, etc.) used as the final pipeline estimator.
    pre_selection_config : PreSelectionConfig or None
        Pre-selection configuration.  ``None`` uses default settings.
    sector_mapping : dict[str, str] or None
        Ticker → sector mapping for :class:`SectorImputer`.

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
    )

    # Flatten pre-selection steps + final optimiser into one pipeline
    # so that get_params() exposes all nested parameters for tuning.
    steps = [*preselection.steps, ("optimizer", optimizer)]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
