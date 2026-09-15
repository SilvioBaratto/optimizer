"""Factory functions for assembling pre-selection / portfolio sklearn Pipelines."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectKExtremes,
    SelectNonDominated,
    SelectNonExpiring,
)
from sklearn.pipeline import Pipeline

from optimizer.pre_selection._config import PreSelectionConfig, SelectKMeasure
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._validation import DataValidator

logger = logging.getLogger(__name__)

# Serialisable SelectKMeasure -> concrete skfolio measure enum member.
_SELECT_K_MEASURE_MAP: dict[SelectKMeasure, object] = {
    SelectKMeasure.SHARPE_RATIO: RatioMeasure.SHARPE_RATIO,
    SelectKMeasure.SORTINO_RATIO: RatioMeasure.SORTINO_RATIO,
    SelectKMeasure.CALMAR_RATIO: RatioMeasure.CALMAR_RATIO,
    SelectKMeasure.MEAN: PerfMeasure.MEAN,
    SelectKMeasure.ANNUALIZED_MEAN: PerfMeasure.ANNUALIZED_MEAN,
    SelectKMeasure.VARIANCE: RiskMeasure.VARIANCE,
    SelectKMeasure.STANDARD_DEVIATION: RiskMeasure.STANDARD_DEVIATION,
    SelectKMeasure.SEMI_DEVIATION: RiskMeasure.SEMI_DEVIATION,
    SelectKMeasure.CVAR: RiskMeasure.CVAR,
    SelectKMeasure.MAX_DRAWDOWN: RiskMeasure.MAX_DRAWDOWN,
}


def build_preselection_pipeline(
    config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    expiration_dates: dict[str, dt.datetime] | None = None,
    outlier_protection_mask: pd.DataFrame | None = None,
) -> Pipeline:
    """Build an sklearn Pipeline for data cleaning and asset pre-selection.

    The pipeline is assembled from *config* and follows this order::

        validate → outliers → SelectComplete → impute → DropZeroVariance
        → DropCorrelated → [SelectKExtremes] → [SelectNonDominated]
        → [SelectNonExpiring]

    Optional steps (in brackets) are only included when the corresponding
    config flag or parameter is set.

    ``SelectComplete`` runs *before* ``impute`` on purpose: it drops assets
    whose history is too short (leading/trailing inception ``NaN``) so the
    imputer never fabricates a full price history for a late-listed asset and
    lets it survive selection.  With ``drop_assets_with_internal_nan=False``
    (default) interior gaps are *kept* by ``SelectComplete`` and filled by the
    imputer afterwards, so genuine full-history assets with a missing day are
    not over-dropped.

    All transformer hyper-parameters are accessible via
    ``pipeline.get_params()`` for cross-validation tuning (e.g.
    ``outliers__winsorize_threshold``).

    Parameters
    ----------
    config : PreSelectionConfig or None
        Pipeline configuration.  Defaults to ``PreSelectionConfig()``
        (sensible defaults for daily equity returns).
    sector_mapping : dict[str, str] or None
        Ticker → sector mapping forwarded to :class:`SectorImputer`.
        When ``None``, global cross-sectional mean imputation is used.
    expiration_dates : dict[str, datetime.datetime] or None
        Ticker → expiration date, forwarded to :class:`SelectNonExpiring`
        (non-serialisable, so a factory keyword rather than a config field).
        Only used when ``config.use_non_expiring`` is set with a positive
        ``expiration_lookahead``.  Without it, ``SelectNonExpiring`` has no
        expiry information and retains every asset.
    outlier_protection_mask : pd.DataFrame or None
        Boolean matrix (dates x tickers) forwarded to :class:`OutlierTreater`
        as ``protected_mask``.  Flags cells that are real economic events (e.g.
        a delisted asset's terminal return) so the outlier stage does not remove
        or winsorise them.  Data-dependent, hence a factory keyword rather than
        a config field.  ``None`` (default) protects nothing.  Produced by
        :func:`optimizer.preprocessing._delisting.delisting_protection_mask`.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if config is None:
        config = PreSelectionConfig()

    if config.outlier_method != "time_series":
        raise ValueError(
            f"Unsupported outlier_method {config.outlier_method!r}. "
            "Only 'time_series' is currently supported."
        )

    steps: list[tuple[str, object]] = [
        (
            "validate",
            DataValidator(max_abs_return=config.max_abs_return),
        ),
        (
            "outliers",
            OutlierTreater(
                winsorize_threshold=config.winsorize_threshold,
                remove_threshold=config.remove_threshold,
                protected_mask=outlier_protection_mask,
            ),
        ),
        (
            "select_complete",
            SelectComplete(
                drop_assets_with_internal_nan=config.drop_internal_nan,
            ),
        ),
        (
            "impute",
            SectorImputer(
                sector_mapping=sector_mapping,
                fallback_strategy=config.imputation_fallback,
            ),
        ),
        (
            "drop_zero_variance",
            DropZeroVariance(threshold=config.zero_variance_threshold),
        ),
        (
            "drop_correlated",
            DropCorrelated(
                threshold=config.correlation_threshold,
                absolute=config.correlation_absolute,
            ),
        ),
    ]

    if config.top_k is not None:
        steps.append(
            (
                "select_k",
                SelectKExtremes(
                    k=config.top_k,
                    measure=_SELECT_K_MEASURE_MAP[config.select_k_measure],
                    highest=config.top_k_highest,
                ),
            )
        )

    if config.use_pareto:
        steps.append(
            (
                "select_pareto",
                SelectNonDominated(
                    min_n_assets=config.pareto_min_assets,
                    threshold=config.pareto_threshold,
                ),
            )
        )

    if config.use_non_expiring and config.expiration_lookahead is not None:
        steps.append(
            (
                "select_non_expiring",
                SelectNonExpiring(
                    expiration_dates=expiration_dates,
                    expiration_lookahead=dt.timedelta(
                        days=config.expiration_lookahead,
                    ),
                ),
            )
        )

    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def build_portfolio_pipeline(
    optimizer: Any,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    expiration_dates: dict[str, dt.datetime] | None = None,
    outlier_protection_mask: pd.DataFrame | None = None,
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
    outlier_protection_mask : pd.DataFrame or None
        Boolean matrix (dates x tickers) forwarded to the pre-selection
        ``OutlierTreater`` so genuine economic events (e.g. delisting returns)
        are exempt from outlier removal/winsorisation.  ``None`` (default)
        protects nothing.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-ready pipeline whose ``fit(X)`` cleans and filters
        returns then optimises, and whose ``predict(X)`` produces
        a skfolio ``Portfolio``.

    Examples
    --------
    >>> from optimizer.optimization import MeanRiskConfig, build_mean_risk
    >>> from optimizer.pre_selection import build_portfolio_pipeline
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
        outlier_protection_mask=outlier_protection_mask,
    )

    # Flatten pre-selection steps + final optimiser into one pipeline
    # so that get_params() exposes all nested parameters for tuning.
    steps = [*preselection.steps, ("optimizer", optimizer)]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
