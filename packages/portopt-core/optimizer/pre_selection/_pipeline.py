"""Factory functions for assembling pre-selection / portfolio sklearn Pipelines."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from sklearn.pipeline import Pipeline

from optimizer.pre_selection._config import PreSelectionConfig, SelectKMeasure
from optimizer.pre_selection._fit_only import (
    FitOnlyDataValidator,
    FitOnlyOutlierTreater,
    FitOnlySectorImputer,
)
from optimizer.pre_selection._guarded import (
    GuardedDropCorrelated,
    GuardedDropZeroVariance,
    GuardedSelectKExtremes,
    GuardedSelectNonDominated,
    GuardedSelectNonExpiring,
)
from optimizer.pre_selection._select_complete import RawSelectComplete
from optimizer.pre_selection._strict_imputer import StrictSectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._validation import DataValidator

logger = logging.getLogger(__name__)

# Serialisable SelectKMeasure -> concrete skfolio measure enum member.
_SELECT_K_MEASURE_MAP: dict[
    SelectKMeasure, PerfMeasure | RatioMeasure | RiskMeasure
] = {
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

    The pipeline is assembled from `config` and follows this order::

        SelectComplete → validate → outliers → impute → DropZeroVariance
        → DropCorrelated → [SelectKExtremes] → [SelectNonDominated]
        → [SelectNonExpiring]

    Optional steps (in brackets) are only included when the corresponding
    config flag or parameter is set.

    ``SelectComplete`` runs first, on the raw frame, and before ``impute``, on
    purpose: it drops assets whose history is too short (leading/trailing
    inception ``NaN``) so the imputer never fabricates a full price history for
    a late-listed asset and lets it survive selection.  Running before
    ``validate`` / ``outliers`` matters too: those steps write ``NaN`` into
    flagged cells, and a flagged first or last observation (a genuine jump on
    the window edge) must not be mistaken for a late listing or an early expiry.
    With ``drop_assets_with_internal_nan=False`` (default) interior gaps are
    *kept* by ``SelectComplete`` and filled by the imputer afterwards, so genuine
    full-history assets with a missing day are not over-dropped.  With ``True``
    an asset is dropped for an interior gap present in the input, but not for a
    cell that ``validate`` / ``outliers`` blank afterwards: those are imputed.

    All transformer hyper-parameters are accessible via
    ``pipeline.get_params()`` for cross-validation tuning (e.g.
    ``outliers__winsorize_threshold``).

    Args:
        config: Pipeline configuration.  Defaults to ``PreSelectionConfig()``
            (sensible defaults for daily equity returns).
        sector_mapping: Ticker → sector mapping forwarded to
            `SectorImputer`.  When ``None``, global cross-sectional
            mean imputation is used.
        expiration_dates: Ticker → expiration date, forwarded to
            `SelectNonExpiring` (non-serialisable, so a factory keyword
            rather than a config field).  Required when
            ``config.use_non_expiring`` is set; assets absent from the mapping
            never expire, so an empty dict retains every asset.  Ignored when
            ``use_non_expiring`` is off.
        outlier_protection_mask: Boolean matrix (dates x tickers) forwarded to
            `OutlierTreater` as ``protected_mask``.  Flags cells that
            are real economic events (e.g. a delisted asset's terminal return)
            so the outlier stage does not remove or winsorise them.
            Data-dependent, hence a factory keyword rather than a config field.
            ``None`` (default) protects nothing.  Produced by
            `delisting_protection_mask`.

    Returns:
        Assembled sklearn Pipeline with output set to pandas.  Fitting on a
        universe that shrinks to one asset, or to none, is valid: the lone
        survivor is returned, and an empty selection yields a frame with no
        columns rather than an error.  Fitting raises `DataError` naming the
        dates on which no asset has an observation once cleaning has run,
        because the imputer has no peer to fill them from.

    Raises:
        ValueError: If ``config.outlier_method`` is unsupported, or if
            ``config.use_non_expiring`` is set without an
            ``expiration_lookahead`` or without ``expiration_dates``.
    """
    steps = _build_steps(
        config,
        sector_mapping,
        expiration_dates,
        outlier_protection_mask,
        fit_only_cleaning=False,
    )
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def _build_steps(
    config: PreSelectionConfig | None,
    sector_mapping: dict[str, str] | None,
    expiration_dates: dict[str, dt.datetime] | None,
    outlier_protection_mask: pd.DataFrame | None,
    *,
    fit_only_cleaning: bool,
) -> list[tuple[str, object]]:
    """Assemble the ordered pre-selection steps.

    ``fit_only_cleaning`` swaps the value-rewriting steps (validate, outliers,
    impute) for variants that clean during ``fit_transform`` only.  A pipeline
    that ends in an optimiser needs this so ``predict`` scores the held-out
    window on realised returns rather than on cleaned ones.
    """
    if config is None:
        config = PreSelectionConfig()

    if config.outlier_method != "time_series":
        raise ValueError(
            f"Unsupported outlier_method {config.outlier_method!r}. "
            "Only 'time_series' is currently supported."
        )

    validator_cls = FitOnlyDataValidator if fit_only_cleaning else DataValidator
    treater_cls = FitOnlyOutlierTreater if fit_only_cleaning else OutlierTreater
    imputer_cls = FitOnlySectorImputer if fit_only_cleaning else StrictSectorImputer

    steps: list[tuple[str, object]] = [
        (
            "select_complete",
            RawSelectComplete(
                drop_assets_with_internal_nan=config.drop_internal_nan,
            ),
        ),
        (
            "validate",
            validator_cls(max_abs_return=config.max_abs_return),
        ),
        (
            "outliers",
            treater_cls(
                winsorize_threshold=config.winsorize_threshold,
                remove_threshold=config.remove_threshold,
                protected_mask=outlier_protection_mask,
            ),
        ),
        (
            "impute",
            imputer_cls(
                sector_mapping=sector_mapping,
                fallback_strategy=config.imputation_fallback,
            ),
        ),
        (
            "drop_zero_variance",
            GuardedDropZeroVariance(threshold=config.zero_variance_threshold),
        ),
        (
            "drop_correlated",
            GuardedDropCorrelated(
                threshold=config.correlation_threshold,
                absolute=config.correlation_absolute,
            ),
        ),
    ]

    if config.top_k is not None:
        steps.append(
            (
                "select_k",
                GuardedSelectKExtremes(
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
                GuardedSelectNonDominated(
                    min_n_assets=config.pareto_min_assets,
                    threshold=config.pareto_threshold,
                ),
            )
        )

    if config.use_non_expiring:
        # Fail at build rather than skip the step: an expiry filter the caller
        # asked for but that cannot run would let expiring contracts through
        # unnoticed.
        if config.expiration_lookahead is None:
            raise ValueError(
                "use_non_expiring requires expiration_lookahead to be set "
                "(number of calendar days)."
            )
        if expiration_dates is None:
            raise ValueError(
                "use_non_expiring requires expiration_dates "
                "(pass an empty dict if no asset expires)."
            )
        steps.append(
            (
                "select_non_expiring",
                GuardedSelectNonExpiring(
                    expiration_dates=expiration_dates,
                    expiration_lookahead=dt.timedelta(
                        days=config.expiration_lookahead,
                    ),
                ),
            )
        )

    return steps


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

    Data cleaning (invalid-value removal, outlier winsorising, imputation)
    applies to the data the optimiser is *fitted* on.  ``predict`` / ``score``
    leave the held-out returns untouched apart from the column selection
    learned at fit, so the resulting ``Portfolio`` reports realised
    performance.  A missing or non-finite held-out observation cannot be scored
    and is valued at zero return.

    Args:
        optimizer: A skfolio optimiser (e.g. from ``build_mean_risk()``)
            used as the final pipeline estimator.
        pre_selection_config: Pre-selection configuration.  ``None`` uses
            default settings.
        sector_mapping: Ticker → sector mapping for `SectorImputer`.
        expiration_dates: Ticker → expiration date, forwarded to the
            pre-selection pipeline's ``SelectNonExpiring`` step
            (non-serialisable, hence a factory keyword rather than a config
            field).  Required when the pre-selection config sets
            ``use_non_expiring``; ignored otherwise.  Assets absent from the
            mapping never expire.
        outlier_protection_mask: Boolean matrix (dates x tickers) forwarded
            to the pre-selection ``OutlierTreater`` so genuine economic events
            (e.g. delisting returns) are exempt from outlier
            removal/winsorisation.  ``None`` (default) protects nothing.

    Returns:
        A fitted-ready pipeline whose ``fit(X)`` cleans and filters
        returns then optimises, and whose ``predict(X)`` produces
        a skfolio ``Portfolio``.

    Raises:
        ValueError: Under the same conditions as
            `build_preselection_pipeline`.

    Examples:
        >>> from optimizer.optimization import MeanRiskConfig, build_mean_risk
        >>> from optimizer.pre_selection import build_portfolio_pipeline
        >>> optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        >>> pipeline = build_portfolio_pipeline(optimizer)
        >>> pipeline.fit(X)
        >>> portfolio = pipeline.predict(X)
        >>> print(portfolio.sharpe_ratio)
    """
    preselection_steps = _build_steps(
        pre_selection_config,
        sector_mapping,
        expiration_dates,
        outlier_protection_mask,
        fit_only_cleaning=True,
    )

    # Flatten pre-selection steps + final optimiser into one pipeline
    # so that get_params() exposes all nested parameters for tuning.
    steps = [*preselection_steps, ("optimizer", optimizer)]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
