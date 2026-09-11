"""Factory function for assembling the pre-selection sklearn Pipeline."""

from __future__ import annotations

import datetime as dt
import logging

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
) -> Pipeline:
    """Build an sklearn Pipeline for data cleaning and asset pre-selection.

    The pipeline is assembled from *config* and follows this order::

        validate → outliers → impute → SelectComplete → DropZeroVariance
        → DropCorrelated → [SelectKExtremes] → [SelectNonDominated]
        → [SelectNonExpiring]

    Optional steps (in brackets) are only included when the corresponding
    config flag or parameter is set.

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
            "select_complete",
            SelectComplete(
                drop_assets_with_internal_nan=config.drop_internal_nan,
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
