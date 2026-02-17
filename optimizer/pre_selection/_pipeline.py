"""Factory function for assembling the pre-selection sklearn Pipeline."""

from __future__ import annotations

from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectKExtremes,
    SelectNonDominated,
    SelectNonExpiring,
)
from sklearn.pipeline import Pipeline

from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._validation import DataValidator


def build_preselection_pipeline(
    config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
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

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if config is None:
        config = PreSelectionConfig()

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
        ("select_complete", SelectComplete()),
        ("drop_zero_variance", DropZeroVariance()),
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
                    highest=config.top_k_highest,
                ),
            )
        )

    if config.use_pareto:
        steps.append(
            (
                "select_pareto",
                SelectNonDominated(min_n_assets=config.pareto_min_assets),
            )
        )

    if config.use_non_expiring and config.expiration_lookahead is not None:
        from datetime import timedelta

        steps.append(
            (
                "select_non_expiring",
                SelectNonExpiring(
                    expiration_lookahead=timedelta(
                        days=config.expiration_lookahead,
                    ),
                ),
            )
        )

    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
