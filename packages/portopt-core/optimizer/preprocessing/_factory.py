"""Factory assembling the time-series return-cleaning pipeline."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from optimizer.preprocessing._config import CleaningConfig, ImputerStrategy
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._regression_imputer import RegressionImputer
from optimizer.preprocessing._validation import DataValidator

__all__ = ["make_cleaning_pipeline"]


def make_cleaning_pipeline(
    config: CleaningConfig | None = None,
    *,
    sector_mapping: dict[str, str] | None = None,
) -> Pipeline:
    """Build a return-cleaning ``sklearn.pipeline.Pipeline`` from a config.

    Steps (each optional per ``config``): ``validate`` (DataValidator),
    ``outliers`` (OutlierTreater), ``impute`` (Sector/Regression imputer).
    The pipeline consumes and returns a return DataFrame — run
    ``prices_to_returns`` (see :func:`optimizer.preprocessing.to_returns`)
    upstream, since that changes data semantics and must stay outside the
    pipeline.

    Parameters
    ----------
    config : CleaningConfig or None, default=None
        Serialisable step configuration.  ``None`` uses defaults (validate +
        outlier-treat, no imputation).
    sector_mapping : dict[str, str] or None, default=None
        Ticker -> sector label, injected into the sector/regression imputer.
        Non-serialisable, so passed here rather than stored on the config.
        ``None`` degrades sector imputation to a global cross-sectional mean.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with at least one step.

    Raises
    ------
    ValueError
        If ``config`` disables every step (empty pipeline).
    """
    cfg = config or CleaningConfig()

    steps: list[tuple[str, object]] = []

    if cfg.validate:
        steps.append(("validate", DataValidator(max_abs_return=cfg.max_abs_return)))

    if cfg.treat_outliers:
        steps.append(
            (
                "outliers",
                OutlierTreater(
                    winsorize_threshold=cfg.winsorize_threshold,
                    remove_threshold=cfg.remove_threshold,
                ),
            )
        )

    if cfg.imputer == ImputerStrategy.SECTOR:
        steps.append(("impute", SectorImputer(sector_mapping=sector_mapping)))
    elif cfg.imputer == ImputerStrategy.REGRESSION:
        steps.append(
            (
                "impute",
                RegressionImputer(
                    n_neighbors=cfg.n_neighbors,
                    min_train_periods=cfg.min_train_periods,
                    sector_mapping=sector_mapping,
                ),
            )
        )

    if not steps:
        raise ValueError(
            "CleaningConfig disables every step; the cleaning pipeline is empty."
        )

    return Pipeline(steps)
