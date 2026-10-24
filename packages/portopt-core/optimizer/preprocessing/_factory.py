"""Factory assembling the time-series return-cleaning pipeline."""

from __future__ import annotations

import pandas as pd
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
    outlier_protection_mask: pd.DataFrame | None = None,
) -> Pipeline:
    """Build a return-cleaning sklearn Pipeline from a config.

    Steps (each optional per config): validate (DataValidator),
    outliers (OutlierTreater), impute (Sector/Regression imputer).
    Run prices_to_returns upstream — that step changes data semantics
    and must stay outside the pipeline.

    The imputer fills interior gaps only. NaN before an asset's first or after
    its last observation (pre-listing, post-delisting) is left in place, so the
    pipeline never invents a return history; this includes a cell that the
    validate or outlier step blanks at the very edge of the window. Drop such
    assets downstream, for example with `build_preselection_pipeline`.

    Args:
        config: Serialisable step configuration. None uses defaults
            (validate + outlier-treat, no imputation).
        sector_mapping: Ticker -> sector label, injected into the
            sector/regression imputer. Non-serialisable, so passed here
            rather than stored on the config. None degrades sector
            imputation to a global cross-sectional mean.
        outlier_protection_mask: Boolean matrix (dates x tickers) forwarded to
            `OutlierTreater` as ``protected_mask``. Flags cells that are real
            economic events (e.g. a delisted asset's terminal return) so the
            outlier step neither removes nor winsorises them (a removed
            terminal return would be refilled by the imputer with a peer mean,
            turning a loss into a gain). Data-dependent, hence a factory
            keyword rather than a config field. None (default) protects
            nothing. Produced by `delisting_protection_mask`.

    Returns:
        Pipeline with at least one step.

    Raises:
        ValueError: If config disables every step (empty pipeline).
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
                    protected_mask=outlier_protection_mask,
                ),
            )
        )

    # Unlike the pre-selection pipeline there is no SelectComplete in front, so
    # the imputer must not touch the pre-listing and post-delisting NaN that a
    # delisted-inclusive panel carries on purpose.
    if cfg.imputer == ImputerStrategy.SECTOR:
        steps.append(
            (
                "impute",
                SectorImputer(sector_mapping=sector_mapping, fill_edges=False),
            )
        )
    elif cfg.imputer == ImputerStrategy.REGRESSION:
        steps.append(
            (
                "impute",
                RegressionImputer(
                    n_neighbors=cfg.n_neighbors,
                    min_train_periods=cfg.min_train_periods,
                    sector_mapping=sector_mapping,
                    fill_edges=False,
                ),
            )
        )

    if not steps:
        raise ValueError(
            "CleaningConfig disables every step; the cleaning pipeline is empty."
        )

    return Pipeline(steps)
